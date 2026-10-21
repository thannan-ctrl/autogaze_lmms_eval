"""HTTP server that answers video questions with AutoGaze's NVILA-HD-Video pipeline.

Run inside AutoGaze's env (e.g. conda env `auto_gaze`):

    python -m autogaze_lmms_eval.server --mode autogaze --nvf 16 --port 8765

One server = one model + one patch-selection mode. Requests are served one at a
time (single GPU, single model), in arrival order. This is also what keeps the
latency numbers clean: AutoGaze's timer (scripts/breakdown/timing.py) is a single
module-global dict, so concurrent requests would mix their timings. Do not swap
HTTPServer for ThreadingHTTPServer.

Endpoints
    GET  /health    -> {"status", "mode", "nvf", "model", "device", "repo_dir", "timing_log"}
    POST /generate  {"video_path", "prompt", "max_new_tokens"?, "do_sample"?, "meta"?}
                    -> {"text", "num_tokens", "timing": {...}}

The per-question path mirrors scripts/breakdown/runner.py:run_question, including
its two-phase timing (reset -> processor -> sync -> snapshot, then reset ->
generate -> sync -> snapshot) and the derived breakdown fields, so the numbers are
directly comparable to the harness's. Each request's timing is also appended to a
JSONL log together with the request's "meta" (lmms-eval task/doc_id), for joining
with lmms-eval's logged samples.

The processor is built with scripts/breakdown/processor.build, whose instrument()
call is required for correct output (SigLIP patch grid, dense keep-all position
fix, codec selector injection) and also installs the processor-side timers.
timing.install_model_hooks adds the ViT/LLM timers and, like the harness by
default, computes prefill logits for the last position only.

--model qwen3_5 / nemotron_omni serve Qwen3.5-9B / Nemotron-3-Nano-Omni through
AutoGaze's model_compat adapters instead (mc_backend.py; modes dense, codec_mv,
codec_mv_hybrid, codec_mv_keep). --model nvila (default) is the path above.
"""
import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REPO_DIR = "/home/thannan/scratch/codec_reproduce/AutoGaze"
# Runtime output goes next to the checkout, never inside it, so the checkout can
# stay read-only while it is being used.
DEFAULT_RUNS_DIR = os.path.join(os.path.dirname(_PKG_ROOT), "runs")
DEFAULT_CACHE_DIR = os.path.join(DEFAULT_RUNS_DIR, "cache")
DEFAULT_TIMING_DIR = os.path.join(DEFAULT_RUNS_DIR, "timing")
MODES = ("dense", "autogaze", "codec", "codec_nvdec", "codec_geo", "codec_ord", "autogaze_singlescale")
MODELS = ("nvila", "qwen3_5", "nemotron_omni")
CODEC_CACHE_ATTRS = ("CACHE_DIR", "NVDEC_CACHE_DIR", "HEVC_AUTOGAZE_CACHE_DIR")


class AutoGazeRunner:
    def __init__(self, repo_dir: str, mode: str, nvf: int, max_batch_size_autogaze: int, device: str,
                 cache_dir: str, timing_log: str, full_seq_logits: bool = False):
        # scripts/breakdown/config.py reads these at import time, so they must be
        # set before anything from the repo is imported.
        os.environ["REPO_DIR"] = repo_dir
        os.environ["FIXED_NUM_VIDEO_FRAMES"] = str(nvf)
        os.environ["MAX_BATCH_SIZE_AUTOGAZE"] = str(max_batch_size_autogaze)
        os.environ["NVILA_DEVICE"] = device
        # `breakdown` is imported as a top-level package from scripts/ (as
        # scripts/nvila_hd_accuracy_breakdown_test.py does); codec_selector also
        # needs the repo root for `from scripts import hevc_to_gaze`.
        for p in (os.path.join(repo_dir, "scripts"), repo_dir):
            if p not in sys.path:
                sys.path.insert(0, p)

        import torch
        from transformers import AutoModel

        from breakdown import codec_selector, config, instrumentation, processor, timing

        # Keep codec encode/decode caches out of the AutoGaze checkout.
        for attr in CODEC_CACHE_ATTRS:
            setattr(codec_selector, attr, os.path.join(cache_dir, attr.lower()))

        self.torch = torch
        self.codec_selector = codec_selector
        self.instrumentation = instrumentation
        self.timing = timing
        self.mode = mode
        self.nvf = nvf
        self.device = device
        self.repo_dir = repo_dir
        self.model_path = config.MODEL_PATH
        self.timing_log = timing_log
        self.request_idx = 0
        os.makedirs(os.path.dirname(os.path.abspath(timing_log)), exist_ok=True)

        print(f"[server] building {mode} processor (nvf={nvf})", flush=True)
        self.proc = processor.build(mode, nvf)
        print(f"[server] loading {config.MODEL_PATH} on {device}", flush=True)
        self.model = AutoModel.from_pretrained(
            config.MODEL_PATH, trust_remote_code=True, dtype=torch.bfloat16, max_batch_size_siglip=32,
        ).to(device)
        self.model.eval()
        self.llm_call_state = timing.install_model_hooks(self.model, full_seq_logits=full_seq_logits)

    def info(self) -> dict:
        return {
            "status": "ok", "mode": self.mode, "nvf": self.nvf, "model": self.model_path,
            "device": self.device, "repo_dir": self.repo_dir, "timing_log": self.timing_log,
        }

    def _sync(self) -> None:
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def generate(self, video_path: str, prompt: str, max_new_tokens: int = 16, do_sample: bool = False) -> dict:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"video not found on the server's filesystem: {video_path}")
        torch, timing = self.torch, self.timing
        text = f"{self.proc.tokenizer.video_token}\n\n{prompt}"
        if self.mode in self.codec_selector.BACKEND_FOR_MODE:
            self.instrumentation.set_codec_video_context(
                video_path, backend=self.codec_selector.BACKEND_FOR_MODE[self.mode]
            )

        # Phase 1: preprocessing (video decode, image prep, patch selection).
        timing.reset()
        self.instrumentation.reset_last_gazing_state()
        t0 = time.time()
        inputs = self.proc(text=text, videos=video_path, return_tensors="pt")
        gazing = self.instrumentation.get_last_num_gazing_each_frame_tiles()
        gazing_per_frame_pos = gazing.float().mean(dim=0).tolist() if gazing is not None else None
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        num_tokens = inputs["input_ids"].shape[1]
        self._sync()
        preproc_ms = (time.time() - t0) * 1000
        pre = timing.snapshot()

        # Phase 2: model.generate (ViT + LLM prefill/decode).
        timing.reset()
        self.llm_call_state["calls_since_reset"] = 0
        gen_t0 = time.time()
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
        self._sync()
        gen_ms = (time.time() - gen_t0) * 1000
        gen = timing.snapshot()

        response = self.proc.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()
        e2e_ms = (time.time() - t0) * 1000
        return {
            "text": response,
            "num_tokens": int(num_tokens),
            "timing": self._breakdown(pre, gen, preproc_ms, gen_ms, e2e_ms, gazing_per_frame_pos),
        }

    @staticmethod
    def _breakdown(pre: dict, gen: dict, preproc_ms: float, gen_ms: float, e2e_ms: float,
                   gazing_per_frame_pos) -> dict:
        """Same derived fields as scripts/breakdown/runner.py:run_question."""
        decode_ms = pre["decode_ms"]
        image_preproc_ms = pre["preprocess_videos_total_ms"] - pre["autogaze_transform_ms"]
        autogaze_ops_ms = pre["autogaze_transform_ms"] + max(pre["gazing_info_total_ms"] - pre["autogaze_model_ms"], 0.0)
        autogaze_model_ms = pre["autogaze_model_ms"]
        other_ms = max(preproc_ms - decode_ms - image_preproc_ms - autogaze_ops_ms - autogaze_model_ms, 0.0)
        selector_glue_ms = max(
            pre["gazing_info_total_ms"] - pre["codec_encode_ms"] - pre["codec_decode_ms"]
            - pre["selector_score_ms"] - pre["selector_rank_ms"],
            0.0,
        )
        return {
            "preproc_ms": preproc_ms,
            "decode_ms": decode_ms,
            "image_preproc_ms": image_preproc_ms,
            "autogaze_ops_ms": autogaze_ops_ms,
            "autogaze_model_ms": autogaze_model_ms,
            "codec_encode_ms": pre["codec_encode_ms"],
            "codec_decode_ms": pre["codec_decode_ms"],
            "selector_score_ms": pre["selector_score_ms"],
            "selector_rank_ms": pre["selector_rank_ms"],
            "selector_csvparse_ms": pre["selector_csvparse_ms"],
            "selector_scorecu_ms": pre["selector_scorecu_ms"],
            "selector_paintmap_ms": pre["selector_paintmap_ms"],
            "gazing_info_total_ms": pre["gazing_info_total_ms"],
            "selector_glue_ms": selector_glue_ms,
            "selector_wall_ms": pre["selector_wall_ms"],
            "selector_wait_ms": pre["selector_wait_ms"],
            "other_ms": other_ms,
            "cpu_ms": decode_ms + image_preproc_ms + autogaze_ops_ms + other_ms,
            "gpu_ms": autogaze_model_ms,
            "generate_ms": gen_ms,
            "vit_ms": gen["vit_ms"],
            "llm_prefill_ms": gen["llm_prefill_ms"],
            "llm_decode_ms": gen["llm_decode_ms"],
            "llm_ms": gen["llm_prefill_ms"] + gen["llm_decode_ms"],
            "llm_calls": gen["llm_calls"],
            "e2e_ms": e2e_ms,
            "gazing_per_frame_pos": gazing_per_frame_pos,
        }

    def log(self, req: dict, out: dict) -> None:
        record = {
            "request_idx": self.request_idx, "time": time.time(), "mode": self.mode, "nvf": self.nvf,
            "meta": req.get("meta"), "video_path": req["video_path"],
            "text": out["text"], "num_tokens": out["num_tokens"], **out["timing"],
        }
        with open(self.timing_log, "a") as f:
            f.write(json.dumps(record) + "\n")
        self.request_idx += 1

    def warmup(self, video_path: str, n: int) -> None:
        """Untimed, unlogged requests before serving, so the first logged request
        doesn't pay CUDA context/cuDNN autotune/JIT costs. Like runner.warmup,
        codec caches are redirected to a throwaway dir meanwhile so warmup doesn't
        pre-populate the cache for questions measured later."""
        saved = {a: getattr(self.codec_selector, a) for a in CODEC_CACHE_ATTRS}
        with tempfile.TemporaryDirectory() as tmp:
            for a in CODEC_CACHE_ATTRS:
                setattr(self.codec_selector, a, os.path.join(tmp, a))
            try:
                for i in range(n):
                    self.generate(video_path, "Describe the video.", max_new_tokens=8)
                    print(f"[server] warmup {i + 1}/{n} done", flush=True)
            finally:
                for a, v in saved.items():
                    setattr(self.codec_selector, a, v)


def make_handler(runner):  # AutoGazeRunner or mc_backend.MCRunner
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, runner.info())
            else:
                self._send(404, {"error": f"unknown path {self.path}"})

        def do_POST(self):
            if self.path != "/generate":
                self._send(404, {"error": f"unknown path {self.path}"})
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                # The bridge's own clock around the whole answer (all model-side
                # work, untimed file reads included), independent of the timers in
                # AutoGaze / model_compat that fill out["timing"]["e2e_ms"].
                # generate() returns decoded text, so the GPU work has finished.
                t0 = time.perf_counter()
                out = runner.generate(
                    video_path=req["video_path"],
                    prompt=req["prompt"],
                    max_new_tokens=int(req.get("max_new_tokens", 16)),
                    do_sample=bool(req.get("do_sample", False)),
                )
                out["timing"]["bridge_server_ms"] = (time.perf_counter() - t0) * 1000
                runner.log(req, out)
                self._send(200, out)
            except Exception as e:
                traceback.print_exc()
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            print(f"[server] {self.address_string()} {fmt % args}", flush=True)

    return Handler


def main() -> None:
    from autogaze_lmms_eval import mc_backend

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="nvila", choices=MODELS)
    ap.add_argument("--repo-dir", default=DEFAULT_REPO_DIR)
    ap.add_argument("--mc-root", default=None,
                    help="non-nvila models: checkout holding model_compat (default: --repo-dir)")
    ap.add_argument("--model-path", default=None,
                    help="non-nvila models: local weights dir or hub id (default: mc_backend.DEFAULT_MODEL_PATH)")
    ap.add_argument("--mode", default="autogaze", choices=MODES + tuple(m for m in mc_backend.ARM_FOR_MODE if m not in MODES))
    ap.add_argument("--nvf", type=int, default=16, help="number of video frames (FIXED_NUM_VIDEO_FRAMES)")
    ap.add_argument("--max-batch-size-autogaze", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="codec encode/decode cache directory")
    ap.add_argument("--timing-log", default=None,
                    help="per-request timing JSONL, appended to (default: timing/<mode>_nvf<nvf>.jsonl)")
    ap.add_argument("--full-seq-logits", action="store_true",
                    help="compute prefill logits for every position (harness FULL_SEQ_LOGITS=1)")
    ap.add_argument("--warmup-video", default=None, help="video to run untimed warmup requests on before serving")
    ap.add_argument("--warmup-n", type=int, default=2)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    timing_log = args.timing_log or os.path.join(DEFAULT_TIMING_DIR, f"{args.mode}_nvf{args.nvf}.jsonl")
    if args.model == "nvila":
        if args.mode not in MODES:
            ap.error(f"--mode {args.mode} needs a non-nvila --model")
        runner = AutoGazeRunner(
            args.repo_dir, args.mode, args.nvf, args.max_batch_size_autogaze, args.device, args.cache_dir,
            timing_log, full_seq_logits=args.full_seq_logits,
        )
    else:
        runner = mc_backend.MCRunner(
            args.model, args.mc_root, args.repo_dir, args.mode, args.nvf, args.device, args.cache_dir,
            timing_log, model_path=args.model_path,
        )
    if args.warmup_video:
        runner.warmup(args.warmup_video, args.warmup_n)
    server = HTTPServer((args.host, args.port), make_handler(runner))
    print(f"[server] ready on http://{args.host}:{args.port} ({runner.info()})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
