"""Server backend for the Qwen3.5-9B, Nemotron-3-Nano-Omni-30B-A3B and
Nemotron-3.5-Super-VL-120B-A12B adapters of AutoGaze's model_compat package
(branch pareto/main, the code behind AutoGaze/RESULTS_LATENCY_ACCURACY.md).

model_compat is imported read-only from the AutoGaze checkout (--repo-dir, at
commit 6ef7314 of pareto/main, or --mc-root if given); nothing in it is edited.
6ef7314 adds the Super VL adapter and multi-GPU timers on top of 9899fd6; the
other changes are off unless an MC_* variable turns them on, so Qwen3.5 and Omni
run as before. Super VL is split over every visible GPU (device_map=auto), so
its server gets all of them. Each request runs model_compat.runner._one, the same
per-question path as `python -m model_compat.runner`, with the pareto "best"
settings of model_compat/slurm/run_pareto.sbatch (v3mv H.264 motion-vector
selector, selector/preprocessing overlap, parallel frame decode). Two things are
changed at runtime from here, both so lmms-eval stays the judge:
  * the prompt: model_compat.data.build_prompt returns lmms-eval's prompt text
    (item["prompt"]) instead of model_compat's own question/options format;
  * max_new_tokens: taken from each request instead of the adapter's constant 8.

Modes (names only, since lmms-eval model_args cannot carry `@k=v`) map to
model_compat arm specs, see ARM_FOR_MODE.
"""
import importlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time

ARM_FOR_MODE = {
    "dense": "dense",
    "codec_mv": "codec",                                         # default keep 20% / 6%
    "codec_mv_hybrid": "codec@first=1.0@chunk=16@rest=0.25",     # recommended strict-L1 point
    "codec_mv_hybrid8": "codec@first=1.0@chunk=8@rest=0.12",     # Super VL's recommended strict-L1 point
    "codec_mv_keep": "codec@first=0.6@rest=0.25",
}
MODELS = ("qwen3_5", "nemotron_omni", "nemotron_super")
DEFAULT_MODEL_PATH = {
    "qwen3_5": "Qwen/Qwen3.5-9B",
    "nemotron_omni": "/home/scratch.thannan_wwfo/model_compat/weights/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
    "nemotron_super": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
}

# run_pareto.sbatch, SETTING=best. setdefault: an explicitly exported value wins
# (and is printed at startup, so a non-default operating point is visible).
COMMON_ENV = {
    "MC_SELECTOR": "v3mv", "MC_SEL_V3_SKIPNONREF": "1", "MC_OVERLAP": "1", "MC_FAST_DECODE": "1",
    "MC_FULL": "1", "MC_NO_PROBE": "1",
}
MODEL_ENV = {
    "qwen3_5": {"MC_QWEN_PROC_DEVICE": "cuda", "MC_QWEN_MAX_PIXELS": "4294967296"},
    # 675 patches = EgoSchema's native 480x360 (the stock 1024 rule upsamples).
    "nemotron_omni": {"MC_MOE": "grouped", "MC_LLM_ATTN": "sdpa", "MC_OMNI_PATCHES": "675"},
    # The Super VL sweep behind RESULTS_LATENCY_ACCURACY.md (RESULTS_LOG.md, "Super VL (4x H100)"):
    # PATCHES=675, FPS=0 (fixed frame count, not the model's 2 fps default), cuDNN attention off.
    "nemotron_super": {"MC_LLM_ATTN": "sdpa", "MC_SUPER_PATCHES": "675", "MC_FPS": "0", "MC_NO_CUDNN_SDPA": "1"},
}
PATH_ENV = {"qwen3_5": "MC_QWEN_PATH", "nemotron_omni": "MC_OMNI_PATH", "nemotron_super": "MC_SUPER_PATH"}
ARM_ENV_KEYS = ("MC_VIT_CTX", "MC_QWEN_VIT_CTX", "MC_COARSE")


def _resolve_model_path(path: str) -> str:
    """A local dir is used as is; a hub id is resolved from $HF_HOME without downloading."""
    if os.path.isdir(path):
        return path
    from huggingface_hub import snapshot_download
    return snapshot_download(path, local_files_only=True)


class MCRunner:
    def __init__(self, model: str, mc_root: str, repo_dir: str, mode: str, nvf: int, device: str,
                 cache_dir: str, timing_log: str, model_path: str = None):
        mc_root = mc_root or repo_dir
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}, expected one of {MODELS}")
        if mode not in ARM_FOR_MODE:
            raise ValueError(f"mode {mode!r} is not available for {model}; expected one of {sorted(ARM_FOR_MODE)}")
        if not os.path.isdir(os.path.join(mc_root, "model_compat")):
            raise FileNotFoundError(f"no model_compat package under {mc_root}: check out AutoGaze commit 6ef7314 "
                                    "(branch pareto/main) or pass --mc-root")

        # model_compat reads these at import time, so set them before importing it.
        for k, v in {**COMMON_ENV, **MODEL_ENV[model]}.items():
            os.environ.setdefault(k, v)
        os.environ.setdefault("MC_NATIVE_BUILD", os.path.join(repo_dir, "native", f"build_{platform.machine()}"))
        self.model_path = _resolve_model_path(model_path or DEFAULT_MODEL_PATH[model])
        os.environ[PATH_ENV[model]] = self.model_path
        if mc_root not in sys.path:
            sys.path.insert(0, mc_root)
        print("[server] model_compat env: " + " ".join(
            f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith("MC_")), flush=True)

        import torch

        from model_compat import data, runner
        # lmms-eval builds the prompt and parses the reply; the adapters call
        # data.build_prompt(item) through the module, so this reaches both.
        data.build_prompt = lambda item: item["prompt"]

        self.torch = torch
        self.runner = runner
        self.adapter_mod = importlib.import_module(f"model_compat.adapters.{model}")
        self.model = model
        self.mode = mode
        self.arm = ARM_FOR_MODE[mode]
        self.nvf = nvf
        self.device = device
        self.mc_root = mc_root
        self.cache_dir = cache_dir
        self.timing_log = timing_log
        self.request_idx = 0
        os.makedirs(os.path.dirname(os.path.abspath(timing_log)), exist_ok=True)

        # Per-arm adapter-level variant env, as runner.main sets it for each arm.
        for k in ARM_ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update(runner.parse_arm(self.arm)[3])

        print(f"[server] loading {model} from {self.model_path} on {device} (mode={mode}, arm={self.arm}, nvf={nvf})",
              flush=True)
        self.adapter = self.adapter_mod.Adapter()
        self.adapter.num_frames = nvf  # the Qwen processor caps pixels per video, so unit_grid depends on it
        self.adapter.load(device)

    def info(self) -> dict:
        return {
            "status": "ok", "mode": self.mode, "arm": self.arm, "nvf": self.nvf, "model": self.model_path,
            "device": self.device, "repo_dir": self.mc_root, "timing_log": self.timing_log,
        }

    def generate(self, video_path: str, prompt: str, max_new_tokens: int = 16, do_sample: bool = False) -> dict:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"video not found on the server's filesystem: {video_path}")
        if do_sample:
            raise ValueError("model_compat adapters decode greedily only (do_sample=True requested)")
        # The adapters read MAX_NEW_TOKENS from their module at generate time.
        self.adapter_mod.MAX_NEW_TOKENS = int(max_new_tokens)
        # Outside model_compat's e2e_ms, like runner.main with MC_FULL=1, but timed
        # here so the bridge's end-to-end cross-check can account for them.
        t = time.perf_counter()
        self.runner._preread(video_path)
        preread_ms = (time.perf_counter() - t) * 1000
        item = {"video_path": video_path, "prompt": prompt}
        try:
            out, ms, frac, _ = self.runner._one(self.adapter, item, self.arm, self.nvf, self.cache_dir)
        finally:
            # Every question pays its own (cold) HEVC fallback encode, as in runner.main.
            t = time.perf_counter()
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            os.makedirs(self.cache_dir, exist_ok=True)
            cleanup_ms = (time.perf_counter() - t) * 1000
        timing = self._breakdown(out, ms, frac)
        timing.update(preread_ms=preread_ms, cleanup_ms=cleanup_ms)
        return {"text": out["response"].strip(), "num_tokens": int(out["seq_len"]), "timing": timing}

    @staticmethod
    def _breakdown(out: dict, ms: dict, frac: float) -> dict:
        """Map model_compat's per-item record onto the fields headline_table.py reads.

        preproc_ms: everything before the ViT (frame decode, selector, HF processor,
        device copy). In codec modes the processor runs while the selector finishes
        (MC_OVERLAP=1), so selector_wall_ms includes it, and selector_wait_ms is the
        part of the selector not hidden behind the processor.
        """
        selector_ms = ms.get("selector_ms", 0.0)
        overlapped = out.get("preprocess_overlapped_ms") or 0.0
        vit, prefill, decode = out["vit_ms"], out["llm_prefill_ms"], out["llm_decode_ms"]
        rec = {
            "preproc_ms": ms.get("decode_ms", 0.0) + selector_ms + out["preprocess_ms"],
            "decode_ms": ms.get("decode_ms", 0.0),
            "selector_wall_ms": selector_ms if "selector_ms" in ms else None,
            "selector_wait_ms": max(selector_ms - overlapped, 0.0) if "selector_ms" in ms else None,
            "image_preproc_ms": out["preprocess_ms"] + overlapped,
            "vit_ms": vit,
            "llm_prefill_ms": prefill,
            "llm_decode_ms": decode,
            "llm_ms": prefill + decode,
            "llm_calls": out["llm_calls"],
            "e2e_ms": ms["e2e_ms"],
            "n_visual_tokens": out["n_visual_tokens"],
            "kept_fraction": frac,
        }
        # Selector internals (mv_fallback_hevc_samples, mp_* timings, activity), kept for audits.
        rec.update({k: v for k, v in ms.items() if k not in rec and k not in ("selector_ms",)})
        return rec

    def log(self, req: dict, out: dict) -> None:
        record = {
            "request_idx": self.request_idx, "time": time.time(), "mode": self.mode, "arm": self.arm,
            "model": self.model, "nvf": self.nvf, "meta": req.get("meta"), "video_path": req["video_path"],
            "text": out["text"], "num_tokens": out["num_tokens"], **out["timing"],
        }
        with open(self.timing_log, "a") as f:
            f.write(json.dumps(record) + "\n")
        self.request_idx += 1

    def warmup(self, video_path: str, n: int) -> None:
        """Untimed, unlogged requests before serving (CUDA context, kernels, and the
        selector's spawn process pool, which starts on first use). The HEVC cache
        goes to a throwaway dir meanwhile."""
        saved = self.cache_dir
        with tempfile.TemporaryDirectory() as tmp:
            self.cache_dir = tmp
            try:
                for i in range(n):
                    self.generate(video_path, "Describe the video.", max_new_tokens=8)
                    print(f"[server] warmup {i + 1}/{n} done", flush=True)
            finally:
                self.cache_dir = saved
