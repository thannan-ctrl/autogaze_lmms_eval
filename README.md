# autogaze_lmms_eval

A bridge that lets [lmms-eval](../lmms-eval) evaluate the AutoGaze NVILA-8B-HD-Video pipeline (in [AutoGaze](../AutoGaze)) without the two sharing a Python env.

```
lmms-eval ──(--model autogaze)──► client.py ──HTTP POST /generate──► server.py ──► AutoGaze
  (its own env)                                                       (auto_gaze env, GPU)
```

- **`server.py`** loads the model and the patch-selection processor (`dense`, `autogaze`, `codec`, `codec_nvdec`, …) once. It answers `{video_path, prompt}` requests using the same code path and per-stage timing as `AutoGaze/scripts/breakdown/runner.py`.
- **`client.py`** is an lmms-eval model plugin, registered through the `lmms_eval.models` entry point. It forwards each question's video path and lmms-eval's own task prompt to the server.

Neither repo is modified, and nothing is written inside this checkout either: codec caches and timing logs default to `../runs/cache/` and `../runs/timing/` (next to this checkout), or wherever `--cache-dir` and `--timing-log` point.

- **`scripts/run_egoschema_headline.sh`** and **`scripts/headline_table.py`** run and summarize the EgoSchema reproduction of `AutoGaze/Codec_Selector_Feasibility.md`'s headline table. The runbook is [`EGOSCHEMA_HEADLINE.md`](EGOSCHEMA_HEADLINE.md).

## Setup

```bash
# lmms-eval env (client side): register --model autogaze. Must run on the same
# node (filesystem) as the server.
conda activate lmms-eval
pip install -e /home/thannan/scratch/codec_reproduce/autogaze_lmms_eval
python -c "from lmms_eval.models import get_model; print(get_model('autogaze'))"
```

The AutoGaze env (server side) needs no install: run the server with `PYTHONPATH` pointing at this checkout, as below. On the GB200 nodes, activate the aarch64 conda first: `eval "$(/home/scratch.thannan_wwfo/miniforge-aarch64/bin/conda shell.bash hook)"`.

## Run

```bash
# 1. start the server (one mode per server; one GPU each)
conda activate auto_gaze
PYTHONPATH=/home/thannan/scratch/codec_reproduce/autogaze_lmms_eval CUDA_VISIBLE_DEVICES=0 \
  python -m autogaze_lmms_eval.server --mode autogaze --nvf 16 --port 8765

# 2. run lmms-eval against it (same machine/filesystem)
export HF_HOME=/home/scratch.thannan_wwfo/hf_cache   # Video-MME videos land in $HF_HOME/videomme/data/
python -m lmms_eval --model autogaze \
  --model_args server_url=http://127.0.0.1:8765,mode=autogaze \
  --tasks videomme --batch_size 1 --limit 8 \
  --log_samples --output_path /home/thannan/scratch/codec_reproduce/lmms_results
```

Drop `--limit 8` for the full 2,700 questions. To run several modes in parallel, start one server per GPU on different ports (for example `--mode codec_nvdec --port 8766`).

`mode=` in `--model_args` is optional. When it is given, the client refuses to run if the server is serving a different mode.

## Differences from AutoGaze's own harness

- **Prompt and parser:** lmms-eval's Video-MME prompt and answer parser are used, not `dataset.build_prompt` / `parse_letter`. Accuracy is therefore comparable to lmms-eval leaderboards, not exactly to `AutoGaze/QUICKSTART.md`.
- **Generation:** `max_new_tokens` comes from the task (16 for Video-MME, vs 8 in the harness). Decoding is greedy (`do_sample=False`).

## Latency

The server times every request with AutoGaze's own timers (`scripts/breakdown/timing.py`), in the same two phases as `runner.run_question`:
1. **Preprocessing:** video decode, image prep, AutoGaze or codec patch selection.
2. **`model.generate`:** ViT, LLM prefill and LLM decode.

The fields are the same as the harness's (`decode_ms`, `image_preproc_ms`, `autogaze_ops_ms`, `autogaze_model_ms`, `codec_*`, `vit_ms`, `llm_prefill_ms`, `llm_decode_ms`, `e2e_ms`, …). Each `/generate` reply carries them under `timing`. They are also appended to `../runs/timing/<mode>_nvf<nvf>.jsonl` (or `--timing-log`), with lmms-eval's `{task, split, doc_id}` under `meta` so rows join with `--log_samples` output.

- **Warmup:** `--warmup-video <some.mp4> --warmup-n 2` runs untimed, unlogged requests first. Codec caches are redirected to a temp directory during warmup, as the harness does. Without warmup, drop the first rows (`request_idx`) yourself.
- **Codec caches:** a cache hit skips the encode, so time codec modes with an empty `--cache-dir`.
- **Logits:** as in the harness, prefill computes logits for the last position only. `--full-seq-logits` turns that off.
- **The log is appended to:** a restarted server adds rows to the same file, with `request_idx` starting again at 0.
