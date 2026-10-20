# Runbook: EgoSchema headline result through lmms-eval

This runbook reproduces the **Headline result** of `AutoGaze/Codec_Selector_Feasibility.md` with lmms-eval and this bridge instead of AutoGaze's own harness. That result is EgoSchema Subset, N=500, nvf=16, four modes, accuracy and latency per question.

Background on the bridge is in `../LMMS_EVAL_BRIDGE.md`. All paths below are under `/home/thannan/scratch/codec_reproduce/`, which is the same directory as `/home/scratch.thannan_wwfo/codec_reproduce/`.

## Rules for the session running this

- **Read-only directories:** `autogaze_lmms_eval/`, `AutoGaze/` and `lmms-eval/`. Run their scripts, but don't edit, install into, or git-modify them. If a code change looks necessary, stop and report it.
- **Where output goes:** every file the run writes goes to `runs/<tag>/`, and the launcher arranges that. The only exception is the download, which goes to `$HF_HOME/egoschema` and `$HF_HOME/hub`.
- **The user runs the prerequisites below,** because they write into `AutoGaze/`.
- **Results files contain the HF token.** lmms-eval writes `HF_TOKEN` into every `*_results.json`, under `config.resolved_cli_args.hf_hub_log_args`. Never paste that field anywhere, and scrub it before sharing.

## Target (from `Codec_Selector_Feasibility.md`)

| Mode | Accuracy | Tokens | Preproc (incl. selector) | of which selector | ViT | LLM | **E2E** |
|---|--:|--:|--:|--:|--:|--:|--:|
| codec_nvdec | 67.0% | 2,783 | 0.86 s | 0.09 s (0.84 s total) | 0.11 s | 0.35 s | **1.37 s** |
| codec | 66.0% | 2,783 | 1.01 s | 0.26 s (1.00 s total) | 0.11 s | 0.36 s | 1.54 s |
| dense | 64.6% | 24,690 | 0.47 s | — | 2.91 s | 0.69 s | 4.11 s |
| autogaze | 66.8% | 1,837 | 4.93 s | 4.07 s | 0.41 s | 0.42 s | 5.81 s |

## Differences from the headline run (by design)

| | Headline (AutoGaze harness) | This run |
|---|---|---|
| Prompt | `Question: …\nA. …\nAnswer with the letter of the correct option only.` | lmms-eval `egoschema_subset`: `{question}\n{options}\nAnswer with the option's letter from the given choices directly.` |
| Answer parser | First A–E letter; no letter counts as wrong | `parse_multi_choice_response`: if no letter is found, it matches option text, or else **picks a random letter** |
| `max_new_tokens` | 8 | 16 (the task doesn't set it, so the client's default is used) |
| AutoGaze code | `cfa11e0` | `0234e8e`. The only `scripts/breakdown` change since is `6ef6165`, which is opt-in and off here. |
| Scheduling | 1 GPU, modes run one after another | 4 GPUs, modes run in parallel |
| Videos | `zobwink/EgoSchema` originals | `lmms-lab/egoschema`, said to be byte-identical |

**Parallel runs share the node's CPUs.** Video decode, and the codec encode and decode, may therefore read slower than in the headline. ViT and LLM times should be close, because each mode has its own GPU.

**Codec caches don't skew EgoSchema.** It has one question per video, and each server starts with an empty cache, so every row is cold.

## Prerequisites (the user runs these, on a GB200 node)

```bash
eval "$(/home/scratch.thannan_wwfo/miniforge-aarch64/bin/conda shell.bash hook)"
conda activate auto_gaze
cd /home/thannan/scratch/codec_reproduce/AutoGaze
git submodule update --init scripts/hevc_dump          # gitlab-master over SSH
cmake -S native -B native/build_$(uname -m) -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build native/build_$(uname -m) -j 32 --target dump_cu codec_raster
nvidia-smi --query-gpu=driver_version --format=csv,noheader   # codec_nvdec needs >= 595.84.01
```

The launcher refuses codec modes until `native/build_aarch64/dump_cu` and `native/build_aarch64/libcodec_raster.so` exist. Without them the selector silently falls back to slower code paths.

**Alignment check:** run this after step 1 below, once the videos exist. It must print PASS:
```bash
mkdir -p AutoGaze/data/egoschema/videos
ln -s $HF_HOME/egoschema/videos/0074f737-11cb-497d-8d07-77c3a8127391.mp4 AutoGaze/data/egoschema/videos/
REPO_DIR=$PWD/AutoGaze FIXED_NUM_VIDEO_FRAMES=16 python AutoGaze/scripts/verify_gazing_alignment.py
```

## Steps

Start with a node that has 4 GPUs, for example `srun --partition=gb200nvl4 --gres=gpu:4 --time=08:00:00 --pty bash`. Use `tmux`, or run the launcher with `nohup … &` and follow `runs/<tag>/logs/`.

`L=/home/thannan/scratch/codec_reproduce/autogaze_lmms_eval/scripts/run_egoschema_headline.sh`. `HF_HOME` defaults to `/home/scratch.thannan_wwfo/hf_cache`. `HF_TOKEN` must be set and must have access to `lmms-lab/egoschema`.

1. **Download the dataset, once.**
   ```bash
   $L --download-only egoschema_download
   ```
   - What it does: starts one dense server and runs 1 question, so lmms-eval downloads the whole `lmms-lab/egoschema` repo and unzips it into `$HF_HOME/egoschema/videos`.
   - It is large. Check free space with `df -h /home/scratch.thannan_wwfo` first.
   - **Do not interrupt it, and do not create `$HF_HOME/egoschema` by hand.** If it gets interrupted, delete `$HF_HOME/egoschema` and re-run. The zips in `$HF_HOME/hub` are reused.

2. **Pick a warmup video that isn't one of the 500 Subset videos** (run in the `lmms-eval` env):
   ```bash
   python - <<'EOF'
   import os
   from datasets import load_dataset
   sub = {d["video_idx"] for d in load_dataset("lmms-lab/egoschema", "Subset", split="test")}
   vdir = os.path.join(os.environ.get("HF_HOME", "/home/scratch.thannan_wwfo/hf_cache"), "egoschema", "videos")
   print(len(sub), "subset ids;", next(os.path.join(vdir, f) for f in sorted(os.listdir(vdir)) if f.rsplit(".", 1)[0] not in sub))
   EOF
   ```
   It should print `500 subset ids; <path>`. Use that path as `W`.

3. **Run the prerequisite alignment check.** It must print PASS.

4. **Smoke test.** About 10 minutes, most of it model loading.
   ```bash
   $L --limit 8 --warmup-video $W egoschema_smoke
   python /home/thannan/scratch/codec_reproduce/autogaze_lmms_eval/scripts/headline_table.py runs/egoschema_smoke
   ```
   Check:
   - every mode reports `done`
   - each `runs/egoschema_smoke/timing_<mode>.jsonl` has 8 rows
   - replies are mostly bare letters
   - token counts are in the right range (dense about 25k, the others about 2–3k)

5. **Full run.** autogaze is the slowest mode, about 500 × 6 s ≈ 50 min, plus loading and warmup.
   ```bash
   $L --warmup-video $W egoschema_headline_$(date +%Y%m%d)
   ```

6. **Summarize.**
   ```bash
   python /home/thannan/scratch/codec_reproduce/autogaze_lmms_eval/scripts/headline_table.py runs/egoschema_headline_<date>
   ```
   This prints the run's table in the headline layout, the headline numbers with the differences, and a reply/prefill table.

7. **Write up.** Add a results section to `../LMMS_EVAL_BRIDGE.md` with:
   - the three tables
   - `runs/<tag>/run_info.txt` (AutoGaze commit, lmms-eval path and commit, node, driver, Slurm job)
   - the differences table above
   - anything that went wrong

## Launcher reference

`run_egoschema_headline.sh [--download-only] [--limit N] [--modes "…"] [--gpus "…"] [--warmup-video P] [--warmup-n N] [--base-port P] [--runs-dir D] [--repo-dir D] <tag>`

- **Servers:** mode *i* runs on GPU *i* (or the *i*-th entry of `--gpus`) on port `8765+i`, with `--nvf 16 --max-batch-size-autogaze 32` (the harness's setting) and `PYTORCH_ALLOC_CONF=expandable_segments:True`.
- **Output in `runs/<tag>/`:**
  - `run_info.txt`
  - `timing_<mode>.jsonl`, the per-question timing written by the server
  - `cache_<mode>/`, the codec caches
  - `lmms_<mode>/`, lmms-eval's results, samples and submission
  - `logs/server_<mode>.log` and `logs/lmms_<mode>.log`
- **Refuses to start when:**
  - `runs/<tag>` already exists (timing logs append, so never reuse a tag)
  - a port is already serving
  - not on aarch64
  - a codec mode is requested but the native tools are missing
  - it's a non-download run and `$HF_HOME/egoschema/videos` is missing
- **A mode counts as failed** if lmms-eval exits non-zero **or** writes no `*_results.json`. lmms-eval catches errors such as a server 500 and still exits 0. If a mode fails, read both of its logs.
- **On exit,** including Ctrl-C, the servers are killed.

## Sanity checks on the full run

- **Row counts:** 500 rows per timing log, and `n-samples.effective == 500` in each results JSON. `headline_table.py` exits with an error if samples and timing rows don't match one to one.
- **Tokens** should match the headline closely: dense ≈ 24,690, autogaze ≈ 1,837, codec and codec_nvdec ≈ 2,783. The prompt wording changes only a few tokens, so a large gap points to a pipeline problem (wrong nvf or wrong processor), not to the prompt.
- **Accuracy** should be within about ±2 points of the headline (1 standard error is about 2.1 at N=500). If the gap is larger, check the "No letter" column first, because those replies were scored by option-text matching or a random guess.
- **Latency:** ViT and LLM should be close to the headline. Higher preprocessing in the codec modes is expected from sharing CPUs. Report it; don't average it away.
