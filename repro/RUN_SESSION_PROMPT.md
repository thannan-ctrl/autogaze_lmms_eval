Run the EgoSchema Subset benchmark (N=500) through lmms-eval and the autogaze_lmms_eval bridge for three models, following autogaze_lmms_eval/README.md (steps 6–8). Read the README fully first, plus LMMS_EVAL_BRIDGE.md §3–§6 for background.

Runs, in this order, each as a smoke test (--limit 8) and then a full run:
1. NVILA (README steps 6–7): `--modes "dense autogaze codec"`, after `tools/verify_alignment.sh dense autogaze codec` prints PASS for each mode.
2. Qwen3.5-9B (README step 8): `SERVER_ENV=autogaze_mc`, `--model qwen3_5 --model-path $QWEN --nvf 128 --modes "dense codec_mv codec_mv_hybrid"`.
3. Nemotron-3-Nano-Omni (README step 8): `SERVER_ENV=autogaze_mc`, `--model nemotron_omni --model-path $OMNI --nvf 64 --modes "dense codec_mv codec_mv_hybrid"`.
Always pass `--modes` and `--warmup-video $W`. Use a new tag for every launch. Set the README step 1 variables (with `SERVER_ENV=autogaze_av19` for run 1) and, for runs 2–3, `export MC_PYTHONPATH=/home/scratch.thannan_wwfo/miniforge-aarch64/envs/autogaze_mc/lib/python3.12/site-packages`.

Results are not yours to report
- The launcher writes the result table itself: at the end of every run it writes runs/<tag>/RESULTS.md (accuracy, the latency breakdown, and an end-to-end latency cross-check against the bridge's own clocks), writes runs/<tag>/SHA256SUMS, and makes runs/<tag>/ read-only. I read the results there myself.
- Don't open RESULTS.md, the timing_*.jsonl / client_timing_*.jsonl logs or lmms-eval's results and samples files, and don't run headline_table.py yourself. Never quote, summarize, compare or interpret accuracy, token or latency numbers, and never write results anywhere.
- To decide whether to continue after a smoke test, use only the launcher's exit code and the `[status]` lines it prints at the end (also in runs/<tag>/logs/headline_table.log). Continue only if every mode reads `timing ok, client log ok, e2e cross-check ok, replies ok`; otherwise stop and tell me which check failed. Server and lmms-eval logs (runs/<tag>/logs/) are fine to read for errors.

Ground rules
- Everything under /home/scratch.thannan_wwfo/codec_reproduce is read-only for you: code (AutoGaze/, lmms-eval/, autogaze_lmms_eval/, tools/, envs/, reference/), docs, .claude/, and results (runs/, lmms_results/). Run the scripts; never edit, create, move, delete, chmod or git-modify anything there. Only the launcher, tools/verify_alignment.sh and lmms-eval write, and only into runs/<new tag>/. Never build, rebuild or run cmake in AutoGaze/native or AutoGaze/scripts/hevc_dump. If a code change, rebuild or install looks necessary, stop and tell me what and why instead of working around it.
- Input data is read-only: $HF_HOME (/home/scratch.thannan_wwfo/hf_cache: egoschema/, hub/ with the model weights, datasets cache). Only the launcher's --download-only writes there. Never delete, move or hand-create anything in it; don't run hf download. If something is missing, stop and tell me.
- /home/scratch.thannan_wwfo/model_compat and /home/scratch.thannan_wwfo/AutoGaze are read-only too.
- The conda installs (/home/scratch.thannan_wwfo/miniforge-aarch64) are read-only: no pip/conda install, uninstall or upgrade in any env.
- Only use the lmms-eval path through the launcher. Don't run AutoGaze's own harnesses (scripts/breakdown/*, model_compat.runner, nvila_hd_accuracy_breakdown_test.py or similar), and don't make numbers any other way.
- If a run breaks, don't clean it up; start a new one under a fresh tag and tell me the old tag. Put your own scratch files in your session scratchpad, nowhere under codec_reproduce/.
- HF_TOKEN must already be set. Check it with `[ -n "$HF_TOKEN" ] && echo set`, never by echoing it. Don't set it or write it anywhere. lmms-eval writes it into every *_results.json (config.resolved_cli_args.hf_hub_log_args): never print, quote or copy that field.
- Before you start, check that you are on an aarch64 GB200 node with at least 3 GPUs that mounts /home/scratch.thannan_wwfo: `uname -m`, `nvidia-smi -L`, and record the driver version. Only the Santa Clara GB200 nodes (partition gb200nvl72_preprod, or gb200nvl72 with --constraint=santa_clara; account blackwell) mount this scratch. If you aren't on such a node, stop and tell me. In each new shell, activate the aarch64 conda: eval "$(/home/scratch.thannan_wwfo/miniforge-aarch64/bin/conda shell.bash hook)".

Order (stop and report at the first failure; don't retry blindly)
1. Check the codec tools: AutoGaze/native/build_aarch64/dump_cu is executable, AutoGaze/native/build_aarch64/libcodec_raster.so exists, and `ldd AutoGaze/native/build_aarch64/dump_cu` shows no "not found". Check `git -C AutoGaze rev-parse HEAD` is 9899fd6829444fdd5ebc86366fd35073816ac635 and the autogaze_mc env passes the README step 3 check (`True`). Check $QWEN, $OMNI and $HF_HOME/egoschema/videos exist. Don't fix anything; stop and tell me.
2. Run 1, smoke then full (the full run takes about an hour), in the background. Watch runs/<tag>/logs/ for server errors; don't poll more often than every few minutes.
3. Run 2, smoke then full.
4. Run 3, smoke then full.

What to give me at the end, and nothing more
- For each launch: the tag, the exact launcher command, whether it exited 0, and the path to its RESULTS.md.
- run_info.txt of each full run, without anything secret.
- Anything that failed or looked off, with the relevant log lines (no result numbers).
