Reproduce the "Headline result" table of AutoGaze/Codec_Selector_Feasibility.md (EgoSchema Subset, N=500, nvf=16, accuracy + per-question latency) using lmms-eval through the autogaze_lmms_eval bridge, for modes dense / autogaze / codec only.

Modes: run only `dense autogaze codec`. Skip codec_nvdec everywhere. Pass `--modes "dense autogaze codec"` to the launcher (GPUs default to 0 1 2), run `tools/verify_alignment.sh dense autogaze codec`, and pass `dense autogaze codec` as the mode arguments to headline_table.py (`headline_table.py runs/<tag> dense autogaze codec`), so codec_nvdec rows are absent from the tables and the write-up. Where the runbook says "4 modes" or "every mode", read it as these three.

Follow autogaze_lmms_eval/EGOSCHEMA_HEADLINE.md step by step. Read it fully first, plus LMMS_EVAL_BRIDGE.md §3–§6 for background.

Ground rules
- AutoGaze/, lmms-eval/, autogaze_lmms_eval/ and .claude/ are read-only. Run their scripts, but never edit, pip install, chmod, or git-modify them. That includes AutoGaze/native/ and AutoGaze/scripts/hevc_dump/: never build, rebuild or re-run cmake there. If a code change or rebuild looks necessary, stop and tell me what and why instead of working around it.
- Only use the lmms-eval path through the launcher. Don't run AutoGaze's own harness (scripts/breakdown/main.py, runner.py, nvila_hd_accuracy_breakdown_test.py or similar), and don't make numbers any other way.
- All run output goes to runs/<tag>/ (the launcher handles this). The alignment check writes its caches to runs/cache/verify_<timestamp>/.
- runs/ and lmms_results/ are read-only for you too. Only the launcher, tools/verify_alignment.sh and lmms-eval write there. Read anything, but never create, edit, move, rename, truncate or delete files in them yourself (no Edit/Write, no shell redirects, rm, mv, sed -i, cleanup of failed runs). If a run is broken, start a new one under a fresh tag and tell me. Put your own scratch files in your session scratchpad, nowhere under codec_reproduce/.
- tools/, REPRO_SPEC.md and this file are read-only. Run tools/verify_alignment.sh, but never edit tools/ and never run tools/build_autogaze_native.sh.
- Input data is read-only: $HF_HOME (/home/scratch.thannan_wwfo/hf_cache, including egoschema/, hub/ and the datasets cache). Only the launcher's download writes there. Never delete, move or hand-create anything in it. The runbook says to delete $HF_HOME/egoschema after an interrupted download; don't do that, stop and tell me instead.
- The conda installs (/home/scratch.thannan_wwfo/miniforge-aarch64) are read-only: no pip/conda install, uninstall or upgrade in any env. Use the envs the scripts activate.
- The only file you may write under codec_reproduce/ is LMMS_EVAL_BRIDGE.md, and only to append the results section at the end. Don't change its existing text.
- HF_TOKEN must already be set. Check it with `[ -n "$HF_TOKEN" ] && echo set`, never by echoing it. Don't set it or write it anywhere. If it's unset, stop and tell me.
- lmms-eval writes my HF_TOKEN into every *_results.json (config.resolved_cli_args.hf_hub_log_args). Never print, quote or copy that field.
- Before you start, check that you are on an aarch64 GB200 node with at least 3 GPUs that mounts /home/scratch.thannan_wwfo: `uname -m`, `nvidia-smi -L`, and record the driver version. Only the Santa Clara GB200 nodes (partition gb200nvl72_preprod, or gb200nvl72 with --constraint=santa_clara; account blackwell) mount this scratch. The Hillsboro gb200nvl4 nodes don't. Their driver is 580.x, which is fine because codec_nvdec (which needs >= 595.84.01) is not run. If you aren't on such a node, stop and tell me. In each new shell, activate the aarch64 conda: eval "$(/home/scratch.thannan_wwfo/miniforge-aarch64/bin/conda shell.bash hook)".

Order (stop and report at the first failure; don't retry blindly)
1. Check the codec tools I built beforehand: AutoGaze/native/build_aarch64/dump_cu must be executable, AutoGaze/native/build_aarch64/libcodec_raster.so must exist, and `ldd AutoGaze/native/build_aarch64/dump_cu` must show no "not found". If anything is missing, stop and tell me. Don't build it.
2. Download: launcher --download-only. First check free space with `df -h /home/scratch.thannan_wwfo`. If $HF_HOME/egoschema/videos already exists, skip the download and say so. The download is long and must not be interrupted. Run it in the background and wait for it to finish.
3. Pick the warmup video (runbook step 2).
4. tools/verify_alignment.sh dense autogaze codec. Each of these modes must print PASS.
5. Smoke test with --limit 8 --modes "dense autogaze codec". Check the runbook's smoke criteria and show me the headline_table.py output. Then continue to the full run unless something looks wrong (e.g. tokens far from dense ≈ 25k / others ≈ 2–3k, mostly non-letter replies, a failed mode).
6. Full run with --modes "dense autogaze codec", in the background. It takes about an hour, with autogaze the slowest. Watch runs/<tag>/logs/ for server errors, and don't poll more often than every few minutes.
7. Run headline_table.py on the full run (with the modes dense autogaze codec) and go through the runbook's sanity checks.

What to give me at the end
- The three tables from headline_table.py.
- run_info.txt, without anything secret.
- For each of the three modes, the gap to the headline in accuracy and E2E, and your explanation. Separate prompt/parser effects, CPU sharing from the parallel modes, and anything that looks like a real pipeline difference.
- Anything that failed or looked off, with the log lines.
- Then append a short results section to LMMS_EVAL_BRIDGE.md (it is writable) with the tables, run_info, the runbook's differences table, and your notes. Say there that codec_nvdec was skipped on purpose (no 595.84+ driver on the nodes that mount scratch).
