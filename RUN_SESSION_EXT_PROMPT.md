Run the first wave of EXTENSION_SPEC.md runs through the autogaze_lmms_eval bridge, and the CPU part of gate 1a, then report back. Read autogaze_lmms_eval/EXTENSION_SPEC.md sections 5-7 and EXTENSION_TODO.md first, and the usage header of autogaze_lmms_eval/scripts/run_ext.sbatch.

What to submit (each job does preflight, a 20-question smoke stage with an automatic gate, the full run, SUMMARY.md, checksums, then makes runs/<tag>/ read-only):

Qwen3.5-9B, H100 (3 jobs, all at once):
  sbatch /home/scratch.thannan_wwfo/codec_reproduce/autogaze_lmms_eval/scripts/run_ext.sbatch --model qwen3_5 --task egoschema_subset --arms main ext_qwen3_5_egoschema_main_20261029
  sbatch /home/scratch.thannan_wwfo/codec_reproduce/autogaze_lmms_eval/scripts/run_ext.sbatch --model qwen3_5 --task lvbench --arms main --shard 0/2 ext_qwen3_5_lvbench_main_s0of2_20261029
  sbatch /home/scratch.thannan_wwfo/codec_reproduce/autogaze_lmms_eval/scripts/run_ext.sbatch --model qwen3_5 --task lvbench --arms main --shard 1/2 ext_qwen3_5_lvbench_main_s1of2_20261029

NVILA, GB200: gate 1b first, then the 7 NExT-QA shards.
  sbatch /home/scratch.thannan_wwfo/codec_reproduce/autogaze_lmms_eval/scripts/run_ext_gb200.sbatch --model nvila --task egoschema_subset --arms gate ext_nvila_egoschema_gate_20261029
Then the NExT-QA main grid in 7 shards (shard I of 7 = questions I*1224 onward, the last runs to the end). Submit shards 0-2
together with the gate (4 GB200), and each further shard only when one of your GB200 jobs has finished, so at most
4 GB200 jobs are queued or running at any time:
  sbatch /home/scratch.thannan_wwfo/codec_reproduce/autogaze_lmms_eval/scripts/run_ext_gb200.sbatch --model nvila --task nextqa_mc_test --arms main --shard 0/7 ext_nvila_nextqa_main_s0of7_20261029
  sbatch /home/scratch.thannan_wwfo/codec_reproduce/autogaze_lmms_eval/scripts/run_ext_gb200.sbatch --model nvila --task nextqa_mc_test --arms main --shard 1/7 ext_nvila_nextqa_main_s1of7_20261029
  sbatch /home/scratch.thannan_wwfo/codec_reproduce/autogaze_lmms_eval/scripts/run_ext_gb200.sbatch --model nvila --task nextqa_mc_test --arms main --shard 2/7 ext_nvila_nextqa_main_s2of7_20261029
  sbatch /home/scratch.thannan_wwfo/codec_reproduce/autogaze_lmms_eval/scripts/run_ext_gb200.sbatch --model nvila --task nextqa_mc_test --arms main --shard 3/7 ext_nvila_nextqa_main_s3of7_20261029
  sbatch /home/scratch.thannan_wwfo/codec_reproduce/autogaze_lmms_eval/scripts/run_ext_gb200.sbatch --model nvila --task nextqa_mc_test --arms main --shard 4/7 ext_nvila_nextqa_main_s4of7_20261029
  sbatch /home/scratch.thannan_wwfo/codec_reproduce/autogaze_lmms_eval/scripts/run_ext_gb200.sbatch --model nvila --task nextqa_mc_test --arms main --shard 5/7 ext_nvila_nextqa_main_s5of7_20261029
  sbatch /home/scratch.thannan_wwfo/codec_reproduce/autogaze_lmms_eval/scripts/run_ext_gb200.sbatch --model nvila --task nextqa_mc_test --arms main --shard 6/7 ext_nvila_nextqa_main_s6of7_20261029
Do NOT submit any other NVILA main runs (egoschema_subset, lvbench). I judge gate 1b from its SUMMARY.md myself and
will tell you whether NVILA continues. If I tell you NVILA stops, submit no further NExT-QA shards.

Not in this wave: Qwen nextqa_mc_test (its smoke gate fails by design of the default prompt; decision pending), mvbench and videomme (not downloaded).

How to submit and watch
- Write each sbatch call exactly as above, as the whole Bash call: literal absolute path, no sbatch options before it, no variables, no `VAR=value` prefix, no `cd`, `&&`, `;`, pipes, `$(...)` or redirects. Only that form runs outside your sandbox.
- Slurm read commands (`squeue --me`, `sacct -j <ids> -X --format=JobID,State,ExitCode,Elapsed`) also only work as bare single commands. Never wrap them in loops or scripts.
- Watch by reading runs/_slurm/run_ext_<jobid>.log, runs/<tag>/preflight.txt and runs/<tag>/{smoke,full}/<arm>/logs/. Don't check more often than every 10 minutes. A full EgoSchema job takes about 1-2 h, an LVBench shard several hours; the 8 h limit applies.
- Success = job COMPLETED, exit 0, `[run_ext] ... done:` in the job log and every `[status]` line ending in "replies ok".
- At most 4 GPUs of each type in use at once (this wave: 3 H100; GB200 up to 4 at a time over 8 jobs).

Gate 1a, EgoSchema half (CPU, on this node, no Slurm), in your session scratchpad $S:
  mkdir -p $S/repro_pin $S/selcache
  git -C /home/scratch.thannan_wwfo/codec_reproduce/AutoGaze archive repro-pin model_compat scripts | tar -x -C $S/repro_pin
  export MC_SEL_V2_TMP=$S/selcache HF_HOME=/home/scratch.thannan_wwfo/hf_cache HF_HUB_OFFLINE=1
  PY=/home/scratch.thannan_wwfo/anaconda3/envs/nemo-omni/bin/python
  SI=/home/scratch.thannan_wwfo/codec_reproduce/autogaze_lmms_eval/scripts/selection_identity.py
  $PY $SI dump --mc-root $S/repro_pin --video-dir /home/scratch.thannan_wwfo/hf_cache/egoschema/videos --n 20 --out $S/pin_egoschema.jsonl
  $PY $SI dump --mc-root /home/scratch.thannan_wwfo/codec_reproduce/AutoGaze --video-dir /home/scratch.thannan_wwfo/hf_cache/egoschema/videos --n 20 --out $S/ext_egoschema.jsonl
  $PY $SI compare $S/pin_egoschema.jsonl $S/ext_egoschema.jsonl
Each dump takes a few minutes. `git archive` only reads the repo; it changes no git state. Report the compare output verbatim (PASS/FAIL line and any DIFF lines). The Video-MME half waits for the Video-MME download.

Results are not yours to report
- Never open, grep, copy or parse SUMMARY.md, RESULTS.md, lmms-eval *_results.json / *samples_*.jsonl, or timing_*.jsonl / client_timing_*.jsonl. Never quote, summarize or compare accuracy, token or latency numbers. I read SUMMARY.md myself.

When something fails
- If preflight fails, report which check failed (preflight.txt lists them) and stop for that job. Don't work around it.
- If a job fails for a reason that looks transient (NODE_FAIL/PREEMPTED, CUDA init error, OOM at model load, time limit), resubmit it once with the same command and a new tag (suffix _r1) and tell me the old tag. Never a second time.
- A failed smoke gate ("[run_ext] ERROR: smoke ... gate failed") is not transient: report the `[status]` line and stop for that job.
- For anything that needs a code, environment, data or settings change, stop and tell me what and why. Never fix it.
- Never delete or clean up a failed run; it stays as evidence.

Ground rules (enforced by your sandbox and permissions; don't test them)
- Everything under /home/scratch.thannan_wwfo/codec_reproduce is read-only for you, as are hf_cache, anaconda3, miniforge-aarch64, model_compat and AutoGaze. No pip/conda installs, no git state changes, no builds.
- No Slurm commands besides those sbatch forms, squeue and sacct: no srun, salloc, scancel, scontrol.
- HF_TOKEN is already in your environment; never print, echo or write it.
- Your own files go in your session scratchpad only.

What to give me at the end, and nothing more
- For each job (11 in total: 3 Qwen, the gate, 7 NExT-QA shards): tag, job id, the exact sbatch command, final state and exit code, path runs/<tag>/SUMMARY.md.
- The first two lines of each runs/<tag>/preflight.txt (settings and shard) and any FAIL lines.
- Gate 1a compare output.
- Anything that failed or looked off, with the relevant log lines (no result numbers), and any abandoned tags.
