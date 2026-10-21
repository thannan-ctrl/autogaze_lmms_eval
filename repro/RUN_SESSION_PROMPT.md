Run lmms-eval's EgoSchema Subset (500 questions) through the autogaze_lmms_eval bridge on H100 for Qwen3.5-9B and Nemotron-3-Nano-Omni, end to end, and report back. Read autogaze_lmms_eval/README.md ("Optional: let an AI agent run it" and "H100: Qwen3.5 and Omni") first.

What to run
- One Slurm job per model. Each job does everything itself: preflight checks, a smoke run (8 questions), an automatic gate, the full run, SUMMARY.md, checksums, and then makes runs/<tag>/ read-only.
- Submit both jobs (they can run at the same time), each with a new tag:
  sbatch /home/scratch.thannan_wwfo/codec_reproduce/autogaze_lmms_eval/scripts/run_eval.sbatch --model qwen3_5 --dataset egoschema qwen3_5_egoschema_<YYYYMMDD_HHMM>
  sbatch /home/scratch.thannan_wwfo/codec_reproduce/autogaze_lmms_eval/scripts/run_eval.sbatch --model nemotron_omni --dataset egoschema nemotron_omni_egoschema_<YYYYMMDD_HHMM>
- Write each call exactly in that form, as the whole Bash call: the literal absolute path, no `sbatch` options before it, no variable as the command, no `VAR=value` prefix, no `cd`, `&&`, `;`, pipes, `$(...)` or redirects. Only that form runs outside your sandbox; anything else cannot write and will fail.
- Watch with `squeue -u $USER` and `sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed`, and by reading the job log runs/_slurm/run_eval_<jobid>.log and runs/<tag>/preflight.txt, runs/<tag>/{smoke,full}/logs/. Don't poll more often than every 5 minutes; a full job takes up to about 2 hours, mostly queue and model load.

Results are not yours to report
- Never open, grep, copy or parse SUMMARY.md, RESULTS.md, lmms-eval *_results.json / *samples_*.jsonl, or timing_*.jsonl / client_timing_*.jsonl. Never quote, summarize or compare accuracy, token or latency numbers. I read SUMMARY.md myself.
- Decide success only from the job state, exit code and the `[status]` / `[run_eval]` lines in the job log.

When something fails
- If preflight fails, report which check failed (preflight.txt lists them) and stop. Don't work around it.
- If a job fails for a reason that looks transient (node failure, NODE_FAIL/PREEMPTED, CUDA init error, out of memory at model load, time limit), resubmit it once with a new tag and tell me the old tag. Don't resubmit a second time.
- For anything that would need a code, environment, data or settings change, stop and tell me what and why. Never try to fix it.
- Never delete or clean up a failed run; it stays as evidence.

Ground rules (enforced by your sandbox and permission rules; don't test them)
- Everything under /home/scratch.thannan_wwfo/codec_reproduce is read-only for you: code, docs, .claude/, and results (runs/). /home/scratch.thannan_wwfo/{hf_cache,anaconda3,miniforge-aarch64,model_compat,AutoGaze} are read-only too. No pip/conda installs, no git state changes, no builds.
- No Slurm commands besides that sbatch form, squeue and sacct: no srun, salloc, scancel or scontrol.
- HF_TOKEN is already in your environment and is passed to the job; never print, echo or write it. lmms-eval writes it into *_results.json, which you don't open anyway.
- Put your own scratch files in your session scratchpad only.

What to give me at the end, and nothing more
- For each job: tag, Slurm job id, the exact sbatch command, final state and exit code, and the path runs/<tag>/SUMMARY.md.
- The contents of each runs/<tag>/preflight.txt.
- Anything that failed or looked off, with the relevant log lines (no result numbers), and any tags you abandoned.
