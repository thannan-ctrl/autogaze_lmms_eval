# Agent session modes: dev and run

A Claude Code session in `$WORK=/home/scratch.thannan_wwfo/codec_reproduce` runs in one of
two modes. The rules are the same in both except for the bridge (`autogaze_lmms_eval/`) and Slurm.

| | **dev** (develop, debug, smoke runs) | **run** (the runs whose results you read) |
|---|---|---|
| Bridge `autogaze_lmms_eval/` | writable; Claude may edit, `git add`, `git commit` (no push, amend, reset) | read-only (file mode and sandbox) |
| Slurm | `sbatch` (any options), `srun`, `sinfo`, `scontrol show`, `scancel`, `squeue`, `sacct` | only bare `sbatch <bridge>/scripts/run_eval.sbatch …` or `run_eval_super.sbatch …`, `squeue`, `sacct` |
| Tags | `dev_` prefix, usually `--smoke-only` | `<model>_egoschema_<YYYYMMDD_HHMM>`, full run |
| Rest of `$WORK` (AutoGaze, lmms-eval, tools, envs, docs, `.claude/`, `runs/`) | read-only | read-only |
| Results (`SUMMARY.md`, `RESULTS.md`, samples, timing logs) | never read by Claude | never read by Claude |

Claude may submit Slurm jobs only in these forms. Pushing is always yours.

## Files

| File | What it is |
|---|---|
| `$WORK/.claude/settings.local.json` | shared base rules, loaded automatically in both modes |
| `$WORK/.claude/settings.dev.json` | dev overlay, passed with `--settings` |
| `$WORK/.claude/settings.run.json` | run overlay, passed with `--settings` |
| `$WORK/.claude/settings.strict_single.json.bak` | the old single strict file, for undoing the install |
| `~/bin/claude_mode.sh` | switches the bridge's file mode and prints the start command |
| `autogaze_lmms_eval/repro/RUN_SESSION_PROMPT.md` | the task to give a run-mode session |

`.claude/` is read-only. The overlay has to be passed with `--settings`. Under the org's
managed settings, Claude Code ignores `excludedCommands` in project files, and without the
overlay's list the session can't reach Slurm.

## Start a dev session

In your own shell, with no Claude session open in `$WORK`:

```bash
~/bin/claude_mode.sh dev          # chmod -R u+w on the bridge
export HF_TOKEN=...               # only needed if the session submits jobs
cd /home/scratch.thannan_wwfo/codec_reproduce
claude --settings /home/scratch.thannan_wwfo/codec_reproduce/.claude/settings.dev.json
# or, to continue an earlier session: claude --resume --settings .../settings.dev.json
```

First, ask the session to check:
- `squeue -u $USER` works;
- the bridge is writable;
- `runs/` and `.claude/` are not writable.

A typical dev loop:
1. Claude edits the bridge and commits.
2. Claude submits a smoke run:
   ```
   sbatch …/run_eval.sbatch --model <m> --dataset egoschema --smoke-only dev_<m>_egoschema_smoke_<YYYYMMDD_HHMM>
   ```
   Super VL uses `run_eval_super.sbatch`.
3. Claude judges the job only from its state, exit code and the `[status]` / `[run_eval]` lines.
4. You push: `git -C $WORK/autogaze_lmms_eval push origin main`.

## Start a run session

Only after:
- every model's smoke run passed the gate;
- the bridge is committed and pushed (`git status` clean, `HEAD` = `origin/main`).

Exit the dev session first, then in your own shell:

```bash
~/bin/claude_mode.sh run          # refuses if the bridge is dirty, warns if HEAD != origin/main,
                                  # then chmod -R a-w on the bridge
export HF_TOKEN=...
cd /home/scratch.thannan_wwfo/codec_reproduce
claude --settings /home/scratch.thannan_wwfo/codec_reproduce/.claude/settings.run.json
# or: claude --resume --settings .../settings.run.json
```

First, ask the session to check:
- the bridge is not writable (`test -w`);
- `sinfo` is refused;
- `squeue -u $USER` works.

Then give it `autogaze_lmms_eval/repro/RUN_SESSION_PROMPT.md` as the task. It submits only in this
form, for example:

```
sbatch /home/scratch.thannan_wwfo/codec_reproduce/autogaze_lmms_eval/scripts/run_eval.sbatch --model qwen3_5 --dataset egoschema qwen3_5_egoschema_<YYYYMMDD_HHMM>
sbatch /home/scratch.thannan_wwfo/codec_reproduce/autogaze_lmms_eval/scripts/run_eval.sbatch --model nemotron_omni --dataset egoschema nemotron_omni_egoschema_<YYYYMMDD_HHMM>
sbatch /home/scratch.thannan_wwfo/codec_reproduce/autogaze_lmms_eval/scripts/run_eval_super.sbatch --model nemotron_super --dataset egoschema nemotron_super_egoschema_<YYYYMMDD_HHMM>
```

No `sbatch` options are allowed, so each job runs on the partition written in its script:
- Qwen and Omni: `H100_80G_HBM3_SXM_AMD_preprod` (H100 SXM, like the reference);
- Super VL: `dgxh100`.

To use a different partition, change the script in dev mode, commit and push it, then switch.
You read the `runs/<tag>/SUMMARY.md` files yourself.

## Switching back

`~/bin/claude_mode.sh dev` makes the bridge writable again. Start the next session with
`settings.dev.json`. A session's mode is fixed when it starts: re-run `claude_mode.sh` and restart
Claude to change it.

## Notes from the first runs

- Slurm commands must be issued alone: no pipes, `&&`, `;` or `| head`. A combined command
  runs in the sandbox, where the controller can't be reached.
- Austin nodes don't mount `/home/scratch.thannan_wwfo`: `H100x8` and `a1u1g-mil-083x` in the
  PCIe partition. On those nodes a job fails within about 20 s with exit `0:53` and no log. In dev
  mode, add `--constraint=santa_clara` when submitting to a mixed partition.
- `srun` without a GPU count asks for a whole node (`exclusive`). For a quick probe use
  `--gres=gpu:1 -n 1 -c 1 --mem=1G -t 00:02:00`.
- In dev mode, H100 PCIe nodes are fine for smoke runs. Full runs belong on H100 SXM, so
  latencies are comparable with the reference.

## Undo the install

```bash
C=/home/scratch.thannan_wwfo/codec_reproduce/.claude
chmod u+w $C $C/*.json && cp $C/settings.strict_single.json.bak $C/settings.local.json \
  && rm $C/settings.dev.json $C/settings.run.json && chmod a-w $C $C/*.json
```
