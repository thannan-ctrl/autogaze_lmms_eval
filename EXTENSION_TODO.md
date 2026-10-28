# Extension TODO (as of 2026-10-28)

Everything still open for `EXTENSION_SPEC.md`. F = 128 frames (Qwen3.5-9B, H100 SXM) / 16 (NVILA, GB200).
"main" = `dense@F` + `codec@first=1.0@chunk=16@rest=0.25@F`. Code: AutoGaze `extension/pooling-nvila`,
`autogaze_lmms_eval` `main`.

## Status so far

- Phase 1 code done: pooling flag, NVILA `mv` backend and `codec_mv_f*_c*_r*` modes, bridge wiring, `run_ext*.sbatch`,
  x265 odd-size fix (`7fd2726`), distinct ports per srun step, `SMOKE_N=20`.
- Benchmarks dropped: MLVU, LongVideoBench (disk), PerceptionTest (disk), TempCompass (human decision).
- 20-question smoke runs (main arms):

| benchmark | Qwen3.5 | NVILA |
|---|---|---|
| egoschema_subset | queued, job 2891909 (redundant once the full run starts) | passed |
| nextqa_mc_test | queued, job 2891910 | passed |
| lvbench | queued, job 2891911 | passed |
| mvbench | not downloaded | not downloaded |
| videomme | not downloaded | not downloaded |

## Human actions

- [ ] `settings.run.json`: allow bare `sbatch` of `run_ext.sbatch` and `run_ext_gb200.sbatch` (both path aliases) in
      `sandbox.excludedCommands` and in the Slurm `hard_deny` rule.
- [ ] Cancel Qwen EgoSchema smoke 2891909 once the full EgoSchema run is submitted (keep ≤4 H100).
- [ ] Delete the TempCompass videos from the HF cache.
- [ ] After each benchmark's full runs: delete its videos (except mvbench and videomme, kept for Phases 3-4).
- [ ] Pin the AutoGaze `extension/pooling-nvila` sha, lmms-eval sha, NVILA revision, env paths and `max_new_tokens`
      in spec section 3 after Phase 1 review.

## Dev session (code and smoke only)

- [x] **Sharding by question in `run_ext.sbatch`** (spec section 6). Done 2026-10-28: `--shard I/N` (lmms-eval
      `--offset/--limit`, last shard runs to the end), one tag per shard, merge with `scripts/summarize_shards.py`.
      Not yet exercised on a GPU: do one `--smoke-only --shard 1/2` run before the first sharded full run. Jobs are capped at 8 h. Estimates from NVILA
      smoke timing (~11 s/question dense, ~6 s codec): EgoSchema ~2.5 h fits; NExT-QA test (~8.5k questions) ~40 h
      and LVBench (~1.5k) ~7 h do not, nor does Video-MME. Blocks the full runs of everything except EgoSchema.
- [x] Selection-identity compare script for gate 1a: `scripts/selection_identity.py` (`dump` / `compare`, CPU only;
      usage in its docstring). Tool smoke on 2 EgoSchema videos passed; the gate itself (≥20+20) is run mode.
- [ ] MVBench: download through the Qwen smoke run, then NVILA smoke; also smoke the `grid` and `pool` arm sets.
- [ ] Video-MME: download through the Qwen smoke run, then NVILA smoke; also smoke `grid`, `pool`, `main_pxrepro`
      (Qwen) and check the per-duration splits come out.
- [ ] Qwen smoke on NExT-QA and LVBench must pass (jobs 2891910, 2891911) before their full runs.

## Run session

### Phase 1 gates
- [ ] 1a: max-pooling selections bit-identical to `repro-pin` on ≥20 Video-MME + ≥20 EgoSchema items (needs the compare script).
- [ ] 1b: NVILA `--arms gate` on egoschema_subset (dense = 24,690 tokens and within CI of 64.0%; native codec within CI of 65.4%). If it fails, NVILA work stops.
- [ ] 1c: NVILA dense@16f fits on the longest Video-MME video.
- [ ] Write `PHASE1.md`, stop for review.

### Phase 2: main grid (answers G), 11 jobs
- [ ] egoschema_subset: Qwen main, NVILA main (can start now)
- [ ] nextqa_mc_test: Qwen main, NVILA main (needs sharding)
- [ ] lvbench: Qwen main, NVILA main (needs sharding)
- [ ] mvbench: Qwen main, NVILA main
- [ ] videomme: Qwen main, Qwen `main_pxrepro`, NVILA main
- [ ] Write `PHASE2.md` (per model × benchmark table, Video-MME splits, `@px=repro` bridge rows), stop for review.

### Phase 3: config grid (answers P, and M across models), videomme + mvbench
- [ ] Qwen `--arms grid`: dense, default codec, 8 configs `first∈{0.5,1.0} × chunk∈{8,16} × rest∈{0.12,0.25}` (10 arms per benchmark)
- [ ] NVILA `--arms grid`: dense, `codec_mv`, native `codec`, the same 8 configs (11 arms per benchmark)
- [ ] Write `PHASE3.md` (Pareto plots accuracy vs tokens and vs latency), stop for review.

### Phase 4: pooling (answers M within a model), videomme + mvbench, Qwen only
- [ ] Qwen `--arms pool`: 8 configs × {mean, first} (16 arms per benchmark)
- [ ] Write `PHASE4.md`, stop for review.

## Deferred (only if time and disk allow, needs spec amendment)
- LongVideoBench from the local val copy, PerceptionTest.

## Operating notes
- Slurm commands must be bare single commands (no `cd &&`, `timeout`, `$()`, `;`, pipes): sandbox exclusion matches
  the command text. Watch jobs by tailing `runs/_slurm/run_ext_<jobid>.log`, not squeue loops.
- ≤4 GPUs per type at once; shard by question, never by arm; never read result files.
- `H100_preprod` node `2u2g-gen-0678` does not mount scratch. Qwen runs on `H100_80G_HBM3_SXM_AMD_preprod`, NVILA on
  `gb200nvl72_preprod` (`gb-nvl-057-*` verified).
