# Task: extend the codec-selector result to more benchmarks, keep ratios and encoders

Follows `REPRO_SPEC.md` and starts only after its EgoSchema reproduction is done. Everything here
is a new configuration, so results from this spec are never mixed into the reproduction report.
Where this spec is silent, `REPRO_SPEC.md` sections 5 (prediction format) and 7 (rules) apply.

## 1. Questions

| id | question | answered by |
|---|---|---|
| G (generalization) | Is the recommended config `codec@first=1.0@chunk=16@rest=0.25` within noise of dense at equal frames on benchmarks beyond EgoSchema / Video-MME, short and hour-long? | Phase 2, main grid |
| P (operating curve) | How do accuracy, visual tokens and latency move across `first`, `chunk`, `rest`? | Phase 3, config grid |
| M (mechanism) | Does a temporal-patch (2-frame tubelet) encoder change how much codec selection helps, and does the way frame-pair scores are pooled matter? | Phase 3 (NVILA vs Qwen at the same selector) + Phase 4 (pooling) |

## 2. Models

| model | encoder | temporal patch | frames (F) | GPU type | path |
|---|---|---|---|---|---|
| Qwen3.5-9B | Qwen ViT, 16 px | 2 (3D conv) | 128 | H100 80GB | `model_compat` adapter `qwen3_5` (existing) |
| NVILA-8B-HD-Video | SigLIP, 14 px, scales 56/112/196/392, 1,060 patches per tile-frame | 1 (per frame) | 64 | GB200 | its own pipeline (`scripts/breakdown`), new codec backend `mv` (Phase 1) |

Omni and Super are out of scope. Latency is compared only within a model, never across models;
cross-model comparisons use accuracy and kept-token fraction.

## 3. Pinned settings (human fills every TODO before Phase 1; never change them)

| setting | value |
|---|---|
| code | `AutoGaze/`, branch `extension/pooling-nvila` @ TODO sha (pinned after Phase 1 review), branched from the reproduction's pinned sha; `lmms-eval` @ TODO sha |
| model revisions | Qwen3.5-9B `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; NVILA-8B-HD-Video @ TODO |
| runtime | per model, same as the reproduction (Qwen: H100 env from `run_eval.sbatch`; NVILA: GB200 env from the EgoSchema headline). TODO: exact env paths |
| resolution | each model's stock processor: Qwen `video_preprocessor_config.json` (`longest_edge` 25,165,824 total pixels per video), i.e. **no** `MC_QWEN_MAX_PIXELS` override; NVILA its stock 4-scale recipe. Exception: the Video-MME bridge arms (section 5) |
| selector | shared H.264 motion-vector scores of model_compat's `v3mv` selector for every codec arm of both models, except the NVILA native reference arm. Qwen: `MC_SELECTOR=v3mv` plus the bridge's `COMMON_ENV`. NVILA: codec backend `mv` in `scripts/breakdown/codec_selector.py`, which takes per-frame 16x16-macroblock scores from `model_compat.selector_v2` (method `mv`) and keeps NVILA's own tiling, 4-scale rasterization and per-frame top-k |
| NVILA schedule | `first/chunk/rest` becomes NVILA's per-frame `gazing_ratio_tile` list over each 16-frame chunk (`config.hybrid_ratios`; `chunk` must divide 16). The thumbnail stream (nvf/2 frames) stays dense, as stock NVILA; report thumbnail and tile tokens separately |
| `chunk` semantics | unchanged code meaning: `chunk` counts frames, `first` applies to the first temporal group of each chunk (a pair for Qwen, one frame for NVILA). Report measured kept-token fraction per arm |
| pooling | frame-pair score pooling `max` (current behaviour) unless the arm says otherwise |
| decoding | greedy; Qwen thinking off; `max_new_tokens` same as the reproduction (TODO value) |
| prompts | each benchmark's lmms-eval default prompt and parser; EgoSchema and Video-MME keep the reproduction's prompts. Video-MME without subtitles |
| latency | as `REPRO_SPEC.md` section 3: wall clock, decode included, `torch.cuda.synchronize()`, warmup items unscored |

## 4. Benchmarks

Main grid, all nine (lmms-eval task names; each must be a `generate_until` task with exactly one video per doc):

| class | task | notes |
|---|---|---|
| anchor | `egoschema_subset` | public 500 |
| anchor | `videomme` | no subtitles; always also report the official short / medium / long splits |
| hour-long | `lvbench`, `mlvu` (dev), `longvideobench_val_v` | `_i` variants return frames, not allowed |
| short | `mvbench` (default `mvbench_doc_to_visual`), `tempcompass` (mc), `nextqa_mc_test`, `perceptiontest_val_mc` | not `perceptiontest_mcppl` (loglikelihood) |

Excluded until their loaders are fixed: `tomato` (empty list on a missing file), `longtimescope`
(`sys.exit` on a missing file), `vinoground` (format unchecked). Ablations (Phases 3-4) run on
`videomme` and `mvbench` only.

## 5. Arms

Arm name = spec string + `@<F>f`, plus `@pool=<p>` when pooling is not `max`, plus `@px=repro`
for the bridge arms. Every arm of an item runs in the same job on the same GPU.

**Main grid (Phase 2), every benchmark, both models:** `dense@<F>f`,
`codec@first=1.0@chunk=16@rest=0.25@<F>f`. Video-MME only, Qwen only: the same two arms at the
reproduction's resolution (`MC_QWEN_MAX_PIXELS=4294967296`), suffix `@px=repro`, to bridge to the
reproduction's numbers.

**Config grid (Phase 3), `videomme` + `mvbench`, both models:**
`first ∈ {0.5, 1.0} × chunk ∈ {8, 16} × rest ∈ {0.12, 0.25}` (8 configs, includes the recommended
point and `hybrid8`), plus `dense@<F>f` and the default `codec@<F>f`. NVILA also gets its native
HEVC selector as a reference arm (`nvila_native_codec@<F>f`) if the native pipeline runs at 64
frames; if it does not, record that and drop the arm.

**Pooling (Phase 4), `videomme` + `mvbench`, Qwen only:** the 8 configs × `pool ∈ {mean, first}`
(16 arms; `max` is already in Phase 3). NVILA has no frame pairs, so no pooling arms.

## 6. Compute and scheduling

- At most 4 GPUs of each type in use at once (≤4 H100 for Qwen, ≤4 GB200 for NVILA).
- Shard items across jobs, never arms. One job loads each arm in turn on its GPU.
- Order: Phase 1 → 2 → 3 → 4. Before any full run of a benchmark, its arms pass a 20-question
  smoke run (`dev_` tag): no bridge errors, every response parsed by the lmms-eval scorer, token
  counts and selector path logged.
- Record GPU-hours per arm per benchmark in each phase report; use the smoke runs to estimate the
  next phase before submitting it.

## 7. Phases (stop after each and wait for review)

**Phase 1: code (no full runs).** In `AutoGaze/` on branch `extension/pooling-nvila`.
Dev mode writes the code, CPU unit tests and `dev_` `--smoke-only` runs only; every gate below
that produces a number (selection identity, NVILA EgoSchema gate, memory check) runs in run mode.
- Pooling flag in `model_compat/selection.py` (`max` | `mean` | `first`), default `max`. Gate: on
  ≥20 Video-MME and ≥20 EgoSchema items, the `max` default produces selections bit-identical to
  the reproduction's pinned sha (dump kept-unit indices per item at that sha before the first
  change, then compare).
- NVILA `mv` codec backend and `codec_mv_f<first>_c<chunk>_r<rest>` modes (decided instead of a
  `model_compat` adapter, so NVILA's input building stays its own). Gate on EgoSchema 500 at
  16 frames (`--arms gate`): `dense` has 24,690 tokens exactly and accuracy within the CI of
  64.0%; the native codec arm reproduces 65.4% within its CI. If either fails, NVILA work stops.
- Memory / context check: NVILA `dense@64f` on the longest Video-MME video. If it does not fit,
  F = 32 for NVILA and this spec is amended before Phase 2.
- Bridge wiring: mode names `codec_mv_f<first>_c<chunk>_r<rest>[_p<pool>]` (Qwen) and
  `codec_mv`, `codec_mv_f<first>_c<chunk>_r<rest>` (NVILA); launcher `--task`; the smoke gate
  (`headline_table.py`) handles any task, including group tasks such as mvbench.
- Job scripts: `scripts/run_ext.sbatch` (Qwen, H100) and `scripts/run_ext_gb200.sbatch` (NVILA,
  GB200), `--model --task --arms main|main_pxrepro|grid|pool|gate [--smoke-only] <tag>`. They
  require AutoGaze on `extension/pooling-nvila` and clean; run mode must allow them before Phase 2.

**Phase 2: main grid.** Section 5 main grid on all nine benchmarks, both models. Answers G.

**Phase 3: config grid.** Answers P, and M at the cross-model level.

**Phase 4: pooling.** Answers M at the within-model level.

## 8. Statistics

- Accuracy per arm with 95% bootstrap CI over items.
- codec vs dense: paired bootstrap 95% CI of the accuracy difference on the same items. "Within
  noise" = CI includes 0; also report against a non-inferiority margin of 1.0 point.
- Latency: mean and median per arm, speedup vs dense at the same frames. Tokens: mean visual
  tokens and kept fraction vs dense.
- Video-MME: overall and per duration split for every arm.

## 9. Rules

- All code changes live on branch `extension/pooling-nvila` in `AutoGaze/`; never commit to the
  branch the reproduction pins. No push, amend, reset or rebase. Any reproduction run requires
  `AutoGaze/` checked out at the reproduction's pinned sha.
- Section 3 and the arm list are fixed. No extra configs, frame counts or benchmarks without a
  spec amendment the human approves.
- Never drop, skip or individually retry items (`REPRO_SPEC.md` section 7).
- Every number in a report comes from the scoring script's output over per-item JSONL; keep raw
  outputs in `logs/`.
- A codec arm significantly above dense is a suspected bug: check selector path, kept-token
  fraction and frame indices before reporting it.

## 10. Reports

One per phase (`PHASE<n>.md`): environment and shas, gate results, per arm accuracy with CI,
paired difference vs dense, tokens, latency, GPU-hours, unparseable count. Phase 2: one table per
model × benchmark plus the Video-MME duration splits and the `@px=repro` bridge rows. Phase 3:
accuracy-vs-tokens and accuracy-vs-latency Pareto plots per model × benchmark. Phase 4: pooling
table (config × pooling). Every report ends with **what does not hold, with suspected causes**.
