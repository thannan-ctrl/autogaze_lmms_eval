#!/usr/bin/env bash
# Run lmms-eval's egoschema_subset (500 questions) through the AutoGaze bridge,
# one server per mode, one GPU per server, all modes in parallel. Reproduces the
# "Headline result" of AutoGaze/Codec_Selector_Feasibility.md; see
# EGOSCHEMA_HEADLINE.md next to this package for the full runbook.
#
# Usage:
#   run_egoschema_headline.sh [options] <tag>
#
# Options:
#   --download-only       start one dense server and run 1 question, so lmms-eval
#                         downloads and unzips lmms-lab/egoschema into
#                         $HF_HOME/egoschema exactly once. Run this first.
#   --limit N             pass --limit N to lmms-eval (smoke test)
#   --modes "M1 M2 ..."   default: "dense autogaze codec codec_nvdec"
#   --gpus "G1 G2 ..."    GPU index per mode, same order (default: 0 1 2 ...)
#   --warmup-video PATH   untimed warmup video for every server (use an EgoSchema
#                         video that is NOT in the Subset)
#   --warmup-n N          warmup requests per server (default 2)
#   --base-port P         mode i listens on P+i (default 8765)
#   --runs-dir DIR        default: <codec_reproduce>/runs
#   --repo-dir DIR        AutoGaze checkout (default: <codec_reproduce>/AutoGaze)
#   --model M             nvila (default) | qwen3_5 | nemotron_omni. The last two run
#                         AutoGaze's model_compat adapters (modes dense, codec_mv,
#                         codec_mv_hybrid, codec_mv_keep; see mc_backend.py)
#   --nvf N               frames per video (default 16; RESULTS_LATENCY_ACCURACY.md
#                         recommends 128 for qwen3_5, 64 for nemotron_omni)
#   --mc-root DIR         checkout holding model_compat (default: --repo-dir, which
#                         must then be at AutoGaze commit 9899fd6, branch pareto/main)
#   --model-path P        weights dir or hub id for qwen3_5 / nemotron_omni
#
# Env for qwen3_5 / nemotron_omni: MC_PYTHONPATH (prepended to the server's
# PYTHONPATH; default: model_compat's vendored fla kernels) and any MC_* override.
#
# Everything the run writes goes to <runs-dir>/<tag>/; nothing is written inside
# this package or the AutoGaze checkout (codec caches included). At the end the
# launcher writes <tag>/RESULTS.md (headline_table.py: accuracy, latency
# breakdown, end-to-end cross-check) and SHA256SUMS, then makes <tag>/ read-only.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PKG_ROOT=$(dirname "$SCRIPT_DIR")
CODEC_ROOT=$(dirname "$PKG_ROOT")

CONDA_BASE=${CONDA_BASE:-/home/scratch.thannan_wwfo/miniforge-aarch64}
SERVER_ENV=${SERVER_ENV:-auto_gaze}
CLIENT_ENV=${CLIENT_ENV:-lmms-eval}
export HF_HOME=${HF_HOME:-/home/scratch.thannan_wwfo/hf_cache}
MC_PYTHONPATH=${MC_PYTHONPATH:-/home/scratch.thannan_wwfo/model_compat/pylibs/fla}
TASK=egoschema_subset
NVF=16
MAX_BATCH_SIZE_AUTOGAZE=32  # the harness value for the headline run (server default is 64)
MODEL=nvila
MC_ROOT=""  # default: $REPO_DIR
MODEL_PATH=""

DOWNLOAD_ONLY=0
LIMIT=""
MODES="dense autogaze codec codec_nvdec"
GPUS=""
WARMUP_VIDEO=""
WARMUP_N=2
BASE_PORT=8765
RUNS_DIR="$CODEC_ROOT/runs"
REPO_DIR="$CODEC_ROOT/AutoGaze"
TAG=""

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; }
die() { echo "[headline] ERROR: $*" >&2; exit 1; }
log() { echo "[headline] $(date +%H:%M:%S) $*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --download-only) DOWNLOAD_ONLY=1 ;;
    --limit) LIMIT=$2; shift ;;
    --modes) MODES=$2; shift ;;
    --gpus) GPUS=$2; shift ;;
    --warmup-video) WARMUP_VIDEO=$2; shift ;;
    --warmup-n) WARMUP_N=$2; shift ;;
    --base-port) BASE_PORT=$2; shift ;;
    --runs-dir) RUNS_DIR=$2; shift ;;
    --repo-dir) REPO_DIR=$2; shift ;;
    --model) MODEL=$2; shift ;;
    --nvf) NVF=$2; shift ;;
    --mc-root) MC_ROOT=$2; shift ;;
    --model-path) MODEL_PATH=$2; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) die "unknown option $1 (see --help)" ;;
    *) [ -z "$TAG" ] || die "more than one tag given: $TAG $1"; TAG=$1 ;;
  esac
  shift
done
[ -n "$TAG" ] || { usage; die "missing <tag>"; }

case "$MODEL" in
  nvila|qwen3_5|nemotron_omni) ;;
  *) die "unknown --model $MODEL (nvila, qwen3_5, nemotron_omni)" ;;
esac

if [ "$DOWNLOAD_ONLY" = 1 ]; then
  MODES=dense
  LIMIT=1
  WARMUP_VIDEO=""
fi
read -r -a MODE_ARR <<< "$MODES"
if [ -n "$GPUS" ]; then
  read -r -a GPU_ARR <<< "$GPUS"
else
  GPU_ARR=()
  for i in "${!MODE_ARR[@]}"; do GPU_ARR+=("$i"); done
fi
[ "${#GPU_ARR[@]}" = "${#MODE_ARR[@]}" ] || die "--gpus has ${#GPU_ARR[@]} entries for ${#MODE_ARR[@]} modes"

# ---- preflight -------------------------------------------------------------
[ "$(uname -m)" = aarch64 ] || die "run this on a GB200 (aarch64) compute node, not $(uname -m)"
[ -x "$CONDA_BASE/bin/conda" ] || die "no conda at $CONDA_BASE"
[ -d "$REPO_DIR/scripts/breakdown" ] || die "no AutoGaze checkout at $REPO_DIR"
[ -n "${HF_TOKEN:-}" ] || log "WARNING: HF_TOKEN is not set; lmms-lab/egoschema needs an authenticated download"
MC_ROOT=${MC_ROOT:-$REPO_DIR}
if [ "$MODEL" != nvila ]; then
  [ -d "$MC_ROOT/model_compat/adapters" ] || die "no model_compat in $MC_ROOT: check out AutoGaze commit 9899fd6 (branch pareto/main)"
  # model_compat's frame sampler loads dump_cu even for dense, and the MV selector
  # falls back to the HEVC path on intra-only windows.
  native="$REPO_DIR/native/build_aarch64"
  [ -x "$native/dump_cu" ] || die "$MODEL needs $native/dump_cu: build AutoGaze/native first (README step 4)"
  for m in "${MODE_ARR[@]}"; do
    case "$m" in
      dense|codec_mv|codec_mv_hybrid|codec_mv_keep) ;;
      *) die "mode $m is not available for $MODEL (dense, codec_mv, codec_mv_hybrid, codec_mv_keep)" ;;
    esac
  done
fi
for m in "${MODE_ARR[@]}"; do
  [ "$MODEL" = nvila ] || break
  case "$m" in
    codec|codec_nvdec|codec_geo|codec_ord)
      # Without these the codec selector silently falls back to slower paths
      # (or fails), so latency would not be comparable.
      native="$REPO_DIR/native/build_aarch64"
      [ -x "$native/dump_cu" ] || die "$m needs $native/dump_cu: build AutoGaze/native first (EGOSCHEMA_HEADLINE.md, prerequisites)"
      [ -f "$native/libcodec_raster.so" ] || die "$m needs $native/libcodec_raster.so: build AutoGaze/native first"
      ;;
  esac
done
[ -z "$WARMUP_VIDEO" ] || [ -f "$WARMUP_VIDEO" ] || die "warmup video not found: $WARMUP_VIDEO"

EGO_DIR="$HF_HOME/egoschema"
if [ "$DOWNLOAD_ONLY" = 1 ]; then
  if [ -d "$EGO_DIR" ]; then
    die "$EGO_DIR already exists ($(find "$EGO_DIR" -name '*.mp4' | wc -l) mp4s); lmms-eval will not download again. If it is half-filled from an interrupted unzip, delete it and re-run."
  fi
else
  [ -d "$EGO_DIR/videos" ] || die "$EGO_DIR/videos missing: run with --download-only first, so 4 parallel runs don't race on the download"
  [ -n "$WARMUP_VIDEO" ] || log "WARNING: no --warmup-video; the first rows of each timing log include warmup costs"
fi

RUN_DIR="$RUNS_DIR/$TAG"
[ ! -e "$RUN_DIR" ] || die "$RUN_DIR already exists; pick a new tag (timing logs are appended to, never reuse them)"
mkdir -p "$RUN_DIR/logs"
# Absolute paths, then work from the run dir: lmms-eval writes the EgoSchema
# submission file relative to the cwd when it doesn't get an output path.
RUN_DIR=$(cd "$RUN_DIR" && pwd)
REPO_DIR=$(cd "$REPO_DIR" && pwd)
MC_ROOT=$(cd "$MC_ROOT" && pwd)
[ -z "$WARMUP_VIDEO" ] || WARMUP_VIDEO=$(cd "$(dirname "$WARMUP_VIDEO")" && pwd)/$(basename "$WARMUP_VIDEO")
cd "$RUN_DIR"

# Never write bytecode into the (read-only) package.
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# "${IN_ENV[@]}" <env> <cmd...>: run cmd in a fresh shell with that conda env
# active. The shell execs cmd, so a backgrounded call's $! is cmd's own pid.
IN_ENV=(bash -c 'eval "$("$0"/bin/conda shell.bash hook)" && conda activate "$1" && shift && exec "$@"' "$CONDA_BASE")

health() {  # health <port>: prints /health JSON, non-zero if unreachable
  "${IN_ENV[@]}" "$CLIENT_ENV" python -c '
import sys, urllib.request
print(urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=5).read().decode())' "$1" 2>/dev/null
}

{
  echo "tag=$TAG"
  echo "date=$(date -Is)"
  echo "host=$(hostname)"
  echo "slurm_job=${SLURM_JOB_ID:-}"
  echo "modes=${MODES} gpus=${GPU_ARR[*]} limit=${LIMIT:-none} warmup_video=${WARMUP_VIDEO:-none} warmup_n=$WARMUP_N"
  echo "model=$MODEL nvf=$NVF max_batch_size_autogaze=$MAX_BATCH_SIZE_AUTOGAZE task=$TASK"
  echo "autogaze_commit=$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  if [ "$MODEL" != nvila ]; then
    echo "model_compat_commit=$(git -C "$MC_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown) mc_root=$MC_ROOT"
    echo "model_path=${MODEL_PATH:-default} mc_pythonpath=$MC_PYTHONPATH"
    echo "mc_env=$(env | grep '^MC_' | sort | tr '\n' ' ')"
  fi
  echo "lmms_eval=$("${IN_ENV[@]}" "$CLIENT_ENV" python -c 'import lmms_eval, os; print(os.path.dirname(os.path.dirname(lmms_eval.__file__)))' 2>/dev/null)"
  echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
} > "$RUN_DIR/run_info.txt"
lmms_dir=$(sed -n 's/^lmms_eval=//p' "$RUN_DIR/run_info.txt")
echo "lmms_eval_commit=$(git -C "$lmms_dir" rev-parse --short HEAD 2>/dev/null || echo unknown)" >> "$RUN_DIR/run_info.txt"
cat "$RUN_DIR/run_info.txt"

# ---- servers ---------------------------------------------------------------
SERVER_PIDS=()
cleanup() {
  for pid in "${SERVER_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${SERVER_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT TERM

for i in "${!MODE_ARR[@]}"; do
  m=${MODE_ARR[$i]}
  port=$((BASE_PORT + i))
  if health "$port" > /dev/null; then die "port $port is already serving: $(health "$port")"; fi
  args=(--model "$MODEL" --repo-dir "$REPO_DIR" --mode "$m" --nvf "$NVF" --max-batch-size-autogaze "$MAX_BATCH_SIZE_AUTOGAZE"
        --port "$port" --cache-dir "$RUN_DIR/cache_$m" --timing-log "$RUN_DIR/timing_$m.jsonl")
  [ -z "$WARMUP_VIDEO" ] || args+=(--warmup-video "$WARMUP_VIDEO" --warmup-n "$WARMUP_N")
  server_pp="$PKG_ROOT"
  if [ "$MODEL" != nvila ]; then
    args+=(--mc-root "$MC_ROOT")
    [ -z "$MODEL_PATH" ] || args+=(--model-path "$MODEL_PATH")
    [ -z "$MC_PYTHONPATH" ] || server_pp="$PKG_ROOT:$MC_PYTHONPATH"
  fi
  log "starting $m server ($MODEL) on GPU ${GPU_ARR[$i]}, port $port"
  CUDA_VISIBLE_DEVICES=${GPU_ARR[$i]} PYTHONPATH="$server_pp" \
    "${IN_ENV[@]}" "$SERVER_ENV" python -u -m autogaze_lmms_eval.server "${args[@]}" > "$RUN_DIR/logs/server_$m.log" 2>&1 &
  SERVER_PIDS+=($!)
done

for i in "${!MODE_ARR[@]}"; do
  m=${MODE_ARR[$i]}
  port=$((BASE_PORT + i))
  until health "$port" > /dev/null; do
    kill -0 "${SERVER_PIDS[$i]}" 2>/dev/null || { tail -n 30 "$RUN_DIR/logs/server_$m.log" >&2; die "$m server exited during startup"; }
    sleep 10
  done
  log "$m server ready: $(health "$port")"
done

# ---- lmms-eval ---------------------------------------------------------------
CLIENT_PIDS=()
for i in "${!MODE_ARR[@]}"; do
  m=${MODE_ARR[$i]}
  port=$((BASE_PORT + i))
  args=(-m lmms_eval --model autogaze
        --model_args "server_url=http://127.0.0.1:$port,mode=$m,client_timing_log=$RUN_DIR/client_timing_$m.jsonl"
        --tasks "$TASK" --batch_size 1 --log_samples --output_path "$RUN_DIR/lmms_$m")
  [ -z "$LIMIT" ] || args+=(--limit "$LIMIT")
  log "starting lmms-eval for $m"
  "${IN_ENV[@]}" "$CLIENT_ENV" python "${args[@]}" > "$RUN_DIR/logs/lmms_$m.log" 2>&1 &
  CLIENT_PIDS+=($!)
done

failed=0
for i in "${!MODE_ARR[@]}"; do
  m=${MODE_ARR[$i]}
  rc=0
  wait "${CLIENT_PIDS[$i]}" || rc=$?
  # lmms-eval catches evaluation errors (e.g. a server 500) and still exits 0,
  # so success also requires that it wrote a results file.
  if [ "$rc" = 0 ] && [ -n "$(find "$RUN_DIR/lmms_$m" -name '*_results.json' 2>/dev/null)" ]; then
    log "$m done"
  else
    log "$m FAILED (exit $rc; see $RUN_DIR/logs/lmms_$m.log and server_$m.log)"
    failed=1
  fi
done

if [ "$DOWNLOAD_ONLY" = 1 ]; then
  log "download finished: $(find "$EGO_DIR/videos" -name '*.mp4' 2>/dev/null | wc -l) videos in $EGO_DIR/videos"
  exit "$failed"
fi

# The bridge writes the result table itself (accuracy, latency breakdown, and the
# end-to-end cross-check against its own clocks), so nobody has to transcribe it.
cleanup  # servers are done; stop them before writing the final files
SERVER_PIDS=()
"${IN_ENV[@]}" "$CLIENT_ENV" python "$SCRIPT_DIR/headline_table.py" "$RUN_DIR" "${MODE_ARR[@]}" \
  > "$RUN_DIR/RESULTS.md" 2> "$RUN_DIR/logs/headline_table.log" \
  || { log "headline_table.py failed (see $RUN_DIR/logs/headline_table.log)"; failed=1; }
[ "$failed" = 0 ] || echo -e "\n**Run status: FAILED** (see logs/)." >> "$RUN_DIR/RESULTS.md"
( cd "$RUN_DIR" && find . -path './cache_*' -prune -o -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum ) \
  > "$RUN_DIR/SHA256SUMS"  # codec caches are left out (large, not results)
# Results are final: make the run folder read-only.
chmod -R a-w "$RUN_DIR"
# Only pass/fail here; the numbers stay in RESULTS.md for the person reading results.
cat "$RUN_DIR/logs/headline_table.log"
log "results table: $RUN_DIR/RESULTS.md (run folder is now read-only; checksums in SHA256SUMS)"
exit "$failed"
