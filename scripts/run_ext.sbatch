#!/usr/bin/env bash
# EXTENSION_SPEC.md runs: one model x one lmms-eval task x one arm set through the AutoGaze bridge, end to end:
# preflight, smoke (20 questions), gate, full run, SUMMARY.md, checksums, then runs/<tag>/ is made read-only.
# Same flow as run_eval.sbatch (the reproduction's script, left untouched), with the extension's code
# (AutoGaze branch extension/pooling-nvila), tasks and arm sets.
#
# Usage:
#   sbatch .../autogaze_lmms_eval/scripts/run_ext.sbatch       --model qwen3_5 --task <task> --arms <set> [--smoke-only] <tag>
#   sbatch .../autogaze_lmms_eval/scripts/run_ext_gb200.sbatch --model nvila   --task <task> --arms <set> [--smoke-only] <tag>
# This file asks Slurm for one H100 (Qwen3.5-9B); run_ext_gb200.sbatch asks for one GB200 (NVILA) and runs this file.
#
# Tasks (EXTENSION_SPEC.md section 4): egoschema_subset videomme lvbench mvbench
#   nextqa_mc_test
# Arm sets (section 5; F = 128 frames for qwen3_5, 16 for nvila):
#   main         dense + recommended codec_mv_f1.0_c16_r0.25 (= codec_mv_hybrid)
#   main_pxrepro qwen3_5, videomme only: main at the reproduction's resolution (MC_QWEN_MAX_PIXELS=4294967296)
#   grid         dense, default codec_mv, the 8 configs first {0.5,1.0} x chunk {8,16} x rest {0.12,0.25};
#                nvila also its native HEVC selector (codec). videomme / mvbench only
#   pool         qwen3_5: the 8 configs with pool mean and first. videomme / mvbench only
#   gate         nvila, egoschema_subset, 16 frames: dense + native codec (Phase 1 adapter gate)
# Resolution is each model's stock processor (Qwen: MC_QWEN_MAX_PIXELS=25165824, the stock per-video cap),
# except main_pxrepro.
#SBATCH --partition=H100_80G_HBM3_SXM_AMD_preprod
#SBATCH --gres=gpu:H100:1,gpu_memory:80G
#SBATCH --time=08:00:00
#SBATCH --job-name=run_ext
#SBATCH --output=/home/scratch.thannan_wwfo/codec_reproduce/runs/_slurm/run_ext_%j.log
set -euo pipefail
for v in $(compgen -e | grep -E '^(MC_|PYTHON)' || true); do unset "$v"; done

WORK=/home/scratch.thannan_wwfo/codec_reproduce
PKG=$WORK/autogaze_lmms_eval
LAUNCHER=$PKG/scripts/run_egoschema_headline.sh
HF_HOME=/home/scratch.thannan_wwfo/hf_cache
WARMUP_VIDEO=$HF_HOME/egoschema/videos/001934bb-81bd-4cd8-a574-0472ef3f6678.mp4  # not in the EgoSchema subset
SMOKE_N=20
EXT_BRANCH=extension/pooling-nvila
PIN_LMMS_EVAL=d6cc2b567db7aeacbbf1af11b6595676e784d1d3
TASKS="egoschema_subset videomme lvbench mvbench nextqa_mc_test"
GRID=""
for f in 0.5 1.0; do for c in 8 16; do for r in 0.12 0.25; do GRID="$GRID codec_mv_f${f}_c${c}_r${r}"; done; done; done

die() { echo "[run_ext] ERROR: $*" >&2; exit 1; }
log() { echo "[run_ext] $(date +%H:%M:%S) $*"; }

MODEL="" TASK="" ARMS="" SMOKE_ONLY=0 TAG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL=${2:-}; shift ;;
    --task) TASK=${2:-}; shift ;;
    --arms) ARMS=${2:-}; shift ;;
    --smoke-only) SMOKE_ONLY=1 ;;
    -*) die "unknown option $1" ;;
    *) [ -z "$TAG" ] || die "more than one tag: $TAG $1"; TAG=$1 ;;
  esac
  shift
done
[[ " $TASKS " == *" $TASK "* ]] || die "--task must be one of: $TASKS"
[[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "tag must be a plain name ([A-Za-z0-9._-]): '$TAG'"
ablation_task() { [ "$TASK" = videomme ] || [ "$TASK" = mvbench ] || die "--arms $ARMS runs on videomme and mvbench only"; }

MC_ENV=()  # extra MC_* exported to the servers (model_compat's setdefault lets them win)
case "$MODEL" in
  qwen3_5)
    ARCH=x86_64 GPU_NAME=H100 CONDA_BASE=/home/scratch.thannan_wwfo/anaconda3 SERVER_ENV=nemo-omni CLIENT_ENV=lmms-eval-bridge
    LMMS_DIR=$WORK/lmms-eval
    MC_NATIVE_BUILD=/home/scratch.thannan_wwfo/model_compat/native_build_x86_64
    MC_PYTHONPATH=/home/scratch.thannan_wwfo/model_compat/pylibs/fla KERNELS="fla causal_conv1d"
    MODEL_PATH=$HF_HOME/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
    NVF=128 RECOMMENDED=codec_mv_f1.0_c16_r0.25 PIXELS=25165824
    case "$ARMS" in
      main) MODES="dense $RECOMMENDED" ;;
      main_pxrepro) [ "$TASK" = videomme ] || die "main_pxrepro is videomme only"; MODES="dense $RECOMMENDED" PIXELS=4294967296 ;;
      grid) ablation_task; MODES="dense codec_mv$GRID" ;;
      pool) ablation_task; MODES=$(for m in $GRID; do printf '%s_pmean %s_pfirst ' "$m" "$m"; done) ;;
      *) die "--arms for qwen3_5: main, main_pxrepro, grid, pool" ;;
    esac
    MC_ENV=(MC_QWEN_MAX_PIXELS=$PIXELS) ;;
  nvila)
    ARCH=aarch64 GPU_NAME=GB200 CONDA_BASE=/home/scratch.thannan_wwfo/miniforge-aarch64 SERVER_ENV=auto_gaze CLIENT_ENV=lmms-eval
    LMMS_DIR=/home/thannan/lmms-eval  # the copy the aarch64 lmms-eval env imports (LMMS_EVAL_BRIDGE.md section 3)
    MC_NATIVE_BUILD=$WORK/AutoGaze/native/build_aarch64 MC_PYTHONPATH="" KERNELS=""
    MODEL_PATH="" NVF=16 RECOMMENDED=codec_mv_f1.0_c16_r0.25  # 64 f exceeds the 32k context (job 2884385)
    case "$ARMS" in
      main) MODES="dense $RECOMMENDED" ;;
      grid) ablation_task; MODES="dense codec_mv codec$GRID" ;;
      gate) [ "$TASK" = egoschema_subset ] || die "gate runs on egoschema_subset"; MODES="dense codec" ;;
      *) die "--arms for nvila: main, grid, gate" ;;
    esac ;;
  *) die "--model must be qwen3_5 or nvila" ;;
esac
SUITE=$WORK/runs/$TAG
[ ! -e "$SUITE" ] || die "$SUITE already exists; use a new tag"

export CONDA_BASE SERVER_ENV CLIENT_ENV HF_HOME MC_NATIVE_BUILD MC_PYTHONPATH
export PYTORCH_ALLOC_CONF=expandable_segments:True
[ ${#MC_ENV[@]} = 0 ] || export "${MC_ENV[@]}"

mkdir -p "$SUITE"
trap 'chmod -R a-w "$SUITE" 2>/dev/null || true' EXIT

# ---- preflight (read-only checks; any failure stops before a GPU is used) ----
PRE=$SUITE/preflight.txt
fail=0
chk() { if out=$("${@:2}" 2>&1); then echo "ok   $1 ${out:+($out)}"; else echo "FAIL $1 ${out:+($out)}"; fail=1; fi; }
inenv() { bash -c 'eval "$("$0"/bin/conda shell.bash hook)" && conda activate "$1" && shift && exec "$@"' "$CONDA_BASE" "$@"; }
{
  echo "tag=$TAG model=$MODEL task=$TASK arms=$ARMS nvf=$NVF modes=$MODES smoke_n=$SMOKE_N smoke_only=$SMOKE_ONLY"
  echo "mc_env=${MC_ENV[*]:-none}"
  echo "host=$(hostname) slurm_job=${SLURM_JOB_ID:-none} date=$(date -Is) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "gpus=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sort | uniq -c | tr -s ' ' | paste -sd';')"
  echo "server_env=$CONDA_BASE/envs/$SERVER_ENV client_env=$CONDA_BASE/envs/$CLIENT_ENV mc_native_build=$MC_NATIVE_BUILD"
  echo "bridge_commit=$(git -C "$PKG" rev-parse HEAD) bridge_dirty=$(git -C "$PKG" status --porcelain --untracked-files=no | wc -l)"
  echo "autogaze_branch=$(git -C "$WORK/AutoGaze" rev-parse --abbrev-ref HEAD) autogaze_commit=$(git -C "$WORK/AutoGaze" rev-parse HEAD)"
  chk "$ARCH node"           test "$(uname -m)" = "$ARCH"
  chk "$GPU_NAME GPU"        bash -c "nvidia-smi -L | grep -q $GPU_NAME"
  chk "HF_TOKEN set"         test -n "${HF_TOKEN:-}"
  chk "AutoGaze on $EXT_BRANCH" test "$(git -C "$WORK/AutoGaze" rev-parse --abbrev-ref HEAD)" = "$EXT_BRANCH"
  chk "AutoGaze code clean"  test -z "$(git -C "$WORK/AutoGaze" status --porcelain --untracked-files=no -- . ':!.claude' ':!.vscode')"
  chk "lmms-eval at pin"     test "$(git -C "$LMMS_DIR" rev-parse HEAD)" = "$PIN_LMMS_EVAL"
  chk "native dump_cu"       test -x "$MC_NATIVE_BUILD/dump_cu"
  chk "warmup video"         test -f "$WARMUP_VIDEO"
  if [ "$MODEL" = nvila ]; then
    chk "native raster lib"  test -f "$MC_NATIVE_BUILD/libcodec_raster.so"
    chk "server env imports" inenv "$SERVER_ENV" env PYTHONPATH="$WORK/AutoGaze" python -c '
import torch, transformers, av, cv2, model_compat.selector_v2
print(torch.__version__, transformers.__version__, "cuda" if torch.cuda.is_available() else "NO CUDA")
raise SystemExit(0 if torch.cuda.is_available() else 1)'
  else
    chk "server env imports" inenv "$SERVER_ENV" env PYTHONPATH="$WORK/AutoGaze${MC_PYTHONPATH:+:$MC_PYTHONPATH}" python -c '
import importlib, sys, torch, transformers, av, model_compat
for m in sys.argv[1:]: importlib.import_module(m)
print(torch.__version__, transformers.__version__, "cuda" if torch.cuda.is_available() else "NO CUDA")
sys.exit(0 if torch.cuda.is_available() else 1)' $KERNELS
    chk "weights"            test -f "$MODEL_PATH/config.json"
  fi
  chk "client env imports"   inenv "$CLIENT_ENV" python -c '
import lmms_eval, autogaze_lmms_eval, os, sys
from lmms_eval.models import get_model; get_model("autogaze")
root = os.path.dirname(os.path.dirname(lmms_eval.__file__)); print(root)
sys.exit(0 if os.path.realpath(root) == os.path.realpath(sys.argv[1]) else f"lmms_eval imported from {root}, expected {sys.argv[1]}")' "$LMMS_DIR"
} > "$PRE"
cat "$PRE"
[ "$fail" = 0 ] || die "preflight failed (see $PRE); nothing was run"

GPU=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | cut -d, -f1)
BASE_PORT=$((20000 + ((${SLURM_JOB_ID:-$$} + ${SLURM_STEP_ID:-0} * 1009) % 4000) * 4))  # step id: srun steps sharing one allocation get distinct ports
launch() {  # launch <stage> <mode> [extra launcher args...]: one arm, into runs/<tag>/<stage>/<mode>/
  mkdir -p "$SUITE/$1"
  local extra=(--mc-pythonpath "$MC_PYTHONPATH")
  [ -z "$MODEL_PATH" ] || extra+=(--model-path "$MODEL_PATH")
  "$LAUNCHER" --model "$MODEL" --task "$TASK" --nvf "$NVF" --modes "$2" --gpus "$GPU" \
    --base-port "$BASE_PORT" --warmup-video "$WARMUP_VIDEO" --server-env "$SERVER_ENV" \
    "${extra[@]}" --runs-dir "$SUITE/$1" "${@:3}" "$2"
}
gate() { grep -qx "\[status\] $2: timing ok, client log ok, e2e cross-check ok, replies ok" "$SUITE/$1/$2/logs/headline_table.log"; }
stage() {  # stage <stage> <die|warn> [extra launcher args...]: every arm, one after another
  local m
  for m in $MODES; do
    log "$1: $m"
    if ! launch "$1" "$m" "${@:3}"; then
      [ "$2" = die ] && die "$1 $m failed (see $SUITE/$1/$m/logs/)"; FAILED=1; log "$1 $m failed (see $SUITE/$1/$m/logs/)"; continue
    fi
    gate "$1" "$m" || { [ "$2" = die ] && die "$1 $m gate failed (see $SUITE/$1/$m/logs/headline_table.log)"
                        FAILED=1; log "$1 $m: status lines not all ok (see $SUITE/$1/$m/logs/headline_table.log)"; }
  done
}

FAILED=0
stage smoke die --limit "$SMOKE_N"
log "smoke passed the gate"
[ "$SMOKE_ONLY" = 1 ] || stage full warn

inenv "$CLIENT_ENV" python "$PKG/scripts/summarize_suite.py" "$SUITE" > "$SUITE/SUMMARY.md" 2> "$SUITE/summary.log" \
  || die "summarize_suite.py failed (see $SUITE/summary.log)"
cat "$SUITE/summary.log"
( cd "$SUITE" && find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum
  find . -mindepth 3 -maxdepth 3 -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum ) > "$SUITE/SHA256SUMS"
[ "$FAILED" = 0 ] || die "full run finished with failed arms; SUMMARY.md has what completed (see the [run_ext] lines above)"
log "done: $SUITE/SUMMARY.md (read-only)"
