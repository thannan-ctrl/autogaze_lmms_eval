#!/usr/bin/env bash
# One model on lmms-eval's egoschema_subset (500 questions) through the AutoGaze
# bridge on H100, end to end: preflight, smoke (8 questions), gate, full run,
# SUMMARY.md, checksums, then the whole runs/<tag>/ is made read-only.
#
# Usage (from anywhere; the agent is allowed exactly this form):
#   sbatch /home/scratch.thannan_wwfo/codec_reproduce/autogaze_lmms_eval/scripts/run_eval.sbatch \
#       --model qwen3_5|nemotron_omni --dataset egoschema [--smoke-only] <tag>
#
# Everything that matters is fixed here, not taken from the caller: partition,
# GPUs, conda envs, weights, arms, frame counts, warmup video. The settings follow
# AutoGaze/RESULTS_LATENCY_ACCURACY.md (pareto/main 9899fd6): its conda env
# (nemo-omni), its x86_64 native build, its weights, Qwen 128 frames, Omni 64
# frames at 675 patches, arms dense and the strict-L1 hybrid. lmms-eval
# (d6cc2b5) supplies prompt, parser and scoring. Like the reference, the arms run
# one after another in the same job on one 80 GB H100 (the partition's nodes have
# one GPU each), so dense and hybrid are compared on the same host; jobs for
# different models run in parallel on different nodes.
#
# Writes only runs/<tag>/ (via run_egoschema_headline.sh) and the Slurm log in
# runs/_slurm/. Only [status] lines go to the log; numbers stay in SUMMARY.md.
#SBATCH --partition=H100_80G_HBM3_SXM_AMD_preprod
#SBATCH --gres=gpu:H100:1,gpu_memory:80G
#SBATCH --time=04:00:00
#SBATCH --job-name=run_eval
#SBATCH --output=/home/scratch.thannan_wwfo/codec_reproduce/runs/_slurm/run_eval_%j.log
set -euo pipefail
# Nothing from the caller's environment steers the run: drop every MC_* and
# Python path override first; everything the run needs is set below.
for v in $(compgen -e | grep -E '^(MC_|PYTHON)' || true); do unset "$v"; done

WORK=/home/scratch.thannan_wwfo/codec_reproduce
PKG=$WORK/autogaze_lmms_eval
LAUNCHER=$PKG/scripts/run_egoschema_headline.sh
CONDA_BASE=/home/scratch.thannan_wwfo/anaconda3
SERVER_ENV=nemo-omni
CLIENT_ENV=lmms-eval-bridge
HF_HOME=/home/scratch.thannan_wwfo/hf_cache
MC_NATIVE_BUILD=/home/scratch.thannan_wwfo/model_compat/native_build_x86_64
MC_FLA=/home/scratch.thannan_wwfo/model_compat/pylibs/fla
WARMUP_VIDEO=$HF_HOME/egoschema/videos/001934bb-81bd-4cd8-a574-0472ef3f6678.mp4
MODES="dense codec_mv_hybrid"
SMOKE_N=8
PIN_AUTOGAZE=9899fd6829444fdd5ebc86366fd35073816ac635
PIN_LMMS_EVAL=d6cc2b567db7aeacbbf1af11b6595676e784d1d3
QWEN_PATH=$HF_HOME/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
OMNI_PATH=/home/scratch.thannan_wwfo/model_compat/weights/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16

die() { echo "[run_eval] ERROR: $*" >&2; exit 1; }
log() { echo "[run_eval] $(date +%H:%M:%S) $*"; }

MODEL="" DATASET="" SMOKE_ONLY=0 TAG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL=${2:-}; shift ;;
    --dataset) DATASET=${2:-}; shift ;;
    --smoke-only) SMOKE_ONLY=1 ;;
    -*) die "unknown option $1" ;;
    *) [ -z "$TAG" ] || die "more than one tag: $TAG $1"; TAG=$1 ;;
  esac
  shift
done
case "$MODEL" in
  qwen3_5) NVF=128 MODEL_PATH=$QWEN_PATH MC_PYTHONPATH=$MC_FLA KERNELS="fla causal_conv1d" ;;  # reference: PYTHONPATH=$WT:pylibs/fla
  nemotron_omni) NVF=64 MODEL_PATH=$OMNI_PATH MC_PYTHONPATH="" KERNELS="mamba_ssm timm open_clip" ;;  # reference: PYTHONPATH=$WT only
  *) die "--model must be qwen3_5 or nemotron_omni" ;;
esac
[ "$DATASET" = egoschema ] || die "--dataset must be egoschema (the only dataset wired up so far)"
[[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "tag must be a plain name ([A-Za-z0-9._-]): '$TAG'"
SUITE=$WORK/runs/$TAG
[ ! -e "$SUITE" ] || die "$SUITE already exists; use a new tag"

export CONDA_BASE SERVER_ENV CLIENT_ENV HF_HOME MC_NATIVE_BUILD MC_PYTHONPATH
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "$SUITE"
trap 'chmod -R a-w "$SUITE" 2>/dev/null || true' EXIT   # results are never left writable

# ---- preflight (read-only checks; any failure stops before a GPU is used) ----
PRE=$SUITE/preflight.txt
fail=0
chk() {  # chk <label> <command...>: record ok/FAIL
  if out=$("${@:2}" 2>&1); then echo "ok   $1 ${out:+($out)}"; else echo "FAIL $1 ${out:+($out)}"; fail=1; fi
}
inenv() { bash -c 'eval "$("$0"/bin/conda shell.bash hook)" && conda activate "$1" && shift && exec "$@"' "$CONDA_BASE" "$@"; }
{
  echo "tag=$TAG model=$MODEL dataset=$DATASET nvf=$NVF modes=$MODES smoke_n=$SMOKE_N smoke_only=$SMOKE_ONLY"
  echo "host=$(hostname) slurm_job=${SLURM_JOB_ID:-none} date=$(date -Is) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "gpus=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sort | uniq -c | tr -s ' ' | paste -sd';')"
  echo "server_env=$CONDA_BASE/envs/$SERVER_ENV client_env=$CONDA_BASE/envs/$CLIENT_ENV"
  echo "model_path=$MODEL_PATH mc_native_build=$MC_NATIVE_BUILD mc_pythonpath=${MC_PYTHONPATH:-none}"
  echo "bridge_commit=$(git -C "$PKG" rev-parse HEAD) bridge_dirty=$(git -C "$PKG" status --porcelain | wc -l)"
  chk "x86_64 node"          test "$(uname -m)" = x86_64
  chk "H100 GPUs"            bash -c 'nvidia-smi -L | grep -q H100'
  chk "1 GPU, 80 GB"          bash -c '[ "$(nvidia-smi -L | wc -l)" -ge 1 ] && [ "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)" -ge 79000 ]'
  chk "HF_TOKEN set"         test -n "${HF_TOKEN:-}"
  chk "AutoGaze at pin"      test "$(git -C "$WORK/AutoGaze" rev-parse HEAD)" = "$PIN_AUTOGAZE"
  chk "lmms-eval at pin"     test "$(git -C "$WORK/lmms-eval" rev-parse HEAD)" = "$PIN_LMMS_EVAL"
  # Code must match the pins; editor/agent config folders (.claude/, .vscode/) don't count, but are recorded.
  for r in AutoGaze lmms-eval; do
    echo "git_status_$r=$(git -C "$WORK/$r" status --porcelain --untracked-files=no | tr '\n' ';')"
    chk "$r code clean" test -z "$(git -C "$WORK/$r" status --porcelain --untracked-files=no -- . ':!.claude' ':!.vscode')"
  done
  chk "native dump_cu"       test -x "$MC_NATIVE_BUILD/dump_cu"
  chk "weights"              test -f "$MODEL_PATH/config.json"
  chk "egoschema videos"     bash -c "[ \$(ls '$HF_HOME/egoschema/videos' | wc -l) -ge 5000 ]"
  chk "warmup video"         test -f "$WARMUP_VIDEO"
  chk "server env imports"   inenv "$SERVER_ENV" env PYTHONPATH="$WORK/AutoGaze${MC_PYTHONPATH:+:$MC_PYTHONPATH}" python -c '
import importlib, sys, torch, transformers, av, model_compat
for m in sys.argv[1:]: importlib.import_module(m)
print(torch.__version__, transformers.__version__, av.__version__, "cuda" if torch.cuda.is_available() else "NO CUDA")
sys.exit(0 if torch.cuda.is_available() else 1)' $KERNELS
  chk "client env imports"   inenv "$CLIENT_ENV" python -c \
      'import lmms_eval, autogaze_lmms_eval, os; from lmms_eval.models import get_model; get_model("autogaze"); print(os.path.dirname(lmms_eval.__file__))'
  chk "warmup not evaluated" inenv "$CLIENT_ENV" python -c '
import glob, sys, pyarrow.parquet as pq
p = glob.glob(sys.argv[1] + "/hub/datasets--lmms-lab--egoschema/snapshots/*/Subset/*.parquet")
ids = set(pq.read_table(p[0], columns=["video_idx"]).column("video_idx").to_pylist())
sys.exit(f"{sys.argv[2]} is one of the {len(ids)} evaluated videos" if sys.argv[2] in ids else 0)' "$HF_HOME" "$(basename "$WARMUP_VIDEO" .mp4)"
} > "$PRE"
cat "$PRE"
[ "$fail" = 0 ] || die "preflight failed (see $PRE); nothing was run"

GPU=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | cut -d, -f1)
BASE_PORT=$((20000 + (${SLURM_JOB_ID:-$$} % 4000) * 4))
launch() {  # launch <stage> <mode> [extra launcher args...]: one arm, into runs/<tag>/<stage>/<mode>/
  mkdir -p "$SUITE/$1"
  "$LAUNCHER" --model "$MODEL" --model-path "$MODEL_PATH" --nvf "$NVF" --modes "$2" --gpus "$GPU" \
    --base-port "$BASE_PORT" --warmup-video "$WARMUP_VIDEO" --server-env "$SERVER_ENV" \
    --mc-pythonpath "$MC_PYTHONPATH" --runs-dir "$SUITE/$1" "${@:3}" "$2"
}
gate() {  # gate <stage> <mode>: the arm's [status] line must be all ok
  grep -qx "\[status\] $2: timing ok, client log ok, e2e cross-check ok, replies ok" "$SUITE/$1/$2/logs/headline_table.log"
}
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
  find smoke full -mindepth 2 -maxdepth 2 -name SHA256SUMS -print0 2>/dev/null | sort -z | xargs -0 -r sha256sum ) > "$SUITE/SHA256SUMS"
[ "$FAILED" = 0 ] || die "full run finished with failed arms; SUMMARY.md has what completed (see the [run_eval] lines above)"
log "done: $SUITE/SUMMARY.md (read-only)"
