#!/usr/bin/env bash
# Hold idle GPUs with salloc and run the 20-question main-arm smoke runs inside them (EXTENSION_SPEC.md section 6).
# Run from a login node that can reach the Slurm controller:
#   bash /home/scratch.thannan_wwfo/codec_reproduce/autogaze_lmms_eval/scripts/hold_and_smoke.sh [queued job ids to cancel...]
# NVILA: 3 GB200 GPUs on one gb200nvl72_cx8 node, one smoke run per GPU in parallel.
# Qwen:  1 H100 on H100_preprod, the three smoke runs one after another.
# Allocations stay held (8 h) after the smoke runs end; release them with scancel <id>.
# Logs: runs/_slurm/run_ext_<alloc>_<model>_<task>.log
set -euo pipefail
WORK=/home/scratch.thannan_wwfo/codec_reproduce
EXT=$WORK/autogaze_lmms_eval/scripts/run_ext.sbatch
LOGS=$WORK/runs/_slurm
TASKS="egoschema_subset nextqa_mc_test lvbench"
DATE=$(date +%Y%m%d_%H%M)

hold() {  # hold <partition> <gpus> <name>: prints the allocation id
  salloc --no-shell -p "$1" --gres=gpu:"$2" -N1 -t 08:00:00 -J "$3" 2>&1 \
    | tee /dev/stderr | sed -n 's/.*Granted job allocation \([0-9]*\).*/\1/p'
}

if [ $# -gt 0 ]; then echo "cancelling queued duplicates: $*"; scancel "$@"; fi

NV=$(hold gb200nvl72_cx8 3 hold_nvila)
[ -n "$NV" ] || { echo "GB200 allocation failed" >&2; exit 1; }
for t in $TASKS; do
  short=${t%%_*}
  nohup srun --jobid="$NV" -N1 -n1 --gres=gpu:1 --exact \
    bash "$EXT" --model nvila --task "$t" --arms main --smoke-only "dev_ext_nvila_${short}_smoke20_$DATE" \
    > "$LOGS/run_ext_${NV}_nvila_${t}.log" 2>&1 &
done
echo "NVILA smoke runs started in allocation $NV"

QW=$(hold H100_preprod 1 hold_qwen)
[ -n "$QW" ] || { echo "H100 allocation failed" >&2; exit 1; }
nohup srun --jobid="$QW" -N1 -n1 --gres=gpu:1 bash -c "
  for t in $TASKS; do
    s=\${t%%_*}
    bash '$EXT' --model qwen3_5 --task \$t --arms main --smoke-only dev_ext_qwen3_5_\${s}_smoke20_$DATE \
      > '$LOGS'/run_ext_${QW}_qwen3_5_\$t.log 2>&1 || echo \"\$t failed\"
  done" > "$LOGS/run_ext_${QW}_qwen3_5_chain.log" 2>&1 &
echo "Qwen smoke runs started in allocation $QW (sequential)"
echo "held: GB200 $NV, H100 $QW. Release with: scancel $NV $QW"
