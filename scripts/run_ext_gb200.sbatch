#!/usr/bin/env bash
# run_ext.sbatch for NVILA-8B-HD-Video on one GB200 (aarch64, the auto_gaze env). This file only asks Slurm
# for that GPU; everything else is run_ext.sbatch.
#
# Usage:
#   sbatch /home/scratch.thannan_wwfo/codec_reproduce/autogaze_lmms_eval/scripts/run_ext_gb200.sbatch \
#       --model nvila --task <task> --arms <set> [--smoke-only] <tag>
# gb200nvl72_preprod is a Santa Clara partition that mounts /home/scratch.thannan_wwfo (LMMS_EVAL_BRIDGE.md).
#SBATCH --partition=gb200nvl72_preprod
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=run_ext_gb200
#SBATCH --output=/home/scratch.thannan_wwfo/codec_reproduce/runs/_slurm/run_ext_%j.log
set -euo pipefail
[[ " $* " == *" --model nvila "* ]] || { echo "[run_ext] ERROR: run_ext_gb200.sbatch is for --model nvila only" >&2; exit 1; }
exec bash /home/scratch.thannan_wwfo/codec_reproduce/autogaze_lmms_eval/scripts/run_ext.sbatch "$@"
