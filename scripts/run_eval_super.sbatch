#!/usr/bin/env bash
# run_eval.sbatch for Nemotron-3.5-Super-VL, whose BF16 weights are split over
# 4x H100 80 GB and need at least 480 GB of RAM to load (the reference sweep's
# resources: AutoGaze/RESULTS_LOG.md, "Super VL (4x H100)"). This file only asks
# Slurm for those resources; everything else is run_eval.sbatch.
#
# Usage (from anywhere; the agent is allowed exactly this form):
#   sbatch /home/scratch.thannan_wwfo/codec_reproduce/autogaze_lmms_eval/scripts/run_eval_super.sbatch \
#       --model nemotron_super --dataset egoschema [--smoke-only] <tag>
# dgxh100_preprod (one DGX H100, viking-dvt-183) has the same H100 SXM 80 GB as dgxh100, whose
# queue was full; the smoke run passed there (job 2883579). Both mount /home/scratch.thannan_wwfo
# (code, envs, videos, this log); H100x8 nodes don't, so a job there dies before writing any log
# (job 2881089, checked with srun on 4u8g-gen-0082).
#SBATCH --partition=dgxh100_preprod
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --job-name=run_eval_super
#SBATCH --output=/home/scratch.thannan_wwfo/codec_reproduce/runs/_slurm/run_eval_%j.log
set -euo pipefail
# The 1-GPU models go through run_eval.sbatch itself, not a 4-GPU allocation.
[[ " $* " == *" --model nemotron_super "* ]] || { echo "[run_eval] ERROR: run_eval_super.sbatch is for --model nemotron_super only" >&2; exit 1; }
exec bash /home/scratch.thannan_wwfo/codec_reproduce/autogaze_lmms_eval/scripts/run_eval.sbatch "$@"
