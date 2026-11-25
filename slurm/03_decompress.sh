#!/bin/bash
# Reconstruct a trajectory from the files written by stage 2-3.
# Example usage:
#   sbatch --account=<project> slurm/03_decompress.sh <prefix> <outdir>
#SBATCH --job-name=compactmd_decompress
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:a100:1

set -euo pipefail

PREFIX=${1:?prefix used in stage 2-3}
OUTDIR=${2:?directory holding the stage 2-3 outputs}
FORMAT=${3:-trr}

# Activate the training environment, e.g.:
#   conda activate compactmd-train
python -m compactmd.decompress \
  -m "$OUTDIR/${PREFIX}_model_bf16.pt" \
  -s "$OUTDIR/${PREFIX}_scaler.pkl" \
  -c "$OUTDIR/${PREFIX}_compressed.pkl" \
  -cog "$OUTDIR/${PREFIX}_cog.npy" \
  -r "$OUTDIR/${PREFIX}_select.pdb" \
  -p "$PREFIX" -o "$OUTDIR" -ox "$FORMAT" \
  -log "$OUTDIR/${PREFIX}_decompress_metrics.csv"
