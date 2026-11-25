#!/bin/bash
# Stages 2-3: train the autoencoder on the selected subset (bf16 mixed precision),
# then compress every frame of the full trajectory.
# Example usage:
#   sbatch --account=<project> slurm/02_train_compress.sh <topology.pdb> <trajectory.trr> <subset.trr> <prefix> <outdir>
#SBATCH --job-name=compactmd_train
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1,nvme:100

set -euo pipefail

TOP=${1:?topology file (pdb/gro)}
FULL_TRAJ=${2:?full trajectory to compress}
SUBSET_TRAJ=${3:?training subset from stage 1 (downsampled_subset.trr)}
PREFIX=${4:?prefix for output files}
OUTDIR=${5:?output directory}

# Settings used in the paper.
EPOCHS=1000
BATCH=128
COMPRESSION=20   # latent size = ceil(3N / COMPRESSION)

# Copy inputs to fast node-local storage if available.
WORK=${LOCAL_SCRATCH:-$(mktemp -d)}/${SLURM_JOB_ID:-local}
mkdir -p "$WORK" "$OUTDIR"
cp "$TOP" "$FULL_TRAJ" "$SUBSET_TRAJ" "$WORK/"

# Activate the training environment, e.g.:
#   conda activate compactmd-train
python -m compactmd.compress \
  -r "$WORK/$(basename "$TOP")" \
  -t "$WORK/$(basename "$SUBSET_TRAJ")" \
  -ct "$WORK/$(basename "$FULL_TRAJ")" \
  -p "$PREFIX" -e "$EPOCHS" -b "$BATCH" -c "$COMPRESSION" \
  -o "$OUTDIR" -log "$OUTDIR/${PREFIX}_metrics.csv"
