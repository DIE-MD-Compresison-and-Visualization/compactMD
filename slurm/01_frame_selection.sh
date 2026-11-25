#!/bin/bash
# Stage 1: select a representative training subset of frames.
# Example usage on a CSC-style cluster:
#   sbatch --account=<project> slurm/01_frame_selection.sh <topology.pdb> <trajectory.trr> <outdir>
#SBATCH --job-name=compactmd_select
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

TOP=${1:?topology file (pdb/gro)}
TRAJ=${2:?trajectory file (trr/xtc)}
OUTDIR=${3:?output directory}

# Keep numerical libraries single-threaded for reproducibility.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Activate the selection environment, e.g.:
#   conda activate compactmd-selection
python -m compactmd.frame_selection --top "$TOP" --traj "$TRAJ" --outdir "$OUTDIR"
# Writes $OUTDIR/downsampled_subset.trr, downsampled_subset_indices.npy,
# selected_params.json and subset_summary.txt.
