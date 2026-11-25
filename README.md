# CompactMD

CompactMD lowers the training and storage cost of autoencoder-based compression of molecular dynamics (MD) trajectories. It keeps the fully connected autoencoder of [compressTraj](https://github.com/SerpentByte/compresstraj) (Wasim, Schäfer & Mondal) and changes how that autoencoder is trained and stored:

1. **Representative-frame selection.** The model is trained on a subset of frames selected in a space of slow collective motions, rather than on every frame. The selection pipeline is: backbone φ/ψ torsions → PCA → TICA → k-means, then a mix of time-uniform and cluster-based sampling.
2. **bfloat16 mixed-precision training.** The trained model and the latent vectors are stored in bfloat16.

The encoder is still applied to **every** frame, so no frames are dropped from the compressed trajectory.

> Frame selection uses backbone torsion angles, so CompactMD currently supports **proteins only**.

## Installation

Stage 1 (selection) and stages 2–3 (training and compression) use separate environments, because PyEMMA needs an older Python.

```bash
git clone https://github.com/DIE-MD-Compresison-and-Visualization/compactMD.git
cd compactMD

conda env create -f environment-selection.yml   # stage 1
conda env create -f environment-train.yml       # stages 2–3
# install the package into each environment:
conda run -n compactmd-selection pip install -e .
conda run -n compactmd-train pip install -e .
```

Stages 2–3 need a GPU with bfloat16 support (for example an NVIDIA A100).

## Usage

Inputs are a topology (`.pdb`/`.gro`) and a trajectory (`.trr`/`.xtc`).

### 1. Select training frames
```bash
conda activate compactmd-selection
python -m compactmd.frame_selection --top topology.pdb --traj trajectory.trr --outdir selection/
```
This writes `selection/downsampled_subset.trr`, the selected frame indices, `selected_params.json` and `subset_summary.txt`. The subset size and the time-uniform/cluster mix are chosen automatically. Run `--help` for all options.

### 2–3. Train and compress
```bash
conda activate compactmd-train
python -m compactmd.compress \
  -r topology.pdb \
  -t selection/downsampled_subset.trr \
  -ct trajectory.trr \
  -p myprotein -e 1000 -b 128 -c 20 -o out/
```
`-t` is the training subset; `-ct` is the full trajectory to compress.
Outputs in `out/`:
- `myprotein_compressed.pkl`: bf16 latent vectors
- `myprotein_model_bf16.pt`: trained model
- `myprotein_scaler.pkl`: normalisation parameters
- `myprotein_cog.npy`: per-frame centre of geometry
- `myprotein_select.pdb`: the selected atoms

All five files are needed for decompression. Count all of them when you report storage cost: the model is often larger than the compressed trajectory.

### Decompress
```bash
python -m compactmd.decompress \
  -m out/myprotein_model_bf16.pt -s out/myprotein_scaler.pkl \
  -c out/myprotein_compressed.pkl -cog out/myprotein_cog.npy \
  -r out/myprotein_select.pdb -p myprotein -o out/ -ox trr
```

Example Slurm scripts for these three steps are in [`slurm/`](slurm/).

## Repository layout
```
compactmd/
  frame_selection.py   stage 1: representative-frame selection
  compress.py          stages 2–3: bf16 training + compression
  decompress.py        reconstruction
  compresstraj/        autoencoder, loss, scaler (from compressTraj, GPL-3.0)
slurm/                 example batch scripts
evaluation/            fidelity metrics (RMSD, PCA/Wasserstein)
```

## Citation
If you use CompactMD, please cite our paper (reference to be added) and compressTraj:

> Wasim, A., Schäfer, L. V., & Mondal, J. compressTraj. bioRxiv (2024). https://doi.org/10.1101/2024.09.15.613125

## License
GPL-3.0. See [LICENSE](LICENSE). CompactMD includes code from compressTraj, which is also GPL-3.0.
