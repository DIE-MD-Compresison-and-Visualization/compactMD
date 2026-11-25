import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
import sys
import os
import random
from tqdm.auto import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from MDAnalysis import Universe, Writer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse 
import shutil
from natsort import natsorted
from glob import glob
import json
from pathlib import Path
import time
from contextlib import contextmanager
from datetime import datetime

from compactmd.compresstraj.classes import *
from compactmd.compresstraj.helpers import *

# Timing context manager
@contextmanager
def timer(name):
    """Context manager to measure execution time"""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    elapsed = end - start
    print(f"{name}: {elapsed:.4f} seconds", flush=True)
    timings[name] = elapsed

# Dictionary to store all timings
timings = {}

# Start total time
total_start = time.perf_counter()

# some stuff that effects the whole code
torch.set_float32_matmul_precision('medium')
set_seed()
# some stuff that effects the whole code

# argument parsing
parser = argparse.ArgumentParser(description="Process input files and model files for compression.")
    
# Add arguments for model, reffile, trajfile, scaler, and output with short and long options
parser.add_argument('-r', '--reffile', type=str, required=True, help='Path to the reference file (pdb/gro)')
parser.add_argument('-t', '--trajfile', type=str, required=True, help='Path to the trajectory file (xtc/trr/dcd/xyz)')
parser.add_argument('-p', '--prefix', type=str, required = True, help='prefix to to the files to be generated.')
parser.add_argument('-e', '--epochs', type=str, help='Number of epochs to train [default=200]', default=200)
parser.add_argument('-b', '--batch', type=str, help='batch size [default=128]', default=128)
parser.add_argument('-l', '--latent', type=str, help='Number of latent dims', default=None)
parser.add_argument('-c', '--compression', type=str, help='Extent of compression to achieve if latent dimension is not specified. [default = 20]', default=20)
parser.add_argument('-sel', '--selection', type=str, default="all",
                    help="a list of selections. the training will treat each selection as a separated entity.")
parser.add_argument('-gid', '--gpuID', type=str, help="select GPU to use [default=0]", default="0")
parser.add_argument('-ckpt', '--ckpt', type=str, help="checkpoint from where to resume training", default=None)
parser.add_argument('--layers', '-layers', type=str, default="4096,1024",
                    help="Comma-separated list of hidden layer sizes for the autoencoder, e.g., '4096,1024'")
parser.add_argument('-o', '--outdir', type=str, help='output directory', default=".")
parser.add_argument('-log','--log-file', type=str, help='CSV file to store metrics [default=compression_metrics.csv]', 
                    default="compression_metrics.csv")
parser.add_argument('-ct', '--compress-trajfile', type=str, default=None, help='Path to trajectory file used only for compression. If not given, uses --trajfile')

args = parser.parse_args()

reffile = args.reffile
trajfile = args.trajfile
compress_trajfile = args.compress_trajfile if args.compress_trajfile is not None else trajfile
prefix = args.prefix
latent = args.latent
compression = int(args.compression)
selection = args.selection
gpu_id = [int(idx) for idx in args.gpuID.split(",")]
layers = list(map(int, args.layers.strip().split(",")))
outdir = args.outdir
log_file = args.log_file

os.makedirs(outdir, exist_ok=True)

#device = torch.device(f"cuda:{'.'.join([str(idx) for idx in gpu_id])}") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device(f"cuda:{gpu_id[0]}") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}.", flush=True)

# INITIALIZATION PHASE
with timer("Initialization Time"):
    # processing trajectory
    traj = Universe(reffile, trajfile)
    scaler = StandardScaler()
    scaler.fit(traj, selection=selection)

    indices = traj.select_atoms(selection).indices
    train_loader = DataLoader(TrajLoader(traj, indices, scaler), batch_size=int(args.batch), shuffle=True, drop_last=True)

    pickle.dump(scaler, open(f"{outdir}/{prefix}_scaler.pkl", "wb"))
    print(f"""scaler saved at {outdir}/{prefix}_scaler.pkl""", flush=True)

    refsuffix = Path(reffile).suffix

    with Writer(f"{outdir}/{prefix}_select{refsuffix}", "w") as w:
        u = Universe(reffile)
        w.write(u.select_atoms(selection))
    print(f"selected atom coordinates saved at {outdir}/{prefix}_select{refsuffix}", flush=True)

    # training AE setup
    N = train_loader.dataset.get_N()
    latent_dim = int(np.ceil(N/compression)) if latent is None else int(args.latent)
    if latent_dim < 2:
        latent_dim = 2

    u = Universe(reffile)

    ae = DenseAutoEncoder(N=N, latent=latent_dim, layers=layers)
    loss_fn = RMSDLoss()
    model = LightAutoEncoder(model=ae, loss_fn=loss_fn, learning_rate=3e-4)

    # restart_ckpt_callback = ModelCheckpoint(
    #     dirpath=outdir,
    #     filename=f"restart_{prefix}",
    #     every_n_epochs=1000,
    #     save_top_k=1,
    #     save_weights_only=False,
    # )

    accelerator="gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(default_root_dir=outdir,max_epochs=int(args.epochs),
                         accelerator=accelerator,
                         devices=gpu_id if accelerator == "gpu" else [0],
                         precision="bf16-mixed", enable_checkpointing=False, logger=None,
                        #  callbacks=[restart_ckpt_callback]
                         )

# TRAINING PHASE
with timer("Training Time"):
    trainer.fit(model, train_loader, ckpt_path=args.ckpt if args.ckpt else None)

    ## saving model and losses
    torch.save(model.model.to(dtype=torch.bfloat16).cpu(), f"{outdir}/{prefix}_model_bf16.pt")
    pickle.dump({"train_loss":model.train_loss},
                  open(f"{outdir}/{prefix}_losses.pkl", "wb"))

    print(f"""Final model saved at {outdir}/{prefix}_model_bf16.pt
Losses saved at {outdir}/{prefix}_losses.pkl""", flush=True)

del train_loader

# COMPRESSION PHASE
with timer("Compression Time"):
    compress_traj = Universe(reffile, compress_trajfile)

    loader = DataLoader(TrajLoader(compress_traj, indices, scaler), batch_size=64, shuffle=False)

    latent = []

    atoms_compress = compress_traj.select_atoms(selection)
    cog = np.array([
        atoms_compress.center_of_geometry()
        for _ in compress_traj.trajectory
    ])
    cog_path = f"{outdir}/{prefix}_cog.npy"
    np.save(cog_path, cog)
    print(f"Saved COG at {cog_path}", flush=True)

    compress_traj.trajectory[0]

    model = torch.load(f"{outdir}/{prefix}_model_bf16.pt", map_location="cpu", weights_only=False)
    encoder = model.encoder
    del model
    encoder = encoder.to(device)
    print("Encoder dtype:", next(encoder.parameters()).dtype)


    with torch.no_grad():
        encoder.eval()
        for Pos in loader:
            Pos = Pos.to(device=device, dtype=torch.float32)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                lat = encoder(Pos)
            latent.append(lat.cpu().to(torch.bfloat16)) 

    with open(f"{outdir}/{prefix}_compressed.pkl", "wb") as f:
        pickle.dump(latent, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved compressed data at {outdir}/{prefix}_compressed.pkl", flush=True)

    with open(f"{outdir}/{prefix}_config.json", "w") as f:
        json.dump({"selection": selection}, f, indent=4)

    print(f"Saved selections to {outdir}/{prefix}_config.json", flush=True)

# Calculate total time
total_time = time.perf_counter() - total_start
timings["Total Time"] = total_time

with timer("Metric Time"):

    # CALCULATE COMPRESSION RATIO
    original_traj_size = os.path.getsize(compress_trajfile)
    compressed_size = os.path.getsize(f"{outdir}/{prefix}_compressed.pkl")
    compression_ratio = original_traj_size / compressed_size

    print("\n" + "="*60)
    print("TIMING SUMMARY")
    print("="*60)
    for name, duration in timings.items():
        print(f"{name:25s}: {duration:10.4f} seconds")
    print("="*60)
    print(f"Compression Ratio: {compression_ratio:.2f}x")
    print(f"Original Size: {original_traj_size / (1024**2):.2f} MB")
    print(f"Compressed Size: {compressed_size / (1024**2):.2f} MB")
    print("="*60 + "\n")

    # SAVE TO CSV using built-in csv module
    metrics_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'prefix': prefix,
        'training_trajectory_file': trajfile,
        'compression_trajectory_file': compress_trajfile,
        'reference_file': reffile,
        'selection': selection,
        'epochs': args.epochs,
        'batch_size': args.batch,
        'latent_dimensions': latent_dim,
        'compression_target': compression,
        'layers': args.layers,
        'initialization_time_sec': timings.get("Initialization Time", 0),
        'training_time_sec': timings.get("Training Time", 0),
        'compression_time_sec': timings.get("Compression Time", 0),
        'total_time_sec': timings.get("Total Time", 0),
        'original_size_mb': original_traj_size / (1024**2),
        'compressed_size_mb': compressed_size / (1024**2),
        'compression_ratio': compression_ratio,
        'device': str(device),
    }

    # Define column order
    fieldnames = list(metrics_data.keys())

    # Create or append to CSV
    file_exists = os.path.exists(log_file)
    with open(log_file, 'a', newline='') as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
            print(f"Metrics saved to new file: {log_file}")
        else:
            print(f"Metrics appended to existing file: {log_file}")
        writer.writerow(metrics_data)

    print(f"\nAll timing and compression metrics saved to: {log_file}")