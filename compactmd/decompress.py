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
from MDAnalysis.analysis import align
import MDAnalysis as mda
import pytorch_lightning as pl
import argparse
import warnings
import shutil
from glob import glob
import time
from contextlib import contextmanager
from datetime import datetime
import csv
from compactmd.compresstraj.classes import *
from compactmd.compresstraj.helpers import *

warnings.filterwarnings("ignore")
set_seed()

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

parser = argparse.ArgumentParser(description="Process input parameters.")
parser.add_argument("-m", "--model", type=str, required=True, help="Path to the model file.")
parser.add_argument("-s", "--scaler", type=str, required=True, help="Path to the scaler file.")
parser.add_argument("-r", "--reffile", type=str, required=True, help="Path to the reference file.")
parser.add_argument('-t', '--trajfile', type=str, help='Path to the trajectory file 1 (xtc)')
parser.add_argument("-c", "--compressed", type=str, required=True, help="Path to the compressed file.")
parser.add_argument("-p", "--prefix", type=str, required=True, help="output file prefix.")
parser.add_argument('-sel', '--selection', type=str, default="all",
                    help="a list of selections. the training will treat each selection as a separated entity.")
parser.add_argument('-gid', '--gpuID', type=str, help="select GPU to use [default=0]", default="0")
parser.add_argument('-o', '--outdir', type=str, help='output directory', default=".")
parser.add_argument('-cog', '--cog', type=str, default=None, help='center of geometry.')
parser.add_argument('-ox', '--outext', type=str, help='output trajectory format [default = xtc]', default="xtc")
parser.add_argument('-log','--log-file', type=str, help='CSV file to store metrics [default=compression_metrics.csv]', 
                    default="compression_metrics.csv")

args = parser.parse_args()

modelpath = args.model
scalerpath = args.scaler
reffile = args.reffile
trajfile = args.trajfile
compressed = args.compressed
prefix = args.prefix
selection = args.selection
gpu_id = [int(idx) for idx in args.gpuID.split(",")]
outdir = args.outdir
cog_path = args.cog
ox_suffix = args.outext
log_file = args.log_file

#device = torch.device(f"cuda:{'.'.join([str(idx) for idx in gpu_id])}") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device(f"cuda:{gpu_id[0]}") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}.", flush=True)

# INITIALIZATION PHASE
with timer("Initialization Time"):
    ref = Universe(reffile)
    select = ref.select_atoms(selection)
    n_atoms = select.n_atoms
    
    model = torch.load(modelpath, map_location="cpu", weights_only=False)
    decoder = model.decoder.to(device)
    decoder.eval()  
    del model
    
    latent = pickle.load(open(compressed, "rb"))
    scaler = pickle.load(open(scalerpath, "rb"))
    
    if cog_path:
        cog = np.load(cog_path)
    
    batch_size = latent[0].shape[0]
    n_frames = sum(lat.shape[0] for lat in latent)

# DECOMPRESSION PHASE
print("Decompressing.....", flush=True)
output_file = f"{outdir}/{prefix}_decompressed.{ox_suffix}"

with timer("Decompression Time"):
    with mda.Writer(output_file, n_atoms) as w:
        with torch.no_grad():
            for frame_idx, lat in enumerate(latent):
                if not torch.is_tensor(lat):
                    lat = torch.from_numpy(lat)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pos = decoder(lat.to(device))
                pos = pos.float().cpu().numpy()
                pos = pos.reshape((pos.shape[0], pos.shape[1]//3, 3))
                pos = scaler.inverse_transform(pos)
                pos -= pos.mean(axis=1, keepdims=True)
                
                if cog_path:
                    cog_batch = cog[frame_idx*batch_size:(frame_idx + 1)*batch_size]
                    pos += np.expand_dims(cog_batch, axis=1)
                
                for p in pos:
                    select.positions = p
                    w.write(select)

print(f"decompressed trajectory saved to {output_file}.", flush=True)

with timer("Metric Time"):
    # Calculate total time
    total_time = time.perf_counter() - total_start
    timings["Total Time"] = total_time

    # Calculate file sizes
    compressed_size = os.path.getsize(compressed)
    decompressed_size = os.path.getsize(output_file)
    expansion_ratio = decompressed_size / compressed_size

    print("\n" + "="*60)
    print("TIMING SUMMARY")
    print("="*60)
    for name, duration in timings.items():
        print(f"{name:25s}: {duration:10.4f} seconds")
    print("="*60)
    print(f"Expansion Ratio: {expansion_ratio:.2f}x")
    print(f"Compressed Size: {compressed_size / (1024**2):.2f} MB")
    print(f"Decompressed Size: {decompressed_size / (1024**2):.2f} MB")
    print(f"Total Frames: {n_frames}")
    print(f"Frames/Second: {n_frames / timings.get('Decompression Time', 1):.2f}")
    print("="*60 + "\n")

    # SAVE TO CSV using built-in csv module
    metrics_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'prefix': prefix,
        'compressed_file': compressed,
        'model_file': modelpath,
        'scaler_file': scalerpath,
        'reference_file': reffile,
        'selection': selection,
        'output_format': ox_suffix,
        'n_atoms': n_atoms,
        'n_frames': n_frames,
        'initialization_time_sec': timings.get("Initialization Time", 0),
        'decompression_time_sec': timings.get("Decompression Time", 0),
        'total_time_sec': timings.get("Total Time", 0),
        'frames_per_second': n_frames / timings.get("Decompression Time", 1),
        'compressed_size_mb': compressed_size / (1024**2),
        'decompressed_size_mb': decompressed_size / (1024**2),
        'expansion_ratio': expansion_ratio,
        'device': str(device),
    }

    # Define column order
    fieldnames = list(metrics_data.keys())

    # Create or append to CSV
    file_exists = os.path.exists(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
            print(f"Metrics saved to new file: {log_file}")
        else:
            print(f"Metrics appended to existing file: {log_file}")
        writer.writerow(metrics_data)

    print(f"\nAll timing and decompression metrics saved to: {log_file}")

if trajfile is None:
    exit(0)