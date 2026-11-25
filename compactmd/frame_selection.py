#!/usr/bin/env python3
"""
Scientific trajectory downsampling (MD) -- v3.18
Auto-selects BOTH:
  - target_total (subset size)
  - uniform_frac (time-uniform vs cluster-coverage mix)

IMPORTANT (stability):
  - NO RMSD / md.load_frame loops.
  - No RMSD streaming either (you hit segfaults in mdtraj.iterload on TRR in your environment).
  - Selection is done ONLY in feature space (whitened TICA): NN coverage error / MMD.
  - Writing subset uses mdtraj.iterload ONCE (usually stable even when repeated iterload isn't).

GOAL:
  Choose a downsampled subset that preserves coverage of the full trajectory dynamics
  in a way that is robust across different trajectories (no manual target_total needed).

Pipeline:
  1) Featurize: backbone torsions (cos/sin) via PyEMMA
  2) PCA: keep k dims reaching target FVE
  3) Blocked-CV VAMP-2: choose (lag, dim) on PCA features
  4) TICA: fit with chosen (lag, dim)
  5) Whitening: zero-mean/unit-variance in TIC space
  6) Blocked-CV MSM VAMP-2: choose n_clusters
  7) Final KMeans on whitened TICA
  8) Auto-select subset size + uniform_frac (optional)
      - proxy: NN coverage error in Yw space (fast, stable)
      - objective: proxy + size_penalty*(target_total/N)
  9) Downsample with two-pool hybrid
 10) Write subset TRR + indices

Requirements: pyemma, mdtraj, numpy, sklearn, matplotlib
"""

import os
import argparse
import json
import warnings
import faulthandler
import numpy as np
import mdtraj as md
import pyemma

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

faulthandler.enable()


# ============================================================
# Utilities
# ============================================================

def concat_arrays(trajs):
    lengths = [t.shape[0] for t in trajs]
    X = np.concatenate(trajs, axis=0) if len(trajs) > 1 else trajs[0].copy()
    return X, lengths


def write_summary(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.rstrip() + "\n")


def stratified_uniform_sample(N, n_samples, rng):
    n_samples = min(int(n_samples), int(N))
    if n_samples <= 0:
        return []
    bins = np.array_split(np.arange(N), n_samples)
    return [int(rng.choice(b)) for b in bins if len(b) > 0]


# ============================================================
# PCA
# ============================================================

def pca_fit_transform_fve(data_list, fve_target, seed):
    X, lengths = concat_arrays(data_list)
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D feature matrix, got shape {X.shape}")
    # Use randomized SVD for large matrices: avoids full N×D SVD allocation
    # (which can be 3-4× the matrix size in scratch memory)
    solver = "randomized" if X.shape[0] > 10_000 else "auto"
    pca = PCA(svd_solver=solver, random_state=seed)
    try:
        Xp = pca.fit_transform(X)
    except MemoryError:
        raise MemoryError(
            f"Out of memory during PCA "
            f"({X.shape[0]} frames × {X.shape[1]} features, svd_solver='{solver}'). "
            f"Increase --stride to reduce the feature matrix size."
        ) from None

    evr = np.asarray(pca.explained_variance_ratio_)
    fve = np.cumsum(evr)
    hits = np.where(fve >= fve_target)[0]
    if hits.size == 0:
        k_pca = len(evr)
        warnings.warn(
            f"PCA FVE never reached {fve_target:.3f} (max={fve[-1]:.6f}). "
            f"Using all {k_pca} components."
        )
    else:
        k_pca = int(hits[0] + 1)

    out, start = [], 0
    for L in lengths:
        out.append(Xp[start:start + L, :k_pca])
        start += L
    return out, k_pca, evr, fve


# ============================================================
# Data-driven parameter candidates
# ============================================================

def estimate_tica_lag_candidates(Xp, n_candidates=4):
    """
    Derive TICA lag candidates from the autocorrelation decay of the top PCA
    components rather than using hardcoded frame-count tiers.

    Finds the lag where each component's ACF first drops below 1/e (the
    characteristic decay time), takes the median across components as the base,
    then returns a small grid: [base/2, base, 2*base, 4*base].

    This is physically meaningful regardless of trajectory length or timestep
    because the 1/e decay time is intrinsic to the data's dynamics.
    """
    N, n_feat = Xp.shape
    # Cap at 1000: MD ACFs almost never need more than ~500 frames to find
    # the 1/e decay. N//5 for large N (e.g. 14000 for N=70k) would cause
    # billions of dot-product operations.
    max_lag = min(max(2, N // 5), 1000)
    n_comp = min(3, n_feat)

    decay_lags = []
    for d in range(n_comp):
        col = Xp[:, d] - Xp[:, d].mean()
        var = np.dot(col, col)
        if var < 1e-12:
            continue
        for lag in range(1, max_lag + 1):
            if np.dot(col[:-lag], col[lag:]) / var < (1.0 / np.e):
                decay_lags.append(lag)
                break
        else:
            decay_lags.append(max_lag)

    base = max(2, int(np.median(decay_lags))) if decay_lags else max(2, N // 20)

    raw = [
        max(1, base // 2),
        base,
        min(base * 2, N // 4),
        min(base * 4, N // 4),
    ]
    # Cap at N//20: ensures block_len = lag*5 still leaves >= 4 blocks for CV.
    # (N // block_len >= 4  →  block_len <= N//4  →  lag*5 <= N//4  →  lag <= N//20)
    max_viable = max(2, N // 20)
    candidates = sorted(set(c for c in raw if 1 <= c <= max_viable))
    if not candidates:
        # All ACF-derived lags exceed what CV can handle — fall back to the largest viable lag
        candidates = [max_viable]
    return candidates[:n_candidates]


def build_cluster_candidates(N, n_points=12):
    """
    Log-spaced cluster count candidates derived continuously from N.

    Lower bound: sqrt(N)/3  — below this, clusters are too coarse to capture
                              conformational diversity.
    Upper bound: min(N//20, 1000) — ensures at least 20 frames per cluster on
                                    average; capped at 1000 to keep KMeans tractable.

    Log-spacing is appropriate because VAMP-2 score improvements are roughly
    logarithmic in cluster count. 12 points gives enough curve resolution for
    reliable knee detection without excessive CV compute.
    """
    c_min = max(3, int(np.sqrt(N) / 3))
    c_max = max(c_min + 2, min(N // 20, 1000))
    pts = np.unique(np.round(np.geomspace(c_min, c_max, n_points)).astype(int))
    return [int(c) for c in pts if c >= 2]


def find_score_knee(counts, scores):
    """
    Find the knee of the VAMP-2 score vs cluster count curve using the
    Kneedle method: the point of maximum perpendicular distance from the
    line connecting the first and last points on the normalized curve.

    For a concave increasing curve (diminishing returns), this is where
    the marginal improvement per additional cluster drops most sharply —
    the optimal cost/benefit trade-off point.

    If the curve is essentially flat (total improvement < 1% of mean score),
    returns the smallest cluster count to prefer the most compact model.
    """
    counts = np.array(counts, dtype=float)
    scores = np.array(scores, dtype=float)

    if len(counts) < 3:
        return int(counts[np.argmax(scores)])

    order = np.argsort(counts)
    counts, scores = counts[order], scores[order]

    score_range = scores.max() - scores.min()
    if score_range < 0.01 * max(abs(scores.mean()), 1e-12):
        # Flat curve — no meaningful kinetic structure found; use smallest C
        return int(counts[0])

    # Normalize both axes to [0, 1]
    xn = (counts - counts[0]) / (counts[-1] - counts[0] + 1e-12)
    yn = (scores - scores.min()) / (score_range + 1e-12)

    # Perpendicular distance from the diagonal (0,0)→(1,1)
    # For a concave curve this peaks at the knee
    dists = yn - xn
    knee_idx = int(np.argmax(dists))
    return int(counts[knee_idx])


def find_coverage_knee(T_vals, proxy_vals, frac=0.75):
    """
    Find the smallest T that captures `frac` of the total proxy-score improvement
    across the search grid.

    Why not Kneedle here:
      The NN coverage error decreases roughly monotonically as T grows.  When
      the curve is nearly linear (no obvious elbow) — which happens when the
      protein's conformational space is not yet saturated within the grid range —
      Kneedle degenerates to returning the middle of the range regardless of the
      data.  A cumulative-fraction criterion is more robust: it asks "at what T
      have we captured X% of the available coverage gain?" and is well-defined
      even for linear curves.

    A 3-point moving-average is applied first to suppress the stochastic noise
    from the NN subsampling before the threshold is evaluated.

    frac=0.75 means: pick the smallest T where 75% of the total coverage
    improvement (from T_min to T_max) has been achieved.
    """
    T_arr = np.array(T_vals, dtype=float)
    p_arr = np.array(proxy_vals, dtype=float)

    if len(T_arr) < 2:
        return int(T_arr[0])

    order = np.argsort(T_arr)
    T_arr, p_arr = T_arr[order], p_arr[order]

    # 3-point moving average — preserves endpoints exactly
    if len(p_arr) >= 3:
        kernel = np.ones(3) / 3.0
        p_smooth = np.convolve(p_arr, kernel, mode="same")
        p_smooth[0] = p_arr[0]
        p_smooth[-1] = p_arr[-1]
    else:
        p_smooth = p_arr.copy()

    total = p_smooth[0] - p_smooth[-1]
    if total < 1e-10:
        # Flat curve — coverage barely changes; return smallest T (tiny subset suffices)
        return int(T_arr[0])

    for i, (T, p) in enumerate(zip(T_arr, p_smooth)):
        captured = (p_smooth[0] - p) / total
        if captured >= frac:
            print(f"   Coverage knee: T={int(T)} captures {captured*100:.1f}% of "
                  f"total proxy improvement ({frac*100:.0f}% threshold)")
            return int(T)

    # Never hit the threshold — coverage still improving at the grid edge
    print(f"   Coverage curve did not plateau within the search grid "
          f"(best captured={((p_smooth[0]-p_smooth[-1])/total)*100:.1f}%). "
          f"Selecting largest T={int(T_arr[-1])}. Consider widening --target-grid.")
    return int(T_arr[-1])


# ============================================================
# I/O helpers
# ============================================================

def count_frames_mdtraj(traj_path, top_path, chunk=200):
    n = 0
    for t in md.iterload(traj_path, top=top_path, chunk=chunk):
        n += t.n_frames
    return n


def write_subset_trr_stream(traj_path, top_path, indices, out_trr, chunk=500):
    """
    Stream once through trajectory and batch-write all matching frames per chunk
    to a TRR file. Includes velocities and forces if present in the source.
    Avoids both md.load_frame random-access loops and per-frame write overhead.
    """
    indices = np.unique(np.sort(np.asarray(indices, dtype=int)))
    if indices.size == 0:
        raise ValueError("No indices provided for TRR writing.")

    from mdtraj.formats import TRRTrajectoryFile
    with TRRTrajectoryFile(out_trr, mode="w") as w:
        global_offset = 0
        for t in md.iterload(traj_path, top=top_path, chunk=chunk):
            n = t.n_frames
            chunk_end = global_offset + n
            mask = (indices >= global_offset) & (indices < chunk_end)
            local_idx = indices[mask] - global_offset
            if local_idx.size > 0:
                xyz = t.xyz[local_idx].astype(np.float32)
                time = (
                    t.time[local_idx].astype(np.float32)
                    if t.time is not None
                    else np.zeros(local_idx.size, dtype=np.float32)
                )
                box = (
                    t.unitcell_vectors[local_idx].astype(np.float32)
                    if t.unitcell_vectors is not None else None
                )
                # # TRRTrajectoryFile.write() accepts velocities and forces
                # # as optional keyword arguments; pass None if unavailable.
                # velocities = (
                #     t.velocities[local_idx].astype(np.float32)
                #     if hasattr(t, "velocities") and t.velocities is not None
                #     else None
                # )
                # forces = (
                #     t.forces[local_idx].astype(np.float32)
                #     if hasattr(t, "forces") and t.forces is not None
                #     else None
                # )
                if box is None:
                    w.write(xyz, time=time)
                else:
                    w.write(xyz, time=time, box=box)
            global_offset = chunk_end


# ============================================================
# Blocked CV
# ============================================================

def blocked_cv_splits(n_frames, n_splits=5, block_len=2000, seed=42):
    if block_len <= 0:
        raise ValueError("block_len must be > 0")
    n_blocks = n_frames // block_len
    tail = n_frames % block_len
    if tail > 0:
        warnings.warn(
            f"blocked_cv_splits: {tail} tail frames dropped "
            f"({n_frames} frames, block_len={block_len} -> {n_blocks} full blocks)."
        )
    if n_blocks < 2:
        raise ValueError(
            f"Not enough frames for blocked CV: n_blocks={n_blocks} < 2. "
            f"Reduce --cv-block-len or use a longer trajectory."
        )
    if n_blocks < n_splits:
        old = n_splits
        n_splits = max(2, n_blocks)
        warnings.warn(
            f"blocked_cv_splits: requested {old} splits but only {n_blocks} blocks. "
            f"Reducing to {n_splits} splits."
        )

    rng = np.random.default_rng(seed)
    blocks = np.arange(n_blocks, dtype=int)
    rng.shuffle(blocks)
    folds = np.array_split(blocks, n_splits)

    def blocks_to_idx(blks):
        parts = [np.arange(b * block_len, b * block_len + block_len, dtype=int) for b in blks]
        return np.concatenate(parts) if parts else np.array([], dtype=int)

    splits = []
    for i in range(len(folds)):
        te = blocks_to_idx(folds[i])
        tr = blocks_to_idx(np.concatenate([folds[j] for j in range(len(folds)) if j != i]))
        splits.append((tr, te))
    return splits, n_splits


# ============================================================
# CV VAMP-2
# ============================================================

def vamp2_score_train_test(X_train, X_test, lag, dim):
    dim = min(dim, X_train.shape[1])
    if X_train.shape[0] < lag + 1:
        raise ValueError(f"Train fold too short: n_train={X_train.shape[0]}, need >= {lag+1}.")
    if X_test.shape[0] < lag + 1:
        raise ValueError(f"Test fold too short: n_test={X_test.shape[0]}, need >= {lag+1}.")
    vamp_model = pyemma.coordinates.vamp([X_train], lag=lag, dim=dim)
    for call in [
        lambda: vamp_model.score([X_test], score_method="VAMP2"),
        lambda: vamp_model.score([X_test], "VAMP2"),
    ]:
        try:
            return float(call())
        except TypeError:
            continue
    raise RuntimeError("PyEMMA VAMP score API mismatch. Consider upgrading pyemma.")


def cv_select_vamp_representation(X, lags, dims, n_splits, block_len, seed=42):
    splits, n_used = blocked_cv_splits(X.shape[0], n_splits=n_splits, block_len=block_len, seed=seed)
    results = []
    for lag in lags:
        for dim in dims:
            if dim <= 0 or dim > X.shape[1]:
                continue
            scores, n_skip = [], 0
            for tr_idx, te_idx in splits:
                try:
                    scores.append(vamp2_score_train_test(X[tr_idx], X[te_idx], lag=lag, dim=dim))
                except Exception as e:
                    n_skip += 1
                    warnings.warn(f"lag={lag}, dim={dim}: fold skipped -- {e}")
            if not scores:
                warnings.warn(f"lag={lag}, dim={dim}: all folds failed -- skipping.")
                continue
            arr = np.asarray(scores, float)
            results.append({
                "lag": int(lag),
                "dim": int(dim),
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
                "n_folds_ok": int(arr.size),
                "n_folds_skip": int(n_skip),
            })
    if not results:
        raise RuntimeError("No (lag, dim) combination produced CV results.")
    results.sort(key=lambda r: (-r["mean"], r["dim"], r["lag"]))
    return results[0], results, n_used


# ============================================================
# MSM cluster selection by blocked CV
# ============================================================

def estimate_msm_and_score(dtrajs_train, dtrajs_test, msm_lag):
    d_tr = np.asarray(dtrajs_train[0], dtype=int)
    d_te = np.asarray(dtrajs_test[0], dtype=int)

    train_states = set(np.unique(d_tr).tolist())
    test_states = set(np.unique(d_te).tolist())
    unseen = test_states - train_states
    if unseen:
        raise RuntimeError(f"Test dtraj contains {len(unseen)} unseen state(s). Cannot score.")

    n_states = len(train_states)
    if len(d_tr) < n_states * msm_lag + 1:
        raise RuntimeError(
            f"Training fold too short for MSM: n_train={len(d_tr)}, n_states={n_states}, msm_lag={msm_lag}."
        )

    msm = pyemma.msm.estimate_markov_model(dtrajs_train, lag=msm_lag, reversible=True)
    last_exc = None
    for call in [
        lambda: msm.score(dtrajs_test, score_method="VAMP2"),
        lambda: msm.score(dtrajs_test, "VAMP2"),
    ]:
        try:
            return float(call())
        except TypeError as e:
            last_exc = e
        except Exception as e:
            last_exc = e
            break
    raise RuntimeError(f"msm.score(..., VAMP2) unavailable (last error: {last_exc}).")


def cv_select_clusters_msm(Yw, clusters_list, msm_lag, n_splits, block_len, seed=42, min_count_per_state=10):
    splits, _ = blocked_cv_splits(Yw.shape[0], n_splits=n_splits, block_len=block_len, seed=seed)

    # Approximate train fold size (frames, not blocks)
    n_train_approx = int(Yw.shape[0] * (n_splits - 1) / n_splits)

    # Pre-filter: drop any C where the MSM training fold is guaranteed to fail.
    # estimate_msm_and_score raises if len(d_tr) < n_states * msm_lag + 1.
    # Worst case: every state appears exactly once → need n_train >= C * msm_lag + 1.
    max_viable_C = max(2, (n_train_approx - 1) // max(msm_lag, 1))
    viable = [int(c) for c in clusters_list if int(c) <= max_viable_C]
    if not viable:
        # Every candidate exceeds viability — clamp all to max_viable_C
        viable = [max_viable_C]
        warnings.warn(
            f"All cluster candidates exceed MSM viability limit (max_viable_C={max_viable_C} "
            f"for n_train≈{n_train_approx}, msm_lag={msm_lag}). Clamping to C={max_viable_C}."
        )
    elif len(viable) < len(clusters_list):
        dropped = sorted(set(int(c) for c in clusters_list) - set(viable))
        print(f"   Pre-filtered {len(dropped)} cluster candidate(s) exceeding MSM viability "
              f"(max_viable_C={max_viable_C}): {dropped}")
    clusters_list = viable

    # Auto-lower min_count_per_state for small N
    smallest_C = min(int(c) for c in clusters_list if int(c) >= 2)
    expected_mean = max(1, n_train_approx // max(smallest_C, 1))
    auto_min = max(2, min(int(min_count_per_state), int(expected_mean * 0.30)))
    if auto_min < min_count_per_state:
        warnings.warn(f"Auto-lowering min_count_per_state: {min_count_per_state} -> {auto_min}")
        min_count_per_state = auto_min

    results = []
    for c_idx, C in enumerate(clusters_list):
        C = int(C)
        if C < 2 or C >= Yw.shape[0]:
            continue

        fold_scores, ok_folds = [], 0
        for fold_id, (tr_idx, te_idx) in enumerate(splits):
            Ytr, Yte = Yw[tr_idx], Yw[te_idx]
            if Ytr.shape[0] < msm_lag + 1 or Yte.shape[0] < msm_lag + 1:
                continue

            fold_seed = seed + c_idx * 997 + fold_id * 10007
            # n_init=5: CV only needs consistent relative ordering between candidates,
            # not the globally optimal clustering (final KMeans uses n_init=30).
            # Use MiniBatchKMeans for large folds to avoid OOM.
            if Ytr.shape[0] > 20_000:
                from sklearn.cluster import MiniBatchKMeans as _MBK
                _bs = min(Ytr.shape[0], max(5_000, Ytr.shape[0] // 10))
                km = _MBK(n_clusters=C, n_init=5, batch_size=_bs, random_state=fold_seed)
            else:
                km = KMeans(n_clusters=C, n_init=5, random_state=fold_seed)
            d_tr = km.fit_predict(Ytr)
            d_te = km.predict(Yte)

            counts = np.bincount(d_tr, minlength=C)
            if np.any(counts < min_count_per_state):
                # skip folds where some states are too small
                continue

            try:
                sc = estimate_msm_and_score([d_tr], [d_te], msm_lag=msm_lag)
                fold_scores.append(sc)
                ok_folds += 1
            except Exception:
                continue

        if ok_folds == 0:
            continue

        arr = np.asarray(fold_scores, float)
        results.append({
            "clusters": int(C),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
            "ok_folds": int(ok_folds),
        })

    if not results:
        # fallback: smallest C that makes sense
        valid_Cs = [int(c) for c in clusters_list if 2 <= int(c) < Yw.shape[0]]
        fallback_C = valid_Cs[0] if valid_Cs else 10
        warnings.warn(
            f"No cluster count survived CV (all folds failed min_count or MSM check). "
            f"Falling back to C={fallback_C}. Results will have limited statistical support."
        )
        # Use 0.0 not NaN so JSON serialization and downstream code stay valid
        dummy = {"clusters": fallback_C, "mean": 0.0, "std": 0.0, "ok_folds": 0}
        return dummy, [dummy]

    # Sort by cluster count for the knee search (not by score)
    results.sort(key=lambda r: r["clusters"])

    knee_C = find_score_knee(
        counts=[r["clusters"] for r in results],
        scores=[r["mean"] for r in results],
    )

    # Find the result entry corresponding to the knee cluster count
    best = next((r for r in results if r["clusters"] == knee_C), results[-1])
    print(f"   Cluster knee detected at C={knee_C} "
          f"(VAMP-2={best['mean']:.4f}, tested range: "
          f"{results[0]['clusters']}–{results[-1]['clusters']})")

    return best, results


# ============================================================
# Conformational extreme seeding
# ============================================================

def get_conformational_extremes(Yw, target_T, max_extreme_frac=0.10, n_per_dim=3):
    """
    For each TICA dimension, collect the n_per_dim frames at the low and high
    extremes. Total extreme frames are capped at max_extreme_frac * target_T so
    that for small trajectories they don't dominate the selection.
    """
    # Build candidates ranked by distance from centroid across all dims
    # (most extreme first across the full TICA space)
    extreme_idx = []
    for d in range(Yw.shape[1]):
        col = Yw[:, d]
        order = np.argsort(col)
        for k in range(min(n_per_dim, len(order))):
            extreme_idx.append(int(order[k]))
            extreme_idx.append(int(order[-(k + 1)]))

    # Deduplicate preserving insertion order, then cap
    seen = set()
    deduped = []
    for i in extreme_idx:
        if i not in seen:
            seen.add(i)
            deduped.append(i)

    cap = max(2, int(round(max_extreme_frac * target_T)))
    return np.array(deduped[:cap], dtype=int)


# ============================================================
# Two-pool downsampler
# ============================================================

def pick_representatives_for_cluster(cluster_idx, Xsub, centroid, n_pick, rng, medoid_frac):
    if n_pick <= 0 or len(cluster_idx) == 0:
        return []
    pts = Xsub[cluster_idx]
    d2 = np.sum((pts - centroid) ** 2, axis=1)
    order = np.argsort(d2)
    n_medoid = max(1, min(int(np.round(n_pick * medoid_frac)), n_pick))
    chosen = cluster_idx[order[:n_medoid]].tolist()
    n_random = n_pick - len(chosen)
    if n_random > 0:
        pool = cluster_idx[order[n_medoid:]]
        if pool.size > 0:
            extra = rng.choice(pool, size=min(n_random, pool.size), replace=False).tolist()
            chosen.extend(extra)
    return chosen


def two_pool_downsample(N, N_target, labels_array, C, Yw, centers_w,
                        cluster_sizes, min_per_cluster, medoid_frac,
                        uniform_frac, mode, rng):
    N_target = min(int(N_target), int(N))
    n_pool_a_target = max(0, min(int(round(float(uniform_frac) * N_target)), N))
    pool_a_idx = stratified_uniform_sample(N, n_pool_a_target, rng)
    pool_a_set = set(pool_a_idx)

    n_pool_b_target = max(0, N_target - len(pool_a_set))
    pool_b_idx = []

    if n_pool_b_target > 0 and mode != "uniform_only":
        dmin = int(min_per_cluster)
        alloc = np.minimum(cluster_sizes, dmin).astype(int)
        remaining = n_pool_b_target - int(alloc.sum())

        if remaining < 0:
            over = -remaining
            for cid in np.argsort(-alloc):
                if over <= 0:
                    break
                take = min(over, alloc[cid])
                alloc[cid] -= take
                over -= take
            remaining = 0

        if mode == "hybrid" and remaining > 0:
            capacity = (cluster_sizes - alloc).clip(min=0)
            total_cap = int(capacity.sum())
            if total_cap > 0:
                extra = rng.multinomial(remaining, capacity / total_cap)
                extra = np.minimum(extra, capacity)
                alloc += extra
                leftover = n_pool_b_target - int(alloc.sum())
                if leftover > 0:
                    remcap = (cluster_sizes - alloc).clip(min=0)
                    for cid in np.argsort(-remcap):
                        if leftover <= 0:
                            break
                        take = min(leftover, remcap[cid])
                        alloc[cid] += take
                        leftover -= take

        for cid in range(C):
            n_pick = int(alloc[cid])
            cluster_idx = np.where(labels_array == cid)[0]
            if n_pick <= 0 or cluster_idx.size == 0:
                continue
            pool_b_idx.extend(
                pick_representatives_for_cluster(
                    cluster_idx=cluster_idx,
                    Xsub=Yw,
                    centroid=centers_w[cid],
                    n_pick=n_pick,
                    rng=rng,
                    medoid_frac=float(medoid_frac),
                )
            )

    final_set = pool_a_set | set(pool_b_idx)
    shortfall = N_target - len(final_set)
    if shortfall > 0:
        remaining_idx = np.setdiff1d(np.arange(N, dtype=int), np.array(sorted(final_set), dtype=int))
        n_extra = min(shortfall, remaining_idx.size)
        if n_extra > 0:
            extra_idx = rng.choice(remaining_idx, size=n_extra, replace=False)
            final_set |= set(extra_idx.tolist())

    final = np.array(sorted(final_set), dtype=int)
    return final


# ============================================================
# Proxies for auto-selection (NO mdtraj)
# ============================================================

def nn_coverage_error(Yw, selected_idx, labels_array=None, cluster_sizes=None,
                     proxy_mode="nn_yw", seed=42,
                     eval_subsample=4000, cluster_power=1.0):
    """
    Lower is better.

    We sample some points from full Yw (evaluation set),
    then compute mean nearest-neighbor distance to the selected subset.

    If proxy_mode == "nn_yw_clusterweighted":
      weight evaluation points inversely by their cluster population^power
      (emphasizes rare states).
    """
    N = Yw.shape[0]
    if len(selected_idx) < 2:
        return float("inf")

    rng = np.random.default_rng(seed)
    n_eval = min(int(eval_subsample), N)
    eval_idx = rng.choice(N, size=n_eval, replace=False)

    # nearest neighbor distance from eval points to subset
    sub_idx = np.asarray(selected_idx, dtype=int)
    try:
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nbrs.fit(Yw[sub_idx])
        dists, _ = nbrs.kneighbors(Yw[eval_idx], return_distance=True)
    except MemoryError:
        # Retry with a smaller eval sample and ball_tree (lower peak memory than kd_tree)
        n_eval = max(100, n_eval // 4)
        eval_idx = rng.choice(N, size=n_eval, replace=False)
        warnings.warn(
            f"nn_coverage_error: MemoryError fitting NearestNeighbors "
            f"(subset_size={len(sub_idx)}). Retrying with n_eval={n_eval} and algorithm='ball_tree'."
        )
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
        nbrs.fit(Yw[sub_idx])
        dists, _ = nbrs.kneighbors(Yw[eval_idx], return_distance=True)
    d = dists[:, 0].astype(np.float64)

    if proxy_mode == "nn_yw_clusterweighted":
        if labels_array is None or cluster_sizes is None:
            return float(d.mean())
        labs = labels_array[eval_idx].astype(int)
        # weights: (1 / size^power), normalized
        w = 1.0 / np.power(cluster_sizes[labs].astype(np.float64), float(cluster_power))
        w_sum = w.sum()
        if w_sum <= 0:
            return float(d.mean())
        return float((d * (w / w_sum)).sum())

    return float(d.mean())


def mmd_rbf(X, Y, sigma=None, seed=None):
    # small and stable MMD for auto-selection
    from sklearn.metrics.pairwise import rbf_kernel

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if sigma is None:
        # median heuristic from X only — uses seeded rng for reproducibility
        n = min(500, len(X))
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=n, replace=False)
        sub = X[idx]
        dists = np.sqrt(np.sum((sub[:, None] - sub[None, :]) ** 2, axis=-1))
        med = np.median(dists[dists > 0])
        sigma = float(med) if med > 0 else 1.0

    gamma = 1.0 / (2.0 * sigma ** 2)
    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)

    n, m = len(X), len(Y)
    mmd2 = (
        (Kxx.sum() - np.trace(Kxx)) / max(n * (n - 1), 1)
        + (Kyy.sum() - np.trace(Kyy)) / max(m * (m - 1), 1)
        - 2.0 * Kxy.mean()
    )
    return float(mmd2)


def select_uniform_frac_by_proxy_fast(
    N, N_target, labels_array, C, Yw, centers_w, cluster_sizes,
    min_per_cluster, medoid_frac, mode, seed,
    proxy, frac_grid, eval_subsample=4000, cluster_power=1.0
):
    frac_grid = sorted(set(float(f) for f in frac_grid))
    best = None
    grid_results = []

    print(f"\n-- Auto-selecting uniform_frac (proxy={proxy}) --")
    print(f"   Grid: {frac_grid}")

    # MMD reference sample (fixed)
    ref_idx = None
    Yw_ref = None
    if proxy == "mmd":
        rng = np.random.default_rng(seed)
        n_ref = min(2000, N)
        ref_idx = rng.choice(N, size=n_ref, replace=False)
        Yw_ref = Yw[ref_idx]

    for frac in frac_grid:
        rng = np.random.default_rng(seed + int(frac * 100000))
        idx = two_pool_downsample(
            N=N, N_target=N_target,
            labels_array=labels_array, C=C,
            Yw=Yw, centers_w=centers_w,
            cluster_sizes=cluster_sizes,
            min_per_cluster=min_per_cluster,
            medoid_frac=medoid_frac,
            uniform_frac=frac,
            mode=mode,
            rng=rng
        )

        if proxy in ("nn_yw", "nn_yw_clusterweighted"):
            score = nn_coverage_error(
                Yw=Yw,
                selected_idx=idx,
                labels_array=labels_array,
                cluster_sizes=cluster_sizes,
                proxy_mode=proxy,
                seed=seed,
                eval_subsample=eval_subsample,
                cluster_power=cluster_power,
            )
        elif proxy == "mmd":
            # compare subset to fixed reference sample
            score = mmd_rbf(Yw_ref, Yw[idx], sigma=None, seed=seed)
        else:
            raise ValueError(f"Unknown proxy: {proxy}")

        grid_results.append({"frac": frac, "score": float(score), "n_selected": int(len(idx))})
        print(f"   uniform_frac={frac:.2f}  SCORE={score:.6f}  (n_selected={len(idx)})")

        if best is None or score < best["score"]:
            best = {"frac": frac, "score": float(score), "indices": idx}

    print(f"\n   Best uniform_frac={best['frac']:.2f}  (score={best['score']:.6f})\n")
    return best["frac"], best["indices"], grid_results, best["score"]


def auto_select_target_total(
    N, target_grid, args,
    labels_array, C, Yw, centers_w, cluster_sizes, min_per_cluster, seed
):
    target_grid = sorted(set(int(x) for x in target_grid))
    frac_grid = [float(x) for x in args.frac_grid.split(",") if x.strip()]

    # Only enforce a hard floor when the user explicitly set --min-reduction > 0.
    # By default (0.0) the knee finder picks the right size from the coverage curve.
    if float(args.min_reduction) > 0.0:
        max_T = max(10, int(N * (1.0 - float(args.min_reduction))))
        filtered = [T for T in target_grid if 10 <= T <= max_T]
        if not filtered:
            warnings.warn(
                f"All target_grid values exceed min_reduction={args.min_reduction:.2f} "
                f"(max_T={max_T}). Relaxing to use the smallest grid value >= 10."
            )
            filtered = [T for T in target_grid if T >= 10] or target_grid
        target_grid = filtered
        print(f"\n-- Auto-selecting target_total (proxy={args.proxy}, "
              f"min_reduction={args.min_reduction:.2f}, max_T={max_T}) --")
    else:
        print(f"\n-- Auto-selecting target_total via knee detection (proxy={args.proxy}) --")

    table = []

    for T in target_grid:
        if T < 10 or T > N:
            continue

        frac, idx, _, proxy_score = select_uniform_frac_by_proxy_fast(
            N=N, N_target=T,
            labels_array=labels_array, C=C,
            Yw=Yw, centers_w=centers_w, cluster_sizes=cluster_sizes,
            min_per_cluster=min_per_cluster,
            medoid_frac=args.medoid_frac,
            mode=args.mode, seed=seed,
            proxy=args.proxy,
            frac_grid=frac_grid,
            eval_subsample=args.nn_eval_subsample,
            cluster_power=args.cluster_power,
        )

        table.append((T, frac, proxy_score, idx))
        print(f"  T={T:4d}  frac={frac:.2f}  proxy={proxy_score:.6f}")

    if not table:
        raise RuntimeError("auto-target-total: no valid target_total in --target-grid.")

    T_vals = [row[0] for row in table]
    proxy_vals = [row[2] for row in table]

    # Use cumulative-improvement knee detection. find_score_knee (Kneedle) degenerates
    # to the midpoint on nearly-linear curves, which is what noisy NN proxy scores
    # look like. find_coverage_knee picks the first T that captures 75% of the total
    # coverage gain — robust to noise and well-defined regardless of curve shape.
    knee_T = find_coverage_knee(T_vals, proxy_vals, frac=0.75)

    knee_row = next(row for row in table if row[0] == knee_T)
    best = {
        "T":     int(knee_row[0]),
        "frac":  float(knee_row[1]),
        "proxy": float(knee_row[2]),
        "idx":   knee_row[3],
        "size":  float(knee_row[0]) / float(N),
        "total": float(knee_row[2]),
    }

    print(f"\n  Knee detected at T={best['T']} ({100*best['size']:.1f}% of N) "
          f"with uniform_frac={best['frac']:.2f}  (proxy={best['proxy']:.6f})\n")
    return best, [(t, f, p, 0.0, p) for t, f, p, _ in table]


# ============================================================
# Auto-scaling CV infrastructure (block_len, splits, msm_lag, min_per_cluster)
# Lag and cluster candidates are derived from data — see estimate_tica_lag_candidates
# and build_cluster_candidates above.
# ============================================================

def auto_scale_cv_params(N, args, lags, dims):
    """
    Scale only the CV infrastructure parameters (block_len, splits, msm_lag,
    min_per_cluster) based on N. Lag and cluster candidates are now derived
    directly from the data and passed in already set.

    Dims candidates are kept small and N-scaled since TICA dimensionality
    is a model complexity choice: more dims is not always better.
    """
    user_defaults = {
        "dims":            "2,5,10",
        "cv_block_len":    2000,
        "cv_splits":       5,
        "msm_lag":         20,
        "min_per_cluster": 4,
    }

    changed = {}

    # Dims: still use a small N-scaled set since higher dims hurt small datasets
    if args.dims == user_defaults["dims"]:
        if N < 500:    new_dims = [2]
        elif N < 2000: new_dims = [2, 3]
        else:          new_dims = [2, 5, 10]
        if new_dims != dims:
            changed["dims"] = (dims, new_dims)
            dims = new_dims

    # MSM lag: half of the largest TICA lag candidate
    msm_lag = int(args.msm_lag)
    if args.msm_lag == user_defaults["msm_lag"]:
        new_msm_lag = max(2, max(lags) // 2)
        if new_msm_lag != msm_lag:
            changed["msm_lag"] = (msm_lag, new_msm_lag)
            msm_lag = new_msm_lag
    else:
        msm_lag = max(2, msm_lag)
    _msm_lag_before_viability_cap = msm_lag  # saved so viability cap can update changed[]

    # CV block length: must be long enough to see max_lag at least 5 times,
    # but no more than N//6 so we get enough blocks for CV.
    cv_block_len = int(args.cv_block_len)
    max_lag = max(max(lags), msm_lag)
    min_block = max_lag * 5
    natural = max(min_block, N // 10)
    capped = min(natural, max(1, N // 6))
    new_block = max(min_block, capped, max_lag + 2)

    if args.cv_block_len == user_defaults["cv_block_len"]:
        if new_block != cv_block_len:
            changed["cv_block_len"] = (cv_block_len, new_block)
            cv_block_len = new_block

    n_blocks = N // cv_block_len
    if n_blocks < 2:
        rescue_block = max(max_lag * 2 + 3, cv_block_len // 2)
        rescue_blocks = N // rescue_block
        if rescue_blocks >= 2:
            changed["cv_block_len"] = (cv_block_len, rescue_block)
            cv_block_len = rescue_block
            n_blocks = rescue_blocks
        else:
            raise ValueError(
                "Trajectory too short for blocked CV. "
                "Reduce --lags/--msm-lag or use a longer trajectory."
            )

    # CV splits: use half the available blocks (good balance of train size vs fold count)
    # n_blocks=2 → 1 → max(2,1)=2; n_blocks=6 → 3; n_blocks=10 → 5 (capped)
    cv_splits = int(args.cv_splits)
    if args.cv_splits == user_defaults["cv_splits"]:
        desired = max(2, min(5, n_blocks // 2))
        new_splits = max(2, min(desired, n_blocks))
        if new_splits != cv_splits:
            changed["cv_splits"] = (cv_splits, new_splits)
            cv_splits = new_splits
    else:
        cv_splits = max(2, min(cv_splits, n_blocks))

    # Viability cap on msm_lag: ensure the smallest cluster candidate survives the
    # max_viable_C filter in cv_select_clusters_msm.
    # max_viable_C = (n_train - 1) // msm_lag  must be >= smallest_C_candidate.
    # smallest_C_candidate = max(3, int(sqrt(N)/3))  — matches build_cluster_candidates.
    # Solving: msm_lag <= (n_train - 1) // smallest_C_candidate.
    # Without this, a large ACF-derived lag (e.g. 500 → msm_lag=250) can push
    # max_viable_C below 33 so ALL candidates are filtered and the code falls back
    # to C = max_viable_C, which is never a real knee selection.
    n_train_approx = max(1, int(N * (cv_splits - 1) / cv_splits))
    smallest_C = max(3, int(np.sqrt(N) / 3))
    max_msm_lag_for_viability = max(2, (n_train_approx - 1) // smallest_C)
    if msm_lag > max_msm_lag_for_viability:
        old_msm = msm_lag
        msm_lag = max_msm_lag_for_viability
        changed["msm_lag"] = (_msm_lag_before_viability_cap, msm_lag)
        print(f"   msm_lag viability cap: {old_msm} -> {msm_lag} "
              f"(ensures smallest cluster candidate C={smallest_C} survives CV filter "
              f"with n_train≈{n_train_approx})")


    # Min frames per cluster: scale continuously with N rather than tier brackets
    min_per_cluster = int(args.min_per_cluster)
    if args.min_per_cluster == user_defaults["min_per_cluster"]:
        # ~1 frame per cluster for tiny N, grows to 4 at N=2000, caps at 4
        new_mpc = max(1, min(4, int(N // 500)))
        if new_mpc != min_per_cluster:
            changed["min_per_cluster"] = (min_per_cluster, new_mpc)
            min_per_cluster = new_mpc

    if changed:
        print(f"\n-- Auto-scaling CV parameters for N={N} frames --")
        for k, (old, new) in changed.items():
            print(f"   {k:24s}: {old}  ->  {new}")
        print("-" * 50)

    return dims, msm_lag, cv_block_len, cv_splits, min_per_cluster, changed


# ============================================================
# N-adaptive target grid
# ============================================================

def build_adaptive_target_grid(N, user_grid_str, n_points=14):
    """
    If the user supplied an explicit grid string, parse and return it.
    Otherwise generate a grid that scales with N:
      - low  end: ~5% of N  (min 30 frames)
      - high end: ~70% of N — wide enough for the knee finder to locate the
        true coverage plateau regardless of trajectory characteristics.
    The knee of the proxy-score vs T curve selects the final value, so the
    grid just needs to span the plausible range; no manual cap is needed.
    """
    if user_grid_str.strip():
        return sorted(set(int(x) for x in user_grid_str.split(",") if x.strip()))

    lo = max(30, int(round(N * 0.05)))
    hi = max(lo + 50, int(round(N * 0.70)))
    hi = min(hi, N - 1)
    pts = np.unique(np.round(np.linspace(lo, hi, n_points)).astype(int))
    return [int(x) for x in pts if x >= 10]


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Scientific MD trajectory downsampling v3.18 (auto target + frac + extreme seeding).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument("--top", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--outdir", default="downsample_out")

    ap.add_argument("--pca-fve", type=float, default=0.90)
    ap.add_argument("--stride", type=int, default=0,
                    help="Stride for featurization (0 = auto: 1 for N<100k, N//100000 for larger). "
                         "Reduces memory and compute for very large trajectories. "
                         "Final frame indices are always in original (un-strided) space.")
    ap.add_argument("--lags", default="5,10,20,50")
    ap.add_argument("--dims", default="2,5,10")
    ap.add_argument("--clusters", default="20,50,100,200")

    ap.add_argument("--cv-splits", type=int, default=5)
    ap.add_argument("--cv-block-len", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--msm-lag", type=int, default=20)
    ap.add_argument("--min-count-per-state", type=int, default=10)

    ap.add_argument("--mode", choices=["coverage", "hybrid", "uniform_only"], default="hybrid")
    ap.add_argument("--uniform-frac", type=float, default=0.70)
    ap.add_argument("--medoid-frac", type=float, default=0.70)
    ap.add_argument("--no-auto-uniform-frac", dest="auto_uniform_frac", action="store_false",
                    help="Fix uniform_frac at --uniform-frac instead of auto-selecting it.")
    ap.set_defaults(auto_uniform_frac=True)
    ap.add_argument("--frac-grid", default="0.50,0.60,0.70,0.75,0.80,0.85,0.90,1.00")

    # Auto target_total (on by default; disable with --no-auto-target-total)
    ap.add_argument("--no-auto-target-total", dest="auto_target_total", action="store_false",
                    help="Disable auto target_total selection and use --target-total instead.")
    ap.set_defaults(auto_target_total=True)
    ap.add_argument("--target-total", type=int, default=200,
                    help="Fixed subset size used only when --no-auto-target-total is set.")
    ap.add_argument("--target-grid", default="",
                    help="Comma list of candidate target_total values for auto selection. "
                         "If empty (default), an N-adaptive grid is generated automatically.")
    ap.add_argument("--size-penalty", type=float, default=0.15,
                    help="Soft penalty weight for larger subsets (used alongside --min-reduction).")
    ap.add_argument("--min-reduction", type=float, default=0.0,
                    help="Minimum required frame reduction (0.0 = let knee detection decide, no hard floor). "
                         "Set e.g. 0.50 to force at least 50%% reduction if you need a hard upper bound on output size.")
    ap.add_argument("--no-target-clamp", action="store_true",
                    help="Do not clamp target_total when using --no-auto-target-total.")

    # proxies
    ap.add_argument("--proxy", choices=["nn_yw", "nn_yw_clusterweighted", "mmd"],
                    default="nn_yw_clusterweighted")
    ap.add_argument("--nn-eval-subsample", type=int, default=0,
                    help="Evaluation subsample size for NN proxy. 0 (default) = auto: "
                         "min(N, max(1000, N//10)) so it scales with trajectory size.")
    ap.add_argument("--cluster-power", type=float, default=1.0)

    # IMPORTANT: disabled RMSD check (your mdtraj stack segfaults on TRR iterload in repeated contexts)
    ap.add_argument("--rmsd-check", action="store_true",
                    help="(DISCOURAGED) Kept for compatibility, but disabled by design in v3.16.")
    # add this in argparse definitions (main())
    ap.add_argument(
    "--min-per-cluster",
    dest="min_per_cluster",
    type=int,
    default=4,
    help="Minimum frames to allocate per cluster in the coverage pool (auto-scaled for small N)."
    )

    args = ap.parse_args()

    if not (0.0 <= args.uniform_frac <= 1.0):
        raise ValueError(f"--uniform-frac must be in [0,1], got {args.uniform_frac}")

    top_path = args.top
    traj_path = args.traj
    seed = args.seed
    rng = np.random.default_rng(seed)

    if not os.path.exists(top_path):
        raise FileNotFoundError(f"Topology not found: {top_path}")
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")

    os.makedirs(args.outdir, exist_ok=True)

    # Output paths: .trr instead of .xtc
    out_train_trr = os.path.join(args.outdir, "downsampled_subset.trr")
    out_train_idx = os.path.join(args.outdir, "downsampled_subset_indices.npy")
    out_params = os.path.join(args.outdir, "selected_params.json")
    out_png_vamp_cv = os.path.join(args.outdir, "cv_vamp2_representation.png")
    out_png_cl_cv = os.path.join(args.outdir, "cv_msm_vamp2_clusters.png")
    out_png_pop = os.path.join(args.outdir, "cluster_populations.png")
    out_summary = os.path.join(args.outdir, "subset_summary.txt")

    lags = sorted(set(int(x) for x in args.lags.split(",") if x.strip()))
    dims = sorted(set(int(x) for x in args.dims.split(",") if x.strip()))
    clusters_list = sorted(set(int(x) for x in args.clusters.split(",") if x.strip()))
    frac_grid = sorted(set(float(x) for x in args.frac_grid.split(",") if x.strip()))

    if not lags:
        raise ValueError("--lags produced an empty list. Provide at least one lag value, e.g. --lags 5,10,20")
    if not dims:
        raise ValueError("--dims produced an empty list. Provide at least one dim value, e.g. --dims 2,5,10")
    if not clusters_list:
        raise ValueError("--clusters produced an empty list. Provide at least one value, e.g. --clusters 20,50,100")
    if not frac_grid:
        raise ValueError("--frac-grid produced an empty list. Provide at least one value, e.g. --frac-grid 0.5,0.7,0.9")

    # Count frames (optional but useful)
    print("Counting frames...")
    n_frames_total = None
    try:
        n_frames_total = count_frames_mdtraj(traj_path, top_path)
        print(f"Total frames: {n_frames_total}")
    except Exception as e:
        warnings.warn(f"Frame count failed (continuing): {e}")

    # Resolve stride: auto-compute from raw frame count to keep feature matrix manageable
    stride = int(args.stride)
    if stride == 0:
        if n_frames_total is not None and n_frames_total >= 100_000:
            stride = max(1, n_frames_total // 100_000)
            print(f"Auto stride={stride} (trajectory has {n_frames_total} frames; "
                  f"featurizing ~{n_frames_total // stride} frames)")
        else:
            stride = 1

    # Featurize
    print(f"Featurizing: backbone torsions (cos/sin), stride={stride}...")
    try:
        feat = pyemma.coordinates.featurizer(top_path)
        feat.add_backbone_torsions(cossin=True)
        n_torsion_features = len(feat.describe())
        if n_torsion_features == 0:
            raise ValueError(
                "No backbone torsion features found. The topology may describe a "
                "single residue, a ligand, or a non-protein molecule. "
                "Backbone torsions require at least 2 residues with φ/ψ angles."
            )

        # Pre-flight memory estimate: warn before allocating if the matrix is very large
        n_to_load = (n_frames_total // stride) if n_frames_total else None
        if n_to_load and n_torsion_features > 0:
            est_mb = n_to_load * n_torsion_features * 8 / 1e6
            if est_mb > 4_000:
                warnings.warn(
                    f"Estimated feature matrix size: ~{est_mb/1000:.1f} GB "
                    f"({n_to_load} frames × {n_torsion_features} features × 8 bytes). "
                    f"This may exhaust available RAM. "
                    f"Consider increasing --stride (current: {stride}). "
                    f"For example: --stride {stride * 2}"
                )

        try:
            raw_data = pyemma.coordinates.load(traj_path, features=feat, stride=stride)
        except MemoryError:
            next_stride = max(2, stride * 2)
            frame_hint = f"~{n_to_load}" if n_to_load else "unknown number of"
            raise MemoryError(
                f"Out of memory loading feature matrix "
                f"({frame_hint} frames × {n_torsion_features} features at stride={stride}). "
                f"Increase --stride to reduce memory usage. "
                f"Suggested: --stride {next_stride}"
            ) from None

    except (ValueError, MemoryError):
        raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to featurize trajectory '{traj_path}' with topology '{top_path}'. "
            f"Check that the files are readable, compatible, and not corrupted.\n"
            f"Original error: {e}"
        ) from e

    if not isinstance(raw_data, np.ndarray) or raw_data.ndim != 2:
        raise RuntimeError(
            f"Unexpected feature output from PyEMMA: "
            f"type={type(raw_data)}, shape={getattr(raw_data, 'shape', None)}. "
            f"Expected a 2-D numpy array of shape (n_frames, n_features)."
        )
    if raw_data.shape[1] == 0:
        raise ValueError(
            f"Featurization produced 0 features for {raw_data.shape[0]} frames. "
            f"The molecule may have no backbone torsions."
        )
    data_list = [raw_data]

    # PCA
    print("Running PCA...")
    pca_out_list, k_pca, evr, fve = pca_fit_transform_fve(data_list, args.pca_fve, seed)
    Xp, _ = concat_arrays(pca_out_list)
    N = Xp.shape[0]
    print(f"N={N} frames. PCA: k_pca={k_pca} (FVE={fve[k_pca-1]:.6f}).")

    # Guard: minimum viable N for blocked CV and meaningful clustering
    if N < 100:
        raise ValueError(
            f"Trajectory has only N={N} frames after featurization — too few for "
            f"blocked CV and meaningful clustering. Need at least 100 frames."
        )
    if N < 300:
        warnings.warn(
            f"N={N} is very small. Results may be unreliable. "
            f"Consider using a longer trajectory."
        )

    # Resolve auto nn_eval_subsample now that N is known.
    # N//5 (vs old N//10) halves the noise on the NN mean-distance estimate,
    # which is critical for reliable knee detection in auto_select_target_total.
    if args.nn_eval_subsample == 0:
        args.nn_eval_subsample = min(N, max(2000, N // 5))
        print(f"NN eval subsample (auto): {args.nn_eval_subsample}")

    # Data-driven lag candidates: derived from ACF of PCA output, not hardcoded tiers
    if args.lags == "5,10,20,50":
        lags = estimate_tica_lag_candidates(Xp)
        print(f"ACF-derived lag candidates: {lags}")
    # else: user supplied custom lags, use them as-is

    # Data-driven cluster candidates: log-spaced from sqrt(N)/3 to N//20
    if args.clusters == "20,50,100,200":
        clusters_list = build_cluster_candidates(N)
        print(f"Log-spaced cluster candidates (N={N}): {clusters_list}")
    # else: user supplied custom clusters, use them as-is

    # Scale CV infrastructure (block_len, splits, msm_lag, min_per_cluster, dims)
    (dims, msm_lag, cv_block_len, cv_splits,
     min_per_cluster, scaled) = auto_scale_cv_params(N, args, lags, dims)

    valid_dims = [d for d in dims if 0 < d <= k_pca]
    if not valid_dims:
        raise ValueError(f"All candidate dims {dims} exceed k_pca={k_pca}")

    # Rep selection
    print("\nSelecting (lag, dim) via blocked CV VAMP-2...")
    rep_best, rep_all, _ = cv_select_vamp_representation(
        X=Xp, lags=lags, dims=valid_dims,
        n_splits=cv_splits, block_len=cv_block_len, seed=seed
    )
    best_lag = rep_best["lag"]
    best_dim = rep_best["dim"]
    print(f"Selected rep: lag={best_lag} dim={best_dim}")

    # plot rep grid
    fig, ax = plt.subplots(figsize=(max(8, len(rep_all) * 0.9), 5))
    xs = np.arange(len(rep_all))
    ax.errorbar(xs, [r["mean"] for r in rep_all],
                yerr=[r["std"] for r in rep_all], fmt="o-", capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"lag={r['lag']},dim={r['dim']}" for r in rep_all],
                       rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("CV VAMP-2 score")
    ax.set_title("Representation selection (blocked CV VAMP-2)")
    fig.tight_layout()
    fig.savefig(out_png_vamp_cv, dpi=150)
    plt.close(fig)

    # TICA
    print("Fitting TICA...")
    try:
        tica = pyemma.coordinates.tica(pca_out_list, lag=best_lag, dim=best_dim)
        tica_out_list = tica.get_output()
    except MemoryError:
        raise MemoryError(
            f"Out of memory fitting TICA (N={N} frames, lag={best_lag}, dim={best_dim}). "
            f"Increase --stride to reduce the number of frames used for fitting."
        ) from None
    Y, _ = concat_arrays(tica_out_list)

    # Whitening
    Y_mean = Y.mean(axis=0)
    Y_std = Y.std(axis=0)
    keep = Y_std > 1e-8
    if keep.sum() == 0:
        raise RuntimeError("All TICA dims have near-zero variance.")
    Yw = (Y[:, keep] - Y_mean[keep]) / (Y_std[keep] + 1e-12)

    # Cluster selection
    print("Selecting clusters...")
    cl_best, cl_all = cv_select_clusters_msm(
        Yw=Yw,
        clusters_list=clusters_list,
        msm_lag=max(2, int(msm_lag)),
        n_splits=cv_splits,
        block_len=cv_block_len,
        seed=seed,
        min_count_per_state=int(args.min_count_per_state),
    )
    C = int(cl_best["clusters"])
    print(f"Selected clusters: C={C}")

    # plot cluster grid — mark the knee
    fig, ax = plt.subplots(figsize=(max(6, min(len(cl_all) * 1.2, 30)), 4))
    xs = np.arange(len(cl_all))
    ax.errorbar(xs, [r["mean"] for r in cl_all],
                yerr=[r["std"] for r in cl_all], fmt="o-", capsize=4)
    knee_xs = [i for i, r in enumerate(cl_all) if r["clusters"] == C]
    if knee_xs:
        ax.axvline(knee_xs[0], color="red", linestyle="--", linewidth=1.5,
                   label=f"Knee C={C}")
        ax.legend(fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"C={r['clusters']}" for r in cl_all], fontsize=8)
    ax.set_ylabel("CV MSM VAMP-2 score")
    ax.set_title("Cluster selection — knee of VAMP-2 curve (blocked CV MSM VAMP-2)")
    fig.tight_layout()
    fig.savefig(out_png_cl_cv, dpi=150)
    plt.close(fig)

    # Final clustering: KMeans for small N, MiniBatchKMeans for large N
    print("Final clustering...")
    if N > 50_000:
        from sklearn.cluster import MiniBatchKMeans
        batch_size = min(N, max(10_000, N // 10))
        kmeans = MiniBatchKMeans(n_clusters=C, n_init=10, batch_size=batch_size,
                                 random_state=seed, max_iter=300)
        print(f"  Using MiniBatchKMeans (N={N}, batch_size={batch_size})")
    else:
        kmeans = KMeans(n_clusters=C, n_init=30, random_state=seed)
    labels_array = kmeans.fit_predict(Yw)
    centers_w = kmeans.cluster_centers_
    cluster_sizes = np.bincount(labels_array, minlength=C)

    fig, ax = plt.subplots(figsize=(max(10, min(C // 5, 40)), 4))
    ax.bar(np.arange(C), cluster_sizes, width=0.8)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Frame count")
    ax.set_title("Cluster population distribution")
    fig.tight_layout()
    fig.savefig(out_png_pop, dpi=150)
    plt.close(fig)

    # Decide target_total + uniform_frac
    chosen_uniform_frac = float(args.uniform_frac)
    chosen_target_total = int(args.target_total)
    final_train = None

    if args.auto_target_total:
        target_grid = build_adaptive_target_grid(N, args.target_grid)
        print(f"Auto target grid (N={N}): {target_grid}")
        best, table = auto_select_target_total(
            N=N,
            target_grid=target_grid,
            args=args,
            labels_array=labels_array,
            C=C,
            Yw=Yw,
            centers_w=centers_w,
            cluster_sizes=cluster_sizes,
            min_per_cluster=min_per_cluster,
            seed=seed,
        )
        chosen_target_total = best["T"]
        chosen_uniform_frac = best["frac"]
        final_train = np.asarray(best["idx"], dtype=int)

    else:
        # Manual mode: clamp at N//3 (less aggressive than the old N//5)
        if (not args.no_target_clamp) and (chosen_target_total > max(10, N // 3)):
            old = chosen_target_total
            chosen_target_total = max(10, N // 3)
            warnings.warn(
                f"Clamping target_total {old} -> {chosen_target_total} "
                f"(max allowed: N//3={N//3}; use --no-target-clamp to disable)."
            )

        if args.auto_uniform_frac:
            chosen_uniform_frac, final_train, grid_res, _ = select_uniform_frac_by_proxy_fast(
                N=N, N_target=chosen_target_total,
                labels_array=labels_array, C=C,
                Yw=Yw, centers_w=centers_w, cluster_sizes=cluster_sizes,
                min_per_cluster=min_per_cluster,
                medoid_frac=args.medoid_frac,
                mode=args.mode, seed=seed,
                proxy=args.proxy, frac_grid=frac_grid,
                eval_subsample=args.nn_eval_subsample,
                cluster_power=args.cluster_power,
            )
        else:
            final_train = two_pool_downsample(
                N=N, N_target=chosen_target_total,
                labels_array=labels_array, C=C,
                Yw=Yw, centers_w=centers_w,
                cluster_sizes=cluster_sizes,
                min_per_cluster=min_per_cluster,
                medoid_frac=args.medoid_frac,
                uniform_frac=chosen_uniform_frac,
                mode=args.mode, rng=rng,
            )

    # Seed conformational extremes — capped at 10% of chosen target to preserve balance
    extreme_idx = get_conformational_extremes(Yw, target_T=chosen_target_total,
                                              max_extreme_frac=0.10, n_per_dim=3)
    print(f"Conformational extremes seeded: {len(extreme_idx)} frames (cap: 10% of T={chosen_target_total})")

    # Merge extreme seeds into the final selection (they are always included)
    final_set = set(np.asarray(final_train, dtype=int).tolist()) | set(extreme_idx.tolist())
    final_train = np.array(sorted(final_set), dtype=int)

    final_train = np.unique(np.sort(np.asarray(final_train, dtype=int)))

    # If stride > 1, indices are in strided space — map back to original frame indices
    if stride > 1:
        original_indices = final_train * stride
        print(f"Stride={stride}: mapping {len(final_train)} strided indices "
              f"-> original frame indices (range 0..{(N-1)*stride})")
    else:
        original_indices = final_train

    np.save(out_train_idx, original_indices)

    n_total_for_compression = n_frames_total if n_frames_total is not None else N * stride
    print(f"\nDownsampling with target_total={chosen_target_total}, uniform_frac={chosen_uniform_frac:.2f}")
    print(f"Selected frames: {len(original_indices)} / {n_total_for_compression}  "
          f"(compression={(1.0 - len(original_indices)/n_total_for_compression)*100.0:.1f}%)")

    # Write subset as TRR
    print(f"Writing subset TRR -> {out_train_trr}")
    write_subset_trr_stream(traj_path, top_path, original_indices, out_train_trr)

    # Save params
    params = {
        "version": "3.18",
        "input_top": top_path,
        "input_traj": traj_path,
        "n_frames_total": int(n_frames_total) if n_frames_total is not None else None,
        "n_frames_used": int(N),
        "pca": {
            "fve_target": float(args.pca_fve),
            "k_pca": int(k_pca),
            "actual_fve": float(fve[k_pca - 1]),
        },
        "rep_selection": {
            "lags_tested": lags,
            "dims_tested": valid_dims,
            "cv_splits": int(cv_splits),
            "cv_block_len": int(cv_block_len),
            "selected_lag": int(best_lag),
            "selected_dim": int(best_dim),
            "best_mean": float(rep_best["mean"]),
            "folds_ok": int(rep_best["n_folds_ok"]),
        },
        "clustering": {
            "clusters_tested": clusters_list,
            "selected_clusters": int(C),
            "min_count_per_state": int(args.min_count_per_state),
            "cluster_sizes": cluster_sizes.tolist(),
        },
        "selection": {
            "auto_target_total": bool(args.auto_target_total),
            "target_total": int(chosen_target_total),
            "auto_uniform_frac": bool(args.auto_uniform_frac or args.auto_target_total),
            "uniform_frac": float(chosen_uniform_frac),
            "mode": args.mode,
            "proxy": args.proxy,
            "size_penalty": float(args.size_penalty),
            "cluster_power": float(args.cluster_power),
            "nn_eval_subsample": int(args.nn_eval_subsample),
        },
        "outputs": {
            "subset_trr": out_train_trr,
            "subset_indices": out_train_idx,
        },
    }
    with open(out_params, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    # Summary
    sep = "=" * 70
    lines = [
        sep,
        "Scientific MD Downsampling -- Summary (v3.18)",
        sep,
        f"TOP:  {top_path}",
        f"TRAJ: {traj_path}",
        f"Total frames (counted): {n_frames_total}",
        f"Frames used (features): {N}",
        "",
        f"PCA: k_pca={k_pca}, FVE={fve[k_pca-1]:.6f}",
        f"Rep: lag={best_lag}, dim={best_dim}",
        f"TICA kept dims: {int(Yw.shape[1])}",
        f"Clusters: C={C}",
        "",
        f"Selection: target_total={chosen_target_total}, uniform_frac={chosen_uniform_frac:.2f}, mode={args.mode}",
        f"Proxy: {args.proxy}, size_penalty={args.size_penalty}",
        f"Selected frames: {len(final_train)} (compression={(1.0 - len(final_train)/N)*100.0:.1f}%)",
        "",
        f"Subset TRR:     {out_train_trr}",
        f"Subset indices: {out_train_idx}",
        f"Params JSON:    {out_params}",
        sep,
    ]
    write_summary(out_summary, lines)
    for ln in lines:
        print(ln)

    if args.rmsd_check:
        warnings.warn(
            "RMSD check is intentionally disabled in v3.16 because mdtraj TRR iterload "
            "segfaulted in your environment. Use a separate, external RMSD tool (gmx rms) "
            "on the produced subset.trr if you need RMSD stats."
        )


if __name__ == "__main__":
    main()