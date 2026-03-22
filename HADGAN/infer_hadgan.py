#!/usr/bin/env python3
"""
HADGAN Inference Script
Loads trained checkpoint and performs anomaly detection.
"""
import sys, os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import math
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.io as sio
import gc

from utils.sliding_window_inference import mean_inference, weighted_inference
from utils.improved_post import (
    compute_residual_map, select_min_energy_bands,
    spatial_detector_from_residual_fast, spectral_detector_from_residual_fast,
    fuse_spatial_spectral
)

# Model classes (same as training script)
from train_hadgan import (
    Encoder, Decoder, LatentDiscriminator, ImageDiscriminator, compute_dz
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def inference(hsi, ref, checkpoint_path, output_dir, window=(120, 120, 239),
              overlap=0, method="mean", lbd=0.5, hsi_path=None, ):
    """Run inference and generate anomaly map"""
    print(f"\n{'='*60}")
    print(f"HADGAN INFERENCE")
    print(f"{'='*60}")
    print(f"HSI Shape: {hsi.shape}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Method: {method}, Overlap: {overlap}%")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Extract inference dataset name from HSI file path
    if hsi_path:
        inference_dataset_name = Path(hsi_path).stem
    else:
        inference_dataset_name = "unknown_inference_dataset"

    Hk, Wk, Bk = window
    batch_size = Hk * Wk

    # Restore model from checkpoint
    print("Loading checkpoint...")
    dz = compute_dz(Bk)
    enc = Encoder(dz=dz, dropout_latent=0.5)
    dec = Decoder(B=Bk)
    dznet = LatentDiscriminator()
    dinet = ImageDiscriminator()

    if checkpoint_path.is_dir():
        # It's a directory - find latest checkpoint
        print(f"Checkpoint path is a directory. Finding latest checkpoint...")
        latest_ckpt = tf.train.latest_checkpoint(str(checkpoint_path))
        if latest_ckpt is None:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")
        resolved_ckpt_path = latest_ckpt
        print(f"Using checkpoint: {resolved_ckpt_path}")
    elif checkpoint_path.exists():
        # It's a file - use it directly
        resolved_ckpt_path = str(checkpoint_path)
        print(f"Using specified checkpoint: {resolved_ckpt_path}")
    else:
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    ckpt_basename = Path(resolved_ckpt_path).name
    training_dataset_name = ckpt_basename.replace("hadgan_ckpt_", "").rsplit("-", 1)[0]

    ckpt = tf.train.Checkpoint(enc=enc, dec=dec, dznet=dznet, dinet=dinet)
    status = ckpt.restore(resolved_ckpt_path)
    print("✅ Checkpoint restored.")

    # Reconstruction
    print(f"Performing {method} inference...")
    if method == "mean":
        Xrec = mean_inference(hsi, enc, dec, window, overlap, batch_size)
    else:
        Xrec = weighted_inference(hsi, enc, dec, window, overlap, batch_size)

    # Residual and post-processing
    print("Computing residual map...")
    residual = compute_residual_map(hsi, Xrec)
    del Xrec
    gc.collect()

    print("Selecting informative bands...")
    bands = select_min_energy_bands(residual, k=3)

    print("Computing spatial detection...")
    dspatial = spatial_detector_from_residual_fast(residual, bands)

    print("Computing spectral detection...")
    dspectral = spectral_detector_from_residual_fast(residual)

    del residual
    gc.collect()

    print("Fusing spatial-spectral maps...")
    fmap = fuse_spatial_spectral(dspatial, dspectral, lam=lbd)

    # evaluation
    pr_auc = average_precision_score(ref.ravel(), fmap.ravel())
    roc = roc_auc_score(ref.ravel(), fmap.ravel())

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"{'='*60}\n")

    # Save numerical results with dataset names
    results_filename = f"{inference_dataset_name}_trained_with_{training_dataset_name}.mat"
    results_path = os.path.join(output_dir, results_filename)
    sio.savemat(results_path, {
        'anomaly_map': fmap,
        'roc_auc': roc,
        'pr_auc': pr_auc
    })
    print(f"✅ Results saved: {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HADGAN Inference Script")
    parser.add_argument('--hsi_path', type=str, required=True,
                        help='Path to HSI .mat file (keys: "data" for HSI, "map" for GT)')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to checkpoint, or to a directory containing checkpoints---the latest one will be used if it\'s a directory.')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Directory to save inference results')
    parser.add_argument('--patch_size', type=int, default=64, choices=[120, 64, 32, 16],
                        help='Patch size (same as training)')
    parser.add_argument('--overlap', type=int, default=0,
                        help='Patch overlap percentage (0-99)')
    parser.add_argument('--method', type=str, default='mean',
                        choices=['mean', 'weighted'],
                        help='Aggregation method for overlapping patches')
    parser.add_argument('--lbd', type=float, default=0.5, help='decide contribution of spatial detection (spectral is 1 - lambda)')


    args = parser.parse_args()

    # Load data
    print(f"Loading HSI and GT from {args.hsi_path}...")
    hsi_mat = sio.loadmat(args.hsi_path)
    hsi = np.array(hsi_mat['data'], dtype=np.float32)
    hsi = (hsi - hsi.min()) / (np.ptp(hsi) + 1e-8)

    ref = np.array(hsi_mat['map'], dtype=bool)

    # Infer
    inference(hsi, ref, Path(args.checkpoint_path), args.output_dir,
              window=(args.patch_size, args.patch_size, hsi.shape[2]),
              overlap=args.overlap, method=args.method, lbd=args.lbd, hsi_path=Path(args.hsi_path))