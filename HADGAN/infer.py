import tensorflow as tf
import numpy as np
import argparse
import math
from utils.postprocessing import *
from utils.sliding_window_inference import *
from utils.adaptive_thresh import *
from sklearn.metrics import roc_auc_score, average_precision_score
from HADGAN.hadgan import Encoder, Decoder, LatentDiscriminator, ImageDiscriminator
import scipy.io as sio
import sys

# with tf.device('/CPU:0'):

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--epoch", type=str, required=True)
parser.add_argument("--method", type=int, required=True)
parser.add_argument("--overlap", type=int, required=True)
args = parser.parse_args()

# --- Loading inference dataset ---
mat=sio.loadmat("/home/ubuntu/aditya/BioSky/Datasets/MOCK/prisma_data.mat")
infer = np.array(mat['data'], dtype=np.float32)
print("Inference image shape:", infer.shape)
infer = (infer - infer.min()) / (np.ptp(infer) + 1e-8)

gt_mat=sio.loadmat("/home/ubuntu/aditya/BioSky/Datasets/MOCK/prisma_gt.mat")
ref = gt_mat['data']  # shape (H,W), binary 0/1
print("Ground truth shape:", ref.shape)

Hk, Wk, Bk = 120, 120, infer.shape[2]
window=(Hk,Wk,Bk)
batch_size=Hk*Wk

# --- Model Initialization ---
dz = int(math.sqrt(Bk) + 1)
enc = Encoder(dz=dz)
dec = Decoder(B=Bk)
dznet = LatentDiscriminator()
dinet = ImageDiscriminator()

ckpt = tf.train.Checkpoint(enc=enc, dec=dec, dznet=dznet, dinet=dinet)
ckpt.restore(tf.train.latest_checkpoint(args.checkpoint)).expect_partial()

if args.method=="mean":
    Xrec=mean_inference(infer,enc,dec,window,args.overlap,batch_size)
else:
    Xrec=weighted_inference(infer,enc,dec,window,args.overlap,batch_size)

residual = compute_residual_map(infer, Xrec)

# --- Run Post-processing Detectors ---
bands = select_min_energy_bands(residual, k=3)
dspatial = spatial_detector_from_residual(residual, bands)
dspectral = spectral_detector_from_residual(residual)
fmap = fuse_spatial_spectral(dspatial, dspectral, lam=0.5)

final_maps[ep+1]=fmap

# --- Evaluate and Visualize ---
tf.print(f"--- Inference completed for Epoch={args.epoch+1}. Evaluating results. ---",output_stream=sys.stdout)
# bests, pr_auc, roc = tune_methods(fmap, ref.reshape(H, W))
threshold,iterative_f1=iterative_f1_threshold(ref.ravel(), fmap.ravel())
bests=(iterative_f1, threshold, None, None)

binary_map=(fmap>threshold).astype(np.uint8)

final_binary_maps[ep+1]=binary_map

# compute continuous-score metrics once for S
pr_auc = average_precision_score(ref.ravel(), fmap.ravel())
roc = roc_auc_score(ref.ravel(), fmap.ravel())

tf.print(f"Loaded Model Tuned results:, {bests}",output_stream=sys.stdout)
tf.print(f"Loaded Model PR-AUC: {pr_auc}, ROC: {roc}",output_stream=sys.stdout)

del Xrec, residual, fmap, binary_map  # free Python references
tf.keras.backend.clear_session()
gc.collect()
tf.experimental.async_clear_error()