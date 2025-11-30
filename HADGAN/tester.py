import sys
import argparse
import math
import time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.covariance import EmpiricalCovariance
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import random
import tensorflow as tf
from tensorflow.keras import Sequential # type:ignore
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU # type:ignore
from tensorflow.keras.callbacks import EarlyStopping # type:ignore
from tensorflow.keras.models import load_model # type:ignore
from tensorflow.keras.utils import serialize_keras_object, deserialize_keras_object # type:ignore
from tensorflow.keras import regularizers # type:ignore
from tqdm import trange
import gc
import subprocess
import h5py
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

from utils.adaptive_thresh import tune_methods,iterative_f1_threshold
from utils.sliding_window_inference import mean_inference, weighted_inference
from utils.augment_patch import augment_patch
from utils.improved_post import *

tf.config.optimizer.set_jit(True)
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
tf.keras.mixed_precision.set_global_policy('mixed_float16')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
# os.environ['TF_DISABLE_MLIR_GRAPH_OPTIMIZATION'] = '1'

bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)
l2_reg = tf.keras.regularizers.L2(1e-5)

# ------------------------ GPU Configuration ------------------------
print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # Efficient allocator

    for gpu in gpus:
        # Enable memory growth (prevents TF from grabbing all memory)
        tf.config.experimental.set_memory_growth(gpu, True)
        
        # Optional: Limit GPU memory per process (e.g., 80% of 23 GB GPU ≈ 11500 MB)
        tf.config.set_logical_device_configuration(
            gpu,
            [tf.config.LogicalDeviceConfiguration(memory_limit=18500)]
        )

    print("GPU configured successfully.")
else:
    print("⚠ No GPU found. Running on CPU.")

# ------------------------------
# Utility / Model components
# ------------------------------
def compute_dz(B):
    return int(math.sqrt(B) + 1)

class HADGAN(tf.keras.Model):
    def __init__(self, enc, dec, dznet, dinet,**kwargs):
        super().__init__(**kwargs)
        self.enc = enc
        self.dec = dec
        self.dznet = dznet
        self.dinet = dinet

    def call(self, x, training=False):
        # only forward through encoder+decoder for inference
        z = self.enc(x, training=training)
        xrec = self.dec(z, training=training)
        return xrec
    
    def get_config(self):
        config = super().get_config()
        # serialize nested models/layers
        config.update({
            "enc": serialize_keras_object(self.enc),
            "dec": serialize_keras_object(self.dec),
            "dznet": serialize_keras_object(self.dznet),
            "dinet": serialize_keras_object(self.dinet),
        })
        return config

    @classmethod
    def from_config(cls, config):
        # deserialize nested models
        enc_cfg = config.pop("enc")
        dec_cfg = config.pop("dec")
        dznet_cfg = config.pop("dznet")
        dinet_cfg = config.pop("dinet")

        enc = deserialize_keras_object(enc_cfg)
        dec = deserialize_keras_object(dec_cfg)
        dznet = deserialize_keras_object(dznet_cfg)
        dinet = deserialize_keras_object(dinet_cfg)
        return cls(enc, dec, dznet, dinet, **config)

class FCBlock(tf.keras.Model):
    def __init__(self, out_dim, batchnorm=False, activation=True, lrelu_slope=0.2,kernel_regularizer=l2_reg,**kwargs):
        super().__init__(**kwargs)
        self.out_dim = out_dim
        self.batchnorm = batchnorm
        self.activation = activation
        self.lrelu_slope = lrelu_slope
        self.kernel_regularizer=kernel_regularizer
        self.fc = Dense(out_dim, use_bias=True, kernel_regularizer=kernel_regularizer)
        self.bn = BatchNormalization(epsilon=1e-5) if batchnorm else None
        self.act=LeakyReLU(negative_slope=lrelu_slope) if activation else None
    def call(self, x, training=False):
        x = self.fc(x)
        if self.bn is not None:
            x = self.bn(x,training=training)
        if self.act is not None:
            x=self.act(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_dim": self.out_dim,
            "batchnorm": self.batchnorm,
            "activation": self.activation,
            "lrelu_slope": self.lrelu_slope,
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer) if self.kernel_regularizer else None
        })
        return config

    @classmethod
    def from_config(cls, config):
        if config.get("kernel_regularizer"):
            config["kernel_regularizer"] = regularizers.deserialize(config["kernel_regularizer"])
        return cls(**config)

class Encoder(tf.keras.Model):
    def __init__(self, d1=1000, d2=1000, dz=None, dropout_latent=0.5,**kwargs): # !
        super().__init__(**kwargs)
        self.d1 = d1
        self.d2 = d2
        self.dz = dz
        self.dropout_latent = dropout_latent
        self.kernel_regularizer=l2_reg
        self.net = Sequential([
            FCBlock(d1, batchnorm=True,kernel_regularizer=l2_reg),
            FCBlock(d2, batchnorm=True,kernel_regularizer=l2_reg),
        ])
        self.latent = Dense(dz, use_bias=False, kernel_regularizer=l2_reg)  # paper sets bias to 0 to avoid pixel bias effects
        self.dropout = Dropout(rate=dropout_latent) if dropout_latent>0 else None
    def call(self, x,training=False):
        h = self.net(x,training=training)
        z = self.latent(h)
        if self.dropout is not None:
            z = self.dropout(z,training=training)
        return z

    def get_config(self):
        config = super().get_config()
        config.update({
            "d1": self.d1, "d2": self.d2, "dz": self.dz, "dropout_latent": self.dropout_latent
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Decoder(tf.keras.Model):
    def __init__(self, d1=1000, d2=1000, B=0,**kwargs):
        super().__init__(**kwargs)
        self.d1 = d1; self.d2 = d2; self.B = B;self.kernel_regularizer=l2_reg
        self.fc1 = FCBlock(d1, batchnorm=False,kernel_regularizer=l2_reg)
        self.fc2 = FCBlock(d2, batchnorm=False,kernel_regularizer=l2_reg)
        self.out = Dense(B, use_bias=False, kernel_regularizer=l2_reg)  # bias kept 0 per paper
    def call(self, z):
        h = self.fc1(z)
        h = self.fc2(h)
        xrec = self.out(h)
        return xrec

    def get_config(self):
        config = super().get_config()
        config.update({"d1": self.d1, "d2": self.d2, "B": self.B})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class LatentDiscriminator(tf.keras.Model):
    def __init__(self, d1=1000, d2=1000,**kwargs):
        super().__init__(**kwargs)
        self.d1 = d1; self.d2 = d2
        self.net = Sequential([
            Dense(d1),
            LeakyReLU(negative_slope=0.2),
            Dense(d2),
            LeakyReLU(negative_slope=0.2),
            Dense(1)
        ])
    def call(self, z,training=False):
        return tf.squeeze(self.net(z,training=training),axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"d1": self.d1, "d2": self.d2})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ImageDiscriminator(tf.keras.Model):
    def __init__(self, d1=1000, d2=1000,**kwargs):
        super().__init__(**kwargs)
        self.d1=d1; self.d2=d2
        self.net = Sequential([
            Dense(d1),
            LeakyReLU(negative_slope=0.2),
            Dense(d2),
            LeakyReLU(negative_slope=0.2),
            Dense(1)
        ])
    def call(self, x,training=False):
        return tf.squeeze(self.net(x,training=training),axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"d1": self.d1, "d2": self.d2})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ------------------------------
# Loss helpers
# ------------------------------
def reconstruction_loss(x, xrec,lambda_sam=1.0):
    batch_size = tf.shape(x)[0]
    x = tf.cast(x, tf.float32)
    xrec=tf.cast(xrec,tf.float32)

    # Base L2,1 loss
    error_matrix=(x-xrec) # (batch_size,B)
    res=tf.math.sqrt(tf.reduce_sum(tf.square(error_matrix),axis=1)+1e-8) # (batch_size,)
    l21_loss=tf.reduce_mean(res)

    # Spectral Angle Mapper (SAM) loss
    dot = tf.reduce_sum(x * xrec, axis=1)
    norm_x = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1) + 1e-8)
    norm_rec = tf.sqrt(tf.reduce_sum(tf.square(xrec), axis=1) + 1e-8)
    cos_angle = dot / (norm_x * norm_rec + 1e-8)
    angle = tf.acos(tf.clip_by_value(cos_angle, -1.0, 1.0))
    sam_loss = tf.reduce_mean(angle)

    # Weighted sum (Hybrid)
    total_loss = l21_loss + lambda_sam * sam_loss
    return total_loss

    # return tf.reduce_mean((x - xrec) ** 2)

def consistency_loss(z, encoder, decoder): 
    rec = encoder(tf.stop_gradient(decoder(z,training=True)),training=True)
    batch_size = tf.shape(z)[0]
    z=tf.cast(z,tf.float32)
    rec=tf.cast(rec,tf.float32)
    error_matrix=(z-rec) # (batch_size,B)
    res=tf.math.sqrt(tf.reduce_sum(tf.square(error_matrix),axis=1)+1e-8) # (batch_size,)
    consis_loss=tf.reduce_sum(res)
    return (1/(2*tf.cast(batch_size, tf.float32)))*consis_loss
    # return tf.reduce_mean((z - rec) ** 2)

def shrink_loss(z):
    z=tf.cast(z,tf.float32)
    return tf.reduce_mean(z**2)

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ------------------------------
# Training loop
# ------------------------------
@tf.function
def train_step(x,
                enc,
                dec,
                dznet,
                dinet,
                opt_enc,
                opt_dec,
                opt_dz,
                opt_di,                
                alpha0=5.0,
                alpha1=1.0,
                alpha2=0.1,
                k_steps=1,
            ):
    """
    hsi_array: numpy array (H, W, B) with float values (recommended normalized per band)
    returns trained models and final detection map
    """
    x = tf.cast(x, tf.float32)

    # 1. Update Discriminators (k_steps times)
    for _ in range(k_steps):
        with tf.GradientTape() as tape:
            z_fake = enc(x, training=False)
            z_real = tf.random.normal(tf.shape(z_fake))
            logits_real = dznet(z_real, training=True)
            logits_fake = dznet(z_fake, training=True)
            loss_dz = bce_logits(tf.ones_like(logits_real), logits_real) + \
                        bce_logits(tf.zeros_like(logits_fake), logits_fake)
            loss_dz = tf.cast(loss_dz, tf.float32)
        # scaled_loss = opt_dz.get_scaled_loss(loss_dz)
        grads  = tape.gradient(loss_dz, dznet.trainable_variables)
        # grads = opt_dz.get_unscaled_gradients(scaled_grads) # UNSCALE GRADIENTS
        opt_dz.apply_gradients(zip(grads, dznet.trainable_variables))
        
        with tf.GradientTape() as tape:
            z_e = enc(x, training=False)
            xrec_detached = dec(z_e, training=False)
            logits_real = dinet(x, training=True)
            logits_fake = dinet(xrec_detached, training=True)
            loss_di = bce_logits(tf.ones_like(logits_real), logits_real) + \
                        bce_logits(tf.zeros_like(logits_fake), logits_fake)
            loss_di = tf.cast(loss_di, tf.float32)
        # scaled_loss = opt_di.get_scaled_loss(loss_di)
        grads  = tape.gradient(loss_di, dinet.trainable_variables)
        # grads = opt_di.get_unscaled_gradients(scaled_grads) # UNSCALE GRADIENTS
        opt_di.apply_gradients(zip(grads, dinet.trainable_variables))

    # 2. Update Generators (1 time)
    with tf.GradientTape(persistent=True) as tape:
        z = enc(x, training=True)
        xrec = dec(z, training=True)
        
        LR = reconstruction_loss(x, xrec)
        Lziz = consistency_loss(z, enc, dec)
        Lzl1 = shrink_loss(z)
        loss_ae = alpha0 * LR + alpha1 * Lziz + alpha2 * Lzl1
        
        logits_enc_adv = dznet(z, training=True)
        loss_enc_adv = bce_logits(tf.ones_like(logits_enc_adv), logits_enc_adv)
        
        logits_dec_adv = dinet(xrec, training=True)
        loss_dec_adv = bce_logits(tf.ones_like(logits_dec_adv), logits_dec_adv)
        
        enc_loss = loss_ae + 1.0 * loss_enc_adv
        dec_loss = loss_ae + 1.0 * loss_dec_adv

        enc_loss = tf.cast(enc_loss, tf.float32)
        dec_loss = tf.cast(dec_loss, tf.float32)
    
    grads_enc  = tape.gradient(enc_loss, enc.trainable_variables)
    opt_enc.apply_gradients(zip(grads_enc, enc.trainable_variables))
    
    grads_dec  = tape.gradient(dec_loss, dec.trainable_variables)
    opt_dec.apply_gradients(zip(grads_dec, dec.trainable_variables))

    # Update Generators
    # with tf.GradientTape() as tape_enc:
    #     z = enc(x, training=True)
    #     xrec = dec(z, training=True)
        
    #     LR = reconstruction_loss(x, xrec)
    #     Lziz = consistency_loss(z, enc, dec)
    #     Lzl1 = shrink_loss(z)
    #     loss_ae = alpha0 * LR + alpha1 * Lziz + alpha2 * Lzl1
        
    #     logits_enc_adv = dznet(z, training=True)
    #     loss_enc_adv = bce_logits(tf.ones_like(logits_enc_adv), logits_enc_adv)
        
    #     enc_loss = loss_ae + 1.0 * loss_enc_adv
    #     enc_loss = tf.cast(enc_loss, tf.float32)

    # with tf.GradientTape() as tape_dec:
    #     z = enc(x, training=True)
    #     xrec = dec(z, training=True)
        
    #     LR = reconstruction_loss(x, xrec)
    #     Lziz = consistency_loss(z, enc, dec)
    #     Lzl1 = shrink_loss(z)
    #     loss_ae = alpha0 * LR + alpha1 * Lziz + alpha2 * Lzl1
        
    #     logits_dec_adv = dinet(xrec, training=True)
    #     loss_dec_adv = bce_logits(tf.ones_like(logits_dec_adv), logits_dec_adv)
        
    #     dec_loss = loss_ae + 1.0 * loss_dec_adv
    #     dec_loss = tf.cast(dec_loss, tf.float32)

    # grads_enc = tape_enc.gradient(enc_loss, enc.trainable_variables)
    # opt_enc.apply_gradients(zip(grads_enc, enc.trainable_variables))

    # grads_dec = tape_dec.gradient(dec_loss, dec.trainable_variables)
    # opt_dec.apply_gradients(zip(grads_dec, dec.trainable_variables))

    del tape

    return LR, Lziz, Lzl1, loss_enc_adv, loss_dec_adv

def train(hsi,infer,ref,window,ov,method="mean"):
    print(f"Dataset name: MOCK_1")
    print(f"Model: HADGAN")

    H,W,B=hsi.shape
    Hk,Wk,Bk=window
    h,w,b=infer.shape

    epochs=300;dropout=0.5;k_steps=4
    lr_enc=5e-5;lr_others=1e-4
    # patch_batch_size=Hk*Wk # per patch batch size
    # batch_size=1 # how many patches to process in a single call to train_hadgan function

    # --- Early stopping parameters ---
    # patience = 50  # Number of epochs to wait for improvement
    # min_delta = 0.0001  # Minimum change to be considered an improvement
    # best_loss = float('inf')
    # epochs_no_improve = 0
    # best_weights=None

    # alpha0=1.0;alpha1=1.0;alpha2=0.1

    stride_h=max(1,(100-ov)*Hk//100)
    stride_w=max(1,(100-ov)*Wk//100)
    
    # --- Model Initialization ---
    dz=compute_dz(Bk)
    enc = Encoder(dz=dz, dropout_latent=dropout)
    dec = Decoder(B=Bk)
    dznet = LatentDiscriminator()
    dinet = ImageDiscriminator()

    # --- Initialize optimizers ---
    opt_enc_base = tf.keras.optimizers.RMSprop(learning_rate=lr_enc)
    opt_dec_base = tf.keras.optimizers.Adam(learning_rate=lr_others)
    opt_dz_base = tf.keras.optimizers.Adam(learning_rate=lr_others)
    opt_di_base = tf.keras.optimizers.Adam(learning_rate=lr_others)

    # Wrap base optimizers in LossScaleOptimizer
    opt_enc = tf.keras.mixed_precision.LossScaleOptimizer(opt_enc_base)
    opt_dec = tf.keras.mixed_precision.LossScaleOptimizer(opt_dec_base)
    opt_dz = tf.keras.mixed_precision.LossScaleOptimizer(opt_dz_base)
    opt_di = tf.keras.mixed_precision.LossScaleOptimizer(opt_di_base)
    # opt_dz=opt_dz_base
    # opt_di=opt_di_base

    # ckpt = tf.train.Checkpoint(enc=enc, dec=dec, dznet=dznet, dinet=dinet,
    #                     #    opt_enc=opt_enc, opt_dec=opt_dec, opt_dz=opt_dz, opt_di=opt_di
    #                     )
    # ckpt_manager = tf.train.CheckpointManager(ckpt, directory="/home/ubuntu/aditya/BioSky/checkpoints", max_to_keep=5)
    # CHECKPOINT_DIR = "/home/ubuntu/aditya/BioSky/HADGAN_Checkpoints"
    # os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # ckpt_prefix = os.path.join(CHECKPOINT_DIR, 'hadgan_ckpt') # Define checkpoint prefix


    # ---- Using generator function ----
    h_starts = list(range(0, H - Hk + 1, stride_h))
    w_starts = list(range(0, W - Wk + 1, stride_w))

    if h_starts[-1] != H - Hk:
        h_starts.append(H - Hk)   # force last patch to align with bottom edge
    if w_starts[-1] != W - Wk:
        w_starts.append(W - Wk)   # force last patch to align with right edge

    # now it takes into account only those patches which have at least 70% pixels with valid data
    def gen_patches():
        for i in h_starts:
            for j in w_starts:
                patch=hsi[i:i+Hk, j:j+Wk, :].astype(np.float32)

                non_zero_mask=patch>1e-6

                # for each spatial position, whether at least one spectral band has a value 
                # above a minimal threshold, indicating the presence of valid HSI data.
                data_present_per_pixel=np.any(non_zero_mask,axis=-1)

                num_data_pixels = np.sum(data_present_per_pixel)
                total_pixels = Hk * Wk

                if num_data_pixels<0.8*total_pixels:
                    continue

                yield patch

    # num_to_show = 3
    # save_dir = "patch_visuals"
    # os.makedirs(save_dir, exist_ok=True)

    # for idx, patch in enumerate(gen_patches()):
    #     if idx >= num_to_show:
    #         break

    #     print(f"\n--- Patch {idx+1} Statistics ---")
    
    #     # 1. Define the mask: Check if ANY band is greater than the threshold (1e-6)
    #     # This correctly identifies a pixel as containing data if any spectral signature is present.
    #     non_zero_mask = patch > 1e-6
    #     data_present_per_pixel = np.any(non_zero_mask, axis=-1)
        
    #     num_data_pixels = np.sum(data_present_per_pixel)
    #     total_pixels = Hk * Wk
        
    #     print(f"Total pixels: {total_pixels}")
    #     print(f"Data pixels (Non-Grey): {num_data_pixels}")
    #     print(f"Min value (All Bands): {np.min(patch)}")
    #     print(f"Max value (All Bands): {np.max(patch)}")
        
    #     # 2. Check the standard deviation (low std suggests constant grey area)
    #     print(f"Average Std Dev across bands: {np.mean(np.std(patch, axis=(0, 1))):.6f}")

    #     # Convert hyperspectral patch to RGB-like image (using 3 representative bands)
    #     rgb_patch = patch[..., [10, 30, 50]]  # choose visible bands; adjust for your dataset
    #     rgb_patch = (rgb_patch - rgb_patch.min()) / (rgb_patch.max() - rgb_patch.min() + 1e-8)

    #     plt.figure(figsize=(4, 4))
    #     plt.imshow(rgb_patch)
    #     plt.title(f"Patch #{idx+1}")
    #     plt.axis('off')
        
    #     save_path = os.path.join(save_dir, f"patch_{idx+1}.png")
    #     plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    #     plt.close()  # close to free memory

    # return

    dataset = tf.data.Dataset.from_generator(
        gen_patches,
        output_signature=tf.TensorSpec(shape=(Hk,Wk,Bk), dtype=tf.float32)
    )

    dataset = dataset.map(lambda x: augment_patch(x),num_parallel_calls=tf.data.AUTOTUNE)

    # Map the patches (Hk, Wk, Bk) into flat pixel batches (Hk*Wk, Bk)
    dataset = dataset.map(lambda x: tf.reshape(x, [-1, Bk]), 
                        num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # --- Patches Extraction ---
    # hsi=tf.convert_to_tensor(hsi[np.newaxis, ...], dtype=tf.float32)
    # patches=tf.image.extract_patches(images=hsi,
    #                                 sizes=[1,Hk,Wk,1],
    #                                 strides=[1,stride_h,stride_w,1],
    #                                 rates=[1,1,1,1],
    #                                 padding='VALID')

    # patches=tf.reshape(patches,[-1,Hk,Wk,Bk])
    # num_patches=patches.shape[0]
    # print("Num patches:", num_patches) # 324
    # print("Patch shape:", patches.shape[1:]) # (120,120,239)
    # dataset=tf.data.Dataset.from_tensor_slices(patches).prefetch(tf.data.AUTOTUNE)

    final_maps={}
    final_binary_maps={}
     
    for ep in trange(epochs, desc="Training HADGAN"):
        total_loss_epoch=0.0
        start=time.time()
        for X_patch in dataset: # each X_patch is of the shape (Hk*Wk,Bk)
        # List to store the final maps for different k_steps
        # final_maps = {}
        # final_binary_maps={}

        # Run training for k=1 to k=5
        # for k in range(1, 6):
            # print(f"\n--- Starting training with k_steps = {k} ---")

            # X_patch_batch=tf.data.Dataset.from_tensor_slices(X_patch).batch(patch_batch_size) # (patch_batch_size,Hk*Wk/patch_batch_size,Bk)

            # for xbatch in X_patch_batch: # xbatch = (Hk*Wk/patch_batch_size,Bk)
            LR, Lziz, Lzl1, loss_enc_adv, loss_dec_adv=train_step(X_patch,enc,dec,dznet,
                                                                  dinet,opt_enc,opt_dec,
                                                                  opt_dz,opt_di,k_steps=k_steps)

                # # Average loss over epoch 
                # avg_loss_epoch = total_loss_epoch / int(num_batches) # !

                # # Check for improvement
                # if avg_loss_epoch < best_loss - min_delta:
                #     best_loss = avg_loss_epoch
                #     epochs_no_improve = 0
                #     best_weights = [enc.get_weights(), dec.get_weights(), dznet.get_weights(), dinet.get_weights()]
                # else:
                #     epochs_no_improve += 1

                # # Check if patience is exceeded
                # if epochs_no_improve >= patience:
                #     print(f'Early stopping triggered after {ep+1} epochs. No improvement for {patience} epochs.')
                #     break

                # end epoch

            # After training, load the best weights
            # if best_weights:
            #     enc.set_weights(best_weights[0])
            #     dec.set_weights(best_weights[1])
            #     dznet.set_weights(best_weights[2])
            #     dinet.set_weights(best_weights[3])

            # final_maps[k] = out['final_map']
            # bests, pr_auc, roc=tune_methods(final_maps[k], ref.reshape(hsi.shape[0],hsi.shape[1]))

            # threshold=bests['iterative_f1'][1]
            # binary_map=(final_maps[k]>threshold).astype(np.uint8)
            # final_binary_maps[k]=binary_map

            # print(f"Tuned results:, {bests}")
            # print(f"pr_auc: {pr_auc}, roc: {roc}")

        if (ep + 1) % 30 == 0 or ep == 0:
            elapsed = time.time() - start
            tf.print(
                f"\nEpoch {ep+1}/{epochs} → LR={LR:.6f} Lziz={Lziz:.6f} Lzl1={Lzl1:.6f} "
                f"adv_enc={loss_enc_adv:.6f} adv_dec={loss_dec_adv:.6f} time={elapsed:.1f}s",
                output_stream=sys.stdout
            )

    # num_k_steps = len(final_maps)
    # fig, axes = plt.subplots(nrows=2, ncols=1 + num_k_steps, figsize=(3 * (1 + num_k_steps), 6))

    # # Plot Ground Truth in the first column
    # axes[0, 0].imshow(ref.reshape(h, w), cmap="gray")
    # axes[0, 0].set_title("Ground Truth")
    # axes[0, 0].axis('off')
    # axes[1, 0].axis('off')

    # # Iterate through k values and plot the continuous and binary maps
    # for i, k in enumerate(sorted(final_maps.keys())):
    #     # Plot the continuous map in the first row, starting from the second column
    #     im_continuous = axes[0, i + 1].imshow(final_maps[k].reshape(h,w), cmap="gray")
    #     axes[0, i + 1].set_title(f"Continuous Map (epoch={k})")
    #     axes[0, i + 1].axis('off')
    #     fig.colorbar(im_continuous, ax=axes[0, i + 1], fraction=0.046, pad=0.04)

    #     # Plot the binary map in the second row, starting from the second column
    #     im_binary = axes[1, i + 1].imshow(final_binary_maps[k].reshape(h,w), cmap="gray")
    #     axes[1, i + 1].set_title(f"Binary Map (epoch={k})")
    #     axes[1, i + 1].axis('off')
    #     fig.colorbar(im_binary, ax=axes[1, i + 1], fraction=0.046, pad=0.04)

    # plt.tight_layout()
    # plt.savefig(f"/home/ubuntu/aditya/BioSky/Mock1_{method}_inference_epoch_comparison.png", dpi=300)
    # plt.close()

    # print(f"Comparison image saved as Mock1_{method}_inference_epoch_comparison.png")

    print("\n--- All training runs complete. Saving the model.... ---")

    # Force build with dummy input (matching your input dimensionality)
    dummy_input = tf.zeros((1, Bk), dtype=tf.float32)  
    hadgan = HADGAN(enc, dec, dznet, dinet)
    _ = hadgan(dummy_input, training=False)  # run once to build

    hadgan.save(f"/home/ubuntu/aditya/BioSky/Logs/train_mock1_{Hk}.keras", include_optimizer=False)

    print(f"Models saved to train_mock1_{Hk}.keras")

# ------------------------------
# Inference 
# ------------------------------
def inference(hsi,ref,window,method="mean"):
    H, W, B = hsi.shape
    Hk,Wk,Bk=window
    MODEL_PATH = f"/home/ubuntu/aditya/BioSky/Logs/train_mock1_{Hk}.keras"

    # --- Define custom objects for loading ---
    # custom_objects = {
    #     'HADGAN': HADGAN,
    #     'Encoder': Encoder,
    #     'Decoder': Decoder,
    #     'LatentDiscriminator': LatentDiscriminator,
    #     'ImageDiscriminator': ImageDiscriminator,
    #     'FCBlock': FCBlock
    # }

    # --- Load the entire model ---
    # print(f"\n--- Loading model for inference ---")
    # loaded_hadgan = load_model(
    #     MODEL_PATH,
    #     custom_objects=custom_objects
    # )

    # # # --- Extract components and run inference ---
    # # We need the original Encoder/Decoder for the custom inference functions
    # enc = loaded_hadgan.enc
    # dec = loaded_hadgan.dec

    # batch_size=Hk*Wk

    # final_maps = {}
    # final_binary_maps={}

    ov=0 # !
    start = time.time()
    # print("Starting reconstruction...")
    # if method=="mean":
    #     Xrec=mean_inference(hsi,enc,dec,window,ov,batch_size)
    # else:
    #     Xrec=weighted_inference(hsi,enc,dec,window,ov,batch_size)

    # del enc, dec

    # print("Reconstruction completed.")
    # savemat="/home/ubuntu/aditya/BioSky/PRE-EVALUATION/pre_eval_res.mat"
    # sio.savemat(savemat, {'data': Xrec})

    # return

    mat=sio.loadmat("/home/ubuntu/aditya/BioSky/PRE-EVALUATION/pre_eval_res.mat")
    Xrec = np.array(mat['data'], dtype=np.float32)
    del mat
    gc.collect() # Force garbage collection
    
    print("Postprocessing started...")
    residual=compute_residual_map(hsi, Xrec)
    del Xrec
    gc.collect() # Force garbage collection

    mask=create_valid_mask(hsi)

    # === NEW: Apply mask to residual map ===
    # residual = residual * mask[:, :, np.newaxis] # Expand mask to (H,W,1) for broadcasting

    # --- Run Post-processing Detectors ---
    bands = select_min_energy_bands(residual, k=3)

    dspatial = spatial_detector_from_residual_fast(residual, bands)
    
    del bands
    gc.collect() # Force garbage collection

    dspectral = spectral_detector_from_residual_fast(residual)

    del residual
    gc.collect() # Force garbage collection

    fmap=fuse_spatial_spectral(dspatial, dspectral, lam=0.5)

    del dspatial, dspectral
    gc.collect() # Force garbage collection

    # fmap[mask == 0] = 0  # keep only inside area

    # from skimage.filters import threshold_otsu

    # threshold = threshold_otsu(fmap)
    # print(f"Threshold using otsu for unsuppressed map: {threshold}")

    # threshold,iterative_f1=iterative_f1_threshold(ref.ravel(), fmap.ravel())

    # binary_map=(fmap>threshold).astype(np.uint8)

    # iterative_f1=f1_score(ref.ravel(),binary_map.ravel())

    # bests=(iterative_f1, threshold, None, None)

    # compute continuous-score metrics once for S
    # pr_auc = average_precision_score(ref.ravel(), fmap.ravel())
    # roc = roc_auc_score(ref.ravel(), fmap.ravel())

    # print(f"Loaded Model Tuned results for original map:, {bests}")
    # print(f"Loaded Model PR-AUC for original map: {pr_auc}, ROC: {roc}")
    
    # he5_path = "/home/ubuntu/aditya/BioSky/PRE-EVALUATION/PRS_L2D_STD_20210516050459_20210516050503_0001.he5"
    s3_path = "/home/ubuntu/data/PRS_L2D_STD_20210516050459_20210516050503_0001.he5"
    # fmap_manmade=create_manmade_mask(hsi,fmap)
    # fmap_manmade[mask == 0] = 0  # keep only inside area

    fmap_manmade=apply_manual_mask(fmap,s3_path)
    # fmap_manmade[mask==0]=0 

    # del fmap

    # --- Evaluate and Visualize ---
    print(f"\n--- Inference completed for overlap parameters = {ov}%. Evaluating results. ---")
    # bests, pr_auc, roc = tune_methods(fmap_manmade, ref.reshape(H, W))

    # from skimage.filters import threshold_otsu

    # threshold = threshold_otsu(fmap_manmade)
    # print(f"Threshold using otsu for unsuppressed map: {threshold}")

    # threshold,iterative_f1=iterative_f1_threshold(ref.ravel(), fmap_manmade.ravel())
    threshold=0.215
    binary_map_manmade=(fmap_manmade>threshold).astype(np.uint8) # !
    # from skimage.filters import threshold_otsu
    # threshold = threshold_otsu(fmap_manmade)
    # binary_map_manmade = (fmap_manmade > threshold).astype(np.uint8)

    # iterative_f1=f1_score(ref.ravel(),binary_map_manmade.ravel())

    # bests=(iterative_f1, threshold, None, None)

    # # compute continuous-score metrics once for S
    # pr_auc = average_precision_score(ref.ravel(), fmap_manmade.ravel())
    # roc = roc_auc_score(ref.ravel(), fmap_manmade.ravel())

    # print(f"Loaded Model Tuned results for original map:, {bests}")
    # print(f"Loaded Model PR-AUC for original map: {pr_auc}, ROC: {roc}")

    def save_geotiff_from_he5(he5_path, output_path, binary_map):
        """
        Save a numpy array (like anomaly map) as GeoTIFF using
        geolocation metadata from a PRISMA HE5 file.
        """
        with h5py.File(he5_path, 'r') as f:
            # Read latitude and longitude grids
            lat = f['HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Latitude'][:]
            lon = f['HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Longitude'][:]

        # Compute geographic bounds
        min_lon, max_lon = lon.min(), lon.max()
        min_lat, max_lat = lat.min(), lat.max()

        # Create affine transform from bounding box
        height, width = binary_map.shape
        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

        # Define coordinate reference system (CRS)
        crs = CRS.from_epsg(4326)  # WGS84

        # Save to GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,  # one band
            dtype=binary_map.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(binary_map, 1)

        print(f"✅ GeoTIFF saved for patch size {Hk}*{Wk}")

    output_path = f"/home/ubuntu/aditya/BioSky/PRE-EVALUATION/evaluation_anomaly_map_{threshold*100}.tif"
    save_geotiff_from_he5(s3_path, output_path, binary_map_manmade)

    elapsed=time.time()-start
    print(f"--- Time elapsed: {elapsed:.1f}s. ---")

    # Change layout to 1 row, 4 columns
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), constrained_layout=True) 

    axes[0].imshow(ref.reshape(1216, 1280), cmap="gray")
    axes[0].set_title("1. Ground Truth Mask")
    axes[0].axis('off') # Hide axes for cleaner image viewing

    im1 = axes[1].imshow(fmap_manmade, cmap="gray", interpolation='none')
    axes[1].set_title("2. Man-Made Filtered Map ")
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(binary_map_manmade, cmap="gray", interpolation='none')
    axes[2].set_title("3. Binary Man-Made Filtered Map")
    axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(fmap, cmap="gray", interpolation='none')
    axes[3].set_title("4. Original Map")
    axes[3].axis('off')
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(f"Anomaly Detection Results with {ov} overlap", fontsize=16, fontweight="bold")

    output_dir = "/home/ubuntu/aditya/BioSky/PRE-EVALUATION"
    output_path = os.path.join(output_dir, f"pre_evaluation_anomaly_map.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    # print(f"Inference results saved to infer_mock2_{Hk}_{ov}_ov.png")

if __name__ == "__main__":
    # print("What to do? 1.Train 2.Test")
    # choice=input()
    
    mat=sio.loadmat("/home/ubuntu/aditya/BioSky/Datasets/MOCK/processed_prisma_data.mat")
    hsi = np.array(mat['data'], dtype=np.float32)
    print("hsi image shape:", hsi.shape)
    hsi = (hsi - hsi.min()) / (np.ptp(hsi) + 1e-8) 

    # mat=sio.loadmat("/home/ubuntu/aditya/BioSky/Datasets/MOCK_2/prisma_data.mat")
    mat=sio.loadmat("/home/ubuntu/aditya/BioSky/PRE-EVALUATION/pre_evaluation.mat")
    infer = np.array(mat['data'], dtype=np.float32)
    print("Infer shape:", infer.shape)
    infer = (infer - infer.min()) / (np.ptp(infer) + 1e-8)  # normalize to [0,1] for stability

    # gt_mat=sio.loadmat("/home/ubuntu/aditya/BioSky/Datasets/MOCK/prisma_gt.mat")
    gt_mat=sio.loadmat("/home/ubuntu/aditya/BioSky/Datasets/MOCK_2/prisma_gt.mat")
    ref = gt_mat['data']  # shape (H,W), binary 0/1
    # ref = ref.astype(np.uint8).reshape(-1)
    print("Ground truth shape:", ref.shape)

    # def save_geotiff_from_he5(he5_path, output_path, fmap):
    #     """
    #     Save a numpy array (like anomaly map) as GeoTIFF using
    #     geolocation metadata from a PRISMA HE5 file.
    #     """
    #     with h5py.File(he5_path, 'r') as f:
    #         # Read latitude and longitude grids
    #         lat = f['HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Latitude'][:]
    #         lon = f['HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Longitude'][:]

    #     # Compute geographic bounds
    #     min_lon, max_lon = lon.min(), lon.max()
    #     min_lat, max_lat = lat.min(), lat.max()

    #     # Create affine transform from bounding box
    #     height, width = fmap.shape
    #     transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

    #     # Define coordinate reference system (CRS)
    #     crs = CRS.from_epsg(4326)  # WGS84

    #     # Save to GeoTIFF
    #     with rasterio.open(
    #         output_path,
    #         'w',
    #         driver='GTiff',
    #         height=height,
    #         width=width,
    #         count=1,  # one band
    #         dtype=fmap.dtype,
    #         crs=crs,
    #         transform=transform,
    #     ) as dst:
    #         dst.write(fmap, 1)

    #     print(f"✅ GeoTIFF saved: {output_path}")

    # output_path = "mock1_gt.tif"
    # he5_path = "/home/ubuntu/aditya/BioSky/Datasets/PRS_L2D_STD_20241205050514_20241205050518_0001.he5"
    # save_geotiff_from_he5(he5_path, output_path, ref)

    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, default=120, choices=[120,64,32,16])
    parser.add_argument('--method', type=str, default='weighted', choices=['mean', 'weighted'])
    parser.add_argument('--choice', type=int, default=1, choices=[1,2])
    args = parser.parse_args()

    window=(args.patch_size,args.patch_size,hsi.shape[2])
    print(f"Patch size: {window[0]}*{window[1]}")

    if args.choice==1:
        set_seeds(42)
        overlap=50 # in %
        train(hsi,infer,ref,window,overlap)
    else:
        set_seeds(42)
        inference(infer,ref,window,method=args.method)