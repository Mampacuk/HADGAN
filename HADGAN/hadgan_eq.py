import sys
import argparse
import math
import time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
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

from utils.adaptive_thresh import tune_methods,iterative_f1_threshold
from utils.sliding_window_inference import mean_inference, weighted_inference

tf.config.optimizer.set_jit(True)
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
tf.keras.mixed_precision.set_global_policy('mixed_float16')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
        
        # Optional: Limit GPU memory per process (e.g., 50% of 23 GB GPU ≈ 11500 MB)
        tf.config.set_logical_device_configuration(
            gpu,
            [tf.config.LogicalDeviceConfiguration(memory_limit=11500)]
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
def reconstruction_loss(x, xrec):
    batch_size = tf.shape(x)[0]
    xrec=tf.cast(xrec,tf.float32)
    error_matrix=(x-xrec) # (batch_size,B)
    res=tf.math.sqrt(tf.reduce_sum(tf.square(error_matrix),axis=1)+1e-8) # (batch_size,)
    rec_loss=tf.reduce_sum(res)
    return (1/(2*tf.cast(batch_size, tf.float32)))*rec_loss
    # return tf.reduce_mean((x - xrec) ** 2)

def consistency_loss(z, encoder, decoder): 
    rec = encoder(decoder(z,training=True),training=True)
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

# ------------------------------
# Postprocessing detectors
# ------------------------------
def compute_residual_map(hsi, recon):
    # hsi, recon are (H, W, B)
    return np.abs(hsi - recon)  # shape (H,W,B)

def select_min_energy_bands(residual, k=3):
    # residual shape (H,W,B)
    H,W,B = residual.shape
    energies = residual.reshape(-1, B).sum(axis=0)  # sum over space for each band
    idx = np.argsort(energies)[:k]
    return idx

def spatial_detector_from_residual(residual, selected_bands): # !
    # residual: (H,W,B)
    H,W,B = residual.shape
    # compute per-band images (grayscale)
    df_list = []
    for b in selected_bands:
        band = residual[:,:,b]

        # Normalize to 8-bit for OpenCV
        band8 = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        
        # Simulating "attribute and edge-preserving filters" with a combination of filters.
        # Here, we use a median filter to reduce salt-and-pepper noise and
        # a bilateral filter for edge-aware smoothing, which is an alternative
        # to the guided filter if it's not available.
        denoised = cv2.medianBlur(band8, 3)
        smoothed = cv2.bilateralFilter(denoised, 5, 75, 75)
        
        # Calculate residual to highlight filtered-out anomalies
        df = np.abs(band8.astype(np.float32) - smoothed.astype(np.float32))
        df_list.append(df)
        
    F = np.mean(np.stack(df_list, axis=0), axis=0)
    
    # Paper uses Guided Filter [cite: 313]
    try:
        F_norm = cv2.normalize(F, None, 0, 255, cv2.NORM_MINMAX).astype('float32')
        # The paper suggests filter size 3 and blur degree 0.5 [cite: 317]
        # guided filter requires an 8-bit or float32 guide and source
        gf = cv2.ximgproc.guidedFilter(guide=F_norm, src=F_norm, radius=3, eps=0.5)
        out = gf
    except Exception:
        # Fallback if ximgproc is not installed
        out = cv2.bilateralFilter(cv2.normalize(F, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'), 5, 75, 75).astype('float32')

    out_norm = (out - out.min()) / (np.ptp(out) + 1e-8)
    return out_norm
    #     # morphological area opening/closing to remove large background components
    #     # area_opening will remove components smaller than area threshold; but paper uses attribute opening/closing
    #     # We choose area thresholds small to preserve small anomalies (paper suggests preserve small area -> so close to 0)
    #     img = band
    #     # convert to 8bit for skimage rank filters
    #     band8 = img_as_ubyte((img - img.min()) / (np.ptp(img)+1e-8))
    #     # area opening/closing using scikit-image (removes small objects)
    #     opened = area_opening(band8, area_threshold=5)  # small threshold; tune per dataset
    #     closed = area_closing(band8, area_threshold=5)
    #     df = np.abs(band8 - opened).astype(np.float32) + np.abs(closed - band8).astype(np.float32)
    #     df_list.append(df)
    # F = np.mean(np.stack(df_list, axis=0), axis=0)
    # # simple guided filter refinement: try opencv ximgproc if available, else use bilateral
    # try:
    #     gf = cv2.ximgproc.guidedFilter(guide=img_as_ubyte(F/255.0), src=img_as_ubyte(F/255.0), radius=3, eps=0.5)
    #     out = gf.astype(np.float32)
    # except Exception:
    #     out = cv2.bilateralFilter(img_as_ubyte(F/255.0), d=5, sigmaColor=75, sigmaSpace=75)
    # # linear coefficients mean step (paper), we simply normalize
    # out_norm = (out - out.min()) / (np.ptp(out) + 1e-8)
    # return out_norm

def spectral_detector_from_residual(residual):
    # residual: (H,W,B) -> flatten pixels and compute Mahalanobis (RX) score
    H,W,B = residual.shape
    X = residual.reshape(-1, B)
    # fit mean & covariance robustly
    cov = EmpiricalCovariance().fit(X)
    mean = cov.location_
    precision = cov.precision_
    dif = (X - mean)
    # Mahalanobis distance per pixel
    m = np.sum(dif.dot(precision) * dif, axis=1)
    return m.reshape(H, W)

def fuse_spatial_spectral(dspatial, dspectral, lam=0.5):
    # normalize each
    ds = (dspatial - dspatial.min()) / (np.ptp(dspatial) + 1e-8)
    dspec = (dspectral - dspectral.min()) / (np.ptp(dspectral) + 1e-8)
    return lam * ds + (1-lam) * dspec

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
                alpha0=1.0,
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
        grads = tape.gradient(loss_dz, dznet.trainable_variables)
        # grads = opt_dz.get_unscaled_gradients(grads) # UNSCALE GRADIENTS
        opt_dz.apply_gradients(zip(grads, dznet.trainable_variables))
        # opt_dz.minimize(loss_dz,dznet.trainable_variables,tape=tape)
        
        with tf.GradientTape() as tape:
            z_e = enc(x, training=False)
            xrec_detached = dec(z_e, training=False)
            logits_real = dinet(x, training=True)
            logits_fake = dinet(xrec_detached, training=True)
            loss_di = bce_logits(tf.ones_like(logits_real), logits_real) + \
                        bce_logits(tf.zeros_like(logits_fake), logits_fake)
            loss_di = tf.cast(loss_di, tf.float32)
        grads = tape.gradient(loss_di, dinet.trainable_variables)
        # grads = opt_di.get_unscaled_gradients(grads) # UNSCALE GRADIENTS
        opt_di.apply_gradients(zip(grads, dinet.trainable_variables))
        # opt_di.minimize(loss_di,dinet.trainable_variables,tape=tape)

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
    
    grads_enc = tape.gradient(enc_loss, enc.trainable_variables)
    # grads_enc = opt_enc.get_unscaled_gradients(grads_enc) # UNSCALE GRADIENTS
    opt_enc.apply_gradients(zip(grads_enc, enc.trainable_variables))
    # opt_enc.minimize(enc_loss,enc.trainable_variables,tape=tape)
    
    grads_dec = tape.gradient(dec_loss, dec.trainable_variables)
    # grads_dec = opt_dec.get_unscaled_gradients(grads_dec) # UNSCALE GRADIENTS
    opt_dec.apply_gradients(zip(grads_dec, dec.trainable_variables))
    # opt_dec.minimize(dec_loss,dec.trainable_variables,tape=tape)

    del tape

    return LR, Lziz, Lzl1, loss_enc_adv, loss_dec_adv

def train(hsi,ref,window,ov):
    print(f"Dataset name: MOCK_1")
    print(f"Model: HADGAN")

    H,W,B=hsi.shape
    Hk,Wk,Bk=window

    epochs=100;dropout=0.5;k_steps=4
    lr_enc=1e-4;lr_others=1e-4
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
    # opt_dz = tf.keras.mixed_precision.LossScaleOptimizer(opt_dz_base)
    # opt_di = tf.keras.mixed_precision.LossScaleOptimizer(opt_di_base)
    opt_dz=opt_dz_base
    opt_di=opt_di_base

    def gen_patches():
        for i in range(0, H - Hk + 1, stride_h):
            for j in range(0, W - Wk + 1, stride_w):
                yield hsi[i:i+Hk, j:j+Wk, :].astype(np.float32)

    dataset = tf.data.Dataset.from_generator(
        gen_patches,
        output_signature=tf.TensorSpec(shape=(Hk,Wk,Bk), dtype=tf.float32)
    )

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

        if (ep + 1) % 50 == 0 or ep == 0:
            elapsed = time.time() - start
            tf.print(
                "Epoch", ep + 1, "/", epochs,
                "→ LR =", LR,
                "Lziz =", Lziz,
                "Lzl1 =", Lzl1,
                "adv_enc =", loss_enc_adv,
                "adv_dec =", loss_dec_adv,
                "time =", elapsed, "s"
                , output_stream=sys.stdout
            )

    print("\n--- All training runs complete. Saving the model.... ---")

    # Force build with dummy input (matching your input dimensionality)
    dummy_input = tf.zeros((1, Bk), dtype=tf.float32)  
    hadgan = HADGAN(enc, dec, dznet, dinet)
    _ = hadgan(dummy_input, training=False)  # run once to build

    # ckpt = tf.train.Checkpoint(enc=enc, dec=dec, dznet=dznet, dinet=dinet,
    #                        opt_enc=opt_enc, opt_dec=opt_dec, opt_dz=opt_dz, opt_di=opt_di)
    # ckpt_manager = tf.train.CheckpointManager(ckpt, directory="/home/ubuntu/aditya/BioSky/checkpoints", max_to_keep=5)
    # ckpt_manager.save()  # write files

    hadgan.save(f"/home/ubuntu/aditya/BioSky/hadgan_mock1.keras", include_optimizer=False)

    print(f"Models saved to hadgan_mock1.keras")
    
    # num_k_steps = len(final_maps)
    # fig, axes = plt.subplots(nrows=2, ncols=1 + num_k_steps, figsize=(20, 5))

    # # Plot Ground Truth in the first column
    # axes[0, 0].imshow(ref.reshape(H, W), cmap="gray")
    # axes[0, 0].set_title("Ground Truth")
    # axes[0, 0].axis('off')
    # axes[1, 0].axis('off')

    # # Iterate through k values and plot the continuous and binary maps
    # for i, k in enumerate(sorted(final_maps.keys())):
    #     # Plot the continuous map in the first row, starting from the second column
    #     im_continuous = axes[0, i + 1].imshow(final_maps[k].reshape(H,W), cmap="gray")
    #     axes[0, i + 1].set_title(f"Continuous Map (k={k})")
    #     axes[0, i + 1].axis('off')
    #     fig.colorbar(im_continuous, ax=axes[0, i + 1], fraction=0.046, pad=0.04)

    #     # Plot the binary map in the second row, starting from the second column
    #     im_binary = axes[1, i + 1].imshow(final_binary_maps[k].reshape(H,W), cmap="gray")
    #     axes[1, i + 1].set_title(f"Binary Map (k={k})")
    #     axes[1, i + 1].axis('off')
    #     fig.colorbar(im_binary, ax=axes[1, i + 1], fraction=0.046, pad=0.04)

    # plt.tight_layout()
    # plt.savefig("/home/ubuntu/aditya/BioSky/Results_train/HADGAN_kstep_l21/sal/hadgan_sal_ksteps_l21_comparison.png", dpi=300)
    # plt.close()

    # print("Comparison image saved as hadgan_sal_ksteps_l21_comparison.png")

    # out = train_hadgan(hsi, epochs=epochs, batch_size=batch_size, dropout=dropout, k_steps=5)

    # print(f"Number of epochs: {epochs}, batch size: {batch_size}, dropout: {dropout}")

    # # # # final detection map:
    # fmap = out['final_map']
    # print("Final map shape:", fmap.shape)

    # bests, pr_auc, roc=tune_methods(fmap, ref.reshape(hsi.shape[0],hsi.shape[1]))
    # print(f"Tuned results:, {bests}")
    # print(f"pr_auc: {pr_auc}, roc: {roc}")

    # threshold=bests['iterative_f1'][1]
    # binary_map=(fmap>threshold).astype(np.uint8)

    # plt.figure(figsize=(12,5))

    # plt.subplot(1,2,1)
    # plt.imshow(ref.reshape(hsi.shape[0], hsi.shape[1]), cmap="gray")
    # plt.title("Ground Truth Mask")

    # plt.subplot(1,2,2)
    # plt.imshow(fmap, cmap="gray")
    # plt.title("Anomaly Detection Map")
    # plt.colorbar()

    # plt.savefig("hadgan_sd_k3.png", dpi=300) # !
    # plt.close()

# ------------------------------
# Inference 
# ------------------------------
def inference(hsi,ref,window,method="mean"):
    # Assuming you want to load the model for k=5 and L21 loss
    MODEL_PATH = "/home/ubuntu/aditya/BioSky/hadgan_mock1.keras"
    H, W, B = hsi.shape
    Hk,Wk,Bk=window

    # --- Define custom objects for loading ---
    custom_objects = {
        'HADGAN': HADGAN,
        'Encoder': Encoder,
        'Decoder': Decoder,
        'LatentDiscriminator': LatentDiscriminator,
        'ImageDiscriminator': ImageDiscriminator,
        'FCBlock': FCBlock
    }

    # --- Load the entire model ---
    print(f"\n--- Loading model for inference ---")
    loaded_hadgan = load_model(
        MODEL_PATH,
        custom_objects=custom_objects
    )

    # --- Extract components and run inference ---
    # We need the original Encoder/Decoder for the custom inference functions
    enc = loaded_hadgan.enc
    dec = loaded_hadgan.dec

    # 1) Weight norms (should be non-zero and varied)
    # print("\nEncoder weight norms:")
    # for v in enc.trainable_variables[:10]:
    #     print(v.name, np.linalg.norm(v.numpy()))

    # print("\nDecoder weight norms:")
    # for v in dec.trainable_variables[:10]:
    #     print(v.name, np.linalg.norm(v.numpy()))

    batch_size=Hk*Wk

    # overlap=[5,20,35,50,65,80,95,99] # 99% overlap means 1% stride

    final_maps = {}
    final_binary_maps={}

    # for ov in overlap:
    ov=50 # !
    start = time.time()
    if method=="mean":
        Xrec=mean_inference(hsi,enc,dec,window,ov,batch_size)
    else:
        Xrec=weighted_inference(hsi,enc,dec,window,ov,batch_size)

    residual = compute_residual_map(hsi, Xrec)

    # --- Run Post-processing Detectors ---
    bands = select_min_energy_bands(residual, k=3)
    dspatial = spatial_detector_from_residual(residual, bands)
    dspectral = spectral_detector_from_residual(residual)
    fmap = fuse_spatial_spectral(dspatial, dspectral, lam=0.5)

    # final_maps[ov]=fmap

    # --- Evaluate and Visualize ---
    print(f"\n--- Inference completed for overlap parameters = {ov}%. Evaluating results. ---")
    # bests, pr_auc, roc = tune_methods(fmap, ref.reshape(H, W))
    threshold,iterative_f1=iterative_f1_threshold(ref.ravel(), fmap.ravel())
    bests=(iterative_f1, threshold, None, None)

    binary_map=(fmap>threshold).astype(np.uint8)

    # final_binary_maps[ov]=binary_map

    # compute continuous-score metrics once for S
    pr_auc = average_precision_score(ref.ravel(), fmap.ravel())
    roc = roc_auc_score(ref.ravel(), fmap.ravel())

    print(f"Loaded Model Tuned results:, {bests}")
    print(f"Loaded Model PR-AUC: {pr_auc}, ROC: {roc}")

    elapsed=time.time()-start
    print(f"--- Time elapsed: {elapsed:.1f}s. ---")

    print("\n--- All inference runs complete. Generating comparison image. ---")

    # num_ov_steps = len(final_maps)
    # fig, axes = plt.subplots(nrows=2, ncols=1 + num_ov_steps, figsize=(30, 6), constrained_layout=True)

    # # Plot Ground Truth in the first column
    # axes[0, 0].imshow(ref.reshape(H, W), cmap="gray")
    # axes[0, 0].set_title("Ground Truth")
    # axes[0, 0].axis('off')
    # axes[1, 0].axis('off')

    # # Iterate through overlap values and plot the continuous and binary maps
    # for i, ov in enumerate(sorted(final_maps.keys())):
    #     # Plot the continuous map in the first row, starting from the second column
    #     im_continuous = axes[0, i + 1].imshow(final_maps[ov].reshape(H,W), cmap="gray")
    #     axes[0, i + 1].set_title(f"Continuous Map (overlap={ov})")
    #     axes[0, i + 1].axis('off')
    #     fig.colorbar(im_continuous, ax=axes[0, i + 1], fraction=0.046, pad=0.04)

    #     # Plot the binary map in the second row, starting from the second column
    #     im_binary = axes[1, i + 1].imshow(final_binary_maps[ov].reshape(H,W), cmap="gray")
    #     axes[1, i + 1].set_title(f"Binary Map (overlap={ov})")
    #     axes[1, i + 1].axis('off')
    #     fig.colorbar(im_binary, ax=axes[1, i + 1], fraction=0.046, pad=0.04)

    # plt.tight_layout()
    # if method == "mean":
    #     out_image = "/home/ubuntu/aditya/BioSky/Results_Inference/Mean_overlap_comparison.png"
    # else:
    #     out_image = "/home/ubuntu/aditya/BioSky/Results_Inference/Weighted_overlap_comparison.png"

    # plt.savefig(out_image, dpi=300)
    # print(f"Comparison image saved as {method}_overlap_comparison.png")
    # plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)

    axes[0].imshow(ref.reshape(H, W), cmap="gray")
    axes[0].set_title("Ground Truth Mask")

    im1 = axes[1].imshow(fmap, cmap="gray")
    axes[1].set_title("Anomaly Detection Map (Loaded k=5)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(binary_map, cmap="gray")
    axes[2].set_title("Binary Anomaly Detection Map (Loaded k=5)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.savefig(f"mock1_{method}_inference", dpi=300)
    plt.close(fig)

    print(f"Inference results saved to mock1_{method}_inference.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='weighted', choices=['mean', 'weighted'])
    args = parser.parse_args()
    # print("What to do? 1.Train 2.Test")
    # choice=input()
    
    # mat=sio.loadmat('/home/ubuntu/aditya/BioSky/Datasets/Sandiego/San_Diego.mat')
    # mat=sio.loadmat('/home/ubuntu/aditya/BioSky/Datasets/HYDICE-urban/HYDICE_urban.mat')
    # mat=sio.loadmat('/home/ubuntu/aditya/BioSky/Datasets/Salinas/Salinas.mat')
    mat=sio.loadmat("/home/ubuntu/aditya/BioSky/Datasets/MOCK/prisma_data.mat")
    # mat=sio.loadmat("/home/ubuntu/aditya/BioSky/Datasets/MOCK_2/prisma_data.mat")
    hsi = np.array(mat['data'], dtype=np.float32)
    print("HSI shape:", hsi.shape)
    hsi = (hsi - hsi.min()) / (np.ptp(hsi) + 1e-8)  # normalize to [0,1] for stability

    # gt_mat = sio.loadmat("/home/ubuntu/aditya/BioSky/Datasets/Sandiego/San_Diego.mat")  # file with ground truth
    # gt_mat=sio.loadmat('/home/ubuntu/aditya/BioSky/Datasets/HYDICE-urban/HYDICE_urban.mat')
    # gt_mat=sio.loadmat("/home/ubuntu/aditya/BioSky/Datasets/Salinas/Salinas_gt.mat")
    gt_mat=sio.loadmat("/home/ubuntu/aditya/BioSky/Datasets/MOCK/prisma_gt.mat")
    # gt_mat=sio.loadmat("/home/ubuntu/aditya/BioSky/Datasets/MOCK_2/prisma_gt.mat")
    ref = gt_mat['data']  # shape (H,W), binary 0/1
    # ref = ref.astype(np.uint8).reshape(-1)
    print("Ground truth shape:", ref.shape)

    window=(120,120,239)

    # if choice=='1':
    # set_seeds(42)
    # overlap=50 # in %
    # train(hsi,ref,window,overlap)
    # else:
    set_seeds(42)
    inference(hsi,ref,window,method=args.method)