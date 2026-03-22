#!/usr/bin/env python3
"""
HADGAN Training Script
Trains on patch-based HSI from .mat files with proper checkpointing.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import math
import time
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.utils import serialize_keras_object, deserialize_keras_object
from tensorflow.keras import regularizers
from tqdm import trange
import scipy.io as sio
from pathlib import Path

# ===== IMPORTS FROM UTILS =====
from utils.augment_patch import augment_patch

# ===== CONFIG =====
# tf.config.optimizer.set_jit(True)
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
tf.keras.mixed_precision.set_global_policy('mixed_float16')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)
l2_reg = tf.keras.regularizers.L2(1e-5)

# ===== GPU CONFIG =====
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        # tf.config.set_logical_device_configuration(
        #     gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=18500)]
        # )
    print("✅ GPU configured successfully.")
else:
    print("⚠ No GPU found. Running on CPU.")

# ===== MODEL CLASSES =====
def compute_dz(B):
    return int(math.sqrt(B) + 1)

class FCBlock(tf.keras.Model):
    def __init__(self, out_dim, batchnorm=False, activation=True, lrelu_slope=0.2,
                 kernel_regularizer=l2_reg, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = out_dim
        self.batchnorm = batchnorm
        self.activation = activation
        self.lrelu_slope = lrelu_slope
        self.kernel_regularizer = kernel_regularizer
        self.fc = Dense(out_dim, use_bias=True, kernel_regularizer=kernel_regularizer)
        self.bn = BatchNormalization(epsilon=1e-5) if batchnorm else None
        self.act = LeakyReLU(alpha=lrelu_slope) if activation else None

    def call(self, x, training=False):
        x = self.fc(x)
        if self.bn is not None:
            x = self.bn(x, training=training)
        if self.act is not None:
            x = self.act(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_dim": self.out_dim,
            "batchnorm": self.batchnorm,
            "activation": self.activation,
            "lrelu_slope": self.lrelu_slope,
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer)
                if self.kernel_regularizer else None
        })
        return config

    @classmethod
    def from_config(cls, config):
        if config.get("kernel_regularizer"):
            config["kernel_regularizer"] = regularizers.deserialize(config["kernel_regularizer"])
        return cls(**config)

class Encoder(tf.keras.Model):
    def __init__(self, d1=1000, d2=1000, dz=None, dropout_latent=0.5, **kwargs):
        super().__init__(**kwargs)
        self.d1 = d1
        self.d2 = d2
        self.dz = dz
        self.dropout_latent = dropout_latent
        self.kernel_regularizer = l2_reg
        self.net = Sequential([
            FCBlock(d1, batchnorm=True, kernel_regularizer=l2_reg),
            FCBlock(d2, batchnorm=True, kernel_regularizer=l2_reg),
        ])
        self.latent = Dense(dz, use_bias=False, kernel_regularizer=l2_reg)
        self.dropout = Dropout(rate=dropout_latent) if dropout_latent > 0 else None

    def call(self, x, training=False):
        h = self.net(x, training=training)
        z = self.latent(h)
        if self.dropout is not None:
            z = self.dropout(z, training=training)
        return z

    def get_config(self):
        config = super().get_config()
        config.update({"d1": self.d1, "d2": self.d2, "dz": self.dz,
                       "dropout_latent": self.dropout_latent})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Decoder(tf.keras.Model):
    def __init__(self, d1=1000, d2=1000, B=0, **kwargs):
        super().__init__(**kwargs)
        self.d1 = d1
        self.d2 = d2
        self.B = B
        self.kernel_regularizer = l2_reg
        self.fc1 = FCBlock(d1, batchnorm=False, kernel_regularizer=l2_reg)
        self.fc2 = FCBlock(d2, batchnorm=False, kernel_regularizer=l2_reg)
        self.out = Dense(B, use_bias=False, kernel_regularizer=l2_reg)

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
    def __init__(self, d1=1000, d2=1000, **kwargs):
        super().__init__(**kwargs)
        self.d1 = d1
        self.d2 = d2
        self.net = Sequential([
            Dense(d1),
            LeakyReLU(alpha=0.2),
            Dense(d2),
            LeakyReLU(alpha=0.2),
            Dense(1, dtype='float32')
        ])

    def call(self, z, training=False):
        return tf.squeeze(self.net(z, training=training), axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"d1": self.d1, "d2": self.d2})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ImageDiscriminator(tf.keras.Model):
    def __init__(self, d1=1000, d2=1000, **kwargs):
        super().__init__(**kwargs)
        self.d1 = d1
        self.d2 = d2
        self.net = Sequential([
            Dense(d1),
            LeakyReLU(alpha=0.2),
            Dense(d2),
            LeakyReLU(alpha=0.2),
            Dense(1, dtype='float32')
        ])

    def call(self, x, training=False):
        return tf.squeeze(self.net(x, training=training), axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"d1": self.d1, "d2": self.d2})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ===== LOSS FUNCTIONS =====
def reconstruction_loss(x, xrec, lambda_sam=1.0):
    """L2,1 norm + Spectral Angle Mapper (SAM) hybrid loss"""
    batch_size = tf.shape(x)[0]
    x = tf.cast(x, tf.float32)
    xrec = tf.cast(xrec, tf.float32)

    # L2,1 loss (encourages compact pixel-wise errors)
    error_matrix = (x - xrec)
    res = tf.math.sqrt(tf.reduce_sum(tf.square(error_matrix), axis=1) + 1e-8)
    l21_loss = tf.reduce_mean(res)

    # SAM loss (preserves spectral consistency)
    dot = tf.reduce_sum(x * xrec, axis=1)
    norm_x = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1) + 1e-8)
    norm_rec = tf.sqrt(tf.reduce_sum(tf.square(xrec), axis=1) + 1e-8)
    cos_angle = dot / (norm_x * norm_rec + 1e-8)
    angle = tf.acos(tf.clip_by_value(cos_angle, -1.0, 1.0))
    sam_loss = tf.reduce_mean(angle)

    total_loss = l21_loss + lambda_sam * sam_loss
    return total_loss

def consistency_loss(z, encoder, decoder):
    """Cycle consistency: z → decode → re-encode → z"""
    rec = encoder(tf.stop_gradient(decoder(z, training=True)), training=True)
    batch_size = tf.shape(z)[0]
    z = tf.cast(z, tf.float32)
    rec = tf.cast(rec, tf.float32)
    error_matrix = (z - rec)
    res = tf.math.sqrt(tf.reduce_sum(tf.square(error_matrix), axis=1) + 1e-8)
    consis_loss = tf.reduce_sum(res)
    return (1 / (2 * tf.cast(batch_size, tf.float32))) * consis_loss

def shrink_loss(z):
    """Latent shrinkage: encourage sparse latent codes"""
    z = tf.cast(z, tf.float32)
    return tf.reduce_mean(z ** 2)

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ===== TRAINING STEP =====
@tf.function
def train_step(x, enc, dec, dznet, dinet, opt_enc, opt_dec, opt_dz, opt_di,
               alpha0=5.0, alpha1=1.0, alpha2=0.1, k_steps=4):
    """Single training iteration"""
    x = tf.cast(x, tf.float32)

    # 1. Update Discriminators (k_steps times)
    for _ in range(k_steps):
        # Latent discriminator
        with tf.GradientTape() as tape:
            z_fake = enc(x, training=False)
            z_real = tf.random.normal(tf.shape(z_fake), dtype=z_fake.dtype)
            logits_real = tf.cast(dznet(z_real, training=True), tf.float32)
            logits_fake = tf.cast(dznet(z_fake, training=True), tf.float32)
            loss_dz = tf.cast(
                bce_logits(tf.ones_like(logits_real), logits_real) +
                bce_logits(tf.zeros_like(logits_fake), logits_fake),
                tf.float32,
            )
            scaled_loss_dz = opt_dz.get_scaled_loss(loss_dz)
        scaled_grads = tape.gradient(scaled_loss_dz, dznet.trainable_variables)
        grads = opt_dz.get_unscaled_gradients(scaled_grads)
        opt_dz.apply_gradients(zip(grads, dznet.trainable_variables))

        # Image discriminator
        with tf.GradientTape() as tape:
            z_e = enc(x, training=False)
            xrec_detached = dec(z_e, training=False)
            logits_real = tf.cast(dinet(x, training=True), tf.float32)
            logits_fake = tf.cast(dinet(xrec_detached, training=True), tf.float32)
            loss_di = tf.cast(
                bce_logits(tf.ones_like(logits_real), logits_real) +
                bce_logits(tf.zeros_like(logits_fake), logits_fake),
                tf.float32,
            )
            scaled_loss_di = opt_di.get_scaled_loss(loss_di)
        scaled_grads = tape.gradient(scaled_loss_di, dinet.trainable_variables)
        grads = opt_di.get_unscaled_gradients(scaled_grads)
        opt_di.apply_gradients(zip(grads, dinet.trainable_variables))

    # 2. Update Generators (1 time)
    with tf.GradientTape(persistent=True) as tape:
        z = enc(x, training=True)
        xrec = dec(z, training=True)

        LR = tf.cast(reconstruction_loss(x, xrec, lambda_sam=1.0), tf.float32)
        Lziz = tf.cast(consistency_loss(z, enc, dec), tf.float32)
        Lzl1 = tf.cast(shrink_loss(z), tf.float32)
        loss_ae = (
            tf.cast(alpha0, tf.float32) * LR +
            tf.cast(alpha1, tf.float32) * Lziz +
            tf.cast(alpha2, tf.float32) * Lzl1
        )

        logits_enc_adv = tf.cast(dznet(z, training=True), tf.float32)
        loss_enc_adv = tf.cast(
            bce_logits(tf.ones_like(logits_enc_adv), logits_enc_adv),
            tf.float32,
        )

        logits_dec_adv = tf.cast(dinet(xrec, training=True), tf.float32)
        loss_dec_adv = tf.cast(
            bce_logits(tf.ones_like(logits_dec_adv), logits_dec_adv),
            tf.float32,
        )

        enc_loss = tf.cast(loss_ae + loss_enc_adv, tf.float32)
        dec_loss = tf.cast(loss_ae + loss_dec_adv, tf.float32)

        scaled_enc_loss = opt_enc.get_scaled_loss(enc_loss)
        scaled_dec_loss = opt_dec.get_scaled_loss(dec_loss)

    scaled_grads_enc = tape.gradient(scaled_enc_loss, enc.trainable_variables)
    grads_enc = opt_enc.get_unscaled_gradients(scaled_grads_enc)
    opt_enc.apply_gradients(zip(grads_enc, enc.trainable_variables))

    scaled_grads_dec = tape.gradient(scaled_dec_loss, dec.trainable_variables)
    grads_dec = opt_dec.get_unscaled_gradients(scaled_grads_dec)
    opt_dec.apply_gradients(zip(grads_dec, dec.trainable_variables))

    del tape
    return LR, Lziz, Lzl1, loss_enc_adv, loss_dec_adv

# ===== TRAINING FUNCTION =====
def train(hsi, output_dir, patch_size=120, epochs=300, overlap=50,
          dropout=0.5, k_steps=4, lr_enc=5e-5, lr_others=1e-4, alpha0=5.0, alpha1=1.0, alpha2=0.1, hsi_path=None):
    """Train HADGAN on HSI patches"""
    print(f"\n{'='*60}")
    print(f"HADGAN TRAINING")
    print(f"{'='*60}")
    print(f"HSI Shape: {hsi.shape}")
    print(f"Patch Size: {patch_size}x{patch_size}")
    print(f"Epochs: {epochs}, Overlap: {overlap}%")
    print(f"K-steps: {k_steps}, Dropout: {dropout}")
    print(f"{'='*60}\n")

    # Extract dataset name from HSI file path
    if hsi_path:
        hsi_filename = Path(hsi_path).stem
    else:
        hsi_filename = "unknown_dataset"

    os.makedirs(output_dir, exist_ok=True)

    H, W, B = hsi.shape
    Hk, Wk, Bk = patch_size, patch_size, B

    stride_h = max(1, (100 - overlap) * Hk // 100)
    stride_w = max(1, (100 - overlap) * Wk // 100)

    # Model initialization
    dz = compute_dz(Bk)
    enc = Encoder(dz=dz, dropout_latent=dropout)
    dec = Decoder(B=Bk)
    dznet = LatentDiscriminator()
    dinet = ImageDiscriminator()

    # Optimizers
    opt_enc = tf.keras.mixed_precision.LossScaleOptimizer(
        tf.keras.optimizers.RMSprop(learning_rate=lr_enc))
    opt_dec = tf.keras.mixed_precision.LossScaleOptimizer(
        tf.keras.optimizers.Adam(learning_rate=lr_others))
    opt_dz = tf.keras.mixed_precision.LossScaleOptimizer(
        tf.keras.optimizers.Adam(learning_rate=lr_others))
    opt_di = tf.keras.mixed_precision.LossScaleOptimizer(
        tf.keras.optimizers.Adam(learning_rate=lr_others))

    # Patch generator with filtering and augmentation
    def gen_patches():
        h_starts = list(range(0, H - Hk + 1, stride_h))
        w_starts = list(range(0, W - Wk + 1, stride_w))
        if h_starts[-1] != H - Hk:
            h_starts.append(H - Hk)
        if w_starts[-1] != W - Wk:
            w_starts.append(W - Wk)

        for i in h_starts:
            for j in w_starts:
                patch = hsi[i:i+Hk, j:j+Wk, :].astype(np.float32)
                yield patch

    dataset = tf.data.Dataset.from_generator(
        gen_patches,
        output_signature=tf.TensorSpec(shape=(Hk, Wk, Bk), dtype=tf.float32)
    )
    dataset = dataset.map(lambda x: augment_patch(x), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x: tf.reshape(x, [-1, Bk]), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Training loop
    for ep in trange(epochs, desc="Training HADGAN"):
        start = time.time()
        for X_patch in dataset:
            LR, Lziz, Lzl1, loss_enc_adv, loss_dec_adv = train_step(
                X_patch, enc, dec, dznet, dinet, opt_enc, opt_dec, opt_dz, opt_di,
                alpha0=alpha0, alpha1=alpha1, alpha2=alpha2, k_steps=k_steps
            )

        if (ep + 1) % 30 == 0 or ep == 0:
            elapsed = time.time() - start
            print(f"Epoch {ep+1}/{epochs} → "
                  f"LR={LR:.6f} Lziz={Lziz:.6f} Lzl1={Lzl1:.6f} "
                  f"adv_enc={loss_enc_adv:.6f} adv_dec={loss_dec_adv:.6f} "
                  f"time={elapsed:.1f}s")

    print("\n✅ Training complete. Saving model...")

    # Save checkpoint
    ckpt = tf.train.Checkpoint(enc=enc, dec=dec, dznet=dznet, dinet=dinet)
    ckpt_path = os.path.join(output_dir, f'hadgan_ckpt_{hsi_filename}')
    save_path = ckpt.save(ckpt_path)
    print(f"✅ Checkpoint saved: {ckpt_path}")

    return enc, dec, dznet, dinet, ckpt_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HADGAN Training Script")
    parser.add_argument('--hsi_path', type=str, required=True,
                        help='Path to HSI .mat file (key: "data")')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--patch_size', type=int, default=64, choices=[120, 64, 32, 16],
                        help='Patch size (HxW)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Training epochs')
    parser.add_argument('--overlap', type=int, default=50,
                        help='Patch overlap percentage (0-99)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate in latent space')
    parser.add_argument('--k_steps', type=int, default=4,
                        help='Discriminator steps per iteration')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--alpha0', type=float, default=5.0, help='alpha0')
    parser.add_argument('--alpha1', type=float, default=1.0, help='alpha1')
    parser.add_argument('--alpha2', type=float, default=0.1, help='alpha2')
    parser.add_argument('--lr_encoder', type=float, default=5e-5, help='learning rate for encoder')
    parser.add_argument('--lr_others', type=float, default=1e-4, help='learning rate for everything other than encoder')

    args = parser.parse_args()
    set_seeds(args.seed)

    # Load data
    print(f"Loading HSI from {args.hsi_path}...")
    hsi_mat = sio.loadmat(args.hsi_path)
    hsi = np.array(hsi_mat['data'], dtype=np.float32)
    hsi = (hsi - hsi.min()) / (np.ptp(hsi) + 1e-8)

    # Train
    train(hsi, args.output_dir, patch_size=args.patch_size,
          epochs=args.epochs, overlap=args.overlap, dropout=args.dropout,
          k_steps=args.k_steps, alpha0=args.alpha0, alpha1=args.alpha1, alpha2=args.alpha2,
          lr_enc=args.lr_encoder, lr_others=args.lr_others, hsi_path=args.hsi_path)