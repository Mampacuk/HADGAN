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

from utils.adaptive_thresh import tune_methods,iterative_f1_threshold
from utils.sliding_window_inference import mean_inference, weighted_inference

tf.config.optimizer.set_jit(False)
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
            "kernel_regularizer": self.kernel_regularizer,
        })
        return config

    @classmethod
    def from_config(cls, config):
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
    error_matrix=(x-xrec) # (batch_size,B)
    res=tf.math.sqrt(tf.reduce_sum(tf.square(error_matrix),axis=1)+1e-8) # (batch_size,)
    rec_loss=tf.reduce_sum(res)
    return (1/(2*tf.cast(batch_size, tf.float32)))*rec_loss
    # return tf.reduce_mean((x - xrec) ** 2)

def consistency_loss(z, encoder, decoder): 
    # L_ziz = (1/dz) * || z - E(De(z)) ||^2  (paper)
    # z_const = tf.stop_gradient(z)  # as paper describes mapping z ~ N(0,I) -> De(z) -> E(De(z)), but we compute gradient w.r.t E in AE update separately
    rec = encoder(decoder(z,training=True),training=True)
    batch_size = tf.shape(z)[0]
    error_matrix=(z-rec) # (batch_size,B)
    res=tf.math.sqrt(tf.reduce_sum(tf.square(error_matrix),axis=1)+1e-8) # (batch_size,)
    consis_loss=tf.reduce_sum(res)
    return (1/(2*tf.cast(batch_size, tf.float32)))*consis_loss
    # return tf.reduce_mean((z - rec) ** 2)

def shrink_loss(z):
    # L_Zl1 = (1/dz) * ||z||^2
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
def train_hadgan(hsi_array,
                 epochs=300,
                 alpha0=1.0,
                 alpha1=1.0,
                 alpha2=0.1,
                 lr_enc=1e-4,  # paper uses RMSprop for encoder; we use RMS for encoder
                 lr_others=1e-4,
                 batch_size=32,
                 dropout=0.5,
                 k_steps=1
                 ):
    """
    hsi_array: numpy array (H, W, B) with float values (recommended normalized per band)
    returns trained models and final detection map
    """

    ## Preprocessing
    H,W,B = hsi_array.shape
    dz = compute_dz(B)
    X = hsi_array.reshape(-1, B).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices(X).batch(batch_size)
    # dataset_iter=iter(tf.data.Dataset.from_tensor_slices(X).batch(batch_size).repeat())

    # models
    enc = Encoder(dz=dz, dropout_latent=dropout)
    dec = Decoder(B=B)
    dznet = LatentDiscriminator()
    dinet = ImageDiscriminator()

    # optimizers
    opt_enc = tf.keras.optimizers.RMSprop(learning_rate=lr_enc)
    opt_dec = tf.keras.optimizers.Adam(learning_rate=lr_others)
    opt_dz = tf.keras.optimizers.Adam(learning_rate=lr_others)
    opt_di = tf.keras.optimizers.Adam(learning_rate=lr_others)

    # Early stopping parameters
    patience = 50  # Number of epochs to wait for improvement
    min_delta = 0.0001  # Minimum change to be considered an improvement
    best_loss = float('inf')
    epochs_no_improve = 0
    best_weights=None

    # training
    start = time.time()
    num_batches=math.ceil(X.shape[0]/batch_size)

    for ep in range(epochs):
        total_loss_epoch = 0.0
        
        for xbatch in dataset:
            x=xbatch

            # 1. Update Discriminators (k_steps times)
            for _ in range(k_steps):
                with tf.GradientTape() as tape:
                    z_fake = enc(x, training=False)
                    z_real = tf.random.normal(tf.shape(z_fake))
                    logits_real = dznet(z_real, training=True)
                    logits_fake = dznet(z_fake, training=True)
                    loss_dz = bce_logits(tf.ones_like(logits_real), logits_real) + \
                              bce_logits(tf.zeros_like(logits_fake), logits_fake)
                grads = tape.gradient(loss_dz, dznet.trainable_variables)
                opt_dz.apply_gradients(zip(grads, dznet.trainable_variables))
                
                with tf.GradientTape() as tape:
                    z_e = enc(x, training=False)
                    xrec_detached = dec(z_e, training=False)
                    logits_real = dinet(x, training=True)
                    logits_fake = dinet(xrec_detached, training=True)
                    loss_di = bce_logits(tf.ones_like(logits_real), logits_real) + \
                              bce_logits(tf.zeros_like(logits_fake), logits_fake)
                grads = tape.gradient(loss_di, dinet.trainable_variables)
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
            
            grads_enc = tape.gradient(enc_loss, enc.trainable_variables)
            grads_dec = tape.gradient(dec_loss, dec.trainable_variables)
            opt_enc.apply_gradients(zip(grads_enc, enc.trainable_variables))
            opt_dec.apply_gradients(zip(grads_dec, dec.trainable_variables))
            
            total_loss_epoch += loss_ae.numpy()
            del tape

        # Average loss over epoch
        avg_loss_epoch = total_loss_epoch / int(num_batches)

        # Check for improvement
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
        if (ep+1) % 50 == 0 or ep == 0:
            elapsed = time.time() - start
            print(f"Epoch {ep+1}/{epochs} LR={LR.numpy():.6f} Lziz={Lziz.numpy():.6f} Lzl1={Lzl1.numpy():.6f} adv_enc={loss_enc_adv.numpy():.6f} adv_dec={loss_dec_adv.numpy():.6f} time={elapsed:.1f}s")

    # After training, load the best weights
    if best_weights:
        enc.set_weights(best_weights[0])
        dec.set_weights(best_weights[1])
        dznet.set_weights(best_weights[2])
        dinet.set_weights(best_weights[3])

    # enc.save_weights(f"/home/ubuntu/aditya/BioSky/Results_train/HADGAN_kstep_l21/sal/enc_sal_k{k_steps}_l21.weights.h5")
    # dec.save_weights(f"/home/ubuntu/aditya/BioSky/Results_train/HADGAN_kstep_l21/sal/dec_sal_k{k_steps}_l21.weights.h5")

    # Force build with dummy input (matching your input dimensionality)
    dummy_input = tf.zeros((1, B), dtype=tf.float32)  
    hadgan = HADGAN(enc, dec, dznet, dinet)
    _ = hadgan(dummy_input, training=False)  # run once to build

    hadgan.save(f"/home/ubuntu/aditya/BioSky/Results_train/HADGAN_kstep_l21/sal/hadgan_sal_k{k_steps}_l21.keras", include_optimizer=True)

    print(f"Models saved to hadgan_sal_k{k_steps}_l21.keras")

    # ===== inference =====
    Xtensor=tf.convert_to_tensor(X,dtype=tf.float32)
    Z = enc(Xtensor,training=False)
    Xrec = dec(Z,training=False).numpy()
    recon = Xrec.reshape(H, W, B)
    residual = compute_residual_map(hsi_array, recon)

    # spatial detector
    bands = select_min_energy_bands(residual, k=3)
    dspatial = spatial_detector_from_residual(residual, bands)
    # spectral detector
    dspectral = spectral_detector_from_residual(residual)
    # fuse
    final_map = fuse_spatial_spectral(dspatial, dspectral, lam=0.5)

    return {
        'encoder': enc, 'decoder': dec, 'dznet': dznet, 'dinet': dinet,
        'recon': recon, 'residual': residual,
        'spatial_map': dspatial, 'spectral_map': dspectral, 'final_map': final_map
    }

def train(hsi,ref):
    print(f"Dataset name: Salinas")
    print(f"Model: HADGAN")

    H=hsi.shape[0]
    W=hsi.shape[1]

    epochs=500
    batch_size=1000
    dropout=0.5

    # List to store the final maps for different k_steps
    final_maps = {}
    final_binary_maps={}

    # Run training for k=1 to k=5
    for k in range(1, 6):
        print(f"\n--- Starting training with k_steps = {k} ---")
        out = train_hadgan(hsi, epochs=epochs, batch_size=batch_size, dropout=dropout, k_steps=k)
        final_maps[k] = out['final_map']
        bests, pr_auc, roc=tune_methods(final_maps[k], ref.reshape(hsi.shape[0],hsi.shape[1]))

        threshold=bests['iterative_f1'][1]
        binary_map=(final_maps[k]>threshold).astype(np.uint8)
        final_binary_maps[k]=binary_map

        print(f"Tuned results:, {bests}")
        print(f"pr_auc: {pr_auc}, roc: {roc}")

    print("\n--- All training runs complete. Generating comparison image. ---")
    
    num_k_steps = len(final_maps)
    fig, axes = plt.subplots(nrows=2, ncols=1 + num_k_steps, figsize=(20, 5))

    # Plot Ground Truth in the first column
    axes[0, 0].imshow(ref.reshape(H, W), cmap="gray")
    axes[0, 0].set_title("Ground Truth")
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')

    # Iterate through k values and plot the continuous and binary maps
    for i, k in enumerate(sorted(final_maps.keys())):
        # Plot the continuous map in the first row, starting from the second column
        im_continuous = axes[0, i + 1].imshow(final_maps[k].reshape(H,W), cmap="gray")
        axes[0, i + 1].set_title(f"Continuous Map (k={k})")
        axes[0, i + 1].axis('off')
        fig.colorbar(im_continuous, ax=axes[0, i + 1], fraction=0.046, pad=0.04)

        # Plot the binary map in the second row, starting from the second column
        im_binary = axes[1, i + 1].imshow(final_binary_maps[k].reshape(H,W), cmap="gray")
        axes[1, i + 1].set_title(f"Binary Map (k={k})")
        axes[1, i + 1].axis('off')
        fig.colorbar(im_binary, ax=axes[1, i + 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("/home/ubuntu/aditya/BioSky/Results_train/HADGAN_kstep_l21/sal/hadgan_sal_ksteps_l21_comparison.png", dpi=300)
    plt.close()

    print("Comparison image saved as hadgan_sal_ksteps_l21_comparison.png")

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
def inference(hsi,ref,Hk=120,Wk=120,method="mean"):
    # Assuming you want to load the model for k=5 and L21 loss
    MODEL_PATH = "/home/ubuntu/aditya/BioSky/Results_train/HADGAN_kstep_l21/sal/hadgan_sal_k5_l21.keras"
    H, W, B = hsi.shape

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
    Bk=204 # !

    overlap=[5,20,35,50,65,80,95,99] # 99% overlap means 1% stride

    final_maps = {}
    final_binary_maps={}

    for ov in overlap:
        start = time.time()
        if method=="mean":
            Xrec=mean_inference(hsi,enc,dec,Hk,Wk,Bk,ov,batch_size)
        else:
            Xrec=weighted_inference(hsi,enc,dec,Hk,Wk,Bk,ov,batch_size)

        residual = compute_residual_map(hsi, Xrec)

        # --- Run Post-processing Detectors ---
        bands = select_min_energy_bands(residual, k=3)
        dspatial = spatial_detector_from_residual(residual, bands)
        dspectral = spectral_detector_from_residual(residual)
        fmap = fuse_spatial_spectral(dspatial, dspectral, lam=0.5)

        final_maps[ov]=fmap

        # --- Evaluate and Visualize ---
        print(f"\n--- Inference completed for overlap parameters = {ov}%. Evaluating results. ---")
        # bests, pr_auc, roc = tune_methods(fmap, ref.reshape(H, W))
        threshold,iterative_f1=iterative_f1_threshold(ref.ravel(), fmap.ravel())
        bests=(iterative_f1, threshold, None, None)

        binary_map=(fmap>threshold).astype(np.uint8)

        final_binary_maps[ov]=binary_map

        # compute continuous-score metrics once for S
        pr_auc = average_precision_score(ref.ravel(), fmap.ravel())
        roc = roc_auc_score(ref.ravel(), fmap.ravel())

        print(f"Loaded Model Tuned results:, {bests}")
        print(f"Loaded Model PR-AUC: {pr_auc}, ROC: {roc}")

        elapsed=time.time()-start
        print(f"--- Time elapsed: {elapsed:.1f}s. ---")

    print("\n--- All inference runs complete. Generating comparison image. ---")

    num_ov_steps = len(final_maps)
    fig, axes = plt.subplots(nrows=2, ncols=1 + num_ov_steps, figsize=(30, 6), constrained_layout=True)

    # Plot Ground Truth in the first column
    axes[0, 0].imshow(ref.reshape(H, W), cmap="gray")
    axes[0, 0].set_title("Ground Truth")
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')

    # Iterate through overlap values and plot the continuous and binary maps
    for i, ov in enumerate(sorted(final_maps.keys())):
        # Plot the continuous map in the first row, starting from the second column
        im_continuous = axes[0, i + 1].imshow(final_maps[ov].reshape(H,W), cmap="gray")
        axes[0, i + 1].set_title(f"Continuous Map (overlap={ov})")
        axes[0, i + 1].axis('off')
        fig.colorbar(im_continuous, ax=axes[0, i + 1], fraction=0.046, pad=0.04)

        # Plot the binary map in the second row, starting from the second column
        im_binary = axes[1, i + 1].imshow(final_binary_maps[ov].reshape(H,W), cmap="gray")
        axes[1, i + 1].set_title(f"Binary Map (overlap={ov})")
        axes[1, i + 1].axis('off')
        fig.colorbar(im_binary, ax=axes[1, i + 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if method == "mean":
        out_image = "/home/ubuntu/aditya/BioSky/Results_Inference/Mean_overlap_comparison.png"
    else:
        out_image = "/home/ubuntu/aditya/BioSky/Results_Inference/Weighted_overlap_comparison.png"

    plt.savefig(out_image, dpi=300)
    print(f"Comparison image saved as {method}_overlap_comparison.png")
    plt.close()

    print(f"Comparison image saved as {method}_overlap_comparison.png")

    # fig, axes = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)

    # axes[0].imshow(ref.reshape(H, W), cmap="gray")
    # axes[0].set_title("Ground Truth Mask")

    # im1 = axes[1].imshow(fmap, cmap="gray")
    # axes[1].set_title("Anomaly Detection Map (Loaded k=5)")
    # fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # im2 = axes[2].imshow(binary_map, cmap="gray")
    # axes[2].set_title("Binary Anomaly Detection Map (Loaded k=5)")
    # fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # fig.savefig("output_mean.png", dpi=300)
    # plt.close(fig)

    # print("Inference results saved to output_mean.png")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--method', type=str, default='weighted', choices=['mean', 'weighted'])
    # args = parser.parse_args()
    print("What to do? 1.Train 2.Test")
    choice=input()
    
    # mat=sio.loadmat('/home/ubuntu/aditya/BioSky/Datasets/Sandiego/San_Diego.mat')
    # mat=sio.loadmat('/home/ubuntu/aditya/BioSky/Datasets/HYDICE-urban/HYDICE_urban.mat')
    # mat=sio.loadmat('/home/ubuntu/aditya/BioSky/Datasets/Salinas/Salinas.mat')
    mat=sio.loadmat("/home/ubuntu/aditya/BioSky/Datasets/MOCK/prisma_data.mat")
    hsi = np.array(mat['data'], dtype=float)
    print("HSI shape:", hsi.shape)
    hsi = (hsi - hsi.min()) / (np.ptp(hsi) + 1e-8)  # normalize to [0,1] for stability

    # gt_mat = sio.loadmat("/home/ubuntu/aditya/BioSky/Datasets/Sandiego/San_Diego.mat")  # file with ground truth
    # gt_mat=sio.loadmat('/home/ubuntu/aditya/BioSky/Datasets/HYDICE-urban/HYDICE_urban.mat')
    # gt_mat=sio.loadmat("/home/ubuntu/aditya/BioSky/Datasets/Salinas/Salinas_gt.mat")
    gt_mat=sio.loadmat("/home/ubuntu/aditya/BioSky/Datasets/MOCK/prisma_gt.mat")
    ref = gt_mat['data']  # shape (H,W), binary 0/1
    ref = ref.astype(np.uint8).reshape(-1)
    print("Ground truth shape:", ref.shape)

    if choice=='1':
        set_seeds(42)
        train(hsi,ref)
    else:
        set_seeds(42)
        inference(hsi,ref,method=args.method)