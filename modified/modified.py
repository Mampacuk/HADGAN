import math
import time
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.covariance import EmpiricalCovariance
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import random
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping

from RGAE.SuperGraph import supergraph
from utils.adaptive_thresh import tune_methods

# ! Trying to use supergraph + hadgan approach
# ! Will try adding L21 norm in reconstruction and consistency losses
# ! Can use supergraph in postprocessing in spatial detector
# ! Use the adaptive weighted thresholding loss function

tf.config.optimizer.set_jit(False)

bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)
l2_reg = tf.keras.regularizers.L2(1e-5) # !

# ------------------------------
# Utility / Model components
# ------------------------------
def compute_dz(B):
    return int(math.sqrt(B) + 1)

class HADGAN(tf.keras.Model):
    def __init__(self, enc, dec, dznet, dinet):
    # def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def call(self, x, training=False):
        # only forward through encoder+decoder for inference
        z = self.enc(x, training=training)
        xrec = self.dec(z, training=training)
        return xrec

class FCBlock(tf.keras.Model):
    def __init__(self, out_dim, batchnorm=False, activation=True, lrelu_slope=0.2,kernel_regularizer=l2_reg):
        super().__init__()
        self.lrelu_slope=lrelu_slope
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

class Encoder(tf.keras.Model):
    def __init__(self, d1=1000, d2=1000, dz=None, dropout_latent=1.0): # !
        super().__init__()
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

class Decoder(tf.keras.Model):
    def __init__(self, d1=1000, d2=1000, B=0):
        super().__init__()
        self.fc1 = FCBlock(d1, batchnorm=False,kernel_regularizer=l2_reg)
        self.fc2 = FCBlock(d2, batchnorm=False,kernel_regularizer=l2_reg)
        self.out = Dense(B, use_bias=False, kernel_regularizer=l2_reg)  # bias kept 0 per paper
    def call(self, z):
        h = self.fc1(z)
        h = self.fc2(h)
        xrec = self.out(h)
        return xrec

class LatentDiscriminator(tf.keras.Model):
    def __init__(self, d1=1000, d2=1000):
        super().__init__()
        self.net = Sequential([
            Dense(d1),
            LeakyReLU(negative_slope=0.2),
            Dense(d2),
            LeakyReLU(negative_slope=0.2),
            Dense(1)
        ])
    def call(self, z,training=False):
        return tf.squeeze(self.net(z,training=training),axis=-1)

class ImageDiscriminator(tf.keras.Model):
    def __init__(self, d1=1000, d2=1000):
        super().__init__()
        self.net = Sequential([
            Dense(d1),
            LeakyReLU(negative_slope=0.2),
            Dense(d2),
            LeakyReLU(negative_slope=0.2),
            Dense(1)
        ])
    def call(self, x,training=False):
        return tf.squeeze(self.net(x,training=training),axis=-1)

# ------------------------------
# Loss helpers
# ------------------------------
def reconstruction_loss(x, xrec): # ! added L21 norm
    batch_size = tf.shape(x)[0]
    error_matrix=(x-xrec) # (batch_size,B)
    res=tf.math.sqrt(tf.reduce_sum(tf.square(error_matrix),axis=1)+1e-8) # (batch_size,)
    rec_loss=tf.reduce_sum(res)
    return (1/(2*tf.cast(batch_size, tf.float32)))*rec_loss
    # return tf.reduce_mean((x - xrec) ** 2)

def consistency_loss(z, encoder, decoder): # ! added L21 norm
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

def fuse_spatial_spectral(dspatial, dspectral, lam=0+.5):
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
                 S=150, # !
                 lambda_=1e-2 # !
                 ):
    """
    hsi_array: numpy array (H, W, B) with float values (recommended normalized per band)
    returns trained models and final detection map
    """

    ## Preprocessing
    H,W,B = hsi_array.shape
    dz = compute_dz(B)
    X = hsi_array.reshape(-1, B).astype(np.float32)
    N=X.shape[0]
    # dataset = tf.data.Dataset.from_tensor_slices(X).batch(batch_size)
    # loader = dataset.batch(batch_size=batch_size, drop_remainder=False)  # paper uses whole-image batch MxN

    # models
    enc = Encoder(dz=dz, dropout_latent=0.5)
    dec = Decoder(B=B)
    # dznet = LatentDiscriminator()
    # dinet = ImageDiscriminator()

    # optimizers
    opt_enc = tf.keras.optimizers.RMSprop(learning_rate=lr_enc)
    opt_dec = tf.keras.optimizers.Adam(learning_rate=lr_others)
    # opt_dz = tf.keras.optimizers.Adam(learning_rate=lr_others)
    # opt_di = tf.keras.optimizers.Adam(learning_rate=lr_others)

    # Early stopping parameters
    patience = 50  # Number of epochs to wait for improvement
    min_delta = 0.0001  # Minimum change to be considered an improvement
    best_loss = float('inf')
    epochs_no_improve = 0
    # best_weights=None

    # ! build supergraph
    SG,_,_= supergraph(hsi_array,S) # SG is the graph Laplacian of size (H*W, H*W)
    X=tf.convert_to_tensor(X,dtype=tf.float32)
    SG=tf.convert_to_tensor(SG,dtype=tf.float32)

    # !
    batch_size = min(batch_size, N)
    batch_num = int(np.ceil(N / batch_size))

    # training
    start = time.time()
    for ep in range(epochs):
        total_loss_epoch = 0 # Variable to accumulate loss over the epoch
        batch_count = 0
        idx=np.random.permutation(N) # shuffle the data
        for j in range(batch_num):
            # x = xbatch # (Npix, B)
            batch_idx=idx[j*batch_size : min((j+1)*batch_size, N)]
            x = tf.gather(X, batch_idx) # get the batch data
            Npix = x.shape[0]
            
            L=tf.gather(tf.gather(SG, batch_idx, axis=0), batch_idx, axis=1) # get the corresponding graph Laplacian

            # ===== 1) Update latent discriminator DZ =====
            # with tf.GradientTape() as tape:
            #     z_fake = tf.stop_gradient(enc(x,training=False))
            #     z_real = tf.random.normal(z_fake.shape) # sample from N(0,I)
            #     logits_real = dznet(z_real,training=True)
            #     logits_fake = dznet(z_fake,training=True)
            #     labels_real = tf.ones_like(logits_real)
            #     labels_fake = tf.zeros_like(logits_fake)
            #     loss_dz = bce_logits(labels_real, logits_real) + bce_logits(labels_fake, logits_fake)
            # grads=tape.gradient(loss_dz, dznet.trainable_variables)
            # opt_dz.apply_gradients(zip(grads, dznet.trainable_variables))

            # ===== 2) Update image discriminator DI =====
            # with tf.GradientTape() as tape:
            #     z_e = enc(x,training=False)
            #     xrec_detached = tf.stop_gradient(dec(z_e,training=False))
            #     logits_real = dinet(x,training=True)
            #     logits_fake = dinet(xrec_detached,training=True)
            #     labels_real = tf.ones_like(logits_real)
            #     labels_fake = tf.zeros_like(logits_fake)
            #     loss_di = bce_logits(labels_real, logits_real) + bce_logits(labels_fake, logits_fake)
            # grads=tape.gradient(loss_di, dinet.trainable_variables)
            # opt_di.apply_gradients(zip(grads, dinet.trainable_variables))

            # ===== 3) Update encoder+decoder (AE) with L = alpha0*LR + alpha1*Lziz + alpha2*Lzl1
            with tf.GradientTape(persistent=True) as tape:
                z = enc(x,training=True)
                xrec = dec(z,training=True)
                LR = reconstruction_loss(x, xrec)
                Lziz = consistency_loss(z, enc, dec)
                Lzl1 = shrink_loss(z)
                loss_ae = alpha0 * LR + alpha1 * Lziz + alpha2 * Lzl1

                # plus adversarial losses to fool discriminators (enc -> latent, dec -> image)
                # encoder adversarial: make dznet(enc(x)) be labeled as real (1)
                # logits_enc_adv = dznet(z,training=True)
                # loss_enc_adv = bce_logits(logits_enc_adv, tf.ones_like(logits_enc_adv))
                # # decoder adversarial: make dinet(dec(z)) be labeled as real (1)
                # logits_dec_adv = dinet(xrec,training=True)
                # loss_dec_adv = bce_logits(logits_dec_adv, tf.ones_like(logits_dec_adv))

                # graph regularization loss # !
                graph_loss = tf.linalg.trace(tf.matmul(tf.matmul(tf.transpose(z), L), z)) 
                loss_ae += (lambda_/tf.cast(tf.shape(x)[0], tf.float32))*graph_loss

                # combine losses
                # enc_loss = loss_ae + 1.0 * loss_enc_adv  # 1.0 is a chosen weight for adversarial term
                # dec_loss = loss_ae + 1.0 * loss_dec_adv
                enc_loss=loss_ae
                dec_loss=loss_ae


            # combine: update encoder with AE loss + encoder adversarial; update decoder with AE loss + decoder adversarial
            # We update both with combined loss but split optimizer steps
            grads_enc = tape.gradient(enc_loss, enc.trainable_variables)
            grads_dec = tape.gradient(dec_loss, dec.trainable_variables)
            opt_enc.apply_gradients(zip(grads_enc, enc.trainable_variables))
            opt_dec.apply_gradients(zip(grads_dec, dec.trainable_variables))

            # total_loss_epoch += (loss_ae + loss_enc_adv + loss_dec_adv).numpy()
            total_loss_epoch+=(loss_ae).numpy()
            batch_count += 1

            del tape  # free memory

        # Average loss over epoch
        avg_loss_epoch = total_loss_epoch / batch_count

        # Check for improvement
        # Check for improvement
        if avg_loss_epoch < best_loss - min_delta:
            best_loss = avg_loss_epoch
            epochs_no_improve = 0
            # best_weights = [enc.get_weights(), dec.get_weights(), dznet.get_weights(), dinet.get_weights()]
        else:
            epochs_no_improve += 1

        # Check if patience is exceeded
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {ep+1} epochs. No improvement for {patience} epochs.')
            break

        # end epoch
        if (ep+1) % 50 == 0 or ep == 0:
            elapsed = time.time() - start
            # print(f"Epoch {ep+1}/{epochs} LR={LR.numpy():.6f} Lziz={Lziz.numpy():.6f} Lzl1={Lzl1.numpy():.6f} adv_enc={loss_enc_adv.numpy():.6f} adv_dec={loss_dec_adv.numpy():.6f} time={elapsed:.1f}s")
            print(f"Epoch {ep+1}/{epochs} LR={LR.numpy():.6f} Lziz={Lziz.numpy():.6f} Lzl1={Lzl1.numpy():.6f} time={elapsed:.1f}s")

    # After training, load the best weights
    # if best_weights:
    #     enc.set_weights(best_weights[0])
    #     dec.set_weights(best_weights[1])
        # dznet.set_weights(best_weights[2])
        # dinet.set_weights(best_weights[3])

    # hadgan = HADGAN(enc, dec, dznet, dinet)
    # hadgan.save("hadganrgae_hu.keras", include_optimizer=True) # !

    # ===== inference =====
    Xtensor=tf.convert_to_tensor(X,dtype=tf.float32)
    Z = enc(Xtensor,training=False)
    Xrec = dec(Z,training=False).numpy()
    recon = Xrec.reshape(H, W, B)
    residual = compute_residual_map(hsi_array, recon)

    # spatial detector
    bands = select_min_energy_bands(residual, k=4)
    dspatial = spatial_detector_from_residual(residual, bands)
    # spectral detector
    dspectral = spectral_detector_from_residual(residual)
    # fuse
    final_map = fuse_spatial_spectral(dspatial, dspectral, lam=0.5)

    # return {
    #     'encoder': enc, 'decoder': dec, 'dznet': dznet, 'dinet': dinet,
    #     'recon': recon, 'residual': residual,
    #     'spatial_map': dspatial, 'spectral_map': dspectral, 'final_map': final_map
    # }
    return {
        'encoder': enc, 'decoder': dec,
        'recon': recon, 'residual': residual,
        'spatial_map': dspatial, 'spectral_map': dspectral, 'final_map': final_map
    }

# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    set_seeds(42)  # for reproducibility

    print(f"Dataset name: San_Deigo")
    print(f"Model: HADGAN")
    print(f"HADGAN + SuperGraph + L21 - DN")

    # mat=sio.loadmat('/home/ubuntu/aditya/BioSky/Datasets/Sandiego/San_Diego.mat')
    # mat=sio.loadmat('/home/ubuntu/aditya/BioSky/Datasets/HYDICE-urban/HYDICE_urban.mat')
    mat=sio.loadmat('/home/ubuntu/aditya/BioSky/Datasets/Salinas/Salinas.mat')
    hsi = np.array(mat['hsi'], dtype=float)
    print("HSI shape:", hsi.shape)

    hsi = (hsi - hsi.min()) / (np.ptp(hsi) + 1e-8)  # normalize to [0,1] for stability
    # plt.imshow(hsi[:, :, 30], cmap='gray')
    # plt.title("Band 30")
    # plt.colorbar()
    # plt.savefig("image.png", dpi=300)
    # plt.close()

    epochs=300
    batch_size=10000
    dropout=0.5
    out = train_hadgan(hsi, epochs=epochs,batch_size=batch_size, dropout=dropout)

    print(f"Number of epochs: {epochs}, batch size: {batch_size}, dropout: {dropout}")

    # # # final detection map:
    fmap = out['final_map']
    print("Final map shape:", fmap.shape)

    # # Save detection map with colorbar
    # # plt.figure(figsize=(6, 6))
    # # im = plt.imshow(fmap, cmap="jet")
    # # plt.colorbar(im, fraction=0.046, pad=0.04)
    # # plt.title("Anomaly Detection Map")
    # # plt.savefig("final_map_with_colorbar.png", dpi=300)
    # # plt.close()
    # # print("Saved detection map as final_map_with_colorbar.png")

    # gt_mat = sio.loadmat("/home/ubuntu/aditya/BioSky/Datasets/Sandiego/San_Diego.mat")  # file with ground truth
    # gt_mat=sio.loadmat('/home/ubuntu/aditya/BioSky/Datasets/HYDICE-urban/HYDICE_urban.mat')
    gt_mat=sio.loadmat("/home/ubuntu/aditya/BioSky/Datasets/Salinas/Salinas_gt.mat")
    ref = gt_mat['hsi_gt']  # shape (H,W), binary 0/1
    ref = ref.astype(np.uint8).reshape(-1)
    print("Ground truth shape:", ref.shape)

    bests, pr_auc, roc=tune_methods(fmap, ref.reshape(hsi.shape[0],hsi.shape[1]))
    print(f"Tuned results:, {bests}")
    print(f"pr_auc: {pr_auc}, roc: {roc}")

    # threshold=bests['iterative_f1'][1]
    # binary_map=(fmap>threshold).astype(np.uint8)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(ref.reshape(hsi.shape[0], hsi.shape[1]), cmap="gray")
    plt.title("Ground Truth Mask")

    plt.subplot(1,2,2)
    plt.imshow(fmap, cmap="gray")
    plt.title("Anomaly Detection Map")
    plt.colorbar()

    plt.savefig("modified_sal.png", dpi=300) # !
    plt.close()