"""
HADGAN PyTorch implementation (usable baseline).
Based on: "Discriminative Reconstruction Constrained GAN for Hyperspectral Anomaly Detection"
"""

import math
import time
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.covariance import EmpiricalCovariance
from skimage.morphology import area_opening, area_closing
from skimage.filters import rank
from skimage import img_as_ubyte
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import scipy.io as sio

import tensorflow as tf
from tf.keras import layers

# ------------------------------
# Utility / Model components
# ------------------------------
def compute_dz(B):
    return int(math.sqrt(B) + 1)

class FCBlock(nn.Module):
    def __init__(self, in_dim, out_dim, batchnorm=False, activation=True, lrelu_slope=0.2):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=True)
        self.bn = nn.BatchNorm1d(out_dim) if batchnorm else None
        self.act = (lambda x: F.leaky_relu(x, negative_slope=lrelu_slope)) if activation else (lambda x: x)
    def forward(self, x):
        x = self.fc(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.act(x)

class Encoder(nn.Module):
    def __init__(self, B, d1=1000, d2=1000, dz=None, dropout_latent=0.0):
        super().__init__()
        self.net = nn.Sequential(
            FCBlock(B, d1, batchnorm=True),
            FCBlock(d1, d2, batchnorm=True),
        )
        self.latent = nn.Linear(d2, dz, bias=False)  # paper sets bias to 0 to avoid pixel bias effects
        self.dropout = nn.Dropout(p=dropout_latent) if dropout_latent>0 else None
    def forward(self, x):
        h = self.net(x)
        z = self.latent(h)
        if self.dropout is not None:
            z = self.dropout(z)
        return z

class Decoder(nn.Module):
    def __init__(self, dz, d1=1000, d2=1000, B=0):
        super().__init__()
        self.fc1 = FCBlock(dz, d1, batchnorm=False)
        self.fc2 = FCBlock(d1, d2, batchnorm=False)
        self.out = nn.Linear(d2, B, bias=False)  # bias kept 0 per paper
    def forward(self, z):
        h = self.fc1(z)
        h = self.fc2(h)
        xrec = self.out(h)
        return xrec

class LatentDiscriminator(nn.Module):
    def __init__(self, dz, d1=1000, d2=1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dz, d1),
            nn.LeakyReLU(0.2),
            nn.Linear(d1, d2),
            nn.LeakyReLU(0.2),
            nn.Linear(d2, 1)
        )
    def forward(self, z):
        return self.net(z).squeeze(-1)

class ImageDiscriminator(nn.Module):
    def __init__(self, B, d1=1000, d2=1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(B, d1),
            nn.LeakyReLU(0.2),
            nn.Linear(d1, d2),
            nn.LeakyReLU(0.2),
            nn.Linear(d2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ------------------------------
# Loss helpers
# ------------------------------
bce_logits = nn.BCEWithLogitsLoss(reduction='mean')

def reconstruction_loss(x, xrec):
    return torch.mean((x - xrec) ** 2)

def consistency_loss(z, encoder, decoder): 
    # L_ziz = (1/dz) * || z - E(De(z)) ||^2  (paper)
    z = z.detach()  # as paper describes mapping z ~ N(0,I) -> De(z) -> E(De(z)), but we compute gradient w.r.t E in AE update separately
    rec = encoder(decoder(z))
    return torch.mean((z - rec) ** 2)

def shrink_loss(z):
    # L_Zl1 = (1/dz) * ||z||^2
    return torch.mean(z**2)

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

def spatial_detector_from_residual(residual, selected_bands):
    # residual: (H,W,B)
    H,W,B = residual.shape
    # compute per-band images (grayscale)
    df_list = []
    for b in selected_bands:
        band = residual[:,:,b]
        # morphological area opening/closing to remove large background components
        # area_opening will remove components smaller than area threshold; but paper uses attribute opening/closing
        # We choose area thresholds small to preserve small anomalies (paper suggests preserve small area -> so close to 0)
        img = band
        # convert to 8bit for skimage rank filters
        band8 = img_as_ubyte((img - img.min()) / (np.ptp(img)+1e-8))
        # area opening/closing using scikit-image (removes small objects)
        opened = area_opening(band8, area_threshold=5)  # small threshold; tune per dataset
        closed = area_closing(band8, area_threshold=5)
        df = np.abs(band8 - opened).astype(np.float32) + np.abs(closed - band8).astype(np.float32)
        df_list.append(df)
    F = np.mean(np.stack(df_list, axis=0), axis=0)
    # simple guided filter refinement: try opencv ximgproc if available, else use bilateral
    try:
        gf = cv2.ximgproc.guidedFilter(guide=img_as_ubyte(F/255.0), src=img_as_ubyte(F/255.0), radius=3, eps=0.5)
        out = gf.astype(np.float32)
    except Exception:
        out = cv2.bilateralFilter(img_as_ubyte(F/255.0), d=5, sigmaColor=75, sigmaSpace=75)
    # linear coefficients mean step (paper), we simply normalize
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
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    hsi_array: numpy array (H, W, B) with float values (recommended normalized per band)
    returns trained models and final detection map
    """

    ## Preprocessing
    H,W,B = hsi_array.shape
    dz = compute_dz(B)
    X = hsi_array.reshape(-1, B).astype(np.float32)
    # convert to torch
    data = torch.from_numpy(X)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=X.shape[0], shuffle=False)  # paper uses whole-image batch MxN

    # models
    enc = Encoder(B=B, dz=dz, dropout_latent=0.0).to(device)
    dec = Decoder(dz=dz, B=B).to(device)
    dznet = LatentDiscriminator(dz=dz).to(device)
    dinet = ImageDiscriminator(B=B).to(device)

    # optimizers
    opt_enc = torch.optim.RMSprop(list(enc.parameters()), lr=lr_enc, weight_decay=1e-5)
    opt_dec = torch.optim.Adam(list(dec.parameters()), lr=lr_others, weight_decay=1e-5)
    opt_dz = torch.optim.Adam(dznet.parameters(), lr=lr_others)
    opt_di = torch.optim.Adam(dinet.parameters(), lr=lr_others)

    # training
    enc.train(); dec.train(); dznet.train(); dinet.train()
    start = time.time()
    torch.autograd.set_detect_anomaly(True)
    for ep in range(epochs):
        for (xbatch,) in loader:
            x = xbatch.to(device)  # (Npix, B)
            Npix = x.shape[0]

            # ===== 1) Update latent discriminator DZ =====
            z_fake = enc(x).detach()
            z_real = tf.random.normal(tf.shape(z_fake))  # sample from N(0,I)
            logits_real = dznet(z_real,training=True)
            logits_fake = dznet(z_fake,training=True)
            loss_dz = bce_logits(logits_real, torch.ones_like(logits_real)) + \
                      bce_logits(logits_fake, torch.zeros_like(logits_fake))
            opt_dz.zero_grad()
            loss_dz.backward()
            opt_dz.step()

            # ===== 2) Update image discriminator DI =====
            with torch.no_grad():
                z_e = enc(x)
                xrec_detached = dec(z_e).detach()
            logits_real_img = dinet(x)
            logits_fake_img = dinet(xrec_detached)
            loss_di = bce_logits(logits_real_img, torch.ones_like(logits_real_img)) + \
                      bce_logits(logits_fake_img, torch.zeros_like(logits_fake_img))
            opt_di.zero_grad()
            loss_di.backward()
            opt_di.step()

            # ===== 3) Update encoder+decoder (AE) with L = alpha0*LR + alpha1*Lziz + alpha2*Lzl1
            z = enc(x)
            xrec = dec(z)
            LR = reconstruction_loss(x, xrec)
            Lziz = consistency_loss(z, enc, dec)
            Lzl1 = shrink_loss(z)
            loss_ae = alpha0 * LR + alpha1 * Lziz + alpha2 * Lzl1

            # plus adversarial losses to fool discriminators (enc -> latent, dec -> image)
            # encoder adversarial: make dznet(enc(x)) be labeled as real (1)
            logits_enc_adv = dznet(z)
            loss_enc_adv = bce_logits(logits_enc_adv, torch.ones_like(logits_enc_adv))
            # decoder adversarial: make dinet(dec(z)) be labeled as real (1)
            logits_dec_adv = dinet(xrec)
            loss_dec_adv = bce_logits(logits_dec_adv, torch.ones_like(logits_dec_adv))

            # combine: update encoder with AE loss + encoder adversarial; update decoder with AE loss + decoder adversarial
            # We update both with combined loss but split optimizer steps
            # Encoder step
            opt_enc.zero_grad()
            opt_dec.zero_grad()

            enc_loss = loss_ae + 1.0 * loss_enc_adv  # 1.0 is a chosen weight for adversarial term
            dec_loss = loss_ae + 1.0 * loss_dec_adv

            enc_loss.backward(retain_graph=True)
            dec_loss.backward()

            opt_enc.step()
            opt_dec.step()

            # Decoder step
            # opt_dec.zero_grad()
            # dec_loss = loss_ae + 1.0 * loss_dec_adv
            # dec_loss.backward()
            # opt_dec.step()

        # end epoch
        if (ep+1) % 50 == 0 or ep == 0:
            elapsed = time.time() - start
            print(f"Epoch {ep+1}/{epochs} LR={LR.item():.6f} Lziz={Lziz.item():.6f} Lzl1={Lzl1.item():.6f} adv_enc={loss_enc_adv.item():.6f} adv_dec={loss_dec_adv.item():.6f} time={elapsed:.1f}s")

    # ===== inference =====
    enc.eval(); dec.eval()
    with torch.no_grad():
        Xtorch = torch.from_numpy(X).to(device)
        Z = enc(Xtorch)
        Xrec = dec(Z).cpu().numpy()
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

# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    # Example: load a toy HSI array (H,W,B) from .npy or create a small random one.
    # Replace with your actual HSI (normalized).
    H, W, B = 100, 100, 189  # e.g., San Diego scene
    # load your HSI here instead of random:
    # hsi = np.load('san_diego.npy')  # shape (H,W,B), dtype float32
    # hsi = np.random.rand(H, W, B).astype(np.float32) * 1.0

    mat=sio.loadmat('abu-beach-1.mat')
    hsi=mat['data']
    hsi = hsi.astype(np.float32) / 1.0  # normalize if needed

    out = train_hadgan(hsi, epochs=1)  # reduce epochs for faster testing

    # final detection map:
    fmap = out['final_map']
    print("Final map shape:", fmap.shape)

    # Save detection map with colorbar
    plt.figure(figsize=(6, 6))
    im = plt.imshow(fmap, cmap="jet")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Anomaly Detection Map")
    plt.savefig("final_map_with_colorbar.png", dpi=300)
    plt.close()
    print("Saved detection map as final_map_with_colorbar.png")


    # If reference mask available, compute AUC:
    # ref = np.load('san_diego_ref.npy').reshape(-1)  # 0/1 per pixel
    # auc = roc_auc_score(ref, fmap.reshape(-1))
    # print("AUC:", auc)
