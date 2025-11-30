import numpy as np
import cv2
from sklearn.covariance import EmpiricalCovariance

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
