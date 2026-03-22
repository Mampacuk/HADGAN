import numpy as np
import cv2
from sklearn.covariance import MinCovDet, LedoitWolf
from sklearn.decomposition import PCA

polygon_points = [(235, 5), (1275, 235), (1030, 1205), (5, 960)]
# polygon_points = [(235, 100), (1180, 235), (1030, 1116), (100, 960)]

def create_valid_mask(hsi, threshold=0.001,polygon_points=polygon_points):
    """
    Create a binary mask for valid (non-empty) pixels in an HSI cube.
    hsi: (H, W, B)
    # """
    mask = np.zeros((hsi.shape[0], hsi.shape[1]), dtype=np.uint8)
    pts = np.array([(y, x) for (x, y) in polygon_points], dtype=np.int32)
    pts = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)

    return mask

def compute_residual_map(hsi, recon):
    # hsi, recon are (H, W, B)
    # Create the difference array directly
    diff = np.empty_like(hsi, dtype=np.float32)
    np.subtract(hsi, recon, out=diff) # diff = hsi - recon
    
    # Calculate absolute value in-place, overwriting the difference array
    np.absolute(diff, out=diff) # diff = |diff|

    return diff

def select_min_energy_bands(residual, k=3):
    # residual shape (H,W,B)
    H,W,B = residual.shape
    X=residual.reshape(-1, B) # shape (N,B)

    # Now we take the k components as decided by PCA 
    pca=PCA(n_components=k)
    X_pca=pca.fit_transform(X) # shape (N,k)
    
    del X

    # Compute band energies in PCA space
    energies = np.sum(X_pca**2, axis=0)
    idx = np.argsort(energies)[-k:]  # top-k informative components

    del X_pca

    # Get actual band indices contributing most to these PCs
    # via PCA loadings (absolute values)
    loadings = np.abs(pca.components_[idx])
    loadings /= np.sum(loadings, axis=1, keepdims=True)
    band_scores = loadings.sum(axis=0)
    top_bands = np.argsort(band_scores)[-k:]

    return top_bands

    # return idx

def spatial_detector_from_residual(residual, selected_bands,mask=None,alpha=0.002): # !
    # residual: (H,W,B)
    H,W,B = residual.shape
    # compute per-band images (grayscale)
    df_list = []

    # === 1️⃣ Process each selected band ===
    for b in selected_bands:
        band = residual[:, :, b]

        # Normalize to 8-bit for OpenCV filters to handle it.
        band8 = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        # --- Multi-scale guided/bilateral filtering ---
        radii = [2, 5, 8]
        filtered_maps = []

        for r in radii:
            try:
                gf = cv2.ximgproc.guidedFilter(
                    guide=band8, src=band8, radius=r, eps=0.1
                )
                filtered_maps.append(gf)
            except Exception:
                # fallback to bilateral if ximgproc not available
                bf = cv2.bilateralFilter(band8, d=5, sigmaColor=75, sigmaSpace=r)
                filtered_maps.append(bf)

        # Combine multi-scale maps
        multi_scale = np.mean(np.stack(filtered_maps, axis=0), axis=0)

        # --- Compute local residual ---
        df = np.abs(band8.astype(np.float32) - multi_scale.astype(np.float32))

        # === Compute Laplacian edges ===
        edges = np.abs(cv2.Laplacian(band8, cv2.CV_32F))

        # === Compute local variance map (texture strength) ===
        mean = cv2.blur(band8.astype(np.float32), (5, 5))
        sqmean = cv2.blur(band8.astype(np.float32)**2, (5, 5))
        var_map = sqmean - mean**2
        var_map = np.clip(var_map, 0, 1e4)

        var_norm = var_map / (var_map.max() + 1e-8)
        var_norm = np.nan_to_num(var_norm, nan=0.0, posinf=1.0, neginf=0.0)

        # === Adaptive weight map ===
        w = np.exp(-alpha * var_norm)  # suppress edges in high-texture zones
        w = np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)

        # === Combine residual and edges with adaptive weighting ===
        df_combined = df + w * edges
        df_combined = np.nan_to_num(df_combined, nan=0.0, posinf=0.0, neginf=0.0)

        df_list.append(df_combined)

    # === 2️⃣ Fuse across bands ===
    F = np.mean(np.stack(df_list, axis=0), axis=0).astype(np.float32)

    # Masking step — ignore tilted edges
    if mask is not None:
        F = F * mask  # zero out unwanted regions

    # === 3️⃣ Morphological cleanup ===
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(F, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    # === 5️⃣ Normalize final output ===
    out_norm = (closed - closed.min()) / (np.ptp(closed) + 1e-8)

    # Reapply mask at the end too (optional)
    if mask is not None:
        out_norm *= mask

    out_norm = np.nan_to_num(out_norm, nan=0.0, posinf=1.0, neginf=0.0)

    return out_norm

def spatial_detector_from_residual_fast(residual, selected_bands, mask=None, alpha=0.002, use_fast_guided=True):
    # Build a composite guide image: mean of selected residual bands (float32)
    guide = np.mean(residual[:, :, selected_bands], axis=-1).astype(np.float32)

    if mask is not None:
        # === MASK THE INPUT GUIDE IMAGE ===
        guide *= mask.astype(np.float32)

    guide8 = cv2.normalize(guide, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # One fast guided filter pass (if available)
    try:
        if use_fast_guided and hasattr(cv2.ximgproc, 'fastGuidedFilter'):
            refined = cv2.ximgproc.fastGuidedFilter(guide8, guide8, radius=6, eps=1e-2, dst=None)
        else:
            refined = cv2.ximgproc.guidedFilter(guide8, guide8, radius=6, eps=1e-2)
        refined = refined.astype(np.float32)
    except Exception:
        # fallback to bilateral but single pass
        refined = cv2.bilateralFilter(guide8, d=9, sigmaColor=75, sigmaSpace=9).astype(np.float32)

    # === REMASK FILTERED RESULT ===
    if mask is not None:
        refined *= mask.astype(np.float32)

    df = np.abs(guide - refined)

    # compute fast local variance using boxFilter
    mean = cv2.boxFilter(guide, ddepth=-1, ksize=(5,5), normalize=True)
    sqmean = cv2.boxFilter(guide*guide, ddepth=-1, ksize=(5,5), normalize=True)
    var_map = np.maximum(sqmean - mean*mean, 1e-6)
    var_map = np.clip(var_map, 0, 1e4)
    var_norm = var_map / (var_map.max() + 1e-8)
    var_norm = np.nan_to_num(var_norm, nan=0.0, posinf=1.0, neginf=0.0)

    # laplacian edges (single)
    edges = np.abs(cv2.Laplacian(guide8, cv2.CV_32F))

    # adaptive weight (no 255 scaling)
    w = np.exp(-alpha * var_norm)
    w = np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)

    df_combined = df + w * edges
    df_combined = np.nan_to_num(df_combined, nan=0.0, posinf=0.0, neginf=0.0)

    if mask is not None:
        df_combined *= mask.astype(np.float32)

    # morphology and normalize (same as before)
    kernel = np.ones((3,3), np.uint8)
    opened = cv2.morphologyEx(df_combined.astype(np.float32), cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    out_norm = (closed - closed.min()) / (np.ptp(closed) + 1e-8)

    if mask is not None:
        out_norm *= mask.astype(np.float32)

    return out_norm

def spectral_detector_from_residual(residual, n_components=20, local=True, window_size=15):
    H, W, B = residual.shape
    X = residual.reshape(-1, B)

    # --- Step 1: PCA to decorrelate ---
    X_pca = PCA(n_components=min(n_components, B)).fit_transform(X)

    # --- Step 2: Robust covariance ---
    cov = MinCovDet().fit(X_pca)
    mean = cov.location_
    precision = cov.precision_

    dif = X_pca - mean
    D_global = np.sum(dif.dot(precision) * dif, axis=1).reshape(H, W)

    if not local:
        return D_global

    # --- Step 3: Local RX (approx) ---
    pad = window_size // 2
    D_local = np.zeros_like(D_global)
    X_reshaped = X_pca.reshape(H, W, -1)

    for i in range(pad, H - pad):
        for j in range(pad, W - pad):
            patch = X_reshaped[i - pad:i + pad + 1, j - pad:j + pad + 1, :].reshape(-1, X_pca.shape[1])
            mu = np.mean(patch, axis=0)
            cov = np.cov(patch, rowvar=False)
            inv = np.linalg.pinv(cov + 1e-6 * np.eye(cov.shape[0]))
            diff = X_reshaped[i, j, :] - mu
            D_local[i, j] = diff @ inv @ diff

    # --- Step 4: Fusion between global and local RX ---
    D = 0.5 * D_global + 0.5 * D_local
    return D

def spectral_detector_from_residual_fast(residual, mask=None, n_components=10, local=True, window_size=15):
    """
    Fast spectral detector:
      1) PCA to n_components
      2) Global RX using Ledoit-Wolf shrinkage covariance (much faster than MCD)
      3) Optional approximate local RX using box-filtered first/second moments
    """
    H, W, B = residual.shape
    X = residual.reshape(-1, B)  # N x B

    # === Data Filtering for Statistics ===
    # Create a filter array based on the mask
    if mask is not None:
        flat_mask = mask.ravel().astype(bool)
        X_filtered = X[flat_mask] # Only include valid pixels for fitting stats
    else:
        X_filtered = X

    # PCA
    n_comp=min(n_components,B)
    pca = PCA(n_components=n_comp)
    pca.fit(X_filtered) # Fit the background model ONLY on valid pixels
    X_pca = pca.transform(X)  # N x n_comp
    X_pca_resh = X_pca.reshape(H, W, n_comp).astype(np.float32)

    # Global RX with Ledoit-Wolf (LW) Covariance
    # Fit LW on the PCA-transformed, filtered data to get robust background statistics
    lw = LedoitWolf().fit(X_pca[flat_mask] if mask is not None else X_pca)
    mean_glob = lw.location_
    precision_glob = np.linalg.pinv(lw.covariance_ + 1e-6 * np.eye(n_comp)) # Use pinv for safety

    del X

    dif = X_pca - mean_glob
    D_global = np.sum((dif @ precision_glob) * dif, axis=1).reshape(H, W)

    # Normalize Global RX first (Critical for balanced fusion)
    D_global_norm = (D_global - D_global.min()) / (D_global.max() - D_global.min() + 1e-8)

    # === Local RX Approximation and Final Score ===
    if local:
        # CRITICAL: Mask the PCA features before local filtering to avoid contaminating
        # the local mean/variance with zero-valued boundary pixels.
        if mask is not None:
            X_pca_resh_masked = X_pca_resh * mask[:, :, np.newaxis].astype(np.float32)
        else:
            X_pca_resh_masked = X_pca_resh
            
        window_size = window_size if window_size % 2 == 1 else window_size + 1 # Ensure odd window size

        # compute local mean and local second moment per PCA channel with boxFilter
        X_chan = [X_pca_resh_masked[..., c] for c in range(n_comp)]
        local_mean = np.stack([cv2.boxFilter(ch, ddepth=-1, ksize=(window_size, window_size), normalize=True) for ch in X_chan], axis=-1)
        local_sqmean = np.stack([cv2.boxFilter((ch*ch).astype(np.float32), ddepth=-1, ksize=(window_size, window_size), normalize=True) for ch in X_chan], axis=-1)

        # local var per channel
        local_var = local_sqmean - local_mean**2
        local_var = np.maximum(local_var, 1e-6)

        # approximate local Mahalanobis using diagonal covariance assumption:
        diff = X_pca_resh_masked - local_mean # USE MASKED DATA HERE for difference
        D_local = np.sum((diff * diff) / local_var, axis=-1)

        # combine global and local (tune weights)
        # Normalize separately before combining to prevent large score ranges from dominating
        D_global_norm = D_global / (D_global.max() + 1e-8)
        D_local_norm = D_local / (D_local.max() + 1e-8)
        
        D = 0.5 * D_global_norm + 0.5 * D_local_norm
        
        # Final normalization (0 to 1)
        D = (D - D.min()) / (D.max() - D.min() + 1e-8)
        
    else:
        D = (D_global - D_global.min()) / (D_global.max() - D_global.min() + 1e-8)
        
    # === FINAL STEP: Apply mask to the result for artifact removal ===
    if mask is not None:
        D *= mask.astype(np.float32)
        
    return D

def fuse_spatial_spectral(dspatial, dspectral, lam=0.5,mode='multiplicative'):
    # normalize each
    ds = (dspatial - dspatial.min()) / (np.ptp(dspatial) + 1e-8)
    dspec = (dspectral - dspectral.min()) / (np.ptp(dspectral) + 1e-8)

    if mode == 'adaptive':
        local_var = cv2.blur((dspatial - cv2.blur(dspatial,(5,5)))**2, (5,5))
        lam_map = cv2.normalize(local_var, None, 0, 1, cv2.NORM_MINMAX)
        D = lam_map * ds + (1 - lam_map) * dspec
    elif mode == 'multiplicative':
        D = (dspatial**lam) * (dspectral**(1-lam))
    else:
        D = lam * ds + (1 - lam) * dspec

    del ds, dspec

    return D

def create_manmade_mask(hsi,fmap=None):
    H,W,B=hsi.shape
    hsi = (hsi - hsi.min()) / (np.ptp(hsi) + 1e-8) 
    all_wavelengths=np.linspace(400,2500,239) # !

    def find_band_index(wavelengths, target_nm):
        return np.argmin(np.abs(wavelengths - target_nm))

    NIR_IDX = find_band_index(all_wavelengths, 850)
    RED_IDX = find_band_index(all_wavelengths, 660)
    GREEN_IDX = find_band_index(all_wavelengths, 550)
    SWIR_IDX = find_band_index(all_wavelengths, 1650)

    del all_wavelengths

    nir=hsi[:,:,NIR_IDX].astype(np.float32)
    red=hsi[:,:,RED_IDX].astype(np.float32)

    # NDVI
    ndvi = (nir - red) / (nir + red + 1e-8)
    veg_mask = (ndvi > 0.3).astype(np.uint8)

    del ndvi,red

    green=hsi[:,:,GREEN_IDX].astype(np.float32)

    # NDWI
    ndwi = (green - nir) / (green + nir + 1e-8)
    water_mask = (ndwi > 0.3).astype(np.uint8)

    del ndwi,green

    swir=hsi[:,:,SWIR_IDX].astype(np.float32)

    # NDBI
    ndbi = (swir - nir) / (swir + nir + 1e-8)
    built_mask = (ndbi > -0.1).astype(np.uint8) # !

    del ndbi,swir,nir

    natural_mask = np.logical_or(veg_mask, water_mask)
    # del veg_mask, water_mask

    final_mask= np.logical_or(built_mask, ~natural_mask)
    # final_mask=~natural_mask # only ndvi and ndwi
    # final_mask=~veg_mask # only ndvi
    # final_mask=~water_mask # only ndwi
    # final_mask= np.logical_and(built_mask, ~natural_mask)

    del built_mask, natural_mask, veg_mask, water_mask

    final_fmap=fmap*final_mask
    del fmap

    return final_fmap
