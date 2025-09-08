import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, percentile_filter, binary_opening, binary_closing, label
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

# ---- methods (slightly adapted from earlier) ----
def local_zscore_threshold(S, sigma=2, z_thr=3.0, min_blob=10):
    S = S.astype(np.float32)
    S_sm = gaussian_filter(S, sigma=1)
    mu = gaussian_filter(S_sm, sigma=sigma, mode='reflect')
    mu2 = gaussian_filter(S_sm*S_sm, sigma=sigma, mode='reflect')
    var = np.clip(mu2 - mu*mu, 1e-8, None)
    z = (S_sm - mu) / np.sqrt(var)
    B = (z > z_thr)
    B = binary_opening(B, structure=np.ones((3,3)))
    B = binary_closing(B, structure=np.ones((3,3)))
    lbl, n = label(B)
    if n > 0:
        sizes = np.bincount(lbl.ravel())
        keep = np.where(sizes >= min_blob)[0]
        if keep.size>0:
            B = np.isin(lbl, keep)
        else:
            B[:] = False
    return B, z

def local_mad_threshold(S, win=15, z_thr=3.5, min_blob=10):
    S = S.astype(np.float32)
    S_sm = gaussian_filter(S, sigma=1)
    med = median_filter(S_sm, size=win, mode='reflect')
    mad = median_filter(np.abs(S_sm - med), size=win, mode='reflect')
    sigma_hat = np.clip(1.4826 * mad, 1e-6, None)
    z = (S_sm - med) / sigma_hat
    B = (z > z_thr)
    B = binary_opening(B, structure=np.ones((3,3)))
    B = binary_closing(B, structure=np.ones((3,3)))
    lbl, n = label(B)
    if n > 0:
        sizes = np.bincount(lbl.ravel())
        keep = np.where(sizes >= min_blob)[0]
        if keep.size>0:
            B = np.isin(lbl, keep)
        else:
            B[:] = False
    return B, z

def local_percentile_threshold(S, win=15, p=99.5, min_blob=10):
    S = S.astype(np.float32)
    S_sm = gaussian_filter(S, sigma=1)
    T = percentile_filter(S_sm, percentile=p, size=win, mode='reflect')
    B = (S_sm > T)
    B = binary_opening(B, structure=np.ones((3,3)))
    B = binary_closing(B, structure=np.ones((3,3)))
    lbl, n = label(B)
    if n > 0:
        sizes = np.bincount(lbl.ravel())
        keep = np.where(sizes >= min_blob)[0]
        if keep.size>0:
            B = np.isin(lbl, keep)
        else:
            B[:] = False
    return B, T

def iterative_f1_threshold(y_true, y_scores):
    thresholds = np.linspace(y_scores.min(), y_scores.max(), 300)
    f1s = []
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred))
    best_idx = np.argmax(f1s)
    return thresholds[best_idx], f1s[best_idx]

# ---- tuner ----
def tune_methods(S, GT):
    bests = {}
    # grids tuned for 100x100
    windows = [9, 15, 25]
    sigmas = [1, 2, 4]
    z_thrs_z = [2.0, 2.5, 3.0, 3.5, 4.0]
    z_thrs_mad = [3.0, 3.5, 4.0, 4.5]
    percentiles = [98, 99, 99.5, 99.9]
    min_blobs = [5, 10, 20]

    # local zscore
    best = (-1, None, None, None)
    for s in sigmas:
        for t in z_thrs_z:
            for mb in min_blobs:
                B, _ = local_zscore_threshold(S, sigma=s, z_thr=t, min_blob=mb)
                f1 = f1_score(GT.ravel().astype(int), B.ravel().astype(int))
                if f1 > best[0]:
                    best = (f1, s, t, mb)
    bests['zscore'] = best

    # local mad
    best = (-1, None, None, None)
    for w in windows:
        for t in z_thrs_mad:
            for mb in min_blobs:
                B, _ = local_mad_threshold(S, win=w, z_thr=t, min_blob=mb)
                f1 = f1_score(GT.ravel().astype(int), B.ravel().astype(int))
                if f1 > best[0]:
                    best = (f1, w, t, mb)
    bests['mad'] = best

    # percentile
    best = (-1, None, None, None)
    for w in windows:
        for p in percentiles:
            for mb in min_blobs:
                B, _ = local_percentile_threshold(S, win=w, p=p, min_blob=mb)
                f1 = f1_score(GT.ravel().astype(int), B.ravel().astype(int))
                if f1 > best[0]:
                    best = (f1, w, p, mb)
    bests['percentile'] = best

    #iterative_f1_threshold
    threshold,iterative_f1=iterative_f1_threshold(GT.ravel(), S.ravel())
    bests['iterative_f1']=(iterative_f1, threshold, None, None)

    # compute continuous-score metrics once for S
    pr_auc = average_precision_score(GT.ravel(), S.ravel())
    roc = roc_auc_score(GT.ravel(), S.ravel())

    return bests, pr_auc, roc
