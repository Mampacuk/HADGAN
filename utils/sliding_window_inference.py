import numpy as np
import tensorflow as tf

def mean_inference(hsi,enc,dec,Hk,Wk,Bk,ov,batch_size):
    H,W,B=hsi.shape

    Xrec=np.zeros((H,W,B),dtype=np.float32)
    K=np.zeros((H,W,B),dtype=np.float32)

    stride_h=max(1,(100-ov)*Hk//100)
    stride_w=max(1,(100-ov)*Wk//100)
    stride_b=max(1,(100-ov)*Bk//100)

    # Compute starting positions (including forced border coverage)
    h_starts = list(range(0, H - Hk + 1, stride_h))
    w_starts = list(range(0, W - Wk + 1, stride_w))
    b_starts = list(range(0, B - Bk + 1, stride_b))

    if h_starts[-1] != H - Hk:
        h_starts.append(H - Hk)   # force last patch to align with bottom edge
    if w_starts[-1] != W - Wk:
        w_starts.append(W - Wk)   # force last patch to align with right edge
    if b_starts[-1] != B - Bk: 
        b_starts.append(B - Bk)

    hsi = np.transpose(hsi.numpy(), (2, 0, 1))
    hsi=tf.convert_to_tensor(hsi[np.newaxis, ..., np.newaxis], dtype=tf.float32)

    patches=tf.image.extract_volume_patches(input=hsi,
                                            ksizes=[1,Bk,Hk,Wk,1],
                                            strides=[1,stride_b,stride_h,stride_w,1],
                                            padding='VALID')

    patches=tf.reshape(patches,[-1,Bk,Hk,Wk])
    num_patches=patches.shape[0]

    # Create a batch dataset for inference
    inference_dataset = tf.data.Dataset.from_tensor_slices(patches).batch(batch_size)

    patch_recons=[]

    for xbatch in inference_dataset:
        X_flat = tf.reshape(xbatch, [-1, Bk])
        Z_batch = enc(X_flat, training=False)
        Xrec_batch = dec(Z_batch, training=False)
        recon = tf.reshape(Xrec_batch, [-1, Hk, Wk, Bk])  # back to patch shape
        patch_recons.append(recon.numpy())

    patch_recons = np.concatenate(patch_recons, axis=0)

    idx=0
    for i in h_starts:
        for j in w_starts:
            for k in b_starts:
                patch_rec=patch_recons[idx]
                Xrec[i:i+Hk,j:j+Wk,k:k+Bk]+=patch_rec
                K[i:i+Hk,j:j+Wk,k:k+Bk]+=1
                idx+=1

    Xrec/=np.maximum(K,1e-8)
    return Xrec

def weighted_inference(hsi, enc, dec, Hk, Wk, Bk, ov,batch_size, sigma=None):
    H, W, B = hsi.shape
    Xrec = np.zeros((H, W, B), dtype=np.float32)
    K = np.zeros((H, W, B), dtype=np.float32)

    stride_h=max(1,(100-ov)*Hk//100)
    stride_w=max(1,(100-ov)*Wk//100)
    stride_b=max(1,(100-ov)*Bk//100)

    # Compute starting positions (including forced border coverage)
    h_starts = list(range(0, H - Hk + 1, stride_h))
    w_starts = list(range(0, W - Wk + 1, stride_w))
    b_starts = list(range(0, B - Bk + 1, stride_b))

    if h_starts[-1] != H - Hk:
        h_starts.append(H - Hk)   # force last patch to align with bottom edge
    if w_starts[-1] != W - Wk:
        w_starts.append(W - Wk)   # force last patch to align with right edge
    if b_starts[-1] != B - Bk: 
        b_starts.append(B - Bk)

    # ! Create a 3D Gaussian weight kernel
    if sigma is None:
        sigma = (Hk + Wk + Bk) / 18.0   # heuristic: ~1/6th of average patch size
    
    # Define a 3D grid
    z, y, x = np.mgrid[0:Hk, 0:Wk, 0:Bk]
    cz, cy, cx = Hk // 2, Wk // 2, Bk // 2

    weights = np.exp(-(((z - cz) ** 2) + ((y - cy) ** 2) + ((x - cx) ** 2)) / (2.0 * sigma**2))
    weights /= weights.max()  # normalize

    for i in h_starts:
        for j in w_starts:
            for k in b_starts:
                patch = hsi[i:i+Hk, j:j+Wk, k:k+Bk]

                X_flat = patch.reshape(-1, Bk).astype(np.float32)
                Xtensor = tf.convert_to_tensor(X_flat, dtype=tf.float32)

                inference_dataset = tf.data.Dataset.from_tensor_slices(Xtensor).batch(batch_size)
                patch_recons = []

                for xbatch in inference_dataset:
                    Z_batch = enc(xbatch, training=False)
                    Xrec_batch = dec(Z_batch, training=False).numpy()
                    patch_recons.append(Xrec_batch)

                patch_recons = np.concatenate(patch_recons, axis=0)
                recon_patch = patch_recons.reshape(Hk, Wk, Bk)

                # Apply 3D Gaussian weights
                weighted_patch = recon_patch * weights

                # Accumulate results and weights
                Xrec[i:i+Hk, j:j+Wk, k:k+Bk] += weighted_patch
                K[i:i+Hk, j:j+Wk, k:k+Bk] += weights

    # 6. Normalize by accumulated weights
    Xrec /= np.maximum(K, 1e-8)
    return Xrec