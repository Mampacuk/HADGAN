import numpy as np
import tensorflow as tf
from tqdm import tqdm

tf.keras.mixed_precision.set_global_policy('mixed_float16')

def mean_inference(hsi,enc,dec,window,ov,batch_size):
    H,W,B=hsi.shape
    Hk,Wk,Bk=window

    hsi=tf.convert_to_tensor(hsi,dtype=tf.float32)

    Xrec = tf.Variable(tf.zeros((H, W, B), dtype=tf.float32))
    K = tf.Variable(tf.zeros((H, W, 1), dtype=tf.float32))

    stride_h=max(1,(100-ov)*Hk//100)
    stride_w=max(1,(100-ov)*Wk//100)

    # ! ---- Using generator function ----
    # Compute starting positions (including forced border coverage)
    h_starts = list(range(0, H - Hk + 1, stride_h))
    w_starts = list(range(0, W - Wk + 1, stride_w))

    if h_starts[-1] != H - Hk:
        h_starts.append(H - Hk)   # force last patch to align with bottom edge
    if w_starts[-1] != W - Wk:
        w_starts.append(W - Wk)   # force last patch to align with right edge

    # def gen_patches():
    #     for i in h_starts:
    #         for j in w_starts:
    #             yield hsi[i:i+Hk, j:j+Wk, :]

    # dataset = (
    #     tf.data.Dataset.from_generator(
    #         gen_patches,
    #         output_signature=tf.TensorSpec(shape=(Hk,Wk,Bk), dtype=tf.float32),
    #     )
    #     .prefetch(tf.data.AUTOTUNE)
    # )

    # ! ---- Using extract patches function ----
    # hsi = tf.transpose(hsi.numpy(), (2, 0, 1))
    # hsi=tf.convert_to_tensor(hsi[tf.newaxis, ..., tf.newaxis], dtype=tf.float32)

    # patches=tf.image.extract_patches(input=hsi,
    #                                         ksizes=[1,Bk,Hk,Wk,1],
    #                                         strides=[1,stride_b,stride_h,stride_w,1],
    #                                         padding='VALID')

    # patches=tf.reshape(patches,[-1,Bk,Hk,Wk])
    # num_patches=patches.shape[0]

    # # Create a batch dataset for inference
    # dataset = tf.data.Dataset.from_tensor_slices(patches).batch(batch_size)

    @tf.function
    def inference_core(xbatch):
        X_flat = tf.reshape(xbatch, [-1, Bk])
        Z_batch = enc(X_flat, training=False)
        Xrec_batch = dec(Z_batch, training=False)
        recon = tf.reshape(Xrec_batch, [-1, Hk, Wk, Bk])  # back to patch shape
        recon=tf.cast(recon,tf.float32)
        return recon

    # patch_recons=[]

    # for xbatch in dataset:
        # X_flat = tf.reshape(xbatch, [-1, Bk])
        # Z_batch = enc(X_flat, training=False)
        # Xrec_batch = dec(Z_batch, training=False)
        # recon = tf.reshape(Xrec_batch, [-1, Hk, Wk, Bk])  # back to patch shape
    #     patch_recons.append(recon)

    # # patch_recons = tf.concat(patch_recons, axis=0)

    # idx=0
    # for i in h_starts:
    #     for j in w_starts:
    #             patch_rec=patch_recons[idx]
    #             patch_rec = tf.cast(patch_rec, tf.float32)

                # Xrec[i:i+Hk, j:j+Wk, :].assign(Xrec[i:i+Hk, j:j+Wk, :] + patch_rec)
                # K[i:i+Hk, j:j+Wk, :].assign(K[i:i+Hk, j:j+Wk, :] + 1.0)

    #             idx+=1

    for i in tqdm(h_starts, desc="Inference Progress"):
        for j in w_starts:
            patch=hsi[i:i+Hk,j:j+Wk,:]
            recon=inference_core(patch)

            Xrec[i:i+Hk, j:j+Wk, :].assign(Xrec[i:i+Hk, j:j+Wk, :] + recon)
            K[i:i+Hk, j:j+Wk, :].assign(K[i:i+Hk, j:j+Wk, :] + 1.0)


    Xrec.assign(Xrec/tf.maximum(K,1e-8))
    return Xrec

# @tf.function
def weighted_inference(hsi, enc, dec, window, ov,batch_size, sigma=None):
    H, W, B = hsi.shape
    Hk,Wk,Bk=window
    
    hsi=tf.convert_to_tensor(hsi,dtype=tf.float32)

    Xrec = tf.Variable(tf.zeros((H, W, B), dtype=tf.float32))
    K = tf.Variable(tf.zeros((H, W, 1), dtype=tf.float32))

    stride_h=max(1,(100-ov)*Hk//100)
    stride_w=max(1,(100-ov)*Wk//100)

    # ! ---- Using generator function ----
    # Compute starting positions (including forced border coverage)
    h_starts = list(range(0, H - Hk + 1, stride_h))
    w_starts = list(range(0, W - Wk + 1, stride_w))

    if h_starts[-1] != H - Hk:
        h_starts.append(H - Hk)   # force last patch to align with bottom edge
    if w_starts[-1] != W - Wk:
        w_starts.append(W - Wk)   # force last patch to align with right edge

    def gen_patches():
        for i in h_starts:
            for j in w_starts:
                yield hsi[i:i+Hk, j:j+Wk, :]

    dataset = (
        tf.data.Dataset.from_generator(
            gen_patches,
            output_signature=tf.TensorSpec(shape=(Hk,Wk,Bk), dtype=tf.float32),
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    # ! ---- Using extract patches function ----
    # hsi = tf.transpose(hsi.numpy(), (2, 0, 1))
    # hsi=tf.convert_to_tensor(hsi[tf.newaxis, ..., tf.newaxis], dtype=tf.float32)

    # patches=tf.image.extract_patches(input=hsi,
    #                                         ksizes=[1,Bk,Hk,Wk,1],
    #                                         strides=[1,stride_b,stride_h,stride_w,1],
    #                                         padding='VALID')

    # patches=tf.reshape(patches,[-1,Bk,Hk,Wk])
    # num_patches=patches.shape[0]

    # # Create a batch dataset for inference
    # dataset = tf.data.Dataset.from_tensor_slices(patches).batch(batch_size)

    # ! Create a 3D Gaussian weight kernel
    if sigma is None:
        sigma = (Hk + Wk) / 12.0   # heuristic: ~1/6th of average patch size
    
    # Define a 3D grid
    y, x = tf.meshgrid(tf.range(Hk, dtype=tf.float32), tf.range(Wk, dtype=tf.float32), indexing='ij')
    cy, cx = Hk / 2.0, Wk / 2.0

    weights = tf.exp(-(((y - cy) ** 2) + ((x - cx) ** 2)) / (2.0 * sigma**2))
    weights=weights[...,tf.newaxis]
    weights /= tf.reduce_max(weights)  # normalize
    
    # Move to GPU explicitly
    weights = tf.constant(weights)  # now it's a tf.Tensor on default device (GPU if available)

    # @tf.function
    # def inference_core(xbatch):
    #     X_flat = tf.reshape(xbatch, [-1, Bk])
    #     Z_batch = enc(X_flat, training=False)
    #     Xrec_batch = dec(Z_batch, training=False)
    #     recon = tf.reshape(Xrec_batch, [-1, Hk, Wk, Bk])  # back to patch shape
    #     recon = tf.cast(recon, tf.float32)
    #     weighted_patch = recon * weights
    #     return weighted_patch

    patch_recons=[]

    for xbatch in dataset:
        X_flat = tf.reshape(xbatch, [-1, Bk])
        Z_batch = enc(X_flat, training=False)
        Xrec_batch = dec(Z_batch, training=False)
        recon = tf.reshape(Xrec_batch, [-1, Hk, Wk, Bk])  # back to patch shape
        recon = tf.cast(recon, tf.float32)
        weighted_patch = recon * weights
        patch_recons.append(weighted_patch)

    idx=0
    for i in h_starts:
        for j in w_starts:
                weighted_patch=patch_recons[idx]

                Xrec[i:i+Hk, j:j+Wk, :].assign(Xrec[i:i+Hk, j:j+Wk, :] + weighted_patch)
                K[i:i+Hk, j:j+Wk, :].assign(K[i:i+Hk, j:j+Wk, :] + weights)

                idx+=1

    # 6. Normalize by accumulated weights
    Xrec.assign(Xrec/tf.maximum(K,1e-8))
    return Xrec