import numpy as np
import scipy.io as sio
import random
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from .RGAE import RGAE
from sklearn.metrics import roc_auc_score
from ..utils.adaptive_thresh import tune_methods
from .ROC import ROC

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

if __name__ == "__main__":
    set_seeds(42)  # for reproducibility

    print(f"Dataset name: San_Deigo")
    print(f"Model: RGAE")

    # mat=sio.loadmat('/home/ubuntu/aditya/BioSky/Datasets/HYDICE-urban/HYDICE_urban.mat')
    # mat=sio.loadmat('/home/ubuntu/aditya/BioSky/Datasets/Sandiego/San_Diego.mat')
    mat=sio.loadmat('/home/ubuntu/aditya/BioSky/Datasets/Salinas/Salinas.mat')

    hsi = np.array(mat['hsi'], dtype=float)
    # data3d = data3d[0:100, 0:100, :]
    # remove_bands = np.hstack((range(6), range(32, 35, 1), range(93, 97, 1), range(106, 113), range(152, 166), range(220, 224)))
    # hsi = np.delete(data3d, remove_bands, axis=2)
    print("HSI shape:", hsi.shape)

    # Normalize the HSI data to [0, 1]
    hsi = (hsi - hsi.min()) / (np.ptp(hsi) + 1e-8)

    # gt_mat = sio.loadmat("/home/ubuntu/aditya/BioSky/Datasets/HYDICE-urban/HYDICE_urban.mat")  # file with ground truth
    # gt_mat = sio.loadmat("/home/ubuntu/aditya/BioSky/Datasets/Sandiego/San_Diego.mat")  # file with ground truth
    gt_mat=sio.loadmat("/home/ubuntu/aditya/BioSky/Datasets/Salinas/Salinas_gt.mat")
    ref = gt_mat['hsi_gt']  # shape (H,W), binary 0/1
    ref = ref.astype(np.uint8).reshape(-1)
    print("Ground truth shape:", ref.shape)

    # Hyperparameters
    lambda_=1e-2  # tradeoff parameter (weight between reconstruction & graph regularization)
    S=150 # number of superpixels
    n_hid=100 # number of hidden units

    y=RGAE(hsi, S, n_hid, lambda_)
    fmap=y.reshape(hsi.shape[0], hsi.shape[1])
    bests, pr_auc, roc=tune_methods(fmap, ref.reshape(hsi.shape[0], hsi.shape[1]))
    # auc=ROC(y, ref, display=True)
    # print("AUC score:", auc)
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

    plt.savefig("rgae_sal.png", dpi=300)
    plt.close()