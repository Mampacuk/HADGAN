import os
import numpy as np
import torch
import torch.optim as optim
import scipy.io as sio
import pdb
from .net import Net
from sklearn.metrics import roc_auc_score, roc_curve
import shutil
from .utils import get_params, img2mask, seed_dict
import random
from progress.bar import Bar
import time 
import torch.nn as nn 
from torch.utils.data import DataLoader
from .data import DatasetHsi
from .block import Block_fold, Block_search
# import cv2 
import matplotlib.pyplot as plt
from utils.adaptive_thresh import tune_methods

dtype = torch.cuda.FloatTensor
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
data_dir = '../../data/'
save_dir = '../../results/'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Data loading / preprocessing
# ------------------------------
def load_hsi_mat(mat_path, dtype=torch.float32):
    """
    Load a .mat with keys 'data' and 'map' like your dataset.
    Returns: img_tensor (1, B, H, W) on CPU, gt numpy array (H, W)
    """

    mat = sio.loadmat(mat_path)
    img_np = np.array(mat['data'], dtype=np.float32)  # shape (H, W, B)
    gt=sio.loadmat(mat_path)
    gt = np.array(gt.get('groundtruth', np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)))
    # transpose to (B, H, W)
    img_np = img_np.transpose(2, 0, 1)
    img_np = img_np - np.min(img_np)
    img_np = img_np / (np.max(img_np) + 1e-12)
    img_var = torch.from_numpy(img_np).type(dtype)
    # add batch dim for DatasetHsi which expects (1, B, H, W)
    img_var = img_var.unsqueeze(0)
    return img_var, gt

# Model builder
# ------------------------------
def build_net(bands: int, patch_size: int = 3, patch_stride: int = 3, embed_dim: int = 64, device=None):
    net = Net(in_chans=bands, embed_dim=embed_dim, patch_size=patch_size,
              patch_stride=patch_stride, mlp_ratio=2.0, attn_drop=0.0, drop=0.0)
    if device is not None:
        net = net.to(device)
    else:
        net = net.cuda() if torch.cuda.is_available() else net
    return net

# ------------------------------
# Training loop
# ------------------------------
def train_gt_had(img_var,
                 seed,
                 device,
                 patch_size=3,
                 patch_stride=3,
                 lr=1e-3,
                 batch_size=64,
                 epochs=150,
                 search_iter=25,
                 avgpool_kernel=(5, 3, 3)
                 ):
    """
    Train the GT-HAD network on one dataset.
    Returns: dict with results (auc, runtime, saved_paths)
    """
    set_seed(seed)

    print(f"Dataset: HYDICE-urban")
    print(f"Model: GT-HAD")

    # Basic sizes
    _, band, row, col = img_var.size()  # img_var shape (1, B, H, W)
    # dataset expects (1, B, H, W) image batch
    block_size = patch_size * patch_stride
    data_set = DatasetHsi(img_var, wsize=block_size, wstride=patch_stride)
    block_fold = Block_fold(wsize=block_size, wstride=patch_stride)
    block_search = Block_search(img_var, wsize=block_size, wstride=patch_stride)

    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, drop_last=False)
    # build network, loss, optimizer
    net = build_net(bands=band, patch_size=patch_size, patch_stride=patch_stride, device=device)

    mse = nn.MSELoss().to(device)
    optimizer = optim.Adam(get_params(net), lr=lr)

    data_num = len(data_set)
    match_vec = torch.zeros((data_num,), device=device)
    search_matrix = torch.zeros((data_num, band, block_size, block_size), device=device)
    search_index = torch.arange(0, data_num, dtype=torch.long, device=device)

    avgpool = nn.AvgPool3d(kernel_size=avgpool_kernel, stride=(1, 1, 1), padding=(2, 1, 1))
    start = time.time()

    for it in range(1, epochs + 1):
        search_flag = (it % search_iter == 0 and it != epochs)
        for batch in data_loader:
            optimizer.zero_grad()
            net_gt = batch['block_gt'].to(device)
            net_input = batch['block_input'].to(device)
            block_idx = batch['index'].cuda() if device.type.startswith('cuda') else batch['index'].long()

            net_out = net(net_input, block_idx=block_idx, match_vec=match_vec)
            if search_flag:
                # store outputs for CMM-search
                search_matrix[block_idx] = net_out.detach()

            loss = mse(net_out, net_gt.to(device))
            loss.backward()
            optimizer.step()

        # CMM search step
        if search_flag:
            match_vec = torch.zeros((data_num,), device=device)
            search_back = block_fold(search_matrix.detach(), data_set.padding, row, col)
            search_back = search_back.to(device)  # <--- ensure same device
            match_vec = block_search(search_back.detach(), match_vec, search_index)

        if it % 15 == 0 or it == 1:
            elapsed = time.time() - start
            print(f"Epoch {it}/{epochs} Loss {loss} time={elapsed:.1f}s")

    torch.save(net.state_dict(),f"gthad_sal.pth")

    # ------------------------
    # Final inference
    # ------------------------
    net.eval()
    infer_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False, drop_last=False)
    infer_res_list = []

    with torch.no_grad():
        for batch in infer_loader:
            infer_in = batch['block_input'].to(device)
            infer_idx = batch['index'].to(device).long()
            infer_out = net(infer_in, block_idx=infer_idx, match_vec=match_vec)
            infer_res = torch.abs(infer_in - infer_out) ** 2
            infer_res = avgpool(infer_res)
            infer_res_list.append(infer_res.cpu())

    infer_res_out = torch.cat(infer_res_list, dim=0)   # (Nblocks, B, k, k)
    infer_res_back = block_fold(infer_res_out, data_set.padding, row, col)  # (B, H, W)
    residual_map = img2mask(infer_res_back)  # (H, W)

    return residual_map

# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    # choose device
    device = get_device()

    img_var,ref=load_hsi_mat(mat_path="/home/ubuntu/aditya/BioSky/Datasets/Pavia.mat")
    print("HSI shape:", img_var.shape)
    print("GT shape:", ref.shape)

    fmap = train_gt_had(img_var, 42, device,epochs=150)
    print("Final map shape:", fmap.shape)

    bests, pr_auc, roc=tune_methods(fmap, ref)
    print(f"Tuned results:, {bests}")
    print(f"pr_auc: {pr_auc}, roc: {roc}")

    # threshold=bests['iterative_f1'][1]
    # binary_map=(fmap>threshold).astype(np.uint8)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(ref, cmap="gray")
    plt.title("Ground Truth Mask")

    plt.subplot(1,2,2)
    plt.imshow(fmap, cmap="hot")
    plt.title("Anomaly Detection Map")
    plt.colorbar()

    plt.savefig("gthad_pavia.png", dpi=300) 
    plt.close()