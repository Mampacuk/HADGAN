import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from skimage.segmentation import slic
from skimage.util import img_as_float
from .PCA import myPCA

def build_knn_graph(data, k=5,sigma=2):
    N,B=data.shape
    nbrs=NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data)
    dist, indices=nbrs.kneighbors(data)

    W=np.zeros((N,N))
    for i in range(N):
        for j, dis in zip(indices[i,1:],dist[i,1:]):
            W[i][j]=np.exp(-dis**2/(2*sigma**2))
            W[j][i]=W[i][j]
    return W

def SLIC(hsi, n_superpixels=150, sigma2=4.0):
    H, W, B = hsi.shape
    N = H * W
    X = hsi.reshape(N, B).astype(np.float32)

    # 1) compute first principal component image
    # We compute the PCA across bands and pick the first PC scores
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X)  # shape (N, 1)
    pc1_img = pc1.reshape(H, W)

    # pc_data = myPCA(hsi)  # shape (H, W, B)
    # pc1_img= pc_data[:, :, -1]  # first principal component image

    pc1_img = (pc1_img - pc1_img.min()) / (pc1_img.ptp() + 1e-12)  # normalize

    # 2) superpixel segmentation on pc1 image
    # convert to float and run slic. compactness can be tuned (default 10)
    labels = slic(pc1_img, n_segments=n_superpixels, compactness=10, start_label=0,channel_axis=None)
    # labels: shape (H, W), values in [0, n_superpixels-1]
    labels_flat = labels.reshape(N)

    # 3) build W: adjacency only inside each superpixel
    Wmat = np.zeros((N, N), dtype=np.float32)

    for lab in np.unique(labels_flat):
        idx = np.where(labels_flat == lab)[0]
        if idx.size <= 1:
            continue
        x = X[idx, :]  # shape (k, B)
        # compute pairwise squared distances (k x k)
        # we can compute efficiently with broadcasting or cdist
        # but for moderate k this is fine:
        diff = x[:, None, :] - x[None, :, :]  # (k, k, B)
        dist2 = np.sum(diff * diff, axis=2)  # (k, k)
        # Gaussian weights
        W_block = np.exp(-dist2 / (2.0 * sigma2))
        # zero out diagonal
        np.fill_diagonal(W_block, 0.0)
        # insert into block
        for i_local, i_global in enumerate(idx):
            Wmat[i_global, idx] = W_block[i_local, :]

    # Symmetrize
    W = (Wmat + Wmat.T) / 2.0

    return W.astype(np.float32)


def supergraph(hsi,S):
    # W=build_knn_graph(hsi.reshape(-1,hsi.shape[2]),k=5,sigma=2)
    W=SLIC(hsi,n_superpixels=S,sigma2=4)
    D=np.diag(W.sum(axis=1))
    L=D-W
    return L,D,W