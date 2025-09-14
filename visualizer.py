from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# mat=sio.loadmat('/home/aditya_ag00/BioSky/Datasets/Sandiego/San_Diego.mat')
# mat=sio.loadmat('/home/aditya_ag00/BioSky/Datasets/HYDICE-urban/HYDICE_urban.mat')
mat=sio.loadmat('/home/aditya_ag00/BioSky/Datasets/Salinas/Salinas.mat')
hsi = np.array(mat['hsi'], dtype=float)

H, W, B = hsi.shape
X = hsi.reshape(-1, B)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)
rgb_pca = X_pca.reshape(H, W, 3)

rgb_pca = (rgb_pca - rgb_pca.min()) / (rgb_pca.max() - rgb_pca.min())

plt.imshow(rgb_pca)
plt.title("Salinas")
plt.axis("off")
plt.savefig("Salinas.png", bbox_inches='tight', pad_inches=0)
plt.close()
