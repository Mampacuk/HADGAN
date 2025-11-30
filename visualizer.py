import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
import matplotlib.pyplot as plt

full_cube=loadmat("/home/ubuntu/aditya/BioSky/Datasets/MOCK_2/prisma_data.mat")
hsi = np.array(full_cube['data'], dtype=float)
print("HSI shape:", hsi.shape)
H,W,B=hsi.shape

hsi = (hsi - hsi.min()) / (hsi.max()-hsi.min())
hsi=hsi.reshape(-1,B)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(hsi)

pca = PCA(n_components=3) 
principal_components = pca.fit_transform(scaled_data)
pca_image = principal_components[:, :3].reshape(H, W, 3)
pca_image=(pca_image-pca_image.min())/(pca_image.max() - pca_image.min())

def onclick(event):
    # Only register left mouse button clicks inside the axes
    if event.inaxes:
        x, y = int(event.xdata), int(event.ydata)
        print(f"Clicked point: ({x}, {y})")
        plt.plot(x, y, 'ro')  # mark the point
        plt.draw()

fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(pca_image)
ax.set_title("Click on the tilted HSI corners to get coordinates")

# Connect event
cid = fig.canvas.mpl_connect('button_press_event', onclick)
# plt.show()
plt.imsave("pca_preview.png", pca_image)
print("Saved PCA preview as pca_preview.png — download and inspect it locally.")

# plt.imshow(pca_image)
# plt.axis("off") 
# plt.savefig("processed_MOCK_1.png",dpi=600,bbox_inches="tight")
# print(f"Comparison image saved as processed_MOCK_1.png")