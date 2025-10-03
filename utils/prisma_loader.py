import h5py
import numpy as np
import scipy.io as sio
from scipy.io import savemat
from PIL import Image

def load_prisma_he5(file_path):
    with h5py.File(file_path, "r") as f:
        # Paths inside the HE5 file
        vnir_path = "HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube"
        swir_path = "HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube"

        # Load data
        vnir_cube = f[vnir_path][:]
        swir_cube = f[swir_path][:]

        vnir_cube=np.transpose(vnir_cube,(0,2,1))
        swir_cube=np.transpose(swir_cube,(0,2,1))

        # Concatenate along spectral dimension (axis=2)
        full_cube = np.concatenate([vnir_cube, swir_cube], axis=2)

        band_info = {
            "vnir_bands": range(vnir_cube.shape[2]),
            "swir_bands": range(vnir_cube.shape[2], full_cube.shape[2])
        }
        
        print("VNIR shape",vnir_cube.shape)
        print("SWIR shape",swir_cube.shape)
        print("VNIR bands:", band_info["vnir_bands"])
        print("SWIR bands:", band_info["swir_bands"])
        
    return full_cube

file_path = "/home/ubuntu/aditya/BioSky/Datasets/PRS_L2D_STD_20201214060713_20201214060717_0001.he5"
full_cube = load_prisma_he5(file_path)
print("Full cube shape:", full_cube.shape)
# savemat("Datasets/MOCK_2/prisma_data.mat", {"data": full_cube})

# print("Full cube shape:", full_cube.shape)

# mat=sio.loadmat("/home/ubuntu/aditya/BioSky/Datasets/MOCK_2/prisma_data.mat")
# print(mat['data'].shape)

# # Load TIFF
# img = Image.open("/home/ubuntu/aditya/BioSky/Datasets/MOCK_2/groundtruth_hyper.tif")
# img_array = np.array(img)

# # Save as MAT
# savemat("Datasets/MOCK_2/prisma_gt.mat", {"data": img_array})

# mat=sio.loadmat("/home/ubuntu/aditya/BioSky/Datasets/MOCK_2/prisma_gt.mat")
# print(mat['data'].shape)
