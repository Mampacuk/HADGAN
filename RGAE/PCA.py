import numpy as np

def myPCA(data):
    """
    Performs Principal Component Analysis (PCA) on hyperspectral data,
    mirroring the provided MATLAB function.

    Args:
        data (np.ndarray): A 3D numpy array of shape (M, N, L),
                           where M and N are spatial dimensions and L is the number of bands.

    Returns:
        np.ndarray: A 3D numpy array of the same shape (M, N, L) containing
                    the PCA-transformed data.
    """
    # Get dimensions
    M, N, L = data.shape

    # Reshape the 3D data cube into a 2D matrix (pixels x bands)
    X = data.reshape(M * N, L)

    # Compute the scatter matrix of the bands (X' * X in MATLAB)
    # This is proportional to the covariance matrix.
    sigma = X.T @ X

    # Compute eigenvectors (V) and eigenvalues (not used here)
    # The columns of V are the eigenvectors.
    # Note: Unlike standard PCA libraries, this doesn't sort the eigenvectors
    # by the magnitude of their corresponding eigenvalues.
    eigenvalues, V = np.linalg.eig(sigma)

    # Project the original data onto the eigenvectors to get the principal components
    Y = X @ V

    # Reshape the result back to a 3D data cube
    Y = Y.reshape(M, N, L)
    
    return Y