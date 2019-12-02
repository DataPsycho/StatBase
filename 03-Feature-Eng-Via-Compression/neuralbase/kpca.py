from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np


def rbf_kernel_pca(features, gamma, n_components):
    """
    RBF kernel PCA implementation.
    :param features: {Numpy ndarray}, shape = [n_samples, n_features]
    :param gamma: float
        Tuning parameter of RBF kernel
    :param n_components:  int
        Number of principal components to return
    :return: X_pc {Numpy, ndarray}, shape = [n_samples, k_features]
    """
    # Calculate pairwise Squared Euclidean distances
    # in the MxN dimentional dataset.
    sq_dists = pdist(features, 'sqeuclidean')
    # # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)
    # Compute the symmetric Kernel Matrix
    kernel = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix.
    sample_size = kernel.shape[0]
    one_n = np.ones((sample_size, sample_size)) / sample_size
    kernel = kernel - one_n.dot(kernel) - kernel.dot(one_n) + one_n.dot(kernel).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    eigvals, eigvecs = eigh(kernel)
    # Reverse sorting in descending order
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    # Collect the top k eigenvectors (projected samples)
    x_pc = np.column_stack((eigvecs[:, i] for i in range(n_components)))
    return x_pc


def rbf_kernel_pca_modified(features, gamma, n_components):
    """
    RBF kernel PCA implementation.
    :param features: {Numpy ndarray}, shape = [n_samples, n_features]
    :param gamma: float
        Tuning parameter of RBF kernel
    :param n_components:  int
        Number of principal components to return
    :return: alphas {Numpy, ndarray}, shape = [n_samples, k_features], lambdas: list Eigenvalues
    """
    # Calculate pairwise Squared Euclidean distances
    # in the MxN dimentional dataset.
    sq_dists = pdist(features, 'sqeuclidean')
    # # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)
    # Compute the symmetric Kernel Matrix
    kernel = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix.
    sample_size = kernel.shape[0]
    one_n = np.ones((sample_size, sample_size)) / sample_size
    kernel = kernel - one_n.dot(kernel) - kernel.dot(one_n) + one_n.dot(kernel).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    eigvals, eigvecs = eigh(kernel)
    # Reverse sorting in descending order
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    # Collect the top k eigenvectors (projected samples)
    alphas = np.column_stack((eigvecs[:, i] for i in range(n_components)))
    # Collect the corresponding eigenvalues
    lambdas = [eigvals[i] for i in range(n_components)]
    return alphas, lambdas
