import numpy as np 

def standardize_data(data: np.ndarray) -> np.ndarray:
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    standardized_data = (data - mean) / std_dev
    return standardized_data

def covariance_matrix(data: np.ndarray) -> np.ndarray:
    centered_data = data - np.mean(data, axis=0)
    cov_matrix = np.dot(centered_data.T, centered_data) / (data.shape[0] - 1)
    return cov_matrix

def pca(data: np.ndarray, k: int) -> np.ndarray:
    standardized_data = standardize_data(data)
    
    cov_mx = covariance_matrix(standardized_data)
    
    eigenvalues, eigenvectors = np.linalg.eig(cov_mx)
    
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    principal_components = sorted_eigenvectors[:, :k]
    
    return principal_components