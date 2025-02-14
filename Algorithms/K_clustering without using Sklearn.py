import numpy as np

def euclidean_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Computes the Euclidean distance between two points."""
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def k_means_clustering(
    points: list[tuple[float, float]], 
    k: int, 
    initial_centroids: list[tuple[float, float]], 
    max_iterations: int
) -> list[tuple[float, float]]:
    
    centroids = np.array(initial_centroids, dtype=np.float64)  # Store centroids as a NumPy array
    
    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]  # Create k empty lists for clusters
        
        # Assign each point to the nearest centroid
        for point in points:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            closest_centroid_idx = np.argmin(distances)  # Get index of the nearest centroid
            clusters[closest_centroid_idx].append(point)
        
        # Compute new centroids
        new_centroids = []
        for cluster in clusters:
            if cluster:  # Avoid empty cluster error
                new_centroids.append(tuple(np.mean(cluster, axis=0)))
            else:
                new_centroids.append(tuple(np.random.rand(2)))  # Random centroid if cluster is empty
        
        # Check for convergence
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        
        centroids = np.array(new_centroids, dtype=np.float64)  # Update centroids
    
    return [tuple(centroid) for centroid in centroids]
