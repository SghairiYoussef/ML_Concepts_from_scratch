import numpy as np

def cross_validation_split(data: np.ndarray, k: int, seed=42) -> list:
    np.random.seed(seed)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    
    folds = np.array_split(data[indices], k)

    result = []
    for i in range(k):
        test_set = folds[i]
        train_set = np.vstack([folds[j] for j in range(k) if j != i])

        result.append([train_set.tolist(), test_set.tolist()])
    return result
