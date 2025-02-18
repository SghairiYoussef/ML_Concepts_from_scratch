import numpy as np

def find_best_classifier(X, y, w, n_features):
    n_samples = X.shape[0]
    best_clf = None
    min_error = float("inf")

    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            for polarity in [1, -1]:
                predictions = np.ones(n_samples)
                predictions[X[:, feature_index] < threshold] = -1
                predictions *= polarity

                error = np.sum(w[predictions != y])

                if error > 0.5:  
                    error = 1 - error
                    polarity *= -1

                if error < min_error:
                    min_error = error
                    best_clf = {
						"polarity": polarity,
                        "threshold": threshold,
						"feature_index": feature_index
                    }

    return best_clf, min_error

def make_predictions(X, clf):
    n_samples = X.shape[0]
    predictions = np.ones(n_samples)
    predictions[X[:, clf["feature_index"]] < clf["threshold"]] = -1
    predictions *= clf["polarity"]
    return predictions

def update_weights(w, alpha, y, predictions):
    w *= np.exp(-alpha * y * predictions)
    w /= np.sum(w)
    return w


def adaboost_fit(X, y, n_clf):
    n_samples, n_features = X.shape
    w = np.full(n_samples, (1 / n_samples))
    clfs = []

    for _ in range(n_clf):
        best_clf, min_error = find_best_classifier(X, y, w, n_features)
        alpha = 0.5 * np.log((1 - min_error) / (min_error + 1e-10))

        predictions = make_predictions(X, best_clf)
        w = update_weights(w, alpha, y, predictions)

        best_clf["alpha"] = alpha
        clfs.append(best_clf)

    return clfs