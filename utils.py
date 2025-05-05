import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler

def generate_and_preprocess_data(n_workers, config):
    problem_type = config['problem_type']
    n_samples = config['n_samples']
    n_features = config['n_features']
    n_informative = config['n_informative_features']
    class_sep = config.get('classification_sep', 0.8)

    print(f"Generating Non-IID data")

    if problem_type == 'logistic':
        X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                   n_informative=n_informative, n_redundant=n_features - n_informative,
                                   n_clusters_per_class=1, flip_y=0.05, class_sep=class_sep,
                                   random_state=203)
        y = 2 * y - 1  # Normalize to -1, 1
    elif problem_type == 'quadratic':
        X, y, coef = make_regression(n_samples=n_samples, n_features=n_features,
                                     n_informative=n_informative, noise=10.0, coef=True, random_state=203)
    else:
        raise NotImplementedError(f"Wrong {problem_type}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_bias = np.hstack([X_scaled, np.ones((X_scaled.shape[0], 1))])
    n_features_bias = X_scaled_bias.shape[1]

    worker_data = []

    # Force non-IID by sorting
    sorted_indices = np.argsort(y)
    indices = sorted_indices

    # Distribute data to workers
    worker_indices = np.array_split(indices, n_workers)
    for i in range(n_workers):
        idx = worker_indices[i]
        X_local_data = X_scaled_bias[idx, :]
        y_local_data = y[idx]
        worker_data.append({'X': X_local_data, 'y': y_local_data})
        print(f"Worker {i}: {len(idx)} samples, Target y range: [{np.min(y_local_data):.2f}, {np.max(y_local_data):.2f}], Mean y: {np.mean(y_local_data):.2f}")




    print(f"Generated {n_samples} samples, {n_features_bias} features")
    return worker_data, n_features_bias, X_scaled_bias, y