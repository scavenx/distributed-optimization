import numpy as np
from obj_problems import logistic_stochastic_gradient, quadratic_stochastic_gradient

class Worker:
    def __init__(self, worker_id, local_data, batch_size, n_features, config):
        self.worker_id = worker_id
        self.X_local = local_data['X']
        self.y_local = local_data['y']
        self.batch_size = batch_size
        self.n_local_samples = self.X_local.shape[0]
        self.config = config
        self.n_features = n_features
        self.x = np.zeros(n_features)

    def get_mini_batch(self):
        # Return empty batch if no local samples
        if self.n_local_samples == 0:
            return np.array([]).reshape(0, self.n_features), np.array([])

        # Ensure batch size doesn't exceed local samples
        effective_batch_size = min(self.batch_size, self.n_local_samples)
        if effective_batch_size <= 0:
             return np.array([]).reshape(0, self.n_features), np.array([])

        # Sample indices without replacement
        replace = effective_batch_size > self.n_local_samples
        idxs = np.random.choice(self.n_local_samples, effective_batch_size, replace=replace)
        return self.X_local[idxs], self.y_local[idxs]

    def compute_gradient(self, model_params=None):
        params_to_use = model_params if model_params is not None else self.x

        X_batch, y_batch = self.get_mini_batch()

        problem_type = self.config['problem_type']
        lambda_reg = self.config['l2_regularization_lambda']
        mu_reg = self.config['strong_convexity_mu']

        if problem_type == 'logistic':
            return logistic_stochastic_gradient(params_to_use, X_batch, y_batch, lambda_reg)
        elif problem_type == 'quadratic':
            return quadratic_stochastic_gradient(params_to_use, X_batch, y_batch, mu_reg)
        else:
            raise NotImplementedError(f"Wrong {problem_type}")