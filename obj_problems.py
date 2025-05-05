import numpy as np
from scipy.special import expit
def logistic_objective(w, X, y, lambda_reg):
    if X.shape[0] == 0:
        return 0.0
    logits = X @ w
    y_logits = y * logits
    log_exp_term = np.maximum(0, -y_logits) + np.log(1 + np.exp(-np.abs(y_logits)))
    data_loss = np.mean(log_exp_term)
    reg_term = (lambda_reg / 2.0) * np.dot(w, w)
    return data_loss + reg_term

def logistic_stochastic_gradient(w, X_batch, y_batch, lambda_reg):
    if X_batch.shape[0] == 0:
        return np.zeros_like(w)
    logits = X_batch @ w
    probabilities = expit(-y_batch * logits)
    grad_data = np.mean(-y_batch[:, np.newaxis] * X_batch * probabilities[:, np.newaxis], axis=0)
    grad_reg = lambda_reg * w
    return grad_data + grad_reg

def logistic_full_gradient(w, workers, lambda_reg):
    """full gradient for across all workers' data."""
    full_grad_data = np.zeros_like(w)
    total_samples = 0
    for worker in workers:
        if worker.X_local.shape[0] > 0:
            logits = worker.X_local @ w
            probabilities = expit(-worker.y_local * logits)
            grad_data_local = np.sum(-worker.y_local[:, np.newaxis] * worker.X_local * probabilities[:, np.newaxis], axis=0)
            full_grad_data += grad_data_local
            total_samples += worker.X_local.shape[0]
    if total_samples == 0: return np.zeros_like(w)
    avg_grad_data = full_grad_data / total_samples
    grad_reg = lambda_reg * w
    return avg_grad_data + grad_reg


def quadratic_objective(w, X, y, mu_reg):
    if X.shape[0] == 0: return 0.0
    predictions = X @ w
    data_loss = 0.5 * np.mean((predictions - y)**2)
    reg_term = (mu_reg / 2.0) * np.dot(w, w) # L2
    return data_loss + reg_term

def quadratic_stochastic_gradient(w, X_batch, y_batch, mu_reg):
    if X_batch.shape[0] == 0:
        return np.zeros_like(w)
    predictions = X_batch @ w
    errors = predictions - y_batch
    grad_data = np.mean(X_batch * errors[:, np.newaxis], axis=0)
    grad_reg = mu_reg * w
    return grad_data + grad_reg

def quadratic_full_gradient(w, workers, mu_reg):
    """full gradient across all workers' data"""
    full_grad_data = np.zeros_like(w)
    total_samples = 0
    for worker in workers:
        if worker.X_local.shape[0] > 0:
            predictions = worker.X_local @ w
            errors = predictions - worker.y_local
            grad_data_local = np.sum(worker.X_local * errors[:, np.newaxis], axis=0)
            full_grad_data += grad_data_local
            total_samples += worker.X_local.shape[0]
    if total_samples == 0: return np.zeros_like(w)
    avg_grad_data = full_grad_data / total_samples
    grad_reg = mu_reg * w
    return avg_grad_data + grad_reg