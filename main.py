import numpy as np
import matplotlib.pyplot as plt
import time
from simulator import Simulator

N_WORKERS = 25         # Number of worker nodes
LOCAL_BATCH_SIZE = 16  # Mini-batch size 'b' per worker
N_ITERATIONS = 10000     # Total number of iterations
LEARNING_RATE_ETA0 = 0.05 # Initial learning rate
SUBOPTIMALITY_THRESHOLD = 0.08 # for reporting

PROBLEM_TYPE = 'quadratic'  # 'logistic', 'quadratic'

N_SAMPLES = N_WORKERS * 500
N_FEATURES = 80
N_INFORMATIVE_FEATURES = 50
CLASSIFICATION_SEP = 0.7


L2_REGULARIZATION_LAMBDA = 1e-4 # for L2
STRONG_CONVEXITY_MU = L2_REGULARIZATION_LAMBDA

if __name__ == "__main__":
    np.random.seed(203)
    sim_config = {
        'n_workers': N_WORKERS,
        'local_batch_size': LOCAL_BATCH_SIZE,
        'n_iterations': N_ITERATIONS,
        'learning_rate_eta0': LEARNING_RATE_ETA0,
        'l2_regularization_lambda': L2_REGULARIZATION_LAMBDA,
        'strong_convexity_mu': STRONG_CONVEXITY_MU,
        'problem_type': PROBLEM_TYPE,
        'n_samples': N_SAMPLES,
        'n_features': N_FEATURES,
        'n_informative_features': N_INFORMATIVE_FEATURES,
        'classification_sep': CLASSIFICATION_SEP,
        'suboptimality_threshold': SUBOPTIMALITY_THRESHOLD,
    }
    simulator = Simulator(sim_config)
    simulator.run_all()
    simulator.plot_results()