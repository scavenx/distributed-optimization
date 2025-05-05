import numpy as np
import networkx as nx
import time
from obj_problems import logistic_objective, quadratic_objective


class CentralizedTrainer:
    def __init__(self, workers, n_features, config):
        self.workers = workers
        self.n_workers = len(workers)
        self.x_global = np.zeros(n_features) # Initialized here
        self.config = config
        self.history = {'objective': [], 'time': []}
        self.n_features = n_features
        self.total_floats_transmitted = 0

    def _get_learning_rate(self, t):
        eta0 = self.config['learning_rate_eta0']
        return eta0 / np.sqrt(t + 1)   #  As in the convex case, using O(1/sqrt(t))

    def _get_objective_func(self):
        problem_type = self.config['problem_type']
        if problem_type == 'logistic':
            return logistic_objective
        elif problem_type == 'quadratic':
            return quadratic_objective
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

    def _get_regularization_param(self):
         return self.config['l2_regularization_lambda']

    def run(self, n_iterations, X_full=None, y_full=None, f_opt=0.0):
        print("\n--- Running Centralized Synchronous mini-batch SGD ---")
        start_time = time.time()
        objective_func = self._get_objective_func()
        reg_param = self._get_regularization_param()
        self.total_floats_transmitted = 0

        # For each iteration
        for t in range(n_iterations):
            gradients = []
            current_model = self.x_global.copy()
            worker_transmissions_up = 0

            # For each worker, compute the gradient
            for worker in self.workers:
                grad = worker.compute_gradient(model_params=current_model)
                gradients.append(grad)
                worker_transmissions_up += self.n_features

            # Calculate the average gradient at central server
            avg_gradient = np.mean(gradients, axis=0)
            eta_t = self._get_learning_rate(t)

            # Update the global model
            self.x_global = self.x_global - eta_t * avg_gradient

            # Calculate the data transmitted
            server_transmissions_down = self.n_workers * self.n_features
            self.total_floats_transmitted += (worker_transmissions_up + server_transmissions_down)

            current_time = time.time() - start_time

            # For full data, calculate the objective
            if X_full is not None and y_full is not None:
                obj_val = objective_func(self.x_global, X_full, y_full, reg_param)
                suboptimality = obj_val - f_opt
                self.history['objective'].append(suboptimality)

            self.history['time'].append(current_time)

        print(f"C-SGD training finished. Time: {time.time() - start_time:.2f}sseconds")
        return self.history, self.x_global

class DecentralizedTrainer:
    def __init__(self, workers, topology, n_features, config):
        self.workers = workers
        self.n_workers = len(workers)
        self.topology = topology
        self.n_features = n_features
        self.config = config
        self.adj = None
        self.degrees = None
        self.W = self._create_mixing_matrix()

        print(f"\n--- Running Decentralized SGD ({self.topology} - {self.config['problem_type']}) ---")
        self.history = {'objective': [], 'consensus_error': [], 'time': []}
        self.total_floats_transmitted = 0

    def _create_mixing_matrix(self):
        """Metropolis-Hastings mixing matrix"""
        adj = np.zeros((self.n_workers, self.n_workers))
        degrees = np.zeros(self.n_workers)
        if self.topology == 'ring':
            for i in range(self.n_workers):
                adj[i, (i + 1) % self.n_workers] = 1
                adj[i, (i - 1 + self.n_workers) % self.n_workers] = 1
        elif self.topology == 'grid':
             side = int(np.sqrt(self.n_workers))
             if side * side != self.n_workers:
                 raise ValueError(f"Warning: N_WORKERS ({self.n_workers}) is not a perfect square.")
             G = nx.grid_2d_graph(side, side, periodic=True)
             node_map = {node: i for i, node in enumerate(sorted(G.nodes()))}
             for u, v in G.edges():
                  u_idx, v_idx = node_map[u], node_map[v]
                  adj[u_idx, v_idx] = 1
                  adj[v_idx, u_idx] = 1
        elif self.topology == 'fully_connected':
             adj = np.ones((self.n_workers, self.n_workers)) - np.eye(self.n_workers)
        else:
             raise ValueError(f"Wrong topology: {self.topology}")

        degrees = np.sum(adj, axis=1)
        self.adj = adj
        self.degrees = degrees

        W = np.zeros((self.n_workers, self.n_workers))
        for i in range(self.n_workers):
            neighbors = np.where(adj[i, :] > 0)[0]
            degree_i = degrees[i]
            for j in neighbors:
                if i != j:
                    degree_j = degrees[j]
                    W[i, j] = 1.0 / (1.0 + max(degree_i, degree_j))
            W[i, i] = 1.0 - np.sum(W[i, neighbors])

        # Sanity checks for mixing matrix assumptions
        if self.n_workers > 0:
            assert np.allclose(np.sum(W, axis=1), 1.0), f"Rows of W do not sum to 1 (Topology: {self.topology})"
            assert np.allclose(W, W.T), f"W is not symmetric (Topology: {self.topology})"
            if self.n_workers > 1:
                eigenvalues = np.linalg.eigvalsh(W)
                rho = np.sort(np.abs(eigenvalues))[-2]
                print(f"Mixing Matrix Spectral gap (1 - rho): {1 - rho:.4f} for topology: {self.topology}")
        return W

    def _get_learning_rate(self, t):
        eta0 = self.config['learning_rate_eta0']
        return eta0 / np.sqrt(t + 1)

    def _get_objective_func(self):
        problem_type = self.config['problem_type']
        if problem_type == 'logistic':
            return logistic_objective
        elif problem_type == 'quadratic':
            return quadratic_objective
        else:
            raise NotImplementedError(f"Wrong {problem_type}")

    def _get_regularization_param(self):
         return self.config['l2_regularization_lambda']

    def run(self, n_iterations, X_full=None, y_full=None, f_opt=0.0):
        start_time = time.time()
        objective_func = self._get_objective_func()
        reg_param = self._get_regularization_param()
        self.total_floats_transmitted = 0

        # For each iteration
        for t in range(n_iterations):
            models_t_list = [worker.x for worker in self.workers]
            models_t = np.array(models_t_list)

            # Compute gradients based on local models and data
            gradients_t_list = [worker.compute_gradient() for worker in self.workers]
            gradients_t = np.array(gradients_t_list)

            iteration_transmission = np.sum(self.degrees) * self.n_features
            self.total_floats_transmitted += iteration_transmission

            # Update models using the mixing matrix
            mixed_models_t = self.W @ models_t
            eta_t = self._get_learning_rate(t)
            new_models = mixed_models_t - eta_t * gradients_t

            # Update each worker's model
            for i, worker in enumerate(self.workers):
                worker.x = new_models[i, :]

            current_time = time.time() - start_time
            avg_model = np.mean(new_models, axis=0)

            # Calculate consensus error
            consensus_error = np.mean([np.linalg.norm(worker.x - avg_model)**2 for worker in self.workers])
            self.history['consensus_error'].append(consensus_error)

            if X_full is not None and y_full is not None:
                obj_val = objective_func(avg_model, X_full, y_full, reg_param)
                suboptimality = obj_val - f_opt
                self.history['objective'].append(suboptimality)

            self.history['time'].append(current_time)

        print(f"Decentralized ({self.topology}) training finished. Time: {time.time() - start_time:.2f}s")
        final_avg_model = np.mean([worker.x for worker in self.workers], axis=0)
        return self.history, final_avg_model