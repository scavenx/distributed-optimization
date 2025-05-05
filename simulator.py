import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.linear_model import Ridge as SklearnRidge

from utils import generate_and_preprocess_data
from worker import Worker
from trainer import CentralizedTrainer, DecentralizedTrainer
from obj_problems import logistic_objective, quadratic_objective

class Simulator:
    def __init__(self, config):
        self.config = config
        self.worker_data, self.n_features, self.X_full, self.y_full = \
            generate_and_preprocess_data(config['n_workers'], config)
        self.workers = self._create_workers()
        self.f_opt = self._compute_reference_optimum()
        self.results = {}
        self.numerical_results = {}

    def _create_workers(self):
        return [Worker(i, self.worker_data[i],
                       self.config['local_batch_size'],
                       self.n_features,
                       self.config)
                for i in range(self.config['n_workers'])]

    def _reset_workers(self):
        self.workers = self._create_workers()

    def _compute_reference_optimum(self):
        problem_type = self.config['problem_type']
        lambda_reg = self.config['l2_regularization_lambda']
        reg_param = lambda_reg
        objective_func = logistic_objective if problem_type == 'logistic' else quadratic_objective
        max_iter_ref = 5000
        tol_ref = 1e-9

        X_no_bias = self.X_full[:, :-1]
        y_target = self.y_full
        n_samples = self.X_full.shape[0]

        # Compute the reference optimum, for suboptimality calculation
        if problem_type == 'logistic':
             sklearn_alpha = lambda_reg * n_samples
             C_param = 1.0/sklearn_alpha if sklearn_alpha > 1e-12 else 1e12
             solver = SklearnLogisticRegression(
                 penalty='l2', C=C_param, fit_intercept=True, solver='saga',
                 max_iter=max_iter_ref, tol=tol_ref, random_state=42)
             solver.fit(X_no_bias, y_target)
             w_opt_nobias = solver.coef_.flatten()
             intercept_opt = solver.intercept_
             w_opt = np.concatenate([w_opt_nobias, intercept_opt])
        elif problem_type == 'quadratic':
             sklearn_alpha = reg_param * n_samples
             solver = SklearnRidge(
                 alpha=sklearn_alpha, fit_intercept=True, solver='saga',
                 max_iter=max_iter_ref, tol=tol_ref, random_state=42)
             solver.fit(X_no_bias, y_target)
             w_opt_nobias = solver.coef_.flatten()
             intercept_opt = solver.intercept_
             w_opt = np.concatenate([w_opt_nobias, [intercept_opt]])
        else:
             raise ValueError("Unknown problem type")

        f_opt_val = objective_func(w_opt, self.X_full, self.y_full, reg_param)
        print(f"Ref f(x*) calculated: {f_opt_val:.6f}")
        return f_opt_val

    def _record_numerical_results(self, label, history, trainer):
        threshold = self.config.get('suboptimality_threshold', 0.05)
        objective_history = np.array(history.get('objective', []))
        iters_to_threshold = -1
        # Check if objective history is not empty, then find the first index where it is below the threshold
        if len(objective_history) > 0:
            reached_indices = np.where(objective_history <= threshold)[0]
            if len(reached_indices) > 0:
                iters_to_threshold = reached_indices[0] + 1

        total_transmission = getattr(trainer, 'total_floats_transmitted', 0)
        avg_worker_transmission = 0
        n_workers_effective = self.config['n_workers']
        # To prevent division by zero
        if n_workers_effective > 0:
             avg_worker_transmission = total_transmission / n_workers_effective

        self.numerical_results[label] = {
            'iterations_to_threshold': iters_to_threshold,
            'total_transmission_floats': total_transmission,
            'avg_worker_transmission_floats': avg_worker_transmission
        }

    def run_all(self):
        print(f"\n=== Starting Simulation: {self.config['problem_type']} ===")
        n_iterations = self.config['n_iterations']

        # --- Centralized ---
        self._reset_workers()
        trainer_cent = CentralizedTrainer(self.workers, self.n_features, self.config)
        hist_cent, _ = trainer_cent.run(n_iterations, self.X_full, self.y_full, self.f_opt)
        self.results['Centralized'] = hist_cent
        self._record_numerical_results('Centralized', hist_cent, trainer_cent)

        # --- Decentralized: Ring ---
        self._reset_workers()
        trainer_d_ring = DecentralizedTrainer(self.workers, 'ring', self.n_features, self.config)
        hist_d_ring, _ = trainer_d_ring.run(n_iterations, self.X_full, self.y_full, self.f_opt)
        self.results['D-SGD (Ring)'] = hist_d_ring
        self._record_numerical_results('D-SGD (Ring)', hist_d_ring, trainer_d_ring)

        # --- Decentralized: Grid ---
        is_perfect_square = int(np.sqrt(self.config['n_workers']))**2 == self.config['n_workers']
        if is_perfect_square and self.config['n_workers'] > 0:
            self._reset_workers()
            trainer_d_grid = DecentralizedTrainer(self.workers, 'grid', self.n_features, self.config)
            actual_topology_label = f"D-SGD ({trainer_d_grid.topology.capitalize()})"
            hist_d_grid, _ = trainer_d_grid.run(n_iterations, self.X_full, self.y_full, self.f_opt)
            self.results[actual_topology_label] = hist_d_grid
            self._record_numerical_results(actual_topology_label, hist_d_grid, trainer_d_grid)
        else:
            print("\nSkipping Grid topology: N_WORKERS is not perfect square")
            self.numerical_results['D-SGD (Grid)'] = {
                'iterations_to_threshold': 'N/A', 'total_transmission_floats': 'N/A',
                'avg_worker_transmission_floats': 'N/A'}

        # --- Decentralized: Fully Connected ---
        self._reset_workers()
        trainer_d_fc = DecentralizedTrainer(self.workers, 'fully_connected', self.n_features, self.config)
        hist_d_fc, _ = trainer_d_fc.run(n_iterations, self.X_full, self.y_full, self.f_opt)
        self.results['D-SGD (Fully Connected)'] = hist_d_fc
        self._record_numerical_results('D-SGD (Fully Connected)', hist_d_fc, trainer_d_fc)



        print("\n=== Simulation Finished ===")
        self.report_numerical_results()

    def report_numerical_results(self):
        print("\n--- Numerical Results ---")
        threshold = self.config.get('suboptimality_threshold', 0.07)
        print(f"Target Suboptimality Gap Threshold: {threshold}")
        sorted_labels = sorted(self.numerical_results.keys(), key=lambda x: (not x.startswith('Centralized'), x))
        print(f"\nIterations to reach suboptimality gap <= {threshold}:")
        max_label_len = max(len(label) for label in sorted_labels) + 2 if sorted_labels else 2
        for label in sorted_labels:
             data = self.numerical_results[label]
             iters = data['iterations_to_threshold']
             if iters == 'N/A': print(f"  {label:<{max_label_len}}: N/A")
             elif iters == -1: print(f"  {label:<{max_label_len}}: > {self.config['n_iterations']} , threshold not reached")
             else: print(f"  {label:<{max_label_len}}: {iters} iterations")

        print(f"\nTotal Data Transmission in floats, over {self.config['n_iterations']} iterations:")
        for label in sorted_labels:
            data = self.numerical_results[label]
            total_tx = data['total_transmission_floats']
            avg_tx = data['avg_worker_transmission_floats']
            if total_tx == 'N/A': print(f"  {label:<{max_label_len}}: Total = N/A, Avg per Worker = N/A")
            else: print(f"  {label:<{max_label_len}}: Total = {total_tx:.3e}, Avg per Worker = {avg_tx:.3e}")

    def plot_results(self):
        iterations = np.arange(1, self.config['n_iterations'] + 1)
        plot_configs = [
             ('objective', f'Suboptimality Gap ($f(\\bar{{x}}_T) - f(x^*)$) - {self.config["problem_type"]}', True),
             ('consensus_error', f'Consensus Error ($(1/N) \sum ||x_{{i,T}} - \\bar{{x}}_T||^2$) - {self.config["problem_type"]}', True),]
        num_plots = len(plot_configs)
        plt.figure(figsize=(7 * num_plots, 6))

        for plot_idx, (metric_key, title, use_log_scale) in enumerate(plot_configs, 1):
            ax = plt.subplot(1, num_plots, plot_idx)
            sorted_labels = sorted(self.results.keys(), key=lambda x: (not x.startswith('Centralized'), x))
            for label in sorted_labels:
                 history = self.results.get(label)
                 if history and metric_key in history:
                     metric_data = history[metric_key]
                     # Skip Centralized for consensus error
                     if metric_key == 'consensus_error' and label == 'Centralized': continue
                     if len(metric_data) == self.config['n_iterations']:
                         values_to_plot = np.array(metric_data)
                         # Prevent plot errors for non-finite values
                         if np.any(~np.isfinite(values_to_plot)):
                               print(f"Warning: Non-finite values found in metric '{metric_key}' for '{label}'. Skipping plot line.")
                               continue
                         if use_log_scale:
                             values_to_plot = np.maximum(values_to_plot, 1e-14)
                         ax.plot(iterations, values_to_plot, label=label, lw=2)
                     else:
                         print(f"Warning: Mismatched data length for metric '{metric_key}' in '{label}'. Expected {self.config['n_iterations']}, got {len(metric_data)}. Skipping.")

            ax.set_xlabel('Iteration (T)')
            ax.set_ylabel('Value (log scale)' if use_log_scale else 'Value')
            if use_log_scale: ax.set_yscale('log')
            ax.set_title(title)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.legend()

        plt.figtext(0.5, 0.01,
                    f"Config: N={self.config['n_workers']}, b={self.config['local_batch_size']}, Problem={self.config['problem_type']}, Non-IID Data, LR0={self.config['learning_rate_eta0']} (Sqrt Decay), $\lambda$={self.config['l2_regularization_lambda']}",
                    ha="center", fontsize=10)
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])
        plt.show()