# === Typing ===
from typing import Callable, Dict
import time
import numpy as np
import os
import sys
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Question 2"))
sys.path.append(base_dir)
from Functions_22_Avino_Lombardi import *

import numpy as np
import time
from typing import Callable, Dict

class MVP(SVM):
    """
    MVP (Modified SMO-based SVM classifier).

    This class implements a simplified Sequential Minimal Optimization (SMO) 
    algorithm for training an SVM using custom kernels and a dual optimization objective.
    """

    def __init__(self, C: float, kernel_func: Callable, kernel_params: Dict, tol: float = 1e-3, max_iter: int = 1000):
        """
        Initialize the MVP model.

        Args:
            C: Regularization parameter.
            kernel_func: Kernel function (e.g., polynomial, Gaussian).
            kernel_params: Dictionary of kernel-specific parameters.
            tol: Tolerance for KKT condition violation and early stopping.
            max_iter: Maximum number of training iterations.
        """
        self.C = C
        self.kernel_func = kernel_func
        self.kernel_params = kernel_params
        self.tol = tol
        self.max_iter = max_iter

        self.alpha = None
        self.b = 0.0
        self.errors = None
        self.K = None
        self.Q = None

        self.X_train = None
        self.y_train = None

        self.n_iter_ = 0
        self.last_cpu_time = None
        self.initial_dual_obj = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the SVM model using SMO algorithm with early stopping.

        Args:
            X: Training data matrix of shape (n_samples, n_features).
            y: Target labels of shape (n_samples,). Must be -1 or +1.
        """
        self.X_train = X
        self.y_train = y
        n_samples = X.shape[0]

        self.alpha = np.zeros(n_samples)
        self.b = 0.0
        self.errors = -y.copy()  # f(x) = 0 all'inizio → errori = -y

        # Precompute full kernel matrix
        self.K = self.kernel_func(X, X, **self.kernel_params)

        # Precompute Q matrix used in dual objective
        self.Q = (y[:, None] * y[None, :]) * self.K

        # Compute initial dual objective value (at alpha = 0)
        self.initial_dual_obj = self._initial_dual_objective(n_samples)

        start = time.time()
        it = 0
        entire_set = True
        alpha_prev = np.copy(self.alpha)

        while it < self.max_iter:
            num_changed = 0
            idxs = range(n_samples) if entire_set else np.where((self.alpha > 0) & (self.alpha < self.C))[0]

            for i in idxs:
                E_i = self._error(i)
                if (y[i] * E_i < -self.tol and self.alpha[i] < self.C) or (y[i] * E_i > self.tol and self.alpha[i] > 0):
                    j = self._second_choice(i, n_samples)
                    if self._optimize_pair(i, j):
                        num_changed += 1

            it += 1

            # Adaptive pass strategy
            if entire_set:
                entire_set = False
            elif num_changed == 0:
                entire_set = True

            # Early stopping: check convergence of alpha
            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.tol:
                print(f"[Early Stopping] Iteration {it}, Δalpha = {diff:.6f} < tol = {self.tol}")
                break

            alpha_prev = np.copy(self.alpha)

        end = time.time()
        self.n_iter_ = it
        self.last_cpu_time = end - start
        self.b = self._compute_b()

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input samples.

        Args:
            X: Input samples matrix.

        Returns:
            Predicted labels (-1 or +1).
        """
        K_test = self.kernel_func(X, self.X_train, **self.kernel_params)
        decision = (self.alpha * self.y_train) @ K_test.T + self.b
        return np.sign(decision)

    def _error(self, i: int) -> float:
        """
        Compute the prediction error for the i-th training point.
        """
        f_xi = np.dot(self.alpha * self.y_train, self.K[:, i])
        return f_xi + self.b - self.y_train[i]

    def _second_choice(self, i: int, n_samples: int) -> int:
        """
        Randomly choose a second index j != i.
        """
        j = i
        while j == i:
            j = np.random.randint(0, n_samples)
        return j

    def _optimize_pair(self, i: int, j: int) -> bool:
        """
        Perform SMO optimization step for a pair (i, j).
        """
        if i == j:
            return False

        alpha_i_old = self.alpha[i]
        alpha_j_old = self.alpha[j]
        y_i = self.y_train[i]
        y_j = self.y_train[j]

        K_ii = self.K[i, i]
        K_jj = self.K[j, j]
        K_ij = self.K[i, j]

        eta = K_ii + K_jj - 2 * K_ij
        if eta <= 0:
            return False

        E_i = self._error(i)
        E_j = self._error(j)

        # Compute bounds L and H
        if y_i != y_j:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
        else:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])

        if L == H:
            return False

        # Compute and clip alpha_j
        alpha_j_new = alpha_j_old + y_j * (E_i - E_j) / eta
        alpha_j_new = np.clip(alpha_j_new, L, H)

        if np.abs(alpha_j_new - alpha_j_old) < 1e-5:
            return False

        # Compute alpha_i
        alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new)

        # Update alphas
        self.alpha[i] = alpha_i_new
        self.alpha[j] = alpha_j_new

        # Update bias
        b1 = self.b - E_i - y_i * (alpha_i_new - alpha_i_old) * K_ii - y_j * (alpha_j_new - alpha_j_old) * K_ij
        b2 = self.b - E_j - y_i * (alpha_i_new - alpha_i_old) * K_ij - y_j * (alpha_j_new - alpha_j_old) * K_jj

        if 0 < alpha_i_new < self.C:
            self.b = b1
        elif 0 < alpha_j_new < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2

        return True

    def _compute_b(self) -> float:
        """
        Compute bias term b using support vectors.
        """
        support_indices = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
        b_values = [
            self.y_train[i] - np.sum(self.alpha * self.y_train * self.K[:, i])
            for i in support_indices
        ]
        return np.mean(b_values) if b_values else 0.0

    def _initial_dual_objective(self, n: int) -> float:
        """
        Compute the initial value of the dual objective function.
        """
        init_alpha = np.zeros(n)
        return -np.sum(init_alpha) + 0.5 * init_alpha.T @ self.Q @ init_alpha

    def dual_objective(self) -> float:
        """
        Compute the final value of the dual objective function.
        """
        return -np.sum(self.alpha) + 0.5 * self.alpha.T @ self.Q @ self.alpha
