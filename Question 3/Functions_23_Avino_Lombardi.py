from typing import Callable, Dict, Union
import time
import numpy as np
from functools import reduce
import os, sys
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Question 2"))
sys.path.append(base_dir)
from Functions_22_Avino_Lombardi import *

class MVP:
    """
    MVP (Most Violating Pair SVM using SMO).

    Implements a Sequential Minimal Optimization (SMO) algorithm to train
    a SVM by choosing the most violating pair (i, j) at each step,
    following a KKT-based selection strategy and checking convergence
    using m and M values.
    """

    # Fix: Changed _init_ to __init__
    def __init__(self, C: float, kernel_func: Callable, kernel_params: Dict, tol: float = 1e-3, max_iter: int = 1000):
        self.C = C
        self.kernel_func = kernel_func
        self.kernel_params = kernel_params
        self.tol = tol # for KKT conditions and convergence
        self.max_iter = max_iter

        self.alpha: np.ndarray | None = None # Alpha multipliers
        self.b: float = 0.0 # Bias term
        # self.gradient: Stores y_k * E_k where E_k = f(x_k) - y_k
        # This is G in the svm_train_mvp code, which is y_k * (sum_l(alpha_l * y_l * K_kl) + b - y_k)
        # If f(x_k) does not include b, then G_k = y_k * sum_l(alpha_l * y_l * K_kl) - 1.
        # The update rule `G += delta_alpha_i * Q[:, i] + delta_alpha_j * Q[:, j]`
        # implies G is directly related to Q @ alpha - 1 or similar gradient forms.
        self.gradient: np.ndarray | None = None

        self.K: np.ndarray | None = None # Kernel matrix (K(x_i, x_j))
        self.Q: np.ndarray | None = None # Q matrix (y_i y_j K_ij)

        self.X_train = None
        self.y_train = None

        self.n_iter_ = 0
        self.last_cpu_time = None
        self.initial_dual_obj = None # Initial dual objective value
        self.final_M_minus_m = None # For final m - M value

        # Support vectors (for more efficient prediction in predict method)
        self.sv_x: np.ndarray | None = None
        self.sv_y: np.ndarray | None = None
        self.sv_lam: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y
        n_samples = X.shape[0]

        self.alpha = np.zeros(n_samples)
        self.b = 0.0 # Initial bias can be set to 0.0

        # Precompute kernel matrix
        self.K = self.kernel_func(X, X, **self.kernel_params)
        # Q matrix: Q_ab = y_a * y_b * K_ab
        self.Q = (y[:, None] * y[None, :]) * self.K

        # Initialize gradient 
        # G_k = y_k * f(x_k) - 1.
        # Initially, f(x_k) = b = 0 (before any alpha updates).
        # So, y_k * (0) - 1 = -1.
        self.gradient = -np.ones(n_samples)

        self.initial_dual_obj = self.dual_objective()

        start = time.time()
        it = 0

        while it < self.max_iter:
            it += 1

            # Select most violating pair using KKT strategy
            i, j, m_val, M_val = self._select_mvp_pair()

            # Check stopping condition based on m and M
            # Convergence if M + tol >= m
            if m_val is not None and M_val is not None and (M_val + self.tol >= m_val):
                print(f"[Converged via KKT gap] Iteration {it}, m-M = {m_val - M_val:.6f} < tol = {self.tol}")
                self.final_M_minus_m = m_val - M_val
                break

            # If no valid pair was found by _select_mvp_pair (e.g., R or S empty, or numerical issues)
            if i is None or j is None:
                print(f"[Early Stopping] Iteration {it}, no violating pair found that allows progress.")
                # Recalculate m and M for final reporting before breaking
                _, _, final_m_val, final_M_val = self._select_mvp_pair(final_check=True)
                self.final_M_minus_m = final_m_val - final_M_val
                break

            success = self._optimize_pair(i, j)

            if not success:
                continue

        end = time.time()
        self.n_iter_ = it
        self.last_cpu_time = end - start

        # Update final M-m if loop finished by max_iter without explicit convergence break
        if self.final_M_minus_m is None:
            _, _, final_m_val, final_M_val = self._select_mvp_pair(final_check=True)
            self.final_M_minus_m = final_m_val - final_M_val

        # Compute final bias from current alpha values for robustness
        self.b = self._compute_b()

        # Store support vectors for faster prediction
        support_vector_indices = np.where(self.alpha > self.tol)[0] # alpha > 0
        self.sv_x = self.X_train[support_vector_indices]
        self.sv_y = self.y_train[support_vector_indices]
        self.sv_lam = self.alpha[support_vector_indices]


    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Predicts class labels for new data.
        '''
        if self.alpha is None or self.X_train is None or self.y_train is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        if self.sv_x is None or len(self.sv_x) == 0:
            # If self.b is 0, predict 1. Otherwise, sign(self.b).
            return np.full(X.shape[0], np.sign(self.b) if self.b != 0 else 1.0)

        # Use only support vectors for prediction
        # K_test should be (n_test_samples, n_sv)
        K_test = self.kernel_func(X, self.sv_x, **self.kernel_params)
        # decision = (sum_{sv} alpha_sv * y_sv * K(x_new, x_sv)) + b
        decision = (self.sv_lam * self.sv_y) @ K_test.T + self.b # (n_test_samples,)

        return np.sign(decision)

    # _error method is not directly used, but can be kept as a utility.
    def _error(self, k: int) -> float:
        """
        Compute the prediction error for the k-th training point.
        f(x_k) = sum_l (alpha_l * y_l * K(x_l, x_k)) + b
        E_k = f(x_k) - y_k
        """
        f_xk = np.dot(self.alpha * self.y_train, self.K[:, k]) + self.b
        return f_xk - self.y_train[k]


    def _select_mvp_pair(self, final_check=False) -> Tuple[Union[int, None], Union[int, None], Union[float, None], Union[float, None]]:
        """
        Select the most violating pair (i, j) according to the KKT-based strategy
        from svm_train_mvp. Also computes m and M for convergence checking.

        Args:
            final_check (bool): If True, only computes and returns m and M without selecting i,j.

        Returns:
            A tuple (i, j, m_val, M_val).
            (None, None, m_val, M_val) if convergence is met or no valid pair found.
        """
        # Define R and S sets
        # R_indices: points where alpha_k can potentially be increased if y_k=1 (alpha_k=0)
        # or decreased if y_k=-1 (alpha_k=C), or points in (0, C).
        
        R_mask = ( (self.alpha > 0) & (self.alpha < self.C) ) | \
                ( (self.alpha == 0) & (self.y_train == 1) ) | \
                ( (self.alpha == self.C) & (self.y_train == -1) )
        R_indices = np.where(R_mask)[0]

        S_mask = ( (self.alpha > 0) & (self.alpha < self.C) ) | \
                ( (self.alpha == 0) & (self.y_train == -1) ) | \
                ( (self.alpha == self.C) & (self.y_train == 1) )
        S_indices = np.where(S_mask)[0]

        # If either set is empty, no valid pair can be formed
        if R_indices.size == 0 or S_indices.size == 0:
            # If no valid indices, calculate theoretical m/M bounds for reporting convergence
            m_val = -np.inf # If R_indices is empty, max over empty set is -inf
            M_val = np.inf  # If S_indices is empty, min over empty set is +inf
            return None, None, m_val, M_val

        # Calculate KKT violation terms: G[k] / -y[k]
        G_over_minus_y_R = [self.gradient[k] / -self.y_train[k] for k in R_indices]
        G_over_minus_y_S = [self.gradient[k] / -self.y_train[k] for k in S_indices]

        # Calculate m and M as defined in svm_train_mvp
        m_val = max(G_over_minus_y_R)
        M_val = min(G_over_minus_y_S)

        if final_check:
            return None, None, m_val, M_val

        # Check stopping condition: M + tol >= m
        if M_val + self.tol >= m_val:
            return None, None, m_val, M_val # Converged

        # Select i and j:
        # i is most violating in R
        i_idx_in_R = np.argmax(G_over_minus_y_R)
        i = R_indices[i_idx_in_R]

        # j is most violating in S
        j_idx_in_S = np.argmin(G_over_minus_y_S)
        j = S_indices[j_idx_in_S]

        # --- Additional checks for optimizability of the selected pair (i,j) ---
        # These checks help prevent infinite loops due to numerical issues
        # where a pair is selected but cannot actually make progress.
        alpha_i_old = self.alpha[i]
        alpha_j_old = self.alpha[j]
        y_i = self.y_train[i]
        y_j = self.y_train[j]

        # eta calculation 
        eta = self.K[i, i] + self.K[j, j] - 2 * self.K[i, j]

        if eta <= 1e-9: # If eta is too small, a meaningful step is not possible
            return None, None, m_val, M_val

        # Compute bounds L and H
        if y_i != y_j:
            L = max(0, alpha_j_old - alpha_i_old)
            H = min(self.C, self.C + alpha_j_old - alpha_i_old)
        else:
            L = max(0, alpha_i_old + alpha_j_old - self.C)
            H = min(self.C, alpha_i_old + alpha_j_old)

        if L == H: # If bounds are the same, no valid step is possible
            return None, None, m_val, M_val

        # Check if alpha_j would actually move after clipping
        E_i = self.gradient[i] / y_i # E_k = self.gradient[k] / y_k
        E_j = self.gradient[j] / y_j
        potential_alpha_j_new_unclipped = alpha_j_old + y_j * (E_i - E_j) / eta
        potential_alpha_j_new = np.clip(potential_alpha_j_new_unclipped, L, H)

        if np.abs(potential_alpha_j_new - alpha_j_old) < 1e-5: # If change is too small
            return None, None, m_val, M_val

        return i, j, m_val, M_val


    def _optimize_pair(self, i: int, j: int) -> bool:
        """
        Perform the SMO optimization step for a selected pair (i, j) and update the gradient.
        """
        if i == j: # Should not happen if _select_mvp_pair is working correctly
            return False

        alpha_i_old = self.alpha[i]
        alpha_j_old = self.alpha[j]
        y_i = self.y_train[i]
        y_j = self.y_train[j]

        K_ii = self.K[i, i]
        K_jj = self.K[j, j]
        K_ij = self.K[i, j]

        eta = K_ii + K_jj - 2 * K_ij

        if eta <= 1e-9: # Using a small positive tolerance
            return False

        E_i = self.gradient[i] / y_i
        E_j = self.gradient[j] / y_j

        # Compute bounds L and H (standard SMO bounds)
        if y_i != y_j:
            L = max(0, alpha_j_old - alpha_i_old)
            H = min(self.C, self.C + alpha_j_old - alpha_i_old)
        else:
            L = max(0, alpha_i_old + alpha_j_old - self.C)
            H = min(self.C, alpha_i_old + alpha_j_old)

        if L == H: # If bounds are the same, no valid step is possible
            return False

        # Compute and clip alpha_j_new
        alpha_j_new_unclipped = alpha_j_old + y_j * (E_i - E_j) / eta
        alpha_j_new = np.clip(alpha_j_new_unclipped, L, H)

        # Check for significant change in alpha_j
        if np.abs(alpha_j_new - alpha_j_old) < 1e-5: # Use a reasonable threshold
            return False

        # Compute alpha_i_new based on alpha_j_new (standard SMO update)
        alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new)

        # Update alphas in the model
        self.alpha[i] = alpha_i_new
        self.alpha[j] = alpha_j_new

        # Update the gradient incrementally
        # G_new = G_old + (alpha_i_new - alpha_i_old) * Q[:, i] + (alpha_j_new - alpha_j_old) * Q[:, j]
        # Q[:, i] is the i-th column of the Q matrix (y_k y_i K_ki for k=0..N-1)
        delta_alpha_i = alpha_i_new - alpha_i_old
        delta_alpha_j = alpha_j_new - alpha_j_old

        self.gradient += delta_alpha_i * self.Q[:, i] + delta_alpha_j * self.Q[:, j]

        # Update bias (self.b) incrementally based on new alphas and gradient (errors)
        # These are derived from the KKT condition y*f(x) - 1 = 0 for free SVs
        b1 = self.b - E_i - y_i * delta_alpha_i * K_ii - y_j * delta_alpha_j * K_ij
        b2 = self.b - E_j - y_i * delta_alpha_i * K_ij - y_j * delta_alpha_j * K_jj

        # Update self.b if one of the updated alphas is a free support vector
        if 0 < self.alpha[i] < self.C:
            self.b = b1
        elif 0 < self.alpha[j] < self.C:
            self.b = b2
        else:
            # If both alphas are at the bounds, take the average (common heuristic)
            self.b = (b1 + b2) / 2.0

        return True

    def _compute_b(self) -> float:
        """
        Compute bias term b using free support vectors (0 < alpha < C).
        """
        # Find indices of non-bound support vectors (0 < alpha < C)
        # Using a small tolerance to account for floating point inaccuracies
        support_indices = np.where((self.alpha > self.tol) & (self.alpha < self.C - self.tol))[0]

        # Precompute alpha_l * y_l, as it's a common term
        alpha_y_product = self.alpha * self.y_train

        if len(support_indices) == 0:
            # If no free support vectors are found, fall back to all support vectors (alpha > tol)
            all_sv_indices = np.where(self.alpha > self.tol)[0]

            if len(all_sv_indices) > 0:
                # Vectorized computation of f_xk for all support vectors
                # K[:, all_sv_indices] selects only the columns corresponding to support vectors
                # The dot product (matrix multiplication) computes f_xk for all of them at once
                f_xk_values = np.dot(alpha_y_product, self.K[:, all_sv_indices])
                
                # Compute b_values = y_k - f(x_k) for these selected support vectors
                b_values = self.y_train[all_sv_indices] - f_xk_values
                
                return np.mean(b_values) # Calculate the mean of these b values
            else:
                return 0.0 # Fallback if no support vectors at all
        else:
            # Vectorized computation of f_xk for free support vectors
            f_xk_values = np.dot(alpha_y_product, self.K[:, support_indices])
            
            # Compute b_values = y_k - f(x_k) for these free support vectors
            b_values = self.y_train[support_indices] - f_xk_values
            
            return np.mean(b_values) # Calculate the mean of these b values


    def dual_objective(self) -> float:
        """
        Compute the current value of the dual objective function:
        sum(alpha) - 0.5 * alpha.T @ Q @ alpha
        Note: The provided example uses -sum(alpha) + 0.5 * alpha.T @ Q @ alpha,
        which is the negative of the standard dual objective (to be minimized).
        Keeping that convention for consistency with previous user code.
        """
        if self.alpha is None or self.Q is None:
            return 0.0 # Or raise an error if called before fit

        return -np.sum(self.alpha) + 0.5 * self.alpha @ self.Q @ self.alpha