# ===========================================
# QUESTION 2
# ===========================================
import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from typing import Callable, Dict, List, Tuple
from itertools import product
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

# ===========================================
#           KERNEL FUNCTIONS
# ===========================================

def gaussian_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute the Gaussian (RBF) kernel between two sets of vectors.

    Args:
        X1: First input matrix of shape (n1, d).
        X2: Second input matrix of shape (n2, d).
        gamma: Bandwidth parameter of the RBF kernel.

    Returns:
        Kernel matrix of shape (n1, n2).
    """
    # Compute squared Euclidean distances
    X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
    X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
    dists_squared = X1_norm + X2_norm - 2 * (X1 @ X2.T)
    return np.exp(-gamma * dists_squared)

def polynomial_kernel(X1: np.ndarray, X2: np.ndarray, p: int) -> np.ndarray:
    """
    Compute the Polynomial kernel between two sets of vectors.

    Args:
        X1: First input matrix of shape (n1, d).
        X2: Second input matrix of shape (n2, d).
        p: Degree of the polynomial.

    Returns:
        Kernel matrix of shape (n1, n2).
    """
    return (X1 @ X2.T + 1) ** p

# ===========================================
#           EVALUATION METRICS
# ===========================================

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Accuracy as a float.
    """
    return np.mean(y_true == y_pred)

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Compute confusion matrix as a labeled DataFrame.

    Args:
        y_true: Ground truth labels (-1 or +1).
        y_pred: Predicted labels (-1 or +1).

    Returns:
        Confusion matrix in DataFrame format.
    """
    labels = [-1, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"True {l}" for l in labels], columns=[f"Pred {l}" for l in labels])

# ===========================================
#              SVM CLASS
# ===========================================

class SVM:
    """
    Support Vector Machine (SVM) classifier using CVXOPT to solve the QP problem.
    """

    def __init__(self, C: float, kernel_func: Callable, kernel_params: Dict):
        """
        Initialize SVM with hyperparameters.

        Args:
            C: Regularization parameter.
            kernel_func: Kernel function.
            kernel_params: Parameters for the kernel function.
        """
        self.C = C
        self.kernel_func = kernel_func
        self.kernel_params = kernel_params
        self.X_train = None
        self.y_train = None
        self.lambdas = None
        self.Q = None
        self.initial_dual_obj = None
        self.b = 0.0
        self.last_num_iter = None
        self.last_cpu_time = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the SVM model to training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Labels array of shape (n_samples,), with values in {-1, +1}.
        """
        self.X_train = X
        self.y_train = y
        self.Q = self._compute_Q(X, y)
        self.lambdas = self._solve_dual(self.Q, y)
        self.b = self._compute_b()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input samples.

        Args:
            X: Input samples matrix.

        Returns:
            Predicted labels (-1 or +1).
        """
        return np.sign(self.decision_function(X))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function values.

        Args:
            X: Input data.

        Returns:
            Decision values.
        """
        K = self.kernel_func(X, self.X_train, **self.kernel_params)
        return K @ (self.lambdas * self.y_train) + self.b

    def _compute_Q(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the Q matrix for the dual problem.

        Args:
            X: Feature matrix.
            y: Labels array.

        Returns:
            Q matrix.
        """
        K = self.kernel_func(X, X, **self.kernel_params)
        return (y[:, None] * y[None, :]) * K

    def _solve_dual(self, Q: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Solve the quadratic programming dual problem using CVXOPT.

        Args:
            Q: Q matrix from dual formulation.
            y: Labels.

        Returns:
            Solution lambdas as a 1D array.
        """
        from time import perf_counter

        n = y.size
        self.initial_dual_obj = self.initial_dual_objective(n)

        P = matrix(Q, tc='d')
        q = matrix(-np.ones(n), tc='d')
        G = matrix(np.vstack([-np.eye(n), np.eye(n)]), tc='d')
        h = matrix(np.hstack([np.zeros(n), self.C * np.ones(n)]), tc='d')
        A = matrix(y.reshape(1, -1), tc='d')
        b = matrix(0.0)

        solvers.options['show_progress'] = False

        start = perf_counter()
        solution = solvers.qp(P, q, G, h, A, b, initvals={'x': matrix(np.zeros(n))})
        end = perf_counter()

        self.last_num_iter = solution.get('iterations', None)
        self.last_cpu_time = end - start

        return np.ravel(solution['x'])

    def _compute_b(self) -> float:
        """
        Compute the bias term 'b' using support vectors on the margin.

        Returns:
            Scalar bias term.
        """
        is_margin_sv = (self.lambdas > 0) & (self.lambdas < self.C)
        idx = np.where(is_margin_sv)[0]

        if len(idx) == 0:
            raise ValueError("No margin support vectors found to compute b.")

        X_sv = self.X_train[idx]
        y_sv = self.y_train[idx]
        K = self.kernel_func(self.X_train, X_sv, **self.kernel_params)
        weighted_K = (self.lambdas * self.y_train)[:, None] * K
        decision_part = np.sum(weighted_K, axis=0)
        b_values = y_sv - decision_part
        return np.mean(b_values)

    def initial_dual_objective(self, n: int) -> float:
        init_lambdas = np.zeros(n)
        return -np.sum(init_lambdas) + 0.5 * init_lambdas.T @ self.Q @ init_lambdas

    def dual_objective(self) -> float:
        """
        Compute the value of the dual objective function.

        Returns:
            Dual objective value.
        """
        return -np.sum(self.lambdas) + 0.5 * self.lambdas.T @ self.Q @ self.lambdas

# ===========================================
#         CROSS-VALIDATION ROUTINE
# ===========================================

def crossval_svm(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    grid: Dict[str, List[float]],
    kernel_func: Callable,
    kernel_param_name: str,
    seed: int = 42,
    use_seed: bool = False
) -> Tuple[List[Dict[str, float]], List[float]]:
    """
    Perform k-fold cross-validation to tune SVM hyperparameters.

    Args:
        X: Input features.
        y: Labels.
        k: Number of folds.
        grid: Dictionary of hyperparameter values to try.
        kernel_func: Kernel function.
        kernel_param_name: Name of kernel parameter in the grid.
        seed: Random seed.
        use_seed: Whether to shuffle using the seed.

    Returns:
        List of hyperparameter dicts and their average validation scores.
    """
    param_list = [dict(zip(grid, vals)) for vals in product(*grid.values())]  # Create all combinations of grid
    avg_scores = []

    kf = KFold(n_splits=k, shuffle=use_seed, random_state=seed if use_seed else None)

    for param in param_list:
        C = param["C"]
        kernel_param = param[kernel_param_name]
        acc = []

        for train_idx, val_idx in kf.split(X):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            model = SVM(C=C, kernel_func=kernel_func, kernel_params={kernel_param_name: kernel_param})
            np.random.seed(seed)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            acc.append(compute_accuracy(y_val, y_pred))

        avg_scores.append(np.mean(acc))

    return param_list, avg_scores
