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





'''
import numpy as np
from cvxopt import matrix, solvers
import time
from typing import Callable, List, Tuple, Dict
from itertools import product

# ==========================
# Gaussian (RBF) Kernel
# ==========================
def gaussian_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    """
    Fully vectorized RBF kernel: returns matrix of shape (X1.shape[0], X2.shape[0])
    """
    X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
    X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
    dists_squared = X1_norm + X2_norm - 2 * (X1 @ X2.T)
    return np.exp(-gamma * dists_squared)

# ==========================
# Polynomial Kernel
# ==========================
def polynomial_kernel(X1: np.ndarray, X2: np.ndarray, p: int) -> np.ndarray:
    """
    Fully vectorized polynomial kernel: returns matrix of shape (X1.shape[0], X2.shape[0])
    """
    return (X1 @ X2.T + 1) ** p

# ==========================
# Kernel Matrix with Labels
# ==========================
def compute_kernel_matrix(
    X: np.ndarray,
    Y: np.ndarray,
    kernel_func: Callable,
    **kwargs
) -> np.ndarray:
    """
    Compute the matrix Q = y_i y_j K(x_i, x_j) for the dual SVM formulation,
    assuming kernel_func supports broadcasting over matrices (vectorized form).

    Args:
        X: Training data of shape (n_samples, n_features)
        Y: Labels in {-1, +1}, shape (n_samples,)
        kernel_func: Kernel function supporting full matrix input (X, X)
        kwargs: Parameters to pass to the kernel function (e.g., gamma, p)

    Returns:
        Q: Dual SVM matrix of shape (n_samples, n_samples)
    """
    # Kernel matrix: shape (n_samples, n_samples)
    K = kernel_func(X, X, **kwargs)

    # Apply label outer product: Q[i,j] = y_i * y_j * K[i,j]
    Q = (Y[:, None] * Y[None, :]) * K
    return Q

# ==========================
# Solve Dual SVM with CVXOPT
# ==========================

def solve_dual_svm(Q: np.ndarray, labels: np.ndarray, C: float):
    """
    Solves the dual SVM quadratic problem using CVXOPT:

        minimize:   -e^T λ + 1/2 λ^T Q λ
        subject to: 0 ≤ λ_i ≤ C, and sum_i λ_i * y_i = 0

    Args:
        Q: (n_samples, n_samples) matrix with Q_ij = y_i y_j K(x_i, x_j)
        labels: (n_samples,) array with values in {-1, +1}
        C: regularization parameter

    Returns:
        lambdas: Optimal dual variables λ (n_samples,)
        num_iter: Number of solver iterations
        cpu_time: Time taken to solve the QP
        solution: Full CVXOPT solution object
    """
    from cvxopt import matrix, solvers

    n = labels.size

    # Convert numpy arrays to CVXOPT matrices (efficiently)
    P = matrix(Q, tc='d')
    q = matrix(-np.ones(n), tc='d')

    # Inequality constraints: 0 ≤ λ ≤ C
    G = matrix(np.vstack([-np.eye(n), np.eye(n)]), tc='d')
    h = matrix(np.hstack([np.zeros(n), C * np.ones(n)]), tc='d')

    # Equality constraint: sum_i λ_i * y_i = 0
    A = matrix(labels.reshape(1, -1), tc='d')
    b = matrix(0.0)

    # Disable CVXOPT output for performance
    solvers.options['show_progress'] = False

    # Solve and time
    start = time.perf_counter()
    solution = solvers.qp(P, q, G, h, A, b)
    end = time.perf_counter()

    # Extract solution
    lambdas = np.ravel(solution['x'])  # flatten to 1D
    return lambdas, solution['iterations'], end - start, solution

# ==========================
# Compute Bias Term b
# ==========================
def compute_b(
    X: np.ndarray,
    Y: np.ndarray,
    lambdas: np.ndarray,
    kernel_func,
    C: float,
    tolerance: float = 1e-5,
    **kwargs
) -> float:
    """
    Compute the bias term b using vectorized KKT conditions.

    Args:
        X: Training data (n_samples, n_features)
        Y: Labels in {-1, +1}
        lambdas: Lagrange multipliers (n_samples,)
        kernel_func: Vectorized kernel function accepting (X1, X2, **kwargs)
        C: Box constraint
        tolerance: Numerical threshold for identifying margin SVs
        kwargs: Kernel-specific parameters

    Returns:
        Bias term b (float)
    """
    # Indices of margin support vectors (0 < λ < C)
    #is_margin_sv = (lambdas > tolerance) & (lambdas < C - tolerance)
    is_margin_sv = (lambdas > 0) & (lambdas < C)
    margin_sv_idx = np.where(is_margin_sv)[0]

    if len(margin_sv_idx) == 0:
        raise ValueError("No margin support vectors found to compute b.")

    # Get margin support vectors
    X_sv = X[margin_sv_idx]                   # shape (m, d)
    Y_sv = Y[margin_sv_idx]                   # shape (m,)

    # Compute full kernel matrix between all training points and margin SVs
    # K[j, i] = K(x_j, x_i_sv) for all i in SVs and all j
    K = kernel_func(X, X_sv, **kwargs)        # shape (n_train, n_sv)

    
    # Weighted sum over training set: sum_j λ_j y_j K(x_j, x_i_sv)
    weighted_K = (lambdas * Y)[:, None] * K   # broadcasting (n,1) * (n, m) → (n, m)
    decision_part = np.sum(weighted_K, axis=0)  # shape (m,)

    # Compute b_i = y_i - sum_j λ_j y_j K(x_j, x_i)
    b_values = Y_sv - decision_part

    return np.mean(b_values)

# ==========================
# Prediction Function
# ==========================
import numpy as np

def predict(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    lambdas: np.ndarray,
    b: float,
    X_test: np.ndarray,
    kernel_func,
    tolerance: float = 1e-5,
    **kwargs
) -> np.ndarray:
    """
    Predict labels for new data points using the trained dual SVM,
    in a fully vectorized way.

    Args:
        X_train: Training data (n_samples, n_features)
        Y_train: Binary labels {-1, +1}, shape (n_samples,)
        lambdas: Lagrange multipliers from dual SVM, shape (n_samples,)
        b: Bias term (float)
        X_test: Test data to classify (m_samples, n_features)
        kernel_func: Kernel function that supports matrix input
        tolerance: Threshold for identifying active support vectors
        kwargs: Kernel-specific parameters (e.g., gamma, p)

    Returns:
        y_pred: Predicted labels for X_test, shape (m_samples,)
    """

    X_sv = X_train           # shape: (n_sv, d)
    Y_sv = Y_train           # shape: (n_sv,)
    lambda_sv = lambdas      # shape: (n_sv,)

    # Compute kernel matrix between test set and support vectors
    # Result: (m_test, n_sv)
    K_test = kernel_func(X_test, X_sv, **kwargs)

    # Weighted decision values: ∑ λ_i y_i K(x, x_i) + b
    weighted_contrib = K_test @ (lambda_sv * Y_sv)  # shape: (m_test,)
    y_pred = np.sign(weighted_contrib + b)

    return y_pred

def compute_dual_objective(lambdas: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute the value of the dual SVM objective function given lambdas and Q.

    Args:
        lambdas: Vector of Lagrange multipliers (n_samples,)
        Q: Matrix Q = y_i y_j K(x_i, x_j) (n_samples, n_samples)

    Returns:
        Dual objective function value (float)
    """
    return - np.sum(lambdas) + 0.5 * lambdas.T @ Q @ lambdas

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the classification accuracy.

    Args:
        y_true: Ground truth labels (array of shape (n_samples,))
        y_pred: Predicted labels (array of shape (n_samples,))

    Returns:
        Accuracy as a float in [0, 1]
    """
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total

from itertools import product
from typing import Callable, Dict, List, Tuple
import numpy as np

from sklearn.model_selection import KFold
from typing import Callable, Dict, List, Tuple
import numpy as np
from itertools import product

def crossval_svm(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    grid: Dict[str, List[float]],
    kernel_func: Callable,
    kernel_param_name: str,
    seed: int = 123,
    use_seed: bool = False
) -> Tuple[List[Dict[str, float]], List[float]]:
    """
    K-Fold Cross-Validation for Dual SVM using CVXOPT.

    Args:
        X: Training features (n_samples, n_features)
        y: Labels in {-1, +1} (n_samples,)
        k: Number of folds
        grid: Dict of hyperparameter values to try (e.g., {"C": [...], "p": [...]})
        kernel_func: Kernel function (e.g., polynomial_kernel or gaussian_kernel)
        kernel_param_name: Name of the kernel parameter in the grid (e.g., "p" or "gamma")
        seed: Seed for reproducibility
        use_seed: Whether to fix the seed or not

    Returns:
        param_list: List of hyperparameter combinations
        avg_scores: Average validation accuracy for each hyperparameter combination
    """
    # Create hyperparameter combinations
    param_list = [dict(zip(grid, vals)) for vals in product(*grid.values())]
    avg_scores = []

    # Set up KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=seed if use_seed else None)

    for param in param_list:
        C = param["C"]
        kernel_param = param[kernel_param_name]
        acc = []

        for train_idx, val_idx in kf.split(X):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Kernel matrix and training
            K_tr = compute_kernel_matrix(X_tr, y_tr, kernel_func, **{kernel_param_name: kernel_param})
            lambdas, _, _, _ = solve_dual_svm(K_tr, y_tr, C)
            b = compute_b(X_tr, y_tr, lambdas, kernel_func, C=C, **{kernel_param_name: kernel_param})

            # Prediction and accuracy
            y_pred = predict(X_tr, y_tr, lambdas, b, X_val, kernel_func, **{kernel_param_name: kernel_param})
            acc.append(compute_accuracy(y_val, y_pred))

        avg_scores.append(np.mean(acc))

    return param_list, avg_scores
'''