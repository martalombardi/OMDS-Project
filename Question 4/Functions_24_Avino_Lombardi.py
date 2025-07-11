import numpy as np
from typing import Callable, Dict, Any, Tuple
from collections import Counter
from itertools import combinations
import time, os, sys

current_notebook_dir = os.getcwd() 
base_dir_q2 = os.path.abspath(os.path.join(current_notebook_dir, "..", "Question 2"))
sys.path.append(base_dir_q2)
from Functions_22_Avino_Lombardi import gaussian_kernel, polynomial_kernel, compute_accuracy

base_dir_q3 = os.path.abspath(os.path.join(current_notebook_dir, "..", "Question 3"))
sys.path.append(base_dir_q3)
from Functions_23_Avino_Lombardi import MVP 


def _get_mvp_decision_function_scores(mvp_model: Any, X_new: np.ndarray) -> np.ndarray:
    """
    Computes the raw decision function scores for new data from a trained MVP model instance.
    This function accesses the necessary attributes (support vectors, bias, kernel)
    from the MVP model to reconstruct the decision function, without requiring
    a decision_function method to be present within MVP itself.

    Args:
        mvp_model: An instance of the trained MVP class.
        X_new: New input data of shape (n_samples, n_features).

    Returns:
        np.ndarray: Raw decision scores for each sample in X_new, shape (n_samples,).
    """
    # Ensure the MVP model is trained and has support vectors
    if mvp_model.sv_x is None or len(mvp_model.sv_x) == 0:
        # If no support vectors are found (e.g., all alpha=0),
        # the decision function is effectively just the bias.
        return np.full(X_new.shape[0], mvp_model.b)

    # Retrieve necessary components from the MVP model
    sv_x = mvp_model.sv_x
    sv_y = mvp_model.sv_y
    sv_lam = mvp_model.sv_lam
    b = mvp_model.b
    kernel_func = mvp_model.kernel_func
    kernel_params = mvp_model.kernel_params

    # Compute the kernel matrix between new data and support vectors
    K_test = kernel_func(X_new, sv_x, **kernel_params)

    # Compute the decision function: sum_l (alpha_l * y_l * K(x_l, x)) + b
    decision = (sv_lam * sv_y) @ K_test.T + b
    return decision

class MulticlassSVM:
    """
    Multiclass SVM classifier using One-vs-Rest (OvR) or One-vs-One (OvO) strategies.
    It uses the MVP (Most Violating Pair) binary SVM as its base estimator,
    accessing its internal components via a helper function to get decision scores.
    """

    def __init__(self, C: float, kernel_func: Callable, kernel_params: Dict,
                 tol: float = 1e-3, max_iter: int = 1000, strategy: str = 'ovr'):

        self.C = C
        self.kernel_func = kernel_func
        self.kernel_params = kernel_params
        self.tol = tol
        self.max_iter = max_iter

        if strategy not in ['ovr', 'ovo']:
            raise ValueError("Strategy must be 'ovr' (One-vs-Rest) or 'ovo' (One-vs-One).")
        self.strategy = strategy

        # Stores newly trained MVP models for the current dataset.
        # For OvR: {class_label: MVP_model_for_that_class}
        # For OvO: {(class1_label, class2_label): MVP_model_for_pair}
        self.models: Dict[Any, Any] = {} # Type hint for MVP model instances
        self.classes_: np.ndarray | None = None # Unique classes in the dataset

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the multiclass SVM model to the training data.
        This method will train new MVP instances internally for the provided dataset.
        """
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes <= 1:
            raise ValueError("Multiclass SVM requires at least two classes.")

        print(f"Starting Multiclass SVM training with {self.strategy.upper()} strategy...")
        start_time = time.time()

        if self.strategy == 'ovr':
            print(f"Training {n_classes} One-vs-Rest (OvR) binary SVMs.")
            for i, c in enumerate(self.classes_):
                print(f"  Training OvR classifier for class '{c}' ({i+1}/{n_classes})...")
                # Create binary labels: +1 for current class, -1 for all others
                y_binary = np.where(y == c, 1, -1)

                # Instantiate and train a NEW MVP model for this binary problem
                model = MVP(C=self.C, kernel_func=self.kernel_func,
                            kernel_params=self.kernel_params, tol=self.tol,
                            max_iter=self.max_iter)
                model.fit(X, y_binary)
                self.models[c] = model # Store the newly trained model associated with this class
            print("OvR training complete.")

        elif self.strategy == 'ovo':
            pairs = list(combinations(self.classes_, 2))
            print(f"Training {len(pairs)} One-vs-One (OvO) binary SVMs.")
            for i, (c1, c2) in enumerate(pairs):
                print(f"  Training OvO classifier for pair ({c1} vs {c2}) ({i+1}/{len(pairs)})...")
                # Filter data for only these two classes
                mask = (y == c1) | (y == c2)
                X_filtered = X[mask]
                y_filtered = y[mask]

                # Create binary labels: +1 for c1, -1 for c2
                y_binary = np.where(y_filtered == c1, 1, -1)

                # Instantiate and train a NEW MVP model for this binary problem
                model = MVP(C=self.C, kernel_func=self.kernel_func,
                            kernel_params=self.kernel_params, tol=self.tol,
                            max_iter=self.max_iter)
                model.fit(X_filtered, y_binary)
                self.models[(c1, c2)] = model # Store the newly trained model associated with this pair
            print("OvO training complete.")

        end_time = time.time()
        print(f"Total multiclass training time: {end_time - start_time:.4f} seconds.")


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for new data.
        """
        if self.models is None or not self.models:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
        if self.classes_ is None:
            raise RuntimeError("Classes not set. Call fit() first.")

        n_samples = X.shape[0]
        # Initialize predictions array with a default type that can hold class labels
        # No need to pre-allocate predictions for OvO as it will be derived from vote_matrix
        # predictions = np.zeros(n_samples, dtype=self.classes_.dtype) # Not needed here

        if self.strategy == 'ovr':
            # For OvR, we use the decision_function scores
            scores = np.zeros((n_samples, len(self.classes_)))
            # Create a mapping from class label to its index in the scores array
            class_to_idx = {cls: i for i, cls in enumerate(self.classes_)}

            for c, model in self.models.items():
                # Use the helper function to get raw scores from the trained MVP model
                scores[:, class_to_idx[c]] = _get_mvp_decision_function_scores(model, X)

            # The predicted class is the one with the highest decision score
            predictions = self.classes_[np.argmax(scores, axis=1)]

        elif self.strategy == 'ovo':
            # For OvO, we use a voting scheme
            # Initialize a vote matrix: rows for samples, columns for classes
            # Stores how many times each class wins a pairwise comparison for each sample
            vote_matrix = np.zeros((n_samples, len(self.classes_)))
            class_to_idx = {cls: i for i, cls in enumerate(self.classes_)}

            for (c1, c2), model in self.models.items():
                # Get decision values for ALL samples from this binary model
                decision_vals_all_samples = _get_mvp_decision_function_scores(model, X)

                # Convert class labels to their corresponding indices for vote_matrix
                idx_c1 = class_to_idx[c1]
                idx_c2 = class_to_idx[c2]

                # Increment votes using boolean indexing (vectorized)
                vote_matrix[decision_vals_all_samples >= 0, idx_c1] += 1
                vote_matrix[decision_vals_all_samples < 0, idx_c2] += 1

            # For each sample, the predicted class is the one with the most votes
            # np.argmax will give the index of the class with max votes
            # Then map back to the actual class label
            predictions = self.classes_[np.argmax(vote_matrix, axis=1)]

        # Return predictions as a 1D array. Let the calling code reshape if needed.
        return predictions