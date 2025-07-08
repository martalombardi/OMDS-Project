from typing import Callable, Tuple, List, Dict, Any
import numpy as np
from sklearn.model_selection import KFold
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError

def unroll_params(W_list: List[np.ndarray], b_list: List[np.ndarray], v: np.ndarray) -> np.ndarray:
    """
    Flattens all parameters (W_list, b_list, v) into a single 1D numpy array.
    This one handles W being a list of matrices, b a list of vectors
    """
    
    flattened_params = []
    for W in W_list:
        flattened_params.append(W.flatten())
    for b in b_list:
        flattened_params.append(b.flatten())
    flattened_params.append(v.flatten())
    return np.concatenate(flattened_params)

def roll_params(flat_params: np.ndarray, W_shapes: List[Tuple[int, ...]], b_shapes: List[Tuple[int, ...]], v_shape: Tuple[int, ...]) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """Reconstructs W_list, b_list, and v original arrays from a flattened array."""
    current_idx = 0
    W_list = []
    for shape in W_shapes:
        size = np.prod(shape)
        W_list.append(flat_params[current_idx : current_idx + size].reshape(shape))
        current_idx += size

    b_list = []
    for shape in b_shapes:
        size = np.prod(shape)
        b_list.append(flat_params[current_idx : current_idx + size].reshape(shape))
        current_idx += size

    size = np.prod(v_shape)
    v = flat_params[current_idx : current_idx + size].reshape(v_shape)

    return W_list, b_list, v

def g1(x: np.ndarray) -> np.ndarray:
    """ Applies hyperbolic tangent element-wise"""
    return np.tanh(x)

def dg1_dx(x: np.ndarray) -> np.ndarray:
    """ Returns the gradient of the hyperbolic tangent """
    return 1 - g1(x)**2

def g2(x: np.ndarray) -> np.ndarray:
    """Applies sigmoid element-wise"""
    return 1 / (1 + np.exp(-x))

def dg2_dx(x: np.ndarray) -> np.ndarray:
    """Returns the gradient of the sigmoid function"""
    sig = g2(x)
    return sig * (1 - sig)

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error loss function: 1/N * sum((y_i - y_hat_i)^2)."""
    return np.mean((y_pred - y_true)**2)

def mse_loss_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Derivative of Mean Squared Error loss function with respect to y_pred.
    For E = 1/N * sum((y_i - y_hat_i)^2), dE/dy_hat_i = 2 * (y_hat_i - y_i) / N.
    """
    return 2*(y_pred - y_true) / y_true.shape[1]

def mape(prediction, Y_true):
    """Mean Absolute Percentage Error (MAPE) metric."""
    return 100*(np.mean(np.abs(((prediction - Y_true) / Y_true))))

def forward(X: np.ndarray,
            W_list: List[np.ndarray],
            b_list: List[np.ndarray],
            v: np.ndarray,
            g: Callable[[np.ndarray], np.ndarray],
            L: int) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Forward propagation for an MLP.

    Parameters:
        X: input of shape [D, N] (D features, N samples)
        W_list: List of weight matrices for hidden layers. Length = (L-1)
                (e.g., if L=3 (2 hidden), W_list has 2 elements: W_H1, W_H2)
        b_list: List of bias vectors for hidden layers. Length = (L-1)
        v: Output layer weights (connecting last hidden layer to output).
           Shape (num_neurons_last_hidden, output_dim).
        g: Activation function for hidden layers.
        L: Total number of layers (hidden layers + output layer).

    Returns:
        out: Network output of shape [output_dim, N]
        a_list: List of pre-activation values (z_math in report) for hidden layers.
                a_list[0] is pre-activation of 1st hidden layer.
        z_list: List of post-activation values (a_math in report) for hidden layers,
                including input X as z_list[0] (a_0).
                z_list[1] is post-activation of 1st hidden layer.
    """

    if L < 3 or L > 5:
        raise ValueError(f"Forward function only supports L = 3, 4, or 5. Got L={L}.")

    # Check consistency between L and W_list/b_list length
    expected_W_b_lists_len = L - 1
    if len(W_list) != expected_W_b_lists_len:
        raise ValueError(f"Expected {expected_W_b_lists_len} weight matrices (for {L-1} hidden layers), got {len(W_list)}.")
    if len(b_list) != expected_W_b_lists_len:
        raise ValueError(f"Expected {expected_W_b_lists_len} bias vectors (for {L-1} hidden layers), got {len(b_list)}.")

    a_list = []  # Stores pre-activation values
    z_list = [X] # Stores post-activation values

    current_input_for_layer = X # network input (a_0)

    # Loop through the (L-1) hidden layers
    for i in range(expected_W_b_lists_len):
        # pre-activation for current hidden layer
        pre_activation = W_list[i] @ current_input_for_layer + b_list[i]
        # post-activation for current hidden layer
        post_activation = g(pre_activation)

        a_list.append(pre_activation) 
        z_list.append(post_activation) 
        current_input_for_layer = post_activation # input for the next layer

    # Output layer (linear activation)
    # corresponds to z_list[L-1]
    out = v.T @ current_input_for_layer

    return out, a_list, z_list


def backward(X: np.ndarray,
             y_true: np.ndarray,
             y_pred: np.ndarray,       # To pass output of forward
             W_list: List[np.ndarray],
             b_list: List[np.ndarray],
             v: np.ndarray,
             a_list: List[np.ndarray], # Stores pre-activation values (z_math) from forward
             z_list: List[np.ndarray], # Stores post-activation values (a_math), where z_list[0] is X (a_0) from forward
             g_prime: Callable[[np.ndarray], np.ndarray],
             L: int) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Backward propagation to compute gradients of MSE term in the loss.
    The regularization term is excluded from here because its gradient computation is straightforward.
    Returns:
        gradients_W: List of gradients for W matrices (for hidden layers).
        gradients_b: List of gradients for b vectors (for hidden layers).
        gradient_v: Gradient for v (output layer weights).
    """

    gradients_W = [np.zeros_like(W) for W in W_list]
    gradients_b = [np.zeros_like(b) for b in b_list]

    if L < 3 or L > 5:
        raise ValueError(f"Backward function only supports L from 3 to 5. Got L={L}.")

    # dE/d_y_hat
    dLoss_dout = mse_loss_prime(y_true, y_pred) 

    # Gradient for v (output layer weights)
    # gradient_v = dE/dv = (dE/dy_hat) @ (dy_hat/dv).T by chain rule
    # dy_hat/dv = z_list[L-1] (post-activation of last hidden layer)
    gradient_v = z_list[L-1] @ dLoss_dout.T

    # delta for output error, propagated to last hidden layer
    delta = v @ dLoss_dout
    delta *= g_prime(a_list[L-2])

    # Loop backward through hidden layers
    for l_idx in reversed(range(L - 1)): 
        # dE/dW_l = delta_l @ (a_{l-1}).T
        # a_{l-1} (post-activation of previous layer) is z_list[l_idx]
        
        gradients_W[l_idx] = delta @ z_list[l_idx].T
        gradients_b[l_idx] = np.sum(delta, axis=1, keepdims=True)

        # Propagate delta to the previous hidden layer (if not the input layer)
        if l_idx > 0: 
            delta = W_list[l_idx].T @ delta
            delta *= g_prime(a_list[l_idx-1])
        # If l_idx is 0, we've propagated back to the input, no further delta needed.

    return gradients_W, gradients_b, gradient_v

def xavier_normal_init(in_dim: int, out_dim: int) -> np.ndarray:
    """
    This function implements the Xavier normal initialization for the weights.
    It samples from a normal distribution with mean 0 and std dev sqrt(2 / (fan_in + fan_out)).
    """
    sigma = np.sqrt(2.0 / (in_dim + out_dim))
    return np.random.randn(in_dim, out_dim) * sigma

def initialize_parameters(input_dim: int, hidden_neurons: List[int], output_dim: int, regularization_factor: float) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Initializes weights and biases for an MLP using Xavier/Glorot Normal initialization.
    """
    W_list = []
    b_list = []

    # Input layer to first hidden layer (hidden_neurons[0])
    # W1 connects input_dim to hidden_neurons[0]
    W1 = xavier_normal_init(input_dim, hidden_neurons[0]).T # <-- Use xavier_normal_init. Transpose if your W is (neurons, input_dim)
    b1 = np.zeros((hidden_neurons[0], 1))
    W_list.append(W1)
    b_list.append(b1)

    # Subsequent hidden layers (if any)
    # Loop for connections between hidden_neurons[i] and hidden_neurons[i+1]
    for i in range(len(hidden_neurons) - 1): # Loop from 0 to num_hidden_layers - 2
        # W_next connects hidden_neurons[i] to hidden_neurons[i+1]
        # fan_in = hidden_neurons[i], fan_out = hidden_neurons[i+1]
        W_next = xavier_normal_init(hidden_neurons[i], hidden_neurons[i+1]).T
        b_next = np.zeros((hidden_neurons[i+1], 1))
        W_list.append(W_next)
        b_list.append(b_next)

    # Output layer weights (v)
    # v connects hidden_neurons[-1] (last hidden layer) to output_dim
    v = xavier_normal_init(hidden_neurons[-1], output_dim) 

    return W_list, b_list, v

def check_gradients_with_central_differences(
    initial_flat_params: np.ndarray,
    X_data_for_check: np.ndarray,
    y_data_for_check: np.ndarray,
    W_shapes_for_check: List[Tuple[int, ...]],
    b_shapes_for_check: List[Tuple[int, ...]],
    v_shape_for_check: Tuple[int, ...],
    activation_func_for_check: Callable[[np.ndarray], np.ndarray],
    activation_prime_for_check: Callable[[np.ndarray], np.ndarray],
    regularization_factor_for_check: float,
    num_layers_for_check: int,
    objective_function_ref: Callable 
):
    """
    Performs a gradient check using central differences.
    Compares the analytical gradient from objective_function
    with a numerical approximation.
    """
    eps = 1.e-6 # Small epsilon for finite difference approximation

    # 1. Get the analytical gradient and loss from your objective_function
    loss_analytical, analytical_gradient = objective_function_ref(
        initial_flat_params,
        X_data_for_check, y_data_for_check,
        W_shapes_for_check, b_shapes_for_check, v_shape_for_check,
        activation_func_for_check, activation_prime_for_check,
        regularization_factor_for_check, num_layers_for_check
    )

    approx_gradient = np.zeros_like(analytical_gradient)

    print("\n--- Gradient Check (Central Differences) ---")
    print(f"Checking {len(initial_flat_params)} parameters...")
    print(f"Analytical loss at initial point: {loss_analytical:.6f}")

    # 2. Iterate over each parameter to compute its approximate gradient
    for i in range(len(initial_flat_params)):
        # Create a perturbation vector for the i-th parameter
        delta_omega = np.zeros_like(initial_flat_params)
        delta_omega[i] = eps

        # Calculate loss at (omega + epsilon)
        loss_plus_eps, _ = objective_function_ref(
            initial_flat_params + delta_omega,
            X_data_for_check, y_data_for_check,
            W_shapes_for_check, b_shapes_for_check, v_shape_for_check,
            activation_func_for_check, activation_prime_for_check,
            regularization_factor_for_check, num_layers_for_check
        )

        # Calculate loss at (omega - epsilon)
        loss_minus_eps, _ = objective_function_ref(
            initial_flat_params - delta_omega,
            X_data_for_check, y_data_for_check,
            W_shapes_for_check, b_shapes_for_check, v_shape_for_check,
            activation_func_for_check, activation_prime_for_check,
            regularization_factor_for_check, num_layers_for_check
        )

        # Central difference formula
        approx_gradient[i] = (loss_plus_eps - loss_minus_eps) / (2 * eps)

    
    # 3. Check the overall norm of the difference
    gradient_difference_norm = np.linalg.norm(analytical_gradient - approx_gradient)

    if gradient_difference_norm > 1.e-4: 
        print(f'\nERROR: Gradient check FAILED! Norm of difference: {gradient_difference_norm:.6e}')
        # Print top N largest absolute differences for debugging
        diffs = np.abs(analytical_gradient - approx_gradient)
        sorted_indices = np.argsort(diffs)[::-1]
        print("  Top 5 largest absolute differences:")
        for k in range(min(5, len(diffs))):
            idx = sorted_indices[k]
            print(f"    Index {idx}: Analytical={analytical_gradient[idx]:.6e}, Approx={approx_gradient[idx]:.6e}, Diff={diffs[idx]:.6e}")
    else:
        print(f'\nGradient check PASSED! Norm of difference: {gradient_difference_norm:.6e}')

    print("------------------------------------------")


def objective_function(flat_params: np.ndarray,
                       input_data: np.ndarray,
                       target_data: np.ndarray,
                       W_shapes: List[Tuple[int, ...]],
                       b_shapes: List[Tuple[int, ...]],
                       v_shape: Tuple[int, ...],
                       activation_func: Callable[[np.ndarray], np.ndarray],
                       activation_prime: Callable[[np.ndarray], np.ndarray],
                       regularization_factor: float,
                       num_layers: int) -> Tuple[float, np.ndarray]:
    """
    Objective function for scipy.optimize.minimize.
    Returns total loss and flattened gradients.
    """
    # Unpack parameters
    W_list, b_list, v = roll_params(flat_params, W_shapes, b_shapes, v_shape)

    # Forward pass
    y_pred, a_list, z_list = forward(input_data, W_list, b_list, v, activation_func, num_layers)

    # Compute MSE loss
    loss = mse_loss(target_data, y_pred)

    # Backward pass 
    grad_W_list, grad_b_list, grad_v = backward(
        X=input_data,
        y_true=target_data,
        y_pred=y_pred,
        W_list=W_list,
        b_list=b_list,
        v=v,
        a_list=a_list,
        z_list=z_list,
        g_prime=activation_prime,
        L=num_layers
    )

    # Add regularization to gradients
    grad_W_list = [gW + 2 * regularization_factor * W for gW, W in zip(grad_W_list, W_list)]
    grad_v += 2 * regularization_factor * v
    # Note: No regularization for biases

    # Flatten gradients
    flattened_gradients = unroll_params(grad_W_list, grad_b_list, grad_v)

    # Add L2 regularization to the loss
    reg_loss = regularization_factor * (
        sum(np.sum(W**2) for W in W_list) + np.sum(v**2)
    )
    loss += reg_loss

    return loss, flattened_gradients


class myMLPRegressor(BaseEstimator, RegressorMixin):
    ''' custom MLP Regressor Class '''
    # default values
    def __init__(self, D_input: int, y_output_dim: int, num_layers: int = 3, # 2 hidden
                 num_neurons: List[int] = [8, 4],
                 activation_func_name: str = 'g1',
                 regularization_factor: float = 0.001,
                 max_iter: int = 5000, print_callback_loss: bool = True, random_state: int = None):

        # Hyperparameters for gridsearch
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.activation_func_name = activation_func_name
        self.regularization_factor = regularization_factor
        self.max_iter = max_iter
        self.print_callback_loss = print_callback_loss

        # Fixed parameters from dataset
        self.D_input = D_input
        self.y_output_dim = y_output_dim

        # Attributes that will be set after fitting (by the fit method)
        self.W_list_ = None
        self.b_list_ = None
        self.v_ = None
        self.W_shapes_ = None
        self.b_shapes_ = None
        self.v_shape_ = None
        self.activation_func_ = None 
        self.activation_prime_ = None
        self.n_iterations_ = None
        self.optimization_message_ = None
        self.final_objective_value_ = None
        self._is_invalid_combo = False # Flag to mark invalid hyperparameter combinations
        self.random_state = random_state # For reproducibility

    def _get_activation_functions(self):
        """Maps activation function name (string) to callable functions."""
        if self.activation_func_name == 'g1':
            return g1, dg1_dx
        elif self.activation_func_name == 'g2':
            return g2, dg2_dx
        else:
            raise ValueError(f"Unknown activation function name: {self.activation_func_name}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        
        X_transposed = X.T # (D, N_samples) 
        y_transposed = y.T # (1, N_samples)

        # Architectural Validation
        expected_hidden_layers = self.num_layers - 1 # L total layers, L-1 hidden layers
        if len(self.num_neurons) != expected_hidden_layers:
            self._is_invalid_combo = True
            return self

        self._is_invalid_combo = False # Reset flag for valid combinations

        # Get activation functions
        self.activation_func_, self.activation_prime_ = self._get_activation_functions()

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialize parameters
        W_init, b_init, v_init = initialize_parameters(
            self.D_input, self.num_neurons, self.y_output_dim, self.regularization_factor
        )
        initial_flat_params = unroll_params(W_init, b_init, v_init)

        # Store shapes of parameters for rolling/unrolling inside objective_function
        self.W_shapes_ = [W.shape for W in W_init]
        self.b_shapes_ = [b.shape for b in b_init]
        self.v_shape_ = v_init.shape

        # Callback function to print mse loss in training
        iteration_count = 0 

        def callback_function(current_flat_params):
            nonlocal iteration_count 
            iteration_count += 1

            if iteration_count % 10 == 0:
                W_list_cb, b_list_cb, v_cb = roll_params(
                    current_flat_params, self.W_shapes_, self.b_shapes_, self.v_shape_
                )
                # Perform forward pass to get y_pred
                y_pred_cb, _, _ = forward(
                    X_transposed, W_list_cb, b_list_cb, v_cb, self.activation_func_, self.num_layers
                )
                # Calculate non-regularized MSE loss
                non_reg_loss = mse_loss(y_transposed, y_pred_cb)
                print(f"  Iteration {iteration_count}: Non-regularized MSE Loss = {non_reg_loss:.6f}")

        callback_arg = callback_function if self.print_callback_loss else None

        result = minimize(
            fun=objective_function,
            x0=initial_flat_params,
            args=(X_transposed, y_transposed, self.W_shapes_, self.b_shapes_, self.v_shape_,
                  self.activation_func_, self.activation_prime_, self.regularization_factor, self.num_layers),
            method='L-BFGS-B',
            jac=True,
            options={'disp': False, 'maxiter': self.max_iter, 'ftol': 1e-5},
            callback=callback_arg # Pass the callback here
        )

        # Store the optimized parameters and optimization details
        self.W_list_, self.b_list_, self.v_ = roll_params(
            result.x, self.W_shapes_, self.b_shapes_, self.v_shape_
        )
        self.n_iterations_ = result.nit
        self.optimization_message_ = result.message
        self.final_objective_value_ = result.fun # final (regularized) loss after optimization

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # If combination flagged invalid during fit, raise an error
        if self._is_invalid_combo:
            raise NotFittedError("This estimator was skipped due to an invalid hyperparameter combination during fit.")
        # Check if the model has actually been trained
        if self.W_list_ is None:
            raise NotFittedError("Model has not been trained yet. Call .fit() first.")

        # Transpose X for forward function
        X_transposed = X.T
        # Perform forward pass with the TRAINED parameters to get predictions
        y_pred, _, _ = forward(X_transposed, self.W_list_, self.b_list_, self.v_, self.activation_func_, self.num_layers)
        # Transpose prediction back to (N_samples,1) for sklearn compatibility
        return y_pred.T.flatten() 

'''
# OLD MANNUAL CV WITHOUT CLASS USAGE
def my_k_fold_CV(
    X_train_norm: np.ndarray, 
    y_train_data: np.ndarray,
    D_input: int,
    y_output_dim: int,
    hyperparameter_grid: Dict[str, Any],
    n_splits: int = 5,
    random_seed: int = 1234,
    max_iter_minimize: int = 500 # reduced for cv, then increased in final full training
) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:

    # --- K-Fold Setup ---
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    X_train_for_kf_split = X_train_norm.T # (N_samples, D_input)

    best_mape = float('inf')
    best_hyperparameters = {}
    best_training_results = {}

    print("Starting Hyperparameter Tuning (using my_k_fold_CV)...\n")

    for L in hyperparameter_grid['num_layers']:
        hidden_neurons_options = hyperparameter_grid['num_neurons_per_layer'][L]

        for neurons_config in hidden_neurons_options:
            for activation_tuple in hyperparameter_grid['activation_function']:
                activation_func, activation_prime = activation_tuple
                activation_name = activation_func.__name__

                for reg_factor in hyperparameter_grid['regularization_factor']:

                    print(f"Testing L={L}, Neurons={neurons_config}, Activation={activation_name}, Reg Factor={reg_factor}")

                    fold_mape_scores = []
                    fold_val_mse_reg_scores = []
                    fold_iterations = [] 
                    fold_initial_obj_values = []
                    fold_final_obj_values = []

                    for fold_idx, (train_index, val_index) in enumerate(kf.split(X_train_for_kf_split)):
                        X_fold_train = X_train_norm[:, train_index]
                        y_fold_train = y_train_data[:, train_index]
                        X_fold_val = X_train_norm[:, val_index]
                        y_fold_val = y_train_data[:, val_index]

                        W_init, b_init, v_init = initialize_parameters(D_input, neurons_config, y_output_dim, reg_factor)
                        initial_flat_params = unroll_params(W_init, b_init, v_init)

                        W_shapes = [W.shape for W in W_init]
                        b_shapes = [b.shape for b in b_init]
                        v_shape = v_init.shape

                        initial_obj_value_this_fold, _ = objective_function(
                            initial_flat_params,
                            X_fold_train, y_fold_train,
                            W_shapes, b_shapes, v_shape,
                            activation_func, activation_prime, reg_factor, L
                        )
                        fold_initial_obj_values.append(initial_obj_value_this_fold)

                        result = minimize(
                            fun=objective_function,
                            x0=initial_flat_params,
                            args=(X_fold_train, y_fold_train, W_shapes, b_shapes, v_shape, activation_func, activation_prime, reg_factor, L),
                            method='L-BFGS-B',
                            jac=True,
                            options={'disp': False, 'maxiter': max_iter_minimize}
                        )

                        final_obj_value_this_fold = result.fun
                        fold_final_obj_values.append(final_obj_value_this_fold)

                        fold_iterations.append(result.nit)

                        W_optimized, b_optimized, v_optimized = roll_params(result.x, W_shapes, b_shapes, v_shape)

                        y_fold_val_pred, _, _ = forward(X_fold_val, W_optimized, b_optimized, v_optimized, activation_func, L)
                        val_mape = mape(y_fold_val, y_fold_val_pred)
                        fold_mape_scores.append(val_mape)

                        val_mse_reg, _ = objective_function(
                            result.x,
                            X_fold_val, y_fold_val,
                            W_shapes, b_shapes, v_shape,
                            activation_func, activation_prime, reg_factor, L
                        )
                        fold_val_mse_reg_scores.append(val_mse_reg)


                    avg_mape = np.mean(fold_mape_scores)
                    avg_val_mse_reg = np.mean(fold_val_mse_reg_scores)
                    avg_iterations = np.mean(fold_iterations)
                    avg_initial_obj_value = np.mean(fold_initial_obj_values)
                    avg_final_obj_value = np.mean(fold_final_obj_values)

                    if avg_mape < best_mape:
                        best_mape = avg_mape
                        best_hyperparameters = {
                            'num_layers': L,
                            'num_neurons_per_layer': neurons_config,
                            'activation_function': activation_name,
                            'regularization_factor': reg_factor,
                            'max_iter_minimize': max_iter_minimize
                        }
                        best_training_results = {
                            'optimization_solver': result.message,
                            'num_iterations': avg_iterations,
                            'initial_objective_function_value': avg_initial_obj_value,
                            'final_objective_function_value': avg_final_obj_value,
                            'average_validation_mape': avg_mape,
                            'average_validation_mse_reg': avg_val_mse_reg,
                            'best_model_params_flat': result.x,
                            'best_model_W_shapes': W_shapes,
                            'best_model_b_shapes': b_shapes,
                            'best_model_v_shape': v_shape
                        }
                    print("-" * 50)

    print("\nHyperparameter Tuning Complete.")
    return best_mape, best_hyperparameters, best_training_results
'''

def final_report_metrics(
    final_L, final_neurons_config, final_activation_name, final_reg_factor, final_train_max_iter,
    result_final_train_message, final_training_iterations,
    initial_train_mse_reg_final_model, final_train_objective_value,
    initial_train_mape_final_model, final_train_mape,
    best_overall_mape_score, 
    test_error_mape, test_error_mse_reg, 
    final_train_mse_no_reg, test_error_mse_no_reg 
):
    """
    Displays all comprehensive performance metrics and generates plots for the final report.
    """

    print("\n--- Comprehensive Performance Metrics for Final Report ---")

    # --- 1. Optimal Configuration ---
    print("\n1. Optimal Model Configuration:")
    print(f"  Non-linearity (Activation Function): {final_activation_name}")
    print(f"  Total Number of Layers (L): {final_L}")
    print(f"  Neurons per Layer (Nl): {final_neurons_config}")
    print(f"  Regularization Factor (λ): {final_reg_factor}")


    # --- 2. Optimization Routine Details (for Final Training) ---
    print("\n2. Optimization Routine Details (for Final Training):")
    print(f"  Optimization Routine: L-BFGS-B")
    print(f"  Max Number of Iterations Parameter: {final_train_max_iter}")
    print(f"  Returned Message: {result_final_train_message}")
    print(f"  Number of Iterations Performed: {final_training_iterations}")
    print(f"  Starting Value of Objective Function: {initial_train_mse_reg_final_model:.4e}")
    print(f"  Final Value of Objective Function: {final_train_objective_value:.4e}")


    # --- 3. Training Set Performance (Initial & Final) ---
    print("\n3. Training Set Performance:")
    print(f"  Initial Training Error (MAPE): {initial_train_mape_final_model:.4f}%")
    print(f"  Initial Training Error (MSE, regularized): {initial_train_mse_reg_final_model:.4f}")
    print(f"  Final Training Error (MAPE): {final_train_mape:.4f}%")
    print(f"  Final Training Error (MSE, regularized): {final_train_objective_value:.4f}")
    print(f"  Final Training Error (MSE, **non-regularized**): {final_train_mse_no_reg:.4f}")

    # --- 4. Validation Set Performance (Average from K-Fold CV)  ---
    print("\n4. Validation Set Performance (Average from K-Fold CV):")
    print(f"  Average Validation Error (MAPE): {best_overall_mape_score:.4f}%")
   

    # --- 5. Test Set Performance (Final) ---
    print("\n5. Test Set Performance:")
    print(f"  Final Test Error (MAPE): {test_error_mape:.4f}%")
    print(f"  Final Test Error (MSE, regularized): {test_error_mse_reg:.4f}")
    print(f"  Final Test Error (MSE, **non-regularized**): {test_error_mse_no_reg:.4f}")

    # Data for plots
    metrics_for_plot = {
        "Training (Final)": {
            "MAPE": final_train_mape,
            "Reg. MSE": final_train_objective_value,
            "MSE (no reg)" : final_train_mse_no_reg,
        },
        "Test (Final)": {
            "MAPE": test_error_mape,
            "Reg. MSE": test_error_mse_reg,
            "MSE (no reg)" : test_error_mse_no_reg,
        },
        "Validation (Average)": {
            "MAPE": best_overall_mape_score,
        },
    }

    # --- Performance Metrics Summary Table ---
    print("\nPerformance Metrics Summary Table for Report (Figure 1 Data):")
    print(f"{'Metric':<25} {'Training (Final)':<20} {'Test (Final)':<20} {'Validation (Avg)':<20}")
    print("-" * 85)

    # MAPE
    print(f"{'MAPE (%)':<25} "
        f"{metrics_for_plot['Training (Final)']['MAPE']:.4f}{'%':<18} "
        f"{metrics_for_plot['Test (Final)']['MAPE']:.4f}{'%':<18}"
        f"{metrics_for_plot['Validation (Average)']['MAPE']:.4f}{'%':<18} "
        )

    # Regularized MSE
    print(f"{'Regularized MSE':<25} "
        f"{metrics_for_plot['Training (Final)']['Reg. MSE']:.4f}{'':<18} "
        f"{metrics_for_plot['Test (Final)']['Reg. MSE']:.4f}{'':<18}")

    # Non-Regularized MSE
    print(f"{'MSE (no reg)':<25} "
        f"{metrics_for_plot['Training (Final)']['MSE (no reg)']:.4f}{'':<18} "
        f"{metrics_for_plot['Test (Final)']['MSE (no reg)']:.4f}{'':<18}")
    
    print("-" * 85)
