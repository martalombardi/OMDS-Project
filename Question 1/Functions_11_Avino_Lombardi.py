from typing import Callable, Tuple, List
import numpy as np


def unroll(W: np.ndarray, b: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Flattens all the neural network parameters inside a long vector --> Call this function when computing gradient!
    NOTE: In flattened_array elements are ordered in this way: W --> b --> v
    """

    W_flattened = np.ravel(W)
    b_flattened = np.ravel(b)
    v_flattened = np.ravel(v)

    return np.concatenate([W_flattened, b_flattened, v_flattened], axis=-1)


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

def roll(
    flattened_array: np.ndarray, 
    W_shape: np.ndarray, 
    b_shape: np.ndarray, 
    v_shape: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstructs W, b and v original arrays
    NOTE: In flattened_array elements are ordered in this way: W --> b --> v

    Returns: (W, b, v)
    """

    # MULTIPLY W SHAPE --> IF W.SHAPE == (3, 4) --> W_ELEMENTS = 3 x 4 = 12
    w_elements = np.prod(W_shape)
    b_elements= np.prod(b_shape)
    v_shape = np.prod(v_shape)

    # reshape: gives a new shape to an array without changing its data.
    W = np.reshape(flattened_array[:w_elements], W_shape)
    b = np.reshape(flattened_array[w_elements:w_elements+b_elements], b_shape)
    v = np.reshape(flattened_array[w_elements+b_elements:], v_shape)

    return W, b, v


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

# --- MSE LOSS FUN ---
def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error loss function: 1/N * sum((y_i - y_hat_i)^2)."""
    return np.mean((y_pred - y_true)**2)

def mse_loss_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Derivative of Mean Squared Error loss function with respect to y_pred.
    For E = 1/N * sum((y_i - y_hat_i)^2), dE/dy_hat_i = 2 * (y_hat_i - y_i) / N.
    """
    return 2*(y_pred - y_true) / y_true.shape[1]

# --- SCORE FUNCTION ---
def mape(prediction, Y_true):
    """Mean Absolute Percentage Error (MAPE) metric."""
    return 100*(np.mean(np.abs(((prediction - Y_true) / Y_true))))

# --- FORWARD PASS ---
def forward(X: np.ndarray,
            W_list: List[np.ndarray],
            b_list: List[np.ndarray],
            v: np.ndarray,         
            g: Callable[[np.ndarray], np.ndarray],
            L: int) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Forward propagation
    
    Parameters:
        X: input of shape [D, N]
        W_list: list of W matrices (L-1 elements)
        b_list: list of b vectors (L-1 elements)
        v: final output weights
        g: activation function
        L: number of total layers (min 2, max 4)

    Returns:
        out: Network output of shape [1, N]
        a_list: List of pre-activation values for hidden layers
        z_list: List of post-activation values for hidden layers (including input as z0 for convenience)
    """
    if L not in {2, 3, 4}:
        raise ValueError("Only L = 2, 3, or 4 are supported.")
    
    if len(W_list) != L - 1:
        raise ValueError(f"Expected {L - 1} weight matrices, got {len(W_list)}.")

    if len(b_list) != L - 1:
        raise ValueError(f"Expected {L - 1} bias vectors, got {len(b_list)}.")
    
    a_list = [] 
    z_list = [X] # z_0 = X, then z_1, z_2, ...

    # first hidden layer output
    a1 = W_list[0] @ X + b_list[0]
    z1 = g(a1)
    a_list.append(a1)
    z_list.append(z1)

    if L >= 3:
        a2 = W_list[1] @ z1 + b_list[1]
        z2 = g(a2)
        a_list.append(a2)
        z_list.append(z2)
        if L == 4:
            a3 = W_list[2] @ z2 + b_list[2]
            z3 = g(a3)
            a_list.append(a3)
            z_list.append(z3)
            out = v.T @ z3
            return out, a_list, z_list  # L=4
        out = v.T @ z2
        return out, a_list, z_list  # L=3
    
    out = v.T @ z1
    return out, a_list, z_list  # L=2


# --- GRADIENT BACKWARD PROPAGATION ---

def backward(X: np.ndarray,
             y_true: np.ndarray,
             W_list: List[np.ndarray],
             b_list: List[np.ndarray],
             v: np.ndarray,
             a_list: List[np.ndarray],
             z_list: List[np.ndarray], # Note: z_list[0] is X (z0)
             g_prime: Callable[[np.ndarray], np.ndarray],
             L: int) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Backward propagation to compute gradients of mse term in the loss
    The regularization term is excluded from here because its gradient computation is straightforward

    Parameters:
        X: input of shape [D, N]
        y_true: true output of shape [1, N]
        W_list: list of W matrices (L-1 elements)
        b_list: list of b vectors (L-1 elements)
        v: final output weights
        a_list: List of pre-activation values from forward pass (a_1, a_2, ...)
        z_list: List of post-activation values from forward pass (z_0=X, z_1, z_2, ...)
        g_prime: derivative of the activation function
        L: number of total layers (min 2, max 4)

    Returns:
        gradients_W: List of gradients for W matrices
        gradients_b: List of gradients for b vectors
        gradient_v: Gradient for v
    """
    gradients_W = [np.zeros_like(W) for W in W_list]
    gradients_b = [np.zeros_like(b) for b in b_list]

    g_func = g1 if g_prime == dg1_dx else g2
    y_pred_for_grad_calc, _, _ = forward(X, W_list, b_list, v, g_func, L)
    # print(f"DEBUG IN BACKWARD: Shape of y_pred_for_grad_calc: {y_pred_for_grad_calc.shape}") # checked: (1, N)
    dLoss_dout = mse_loss_prime(y_true, y_pred_for_grad_calc)
    # print(f"DEBUG IN BACKWARD: Shape of dLoss_dout: {dLoss_dout.shape}") # checked: (1, N) 

    # Gradient for v
    gradient_v = z_list[L-1] @ dLoss_dout.T

    # Backpropagate through the output layer's last activation (z_{L-1})
    delta_current_layer = v @ dLoss_dout
    delta_current_layer = delta_current_layer * g_prime(a_list[L-2]) # a_list[L-2] is a_{L-1}

    # Gradients for W_{L-1} and b_{L-1}
    gradients_W[L-2] = delta_current_layer @ z_list[L-2].T # z_list[L-2] is z_{L-2}
    gradients_b[L-2] = np.sum(delta_current_layer, axis=1, keepdims=True)

    if L == 4:
        # Backpropagate to layer 2 (W_2, b_2)
        delta_prev_layer = W_list[2].T @ delta_current_layer
        delta_prev_layer = delta_prev_layer * g_prime(a_list[1]) # a_list[1] is a_2

        # Gradients for W_2 and b_2
        gradients_W[1] = delta_prev_layer @ z_list[1].T # z_list[1] is z_1
        gradients_b[1] = np.sum(delta_prev_layer, axis=1, keepdims=True)

        # Backpropagate to layer 1 (W_1, b_1)
        delta_prev_layer = W_list[1].T @ delta_prev_layer
        delta_prev_layer = delta_prev_layer * g_prime(a_list[0]) # a_list[0] is a_1

        # Gradients for W_1 and b_1
        gradients_W[0] = delta_prev_layer @ z_list[0].T # z_list[0] is X (z_0)
        gradients_b[0] = np.sum(delta_prev_layer, axis=1, keepdims=True)

    elif L == 3:
        # Backpropagate to layer 1 (W_1, b_1)
        delta_prev_layer = W_list[1].T @ delta_current_layer
        delta_prev_layer = delta_prev_layer * g_prime(a_list[0]) # a_list[0] is a_1

        # Gradients for W_1 and b_1
        gradients_W[0] = delta_prev_layer @ z_list[0].T # z_list[0] is X (z_0)
        gradients_b[0] = np.sum(delta_prev_layer, axis=1, keepdims=True)

    return gradients_W, gradients_b, gradient_v


def initialize_parameters(input_dim: int, hidden_neurons: List[int], output_dim: int, regularization_factor: float) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Initializes weights and biases for an MLP.

    Parameters:
        input_dim: Dimension of the input features.
        hidden_neurons: List of number of neurons in each hidden layer.
        output_dim: Dimension of the output layer (should be 1 for regression).
        regularization_factor: Lambda for regularization (not used in initialization but passed for completeness).

    Returns:
        W_list: List of initialized weight matrices.
        b_list: List of initialized bias vectors.
        v: Initialized output weights.
    """
    W_list = []
    b_list = []

    # Input to first hidden layer
    W1 = np.random.randn(hidden_neurons[0], input_dim) * np.sqrt(1. / input_dim)
    b1 = np.zeros((hidden_neurons[0], 1))
    W_list.append(W1)
    b_list.append(b1)

    # Subsequent hidden layers
    for i in range(len(hidden_neurons) - 1):
        W_next = np.random.randn(hidden_neurons[i+1], hidden_neurons[i]) * np.sqrt(1. / hidden_neurons[i])
        b_next = np.zeros((hidden_neurons[i+1], 1))
        W_list.append(W_next)
        b_list.append(b_next)

    # Output layer weights (v)
    # Initialize v as a column vector (neurons_in_last_hidden_layer, output_dim)
    v = np.random.randn(hidden_neurons[-1], output_dim) * np.sqrt(1. / hidden_neurons[-1])

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
    objective_function_ref: Callable # Pass the objective_function itself as an argument
):
    """
    Performs a gradient check using central differences.
    Compares the analytical gradient from objective_function
    with a numerical approximation.

    Args:
        initial_flat_params: The flattened parameter vector (omega0).
        X_data_for_check: Input features for the gradient check.
        y_data_for_check: Ground truth for the gradient check.
        W_shapes_for_check: List of shapes for weight matrices.
        b_shapes_for_check: List of shapes for bias vectors.
        v_shape_for_check: Shape for the final layer weight vector.
        activation_func_for_check: The activation function used.
        activation_prime_for_check: The derivative of the activation function.
        regularization_factor_for_check: The regularization factor (lambda).
        num_layers_for_check: The total number of layers (L).
        objective_function_ref: A reference to your objective_function.
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
    # This loop can be very slow for many parameters.
    # Consider checking a random subset of parameters if it's too slow.
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