from typing import Callable, Tuple
import numpy as np

def unroll(W: np.ndarray, b: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Flattens alle the neural network parameters inside a long vector --> Call this function when computing gradient!
    NOTE: In flattened_array elements are ordered in this way: W --> b --> v
    """

    W_flattened = np.ravel(W)
    b_flattened = np.ravel(b)
    v_flattened = np.ravel(v)

    return np.concatenate([W_flattened, b_flattened, v_flattened], axis=-1)

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

def g1(x: np.ndarray) -> np.ndarray:
    """
    Applies hyperbolic tangent element-wise
    """

    return np.tanh(x)

def dg1_dx(x: np.ndarray) -> np.ndarray:
    """
    Returns the gradient of the hyperbolic tangent
    """

    return 1 - g1(x)**2

def g2(x: np.ndarray) -> np.ndarray:
    """
    Applies sigmoid element-wise
    """
    return 1 / (1 + np.exp(-x))

def dg2_dx(x: np.ndarray) -> np.ndarray:
    """
    Returns the gradient of the sigmoid function
    """
    sig = g2(x)
    return sig * (1 - sig)

# BUILD MODEL
import numpy as np
from typing import Callable, List

def forward(X: np.ndarray,
            W_list: List[np.ndarray],
            b_list: List[np.ndarray],
            v: np.ndarray,         
            g: Callable[[np.ndarray], np.ndarray],
            L: int) -> np.ndarray:
    """
    Forward propagation with 2 to 4 layers using if, no for-loops.
    
    Parameters:
        X: input of shape [D, N]
        W_list: list of W matrices (L-1 elements)
        b_list: list of b vectors (L-1 elements)
        v: final output weights
        g: activation function
        L: number of total layers (min 2, max 4)
    """
    if L not in {2, 3, 4}:
        raise ValueError("Only L = 2, 3, or 4 are supported.")
    
    if len(W_list) != L - 1:
        raise ValueError(f"Expected {L - 1} weight matrices, got {len(W_list)}.")

    if len(b_list) != L - 1:
        raise ValueError(f"Expected {L - 1} bias vectors, got {len(b_list)}.")

    a1 = W_list[0] @ X + b_list[0]
    z1 = g(a1)

    if L >= 3:
        a2 = W_list[1] @ z1 + b_list[1]
        z2 = g(a2)
        if L == 4:
            a3 = W_list[2] @ z2 + b_list[2]
            z3 = g(a3)
            out = v @ z3
            return out
        out = v @ z2
        return out
    
    out = v @ z1
    return out

def E(
        omega: np.ndarray,  # FIRST ARGUMENT!
        Dataset: Tuple[np.ndarray, np.ndarray], 
        REGULARIZATION_TERM,
        W_shape, 
        b_shape, 
        v_shape
    ) -> np.ndarray:
    """
    Loss function to optimize

    Parameters:
    omega: unrolled parameters of MLP
    Dataset: [X, Y]
    reg_term: regularization term
    """

    X = Dataset[0]
    Y = Dataset[1]

    W, b, v = roll(omega, W_shape, b_shape, v_shape)

    out = forward(X, W, b, v)

    n_points = X.shape[1]
    error = 1 / (2 * n_points) * np.sum((out - Y)**2)
    regularization = 0.5 * REGULARIZATION_TERM * (np.sum(W**2) + np.sum(v**2))

    return error + regularization


def dE_dOmega(omega: np.ndarray, Dataset: Tuple[np.ndarray, np.ndarray], W_shape, b_shape, v_shape) -> np.ndarray:
    """
    Computes the gradient of error funcion w.r.t all MLP parameters
    """

    # UNPACK DATASET
    X = Dataset[0]
    Y = Dataset[1]

    # UNPACK MLP PARAMETERS
    W, b, v = roll(omega, W_shape, b_shape, v_shape)

    # COMPUTE GRADIENTS
    grad_v = np.array([[dE_dv(W,b,v,X,Y,j)] for j in range(v.shape[0])])
    grad_b = np.array([[dE_db(W,b,v,X,Y,j)] for j in range(b.shape[0])])
    grad_W = np.array([[dE_dW(W,b,v,X,Y,h,j) for h in range(W.shape[1])] for j in range(W.shape[0])])

    #print(f"v shape: {v.shape} -- grad v shape: {grad_v.shape}")
    #print(f"b shape: {b.shape} -- grad b shape: {grad_b.shape}")
    #print(f"W shape: {W.shape} -- grad W shape: {grad_W.shape}") 

    # UNROLL GRADIENTS
    return unroll(grad_W, grad_b, grad_v)

def dE_dv(
    W: np.ndarray,
    b: np.ndarray,
    v: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    j: np.ndarray,
    REGULARIZATION_TERM,
    g
) -> np.ndarray:
    """
    Computes the gradient of error function w.r.t. v^j
    """
    
    t1 = np.dot(v, g(np.dot(W, X) + b)) - Y
    t2 = g(np.dot(W[j, :], X) + b[j])
    t3 = REGULARIZATION_TERM * v[j]
    
    return (1/X.shape[1]) * np.sum(t1 * t2) + t3
    
def dE_db(
    W: np.ndarray,
    b: np.ndarray,
    v: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    j: np.ndarray,
    g,
    dg_dx
) -> np.ndarray:
    """
    Computes the gradient of error function w.r.t. b^j
    """

    t1 = np.dot(v, g(np.dot(W, X) + b)) - Y
    t2 = v[j] * dg_dx(np.dot(W[j,:], X) + b[j])

    return (1/X.shape[1]) * np.sum(t1 * t2)

def dE_dW(
    W: np.ndarray,
    b: np.ndarray,
    v: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    h: np.ndarray,
    j: np.ndarray,
    REGULARIZATION_TERM,
    g,
    dg_dx
) -> np.ndarray:
    """
    Computes gradient of error function w.r.t. w^jh
    """

    t1 = np.dot(v, g(np.dot(W, X) + b)) - Y
    t2 = v[j] * dg_dx(np.dot(W[j,:], X) + b[j])
    t3 = X[h,:]
    t4 = REGULARIZATION_TERM * W[j, h]

    return (1/X.shape[1]) * np.sum(t1 * t2 * t3) + t4


def mape(prediction, Y_test):
    100*(np.mean(np.abs(((prediction - Y_test) / Y_test))))