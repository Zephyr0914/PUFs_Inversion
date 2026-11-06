import numpy as np
import sklearn
from sklearn.metrics.pairwise import polynomial_kernel

# You are allowed to import any submodules of sklearn e.g. metrics.pairwise to construct kernel Gram matrices
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_kernel, my_decode etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here

POLY_DEGREE = 3      # d
POLY_COEF0  = 0.16   # c
################################
# Non Editable Region Starting #
################################
def my_kernel( X1, Z1, X2, Z2 ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to compute Gram matrices for your proposed kernel
	# Your kernel matrix will be used to train a kernel ridge regressor
    X1 = np.asarray(X1, dtype=float).reshape(-1, 1)
    X2 = np.asarray(X2, dtype=float).reshape(-1, 1)
    Z1 = np.asarray(Z1, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)

    # Polynomial kernel on z: (z1^T z2 + c)^d
    K_z = polynomial_kernel(
        Z1,
        Z2,
        degree=POLY_DEGREE,
        coef0=POLY_COEF0
    )  # shape: (n1, n2)

    # Outer product on x: x1 * x2
    K_x = X1 @ X2.T            # shape: (n1, n2)

    # Final kernel: x1 x2 * K_z + 1
    G = K_x * K_z + 1.0
	
    return G


import numpy as np
from numpy.linalg import svd
from sklearn.linear_model import LinearRegression

def build_A():
    """
    Constructs the sparse transition matrix A (size 33 x 128) that maps
    multiplexer delays to the linear model coefficients.
    u = A * x
    """
    k = 32
    rows = k + 1
    cols = 4 * k
    A = np.zeros((rows, cols))
    
    # Offsets for the 4 delay components per multiplexer
    off_a, off_b, off_c, off_d = 0, k, 2*k, 3*k
    
    for i in range(k):
        # Column indices for the current stage i
        p, q, r, s = i + off_a, i + off_b, i + off_c, i + off_d
        
        # Coefficients for alpha_i: 0.5, -0.5, 0.5, -0.5
        # Coefficients for beta_i:  0.5, -0.5, -0.5, 0.5
        
        # Contribution from alpha_i to row i
        A[i, p] += 0.5
        A[i, q] -= 0.5
        A[i, r] += 0.5
        A[i, s] -= 0.5
        
        if i > 0:
            # Contribution from beta_{i-1} to row i
            prev = i - 1
            pp, qq, rr, ss = prev + off_a, prev + off_b, prev + off_c, prev + off_d
            A[i, pp] += 0.5
            A[i, qq] -= 0.5
            A[i, rr] -= 0.5
            A[i, ss] += 0.5

    # Final row k: corresponds to u_k = beta_{k-1}
    last_idx = k - 1
    pp, qq, rr, ss = last_idx + off_a, last_idx + off_b, last_idx + off_c, last_idx + off_d
    A[k, pp] += 0.5
    A[k, qq] -= 0.5
    A[k, rr] -= 0.5
    A[k, ss] += 0.5
    
    return A

def solve_nnls(A, target):
    """
    Solves argmin_x ||Ax - target||^2 subject to x >= 0
    Using sklearn LinearRegression with positive=True
    """
    # fit_intercept=False is crucial because our model u = Ax has no bias term.
    # positive=True enforces non-negative coefficients (the delays).
    reg = LinearRegression(positive=True, fit_intercept=False)
    reg.fit(A, target)
    return reg.coef_

################################
# Non Editable Region Starting #
################################
def my_decode( w ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to invert a PUF linear model to get back delays
	# w is a single 1089-dim vector (last dimension being the bias term)
	# The output should be eight 32-dimensional vectors
     # 1. Reshape model to 33x33 matrix
    w_matrix = np.asarray(w, dtype=float).reshape(33, 33)
    
    # 2. SVD to find best Rank-1 Approximation
    # W approx sigma * u_vec * v_vec^T
    # Using numpy.linalg.svd
    U, S, Vt = svd(w_matrix, full_matrices=False)
    
    sigma = np.sqrt(S[0])
    u_hat = U[:, 0] * sigma
    v_hat = Vt[0, :] * sigma # Vt is already transposed in numpy SVD
    
    # Sign Correction Heuristic:
    # If the majority of the vector is negative, flip both to make them positive.
    # This helps the NNLS solver find a better fit.
    if np.sum(u_hat) < 0:
        u_hat = -u_hat
        v_hat = -v_hat

    # 3. Solve for delays using sklearn (NNLS equivalent)
    A = build_A()
    
    # Recover delays for first arbiter
    x1 = solve_nnls(A, u_hat)
    
    # Recover delays for second arbiter
    x2 = solve_nnls(A, v_hat)
    
    # 4. Format Output (8 vectors of size 32)
    k = 32
    
    # Unpack x1 (First Arbiter)
    a = x1[0:k]
    b = x1[k:2*k]
    c = x1[2*k:3*k]
    d = x1[3*k:4*k]
    
    # Unpack x2 (Second Arbiter)
    p = x2[0:k]
    q = x2[k:2*k]
    r = x2[2*k:3*k]
    s = x2[3*k:4*k]

    return a, b, c, d, p, q, r, s

