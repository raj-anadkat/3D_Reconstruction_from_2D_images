import numpy as np

def least_squares_estimation(X1, X2):
    """
      args: X1: NX3 matrix (p)
            X2: NX3 matrix (q)
      output: Essential matrix (3X3)

      Implementing 8 point algorithm
    """
    # qT(E)p = 0, finding A (nx8) vector.
    n = len(X1)
    p = X1
    q = X2
    A = np.zeros((n, 9))

    for i in range(n):
        A[i, 0:3] = p[i, 0] * q[i, :]
        A[i, 3:6] = p[i, 1] * q[i, :]
        A[i, 6:9] = p[i, 2] * q[i, :]

    # finding E' (9X1) using SVD setting h to be the smallest right singular vector
    [U, S, Vt] = np.linalg.svd(A)
    V = Vt.T
    E = V[:, 8]

    # E obtained is 9X1 is [e1 , e2 , e3] , reshaping it correctly to 3X3
    E = np.array(E, dtype=np.float32).reshape(3, 3).T

    # Re decomposing by applying SVD again
    Ua, Sa, Vta = np.linalg.svd(E)
    k = [1, 1, 0]
    Sa = np.diag(k)
    E = Ua @ Sa @ Vta

    """
    """
    return E
