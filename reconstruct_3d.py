import numpy as np


def reconstruct3D(transform_candidates, calibrated_1, calibrated_2):
    """This functions selects (T,R) among the 4 candidates transform_candidates
  such that all triangulated points are in front of both cameras.
  """
    print(np.shape(calibrated_1))
    # shape of caliberated = nX3

    best_num_front = -1
    best_candidate = None
    best_lambdas = None
    for candidate in transform_candidates:
        R = candidate['R']
        T = candidate['T']

        lambdas = np.zeros((2, calibrated_1.shape[0]))

        """ YOUR CODE HERE
            """
        j = 0
        for point1, point2 in zip(calibrated_1, calibrated_2):

            point1 = point1.reshape(3, 1)
            point2 = point2.reshape(3, 1)

            A = np.column_stack((point2, -(R @ point1)))
            B = T

            # solving for lambda 1, 2 through pseudo inverse of Ax = B
            lambda_values = (np.linalg.inv(A.T @ A) @ A.T @ B).reshape(2, 1)

            lambdas[0, j] = lambda_values[0]
            lambdas[1, j] = lambda_values[1]

            j += 1

        """ END YOUR CODE
        """
        num_front = np.sum(np.logical_and(lambdas[0] > 0, lambdas[1] > 0))

        if num_front > best_num_front:
            best_num_front = num_front
            best_candidate = candidate
            best_lambdas = lambdas
            print("best", num_front, best_lambdas[0].shape)
        else:
            print("not best", num_front)

    P1 = best_lambdas[1].reshape(-1, 1) * calibrated_1
    P2 = best_lambdas[0].reshape(-1, 1) * calibrated_2
    T = best_candidate['T']
    R = best_candidate['R']
    return P1, P2, T, R
