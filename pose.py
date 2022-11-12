import numpy as np


def pose_candidates_from_E(E):
    R_90_pos = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R_90_neg = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])


    # Note: each candidate in the above list should be a dictionary with keys "T", "R"
    """ YOUR CODE HERE
    """
    U, S, Vt = np.linalg.svd(E)

    # T is the third column of U (2 solutions)
    T1 = U[:, 2]
    T2 = -T1

    # finding R matrix (2 solutions)
    R1 = U @ R_90_pos.T @ Vt
    R2 = U @ R_90_neg.T @ Vt

    # solutions:

    dict1 = dict([("R", R1), ("T", T1)])
    dict2 = dict([("R", R2), ("T", T1)])
    dict3 = dict([("R", R1), ("T", T2)])
    dict4 = dict([("R", R2), ("T", T2)])

    transform_candidates = [dict1, dict2, dict3, dict4]

    return transform_candidates
