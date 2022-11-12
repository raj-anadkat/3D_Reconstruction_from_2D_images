from lse import least_squares_estimation
import numpy as np

def ransac_estimator(X1, X2, num_iterations=60000):
    sample_size = 8
    eps = 10**-4
    best_num_inliers = -1
    best_inliers = None
    best_E = None

    for i in range(num_iterations):

        # permuted indices randomly selects a permutation of points of X's
        permuted_indices = np.random.RandomState(seed=(i * 10)).permutation(np.arange(X1.shape[0]))

        # sample_indices selects first 8 of those indices, if x > 8, it'll only take 8
        sample_indices = permuted_indices[:sample_size]

        # test_indices are the ones on which you'll find the residuals having found E from the other 8 points (sample)
        test_indices = permuted_indices[sample_size:]

        """ YOUR CODE HERE
        """
        # assigning 8 sample points to x1 and x2
        x1 = X1[sample_indices]
        x2 = X2[sample_indices]

        # matching points according to the order (nx3)(3x3)(3xn) (qtEp)
        p = X1[test_indices].T
        q = X2[test_indices]

        # finding essential matrix out of those 8 correspondences.
        E = least_squares_estimation(x1, x2)

        # unit vector in z direction
        e3 = [0, 0, 1]

        # e3_hat is the skew symmetric matrix of vector e3 (ed)
        e3_hat = np.array([[0, -e3[2], e3[1]], [e3[2], 0, -e3[0]], [-e3[1], e3[0], 0]])

        residual = []
        # finding residuals for the rest of the samples where residual = d(x2, epi(x1))2 + d(x1, epi(x2))2
        # first term of the residual: d(x2, epi(x1))2  , where x2 =q and x1= p in qT(E)p = 0
        for i in range(len(q)):
            d1 = ((q[i, :] @ E @ p[:, i]) ** 2) / ((np.linalg.norm(e3_hat @ E @ p[:, i])) ** 2)
            d2 = ((p.T[i, :] @ E.T @ q.T[:, i]) ** 2) / ((np.linalg.norm(e3_hat @ E.T @ q.T[:, i])) ** 2)
            residual.append(d1 + d2)

        # counting how many residuals are lower than ε = 10−4 (consensus set)
        count = 0
        inlier_val = []
        j = 0
        best_num_indices_test = []

        for vals in residual:
            if vals < eps:
                count += 1
                inlier_val.append(j)  #
            j = j + 1

        if count > best_num_inliers:
            best_num_inliers = count
            best_E = E
            best_inlier_test = inlier_val
            for k in inlier_val:
                best_num_indices_test.append(test_indices[k])

            b = len(sample_indices)
            d = len(best_num_indices_test)
            sample_indices_1 = np.array(sample_indices, dtype=int).reshape(b, 1)
            best_num_indices_test_1 = np.array(best_num_indices_test, dtype=int).reshape(d, 1)
            best_inliers = np.append(sample_indices_1, best_num_indices_test_1)

    return best_E, best_inliers