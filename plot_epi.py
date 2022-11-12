import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_lines(lines, h, w):
    """ Utility function to plot lines
    """

    for i in range(lines.shape[1]):
        # plt.close('all')
        if abs(lines[0, i] / lines[1, i]) < 1:
            y0 = -lines[2, i] / lines[1, i]
            yw = y0 - w * lines[0, i] / lines[1, i]
            plt.plot([0, w], [y0, yw])


        else:
            x0 = -lines[2, i] / lines[0, i]
            xh = x0 - h * lines[1, i] / lines[0, i]
            plt.plot([x0, xh], [0, h])


def plot_epipolar_lines(image1, image2, uncalibrated_1, uncalibrated_2, E, K, plot=True):
    """ Plots the epipolar lines on the images
    """
    # uncaliberated points have the shape 3xN
    """ YOUR CODE HERE
    """

    epipolar_lines_in_1 = []
    epipolar_lines_in_2 = []
    print(K)
    K_inv = np.linalg.inv(K)
    # fundamental matrix
    F = (K_inv.T @ E @ K_inv)

    for i in range(len(uncalibrated_1.T)):
        epipolar_lines_in_2.append(F @ uncalibrated_1[:, i])
        epipolar_lines_in_1.append(F.T @ uncalibrated_2[:, i])

    epipolar_lines_in_1 = np.array(epipolar_lines_in_1, dtype=np.float32).T
    epipolar_lines_in_2 = np.array(epipolar_lines_in_2, dtype=np.float32).T

    """ END YOUR CODE
    """

    if (plot):

        plt.figure(figsize=(6.4 * 3, 4.8 * 3))
        ax = plt.subplot(1, 2, 1)
        ax.set_xlim([0, image1.shape[1]])
        ax.set_ylim([image1.shape[0], 0])
        plt.imshow(image1[:, :, ::-1])
        plot_lines(epipolar_lines_in_1, image1.shape[0], image1.shape[1])

        ax = plt.subplot(1, 2, 2)
        ax.set_xlim([0, image1.shape[1]])
        ax.set_ylim([image1.shape[0], 0])
        plt.imshow(image2[:, :, ::-1])
        plot_lines(epipolar_lines_in_2, image2.shape[0], image2.shape[1])

    else:
        return epipolar_lines_in_1, epipolar_lines_in_2
