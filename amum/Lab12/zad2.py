import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as animation

matrix = np.asarray([[2, 1],
                     [-2, 1],
                     [-2, -1],
                     [2, -1]])

t = 0
k = 0
angle = np.radians(10)

fig, ax = plt.subplots(figsize=(10, 7))


def rotate(matrix, t):
    trans_matrix = np.array([[np.cos(t * angle), -np.sin(t * angle)],
                             [np.sin(t * angle), np.cos(t * angle)]])
    return matrix @ trans_matrix


def stretch(matrix, k):
    trans_matrix = np.array([[1 + k * 0.01, 0],
                             [0, 1 + k * 0.01]])
    return matrix @ trans_matrix


def angled(matrix, k):
    trans_matrix = np.array([[1, 1 + k * 0.1],
                             [1 + k * 0.1, 1]])
    return matrix @ trans_matrix


def animate_rotate(t):
    plt.clf()

    matrix_rotated = rotate(matrix, t)
    plt.fill(matrix_rotated[:, 0], matrix_rotated[:, 1])
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)


def animate_stretch(k):
    plt.clf()

    matrix_stretched = stretch(matrix, k)
    plt.fill(matrix_stretched[:, 0], matrix_stretched[:, 1])
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)


def animate_angle(k):
    plt.clf()

    matrix_angled = angled(matrix, k)
    plt.fill(matrix_angled[:, 0], matrix_angled[:, 1])
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)


# ani = animation(fig, animate_rotate, frames=100, interval=0.1, repeat=False)
ani = animation(fig, animate_stretch, frames=100, interval=0.1, repeat=True)
# ani = animation(fig, animate_angle, frames=100, interval=0.1, repeat=True)

plt.show()