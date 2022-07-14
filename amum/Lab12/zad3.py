import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as animation

matrix = np.asarray([[2, 1],
                     [-2, 1],
                     [-2, -1],
                     [2, -1]])


def expand(matrix):
    ones = np.ones(len(matrix)).reshape(len(matrix), 1)
    return np.append(matrix, ones, axis=1)


matrix = expand(matrix)

t = 0
angle = np.radians(10)

fig, ax = plt.subplots(figsize=(10, 7))

def translation(matrix, t):
    trans_matrix = np.array([[np.cos(t * angle), -np.sin(t * angle), 1 + t * 0.05],
                             [np.sin(t * angle), np.cos(t * angle), 1 + t * 0.05],
                             [0, 0, 1]])
    return matrix @ trans_matrix.T

def animate_translation(t):
    plt.cla()

    matrix_translated = translation(matrix, t)
    plt.fill(matrix_translated[:, 0], matrix_translated[:, 1])
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)


ani = animation(fig, animate_translation, frames=100, interval=0.1, repeat=True)

plt.show()
