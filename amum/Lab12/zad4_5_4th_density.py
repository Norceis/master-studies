import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as animation


matrix = np.asarray([[2, 1, 1],
                    [-2, 1, 1],
                    [2, -1, 1],
                    [-2, -1, 1],
                    [2, 1, -1],
                    [-2, 1, -1],
                    [2, -1, -1],
                    [-2, -1, -1]]
                   )



t = 0
angle = np.radians(10)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection='3d')


def translation(matrix, t):
    trans_matrix = np.array([[np.cos(t * angle), -np.sin(t * angle), 1 + t * 0.05],
                             [np.sin(t * angle), np.cos(t * angle), 1 + t * 0.05],
                             [0, 0, 1]])
    return matrix @ trans_matrix

def translation_wicked(matrix, t):
    trans_matrix = np.array([[np.cos(t * angle), -np.sin(t * angle), 1 + t * 0.05],
                             [np.sin(t * angle), np.cos(t * angle), 1 + t * 0.05],
                             [0, 0, 1]])
    return matrix @ np.linalg.pinv(trans_matrix)


def animate_translation(t):
    plt.cla()

    if t<100:
        matrix_translated = translation(matrix, t)
    else:
        matrix_translated = translation(matrix, 200-t)

    ax.scatter(matrix_translated[:, 0], matrix_translated[:, 1], matrix_translated[:, 2])

    ax.plot([matrix_translated[0, 0], matrix_translated[1, 0]], [matrix_translated[0, 1], matrix_translated[1, 1]], [matrix_translated[0, 2], matrix_translated[1, 2]])
    ax.plot([matrix_translated[2, 0], matrix_translated[3, 0]], [matrix_translated[2, 1], matrix_translated[3, 1]], [matrix_translated[2, 2], matrix_translated[3, 2]])
    ax.plot([matrix_translated[4, 0], matrix_translated[5, 0]], [matrix_translated[4, 1], matrix_translated[5, 1]], [matrix_translated[4, 2], matrix_translated[5, 2]])
    ax.plot([matrix_translated[6, 0], matrix_translated[7, 0]], [matrix_translated[6, 1], matrix_translated[7, 1]], [matrix_translated[6, 2], matrix_translated[7, 2]])

    ax.plot([matrix_translated[0, 0], matrix_translated[2, 0]], [matrix_translated[0, 1], matrix_translated[2, 1]], [matrix_translated[0, 2], matrix_translated[2, 2]])
    ax.plot([matrix_translated[1, 0], matrix_translated[3, 0]], [matrix_translated[1, 1], matrix_translated[3, 1]], [matrix_translated[1, 2], matrix_translated[3, 2]])
    ax.plot([matrix_translated[4, 0], matrix_translated[6, 0]], [matrix_translated[4, 1], matrix_translated[6, 1]], [matrix_translated[4, 2], matrix_translated[6, 2]])
    ax.plot([matrix_translated[5, 0], matrix_translated[7, 0]], [matrix_translated[5, 1], matrix_translated[7, 1]], [matrix_translated[5, 2], matrix_translated[7, 2]])

    ax.plot([matrix_translated[0, 0], matrix_translated[4, 0]], [matrix_translated[0, 1], matrix_translated[4, 1]], [matrix_translated[0, 2], matrix_translated[4, 2]])
    ax.plot([matrix_translated[1, 0], matrix_translated[5, 0]], [matrix_translated[1, 1], matrix_translated[5, 1]], [matrix_translated[1, 2], matrix_translated[5, 2]])
    ax.plot([matrix_translated[2, 0], matrix_translated[6, 0]], [matrix_translated[2, 1], matrix_translated[6, 1]], [matrix_translated[2, 2], matrix_translated[6, 2]])
    ax.plot([matrix_translated[3, 0], matrix_translated[7, 0]], [matrix_translated[3, 1], matrix_translated[7, 1]], [matrix_translated[3, 2], matrix_translated[7, 2]])

    ax.set_xlim3d(-12, 12)
    ax.set_ylim3d(-12, 12)
    ax.set_zlim3d(-12, 12)


ani = animation(fig, animate_translation, frames=200, interval=0.1, repeat=True)
plt.show()
