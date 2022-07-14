import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as animation
import mpl_toolkits.mplot3d.axes3d as p3

def f(x, y, t): return 0.5*np.sin(x**3) + 0.25*np.sin((y - t)**2)

fig, ax = plt.subplots(figsize=(10, 7))

def animate(t):
    plt.clf()

    xx, yy = np.meshgrid(np.arange(-np.pi / 2, np.pi / 2, 0.02),
                         np.arange(-np.pi, np.pi / 2, 0.02))

    Z = np.asarray(f(xx, yy, t))

    ax = p3.Axes3D(fig)

    ax.plot_surface(xx, yy, Z, cmap='gist_ncar')

ani = animation(fig, animate, frames=80, interval=0.01, repeat=False)
plt.show()
