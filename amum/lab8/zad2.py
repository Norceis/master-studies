import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as animation

def f(x, y, t): return 0.5*np.sin(x**3) + 0.25*np.sin((y - t*0.05)**2)

fig, ax = plt.subplots(figsize=(10, 7))

def animate(t):
    plt.clf()

    xx, yy = np.meshgrid(np.arange(-np.pi / 2, np.pi / 2, 0.02),
                         np.arange(-np.pi, np.pi / 2, 0.02))

    Z = np.asarray(f(xx, yy, t))

    plt.contour(xx, yy, Z, colors='black')
    plt.contourf(xx, yy, Z, cmap='jet')

ani = animation(fig, animate, frames=80, interval=0.1, repeat=False)
plt.show()


