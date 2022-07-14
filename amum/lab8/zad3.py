import numpy as np
import matplotlib.pyplot as plt

def f(x,y): return 0.5*np.sin(x**3) + 0.25*np.sin((y + np.pi)**2)

n = 100
x = np.linspace(-np.pi/2,np.pi/2,n)
y = np.linspace(-np.pi,np.pi/2,n)

xx, yy = np.meshgrid(np.arange(-np.pi/2, np.pi/2, 0.02),
                     np.arange(-np.pi, np.pi/2, 0.02))

Z = np.asarray(f(xx, yy))

plt.rcParams["figure.figsize"] = (10,8)
ax = plt.axes(projection ='3d')

ax.plot_surface(xx, yy, Z, cmap='gist_ncar')

plt.show()


