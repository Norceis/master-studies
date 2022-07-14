import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)

with open('animals.npz', 'rb') as f:
    animals = np.load(f)['animals']

weight = []
height = []
samples = []

for element in animals:
    weight.append(float(element[2]))
    height.append(int(element[3]))
    samples.append([float(element[2]), int(element[3])])

images = [image.imread('0.png'),
          image.imread('1.jpg'),
          image.imread('2.png'),
          image.imread('3.jpg'),
          image.imread('4.jpg'),
          image.imread('5.jpg'),
          image.imread('6.jpg'),
          image.imread('7.png'),
          image.imread('8.jpg')]

fig, ax = plt.subplots()
sc = plt.scatter(weight, height)

for i in range(8):
    imagebox = OffsetImage(images[i], zoom=0.2)
    imagebox.image.axes = ax
    annot = AnnotationBbox(imagebox, samples[i],
                        xybox=(120, -80),
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0.5,
                        arrowprops=dict(
                           arrowstyle="->",
                           connectionstyle="angle,angleA=0,angleB=90,rad=3")
                        )
    ax.add_artist(annot)
    annot.set_visible(False)

def hover(event):
    if sc.contains(event)[0]:
        ind, = sc.contains(event)[1]["ind"]
        w,h = fig.get_size_inches()*fig.dpi
        ws = (event.x > w/2.)*-1 + (event.x <= w/2.)
        hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
        annot.xybox = (120*ws, -80*hs)
        annot.set_visible(True)
        annot.xy =(weight[ind], height[ind])
        imagebox.set_data(images[ind])
    else:
        annot.set_visible(False)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', hover)
plt.show()