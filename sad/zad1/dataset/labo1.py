# analiza analogiczna do punktu poprzedniego, ale biorąc pod uwagę zmiany w latach
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

with open('dane-ludnosc.npz', 'rb') as f:
    content = np.load(f)
    columnsludnosc, dataludnosc = content['columns'].tolist(), content['data'].tolist()
with open('dane.npz', 'rb') as f:
    content = np.load(f)
    columns, data = content['columns'].tolist(), content['data'].tolist()
listaludnosci = sorted([[x[0], x[1], x[2], x[3], x[4], x[5]] for x in dataludnosc if x[1] != 'POLSKA'],
                       key=lambda a: a[1])
lista = sorted([[x[0], x[1], x[2], x[3], x[4], x[5]] for x in data if x[1] != 'POLSKA'], key=lambda a: a[1])

listaludnosci = np.array(listaludnosci)
lista = np.array(lista)

listaludnosci2, listaludnosci3, listaludnosci4 = listaludnosci[:, 1], listaludnosci[:, 5], listaludnosci[:, 4]
lista2, lista3 = lista[:, 1], lista[:, 4]

listaludnosci5 = []
lista4, lista5, lista6 = [], [], []

for i in lista2:
    if i not in lista4:
        lista4 += [i]
for i in lista3:
    lista5 += [float(i)]
for i in listaludnosci3:
    listaludnosci5 += [float(i)]
for i in listaludnosci4[0:22]:
    lista6 += [int(i)]

listawydatkow = []
wydatki = []
for i, j in zip(lista5, listaludnosci5):
    wydatki.append(int((int(round(i)) / (int(round(j))))))
for i in range(0, len(wydatki), 22):
    listawydatkow.append(wydatki[i:i + 22])

# listawydatkow


# lista4 to x, lista6 to y, listawydatkow to z

result = np.array(listawydatkow)
fig = plt.figure(figsize=(5, 5), dpi=150)
ax1 = fig.add_subplot(111, projection='3d')

xlabels = np.array(lista4)
xpos = np.arange(xlabels.shape[0])

ylabels = np.array(lista6)
ypos = np.arange(ylabels.shape[0])

xposM, yposM = np.meshgrid(xpos, ypos, copy=False)
zpos = result
zpos = zpos.ravel()

dx = 0.5
dy = 0.5
dz = zpos

ax1.w_xaxis.set_ticks(xpos + dx / 2.)
ax1.w_xaxis.set_ticklabels(xlabels)

ax1.w_yaxis.set_ticks(ypos + dy / 2.)
ax1.w_yaxis.set_ticklabels(ylabels)
# ax1.invert_yaxis()


plt.xticks(rotation=90)
plt.yticks(rotation=90)
values = np.linspace(0.2, 1., xposM.ravel().shape[0])
colors = cm.rainbow(values)
ax1.bar3d(xposM.ravel(), yposM.ravel(), dz * 0, dx, dy, dz, color=colors)
plt.show()
