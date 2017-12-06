import numpy as np
from matplotlib import pyplot as plt


class node:

    def __init__(self, coordinates, index):
        self.coordinates = coordinates
        self.index = index

    def setCoordinates(self, coordinates):
        self.coordinates = coordinates


# Define Mesh Size Parameters
mrl = 10
nx = mrl
ny = mrl
x = np.linspace(0, 10, nx)
y = np.linspace(0, 10, ny)
x, y = np.meshgrid(x, y)

nodes = []
for i in range(0, x.shape[0]):
    for j in range(0, x.shape[1]):
        # nodes.append((x[i,j],y[i,j]))

        nodes.append(node((x[i, j], y[i, j]), i*ny+j))


plt.figure()
plt.plot(x, y, '.')
plt.show()
