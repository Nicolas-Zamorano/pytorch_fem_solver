import matplotlib.pyplot as plt
import numpy as np

from Triangulation import Triangulation

c4n, n4e = Triangulation(3)

nodes_array = np.array(c4n)
triangles_array = np.array(n4e)

fig, ax = plt.subplots()
ax.triplot(nodes_array[:, 0], nodes_array[:, 1], triangles_array, color='black', lw=1)
ax.plot(nodes_array[:, 0], nodes_array[:, 1], 'o', color='red')
ax.set_aspect('equal')
plt.show()
