import torch
import tensordict as td
import triangle as tr
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

# Definición del problema
h = 0.5**1

fracture_2d_data = {"vertices" : [[-1., 0.],
                                  [ 1., 0.],
                                  [-1., 1.],
                                  [ 1., 1.],
                                  [ 0., 0.],
                                  [ 0., .5],
                                  [ 0., 1.]],
                    "segments" : [[0, 1],
                                  [1, 3],
                                  [2, 3],
                                  [0, 2],
                                  [4, 5],
                                  [5, 6]]
                    }

# Triangulación 2D
fracture_triangulation = td.TensorDict(tr.triangulate(fracture_2d_data, 
                                                      "pqsena"+str(h)
                                                      ))

# Primeros tres vértices en 2D y 3D
fracture_2D_vertices = fracture_triangulation["vertices"][:3]  # (3,2)

# fracture_3D_vertices = torch.tensor( [[-1., 0., 0.],    
#                                       [ 1., 0., 0.],    
#                                       [-1., 1., 0.]])

fracture_3D_vertices = torch.tensor([[ 0., 0.,-1.], 
                                     [ 0., 0., 1.], 
                                     [ 0., 1.,-1.]])

# Construcción de la matriz extendida de 2D
hat_V = torch.cat([fracture_2D_vertices.T, torch.ones(1, 3)], dim=0)  # (3,3)
V = fracture_3D_vertices.T  # (3,3)

# Matriz de transformación
T = V @ torch.linalg.inv(hat_V)  # (3,3)

# Separar A y b
A = T[:, :2]  # (3x2)
b = T[:, [-1]]  # (3x1)

# Definir función de mapeo
def fracture_map(x):
    return (A @ x.mT + b).mT  # Output: (N, 3)

# Aplicar transformación a vértices 2D
vertices_3D = fracture_map(fracture_triangulation["vertices"])

triangles = fracture_triangulation["triangles"]

# Separar coordenadas
x, y, z = vertices_3D[:,0], vertices_3D[:,1], vertices_3D[:,2]

# Crear figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Graficar la superficie triangulada
ax.plot_trisurf(x, y, triangles, z, cmap='viridis', edgecolor='k')

# Opcional: mejorar visualización
ax.set_box_aspect([1,1,1])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.show()

