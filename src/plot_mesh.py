import matplotlib.pyplot as plt

from fem import Mesh, Elements, Basis
import skfem 
import torch

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)

k_ref = 0
k_int = 3
k_test = 1
q = 2


mesh_H = skfem.MeshTri1().refined(k_ref)

V_H = Basis(Mesh(torch.tensor(mesh_H.p).T, torch.tensor(mesh_H.t).T), Elements(P_order = k_int, int_order = q))

nodes_array_H = V_H.mesh.coords4nodes.cpu()
triangles_array_H = V_H.mesh.nodes4elements.cpu()
dofs_array_H = V_H.coords4global_dofs.cpu()

fig, ax = plt.subplots()
ax.triplot(nodes_array_H[:, 0], nodes_array_H[:, 1], triangles_array_H, color='black', lw = 1)
ax.plot(dofs_array_H[:, 0], dofs_array_H[:, 1], 'o', color='red')
ax.set_aspect('equal')

mesh_h = mesh_H.refined(k_int - k_test)

V_h = Basis(Mesh(torch.tensor(mesh_h.p).T, torch.tensor(mesh_h.t).T), Elements(P_order = k_test, int_order = q))

nodes_array_h = V_h.mesh.coords4nodes.cpu()
triangles_array_h = V_h.mesh.nodes4elements.cpu()
dofs_array_h = V_h.coords4global_dofs.cpu()

fig, ax = plt.subplots()
ax.triplot(nodes_array_h[:, 0], nodes_array_h[:, 1], triangles_array_h, color='black', lw = 1)
ax.plot(dofs_array_h[:, 0], dofs_array_h[:, 1], 'o', color='red')
ax.set_aspect('equal')

plt.show()
