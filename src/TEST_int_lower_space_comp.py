import torch 
import skfem
from fem import Mesh_Tri, Element_Tri, Basis
from skfem.helpers import grad, dot

mesh_sk_H = skfem.MeshTri1().refined(0) 

V_H_sk = skfem.Basis(mesh_sk_H, skfem.ElementTriP2(), intorder = 3)

f = lambda x, y : 1.

@skfem.BilinearForm
def a_sk(u, v, w):
    return dot(grad(u), grad(v))

@skfem.LinearForm
def l_sk(v, w):
    return f(*w.x) * v

A_sk = a_sk.assemble(V_H_sk)
b_sk = l_sk.assemble(V_H_sk)

u_sk = skfem.solve(*skfem.condense(A_sk, b_sk, D = V_H_sk.get_dofs()))
u_sk_torch = torch.tensor(u_sk).unsqueeze(-1)

mesh_sk_h = mesh_sk_H.refined(1)

V_h_sk = skfem.Basis(mesh_sk_h, skfem.ElementTriP1(), intorder = 3)
                 
I_H_u_sk = torch.tensor(V_h_sk.interpolate(u_sk))
I_H_u_grad_sk = torch.tensor(grad(V_h_sk.interpolate(u_sk)))

mesh_H = Mesh_Tri(torch.tensor(mesh_sk_H.p).T,  torch.tensor(mesh_sk_H.t).T)

V_H = Basis(mesh_H, Element_Tri(P_order = 2, int_order = 3))

def a(elements):
    return elements.v_grad @ elements.v_grad.mT

def l(elements):
    return f(*elements.integration_points) * elements.v

A = V_H.reduce(V_H.integrate_bilineal_form(a))
b = V_H.reduce(V_H.integrate_lineal_form(l))

u = torch.zeros(V_H.basis_parameters["linear_form_shape"])
u[V_H.basis_parameters["inner_dofs"]] = torch.linalg.solve(A, b)

mesh_h = Mesh_Tri(torch.tensor(mesh_sk_h.p).T, torch.tensor(mesh_sk_h.t).T)

V_h = Basis(mesh_h, Element_Tri(P_order = 1, int_order = 3))

I_H_u, I_H_u_grad = V_H.interpolate(V_h, u)

I_u_values, I_u_count = torch.unique(I_H_u.reshape(-1), return_counts=True)
I_u_sk_values, I_u_sk_count = torch.unique(I_H_u_sk.reshape(-1), return_counts=True)

print("Values of I_u   ", I_u_values.numpy(),"Repeat:", I_u_count.numpy())
print("Values of I_u_sk", I_u_sk_values.numpy(),"Repeat:", I_u_sk_count.numpy())

I_u_grad_values, I_u_grad_count = torch.unique(I_H_u_grad.reshape(-1), return_counts=True)
I_u_grad_sk_values, I_u_grad_sk_count = torch.unique(I_H_u_grad_sk.reshape(-1), return_counts=True)

print("Values of I_u_grad   ", I_u_grad_values.numpy(),"Repeat:", I_u_grad_count.numpy())
print("Values of I_u_grad_sk", I_u_grad_sk_values.numpy(),"Repeat:", I_u_grad_sk_count.numpy())

print("u error   norm:", (torch.norm(u_sk_torch - u)/torch.norm(u_sk_torch)))
print("I_u error norm:", (torch.norm(I_H_u.squeeze(-1).squeeze(-1) - I_H_u_sk)/torch.norm(I_H_u_sk)).item())
print("I_u_grad  norm:", (torch.norm(I_H_u_grad.squeeze(-2).permute((2,0,1)) - I_H_u_grad_sk)/torch.norm(I_H_u_grad_sk)).item())
