from torch_fem import MeshTri, Basis, ElementTri
import triangle

mesh_data = triangle.triangulate(
    {"vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]}, "ne"
)

mesh = MeshTri(triangulation=mesh_data)

elements = ElementTri(polynomial_order=3, integration_order=6)

basis = Basis(mesh=mesh, element=elements)

print(basis.coords_4_global_dofs)
