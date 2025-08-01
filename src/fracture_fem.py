import torch
import tensordict as td
from fem import Abstract_Mesh, Abstract_Element, Abstract_Basis

torch.set_default_dtype(torch.float64)


class Fractures(Abstract_Mesh):
    def __init__(self, triangulations: list, fractures_3D_data: torch.Tensor):

        self.fractures_3D_data = fractures_3D_data

        self.edges_permutations = torch.tensor([[0, 1], [1, 2], [0, 2]])

        self.local_triangulations, self.edges_parameters = self.stack_triangulations(
            triangulations, fractures_3D_data
        )

        self.mesh_parameters = self.compute_mesh_parameters(self.local_triangulations)

    def compute_mesh_parameters(self, triangulations):

        nb_fractures, nb_nodes, nb_dimensions = triangulations["vertices"].shape
        _, nb_simplex, nb_size4simplex = triangulations["triangles"].shape

        mesh_parameters = {
            "nb_fractures": nb_fractures,
            "nb_nodes": nb_nodes,
            "nb_dimensions": nb_dimensions,
            "nb_simplex": nb_simplex,
            "nb_size4simplex": nb_size4simplex,
        }

        return mesh_parameters

    def stack_triangulations(self, fracture_triangulations: list, fractures_3D_data):

        stack_vertices = torch.stack(
            [triangulation["vertices"] for triangulation in fracture_triangulations],
            dim=0,
        )
        stack_vertex_markers = torch.stack(
            [
                triangulation["vertex_markers"]
                for triangulation in fracture_triangulations
            ],
            dim=0,
        )

        stack_triangles = torch.stack(
            [triangulation["triangles"] for triangulation in fracture_triangulations],
            dim=0,
        )

        stack_edges = torch.stack(
            [triangulation["edges"] for triangulation in fracture_triangulations], dim=0
        )
        stack_edge_markers = torch.stack(
            [
                triangulation["edge_markers"]
                for triangulation in fracture_triangulations
            ],
            dim=0,
        )

        stack_edge_parameters = torch.stack(
            [
                td.TensorDict(self.compute_edges_values(triangulation))
                for triangulation in fracture_triangulations
            ],
            dim=0,
        )

        stack_coords4triangles = stack_vertices[
            torch.arange(stack_vertices.shape[0])[:, None, None], stack_triangles
        ]

        fractures_2D_vertices = stack_vertices[:, :3, :]

        fractures_3D_vertices = fractures_3D_data[:, :3, :].mT

        hat_V = torch.cat(
            [
                fractures_2D_vertices.mT,
                torch.ones_like(fractures_3D_vertices[:, [-1], :]),
            ],
            dim=-2,
        )

        A_b = fractures_3D_vertices @ torch.linalg.inv(hat_V)

        # Separar A y b
        fractures_map_jacobian = A_b[..., :2]
        b = A_b[..., [-1]]

        fractures_map_jacobian_int = fractures_map_jacobian.unsqueeze(-3).unsqueeze(-3)
        b_int = b.unsqueeze(-3).unsqueeze(-3)

        # Definir función de mapeo
        def fractures_map(x):
            return (fractures_map_jacobian @ x.mT + b).mT

        def fractures_map_int(x):
            return (fractures_map_jacobian_int @ x.mT + b_int).mT

        det_fractures_map_jacobian = torch.norm(
            torch.linalg.cross(*torch.split(fractures_map_jacobian, 1, dim=-1), dim=-2),
            dim=-2,
            keepdim=True,
        )

        fractures_map_jacobian_inv = (
            torch.linalg.inv(fractures_map_jacobian.mT @ fractures_map_jacobian)
            @ fractures_map_jacobian.mT
        )

        stack_vertices_3D = fractures_map(stack_vertices)

        stack_coords_3D4triangles = stack_vertices_3D[
            torch.arange(stack_vertices_3D.shape[0])[:, None, None], stack_triangles
        ]

        stack_normal4inner_edges_3D = fractures_map_int(
            stack_edge_parameters["normal4inner_edges"].unsqueeze(-2)
        )

        stack_triangulation = td.TensorDict(
            vertices=stack_vertices,
            vertices_3D=stack_vertices_3D,
            vertex_markers=stack_vertex_markers,
            triangles=stack_triangles,
            edges=stack_edges,
            edge_markers=stack_edge_markers,
            fractures_map=fractures_map,
            coords4triangles=stack_coords4triangles,
            stack_coords_3D4triangles=stack_coords_3D4triangles,
            fractures_map_jacobian=fractures_map_jacobian,
            det_fractures_map_jacobian=det_fractures_map_jacobian,
            fractures_map_jacobian_inv=fractures_map_jacobian_inv,
            fractures_map_int=fractures_map_int,
            normal4inner_edges_3D=stack_normal4inner_edges_3D,
        )

        return stack_triangulation, stack_edge_parameters

    def compute_edges_values(self, triangulation: dict):

        nodes4elements = triangulation["triangles"]

        coords4nodes = triangulation["vertices"]

        coords4elements = coords4nodes[nodes4elements]

        nodes4edges, _ = torch.sort(
            nodes4elements[..., self.edges_permutations], dim=-1
        )

        coords4edges = coords4nodes[nodes4edges]

        coords4edges_1, coords4edges_2 = torch.split(coords4edges, 1, dim=-2)

        elements_diameter = torch.min(
            torch.norm(coords4edges_2 - coords4edges_1, dim=-1, keepdim=True),
            dim=-2,
            keepdim=True,
        )[0]

        nodes4unique_edges = triangulation["edges"]
        boundary_mask = triangulation["edge_markers"].squeeze(-1)

        edges_idx = self.get_edges_idx(nodes4elements, nodes4unique_edges)

        nodes4boundary_edges = nodes4unique_edges[boundary_mask == 1]
        nodes4inner_edges = nodes4unique_edges[boundary_mask != 1]
        nodes4boundary = torch.nonzero(triangulation["vertex_markers"])[:, [0]]

        elements4boundary_edges = (
            (
                nodes4boundary_edges.unsqueeze(-2).unsqueeze(-2)
                == nodes4elements.unsqueeze(-1).unsqueeze(-4)
            )
            .any(dim=-2)
            .all(dim=-1)
            .float()
            .argmax(dim=-1, keepdim=True)
        )
        elements4inner_edges = torch.nonzero(
            (
                nodes4inner_edges.unsqueeze(-2).unsqueeze(-2)
                == nodes4elements.unsqueeze(-1).unsqueeze(-4)
            )
            .any(dim=-2)
            .all(dim=-1),
            as_tuple=True,
        )[1].reshape(-1, 2)

        nodes_idx4boundary_edges = torch.nonzero(
            (nodes4unique_edges.unsqueeze(-2) == nodes4boundary_edges.unsqueeze(-3))
            .all(dim=-1)
            .any(dim=-1)
        )

        # compute inner edges normal vector

        coords4inner_edges = coords4nodes[nodes4inner_edges]

        coords4inner_edges_1, coords4inner_edges_2 = torch.split(
            coords4inner_edges, 1, dim=-2
        )

        inner_edges_vector = coords4inner_edges_2 - coords4inner_edges_1

        inner_edges_length = torch.norm(inner_edges_vector, dim=-1, keepdim=True)

        normal4inner_edges = (
            inner_edges_vector[..., [1, 0]]
            * torch.tensor([-1.0, 1.0])
            / inner_edges_length
        )

        inner_elements_centroid = coords4elements[elements4inner_edges].mean(dim=-2)

        inner_elements_centroid_1, inner_elements_centroid_2 = torch.split(
            inner_elements_centroid, 1, dim=-2
        )

        inner_direction_mask = (
            normal4inner_edges * (inner_elements_centroid_2 - inner_elements_centroid_1)
        ).sum(dim=-1)

        normal4inner_edges[inner_direction_mask < 0] *= -1

        edges_parameters = {
            "nodes4edges": nodes4edges,
            "edges_idx": edges_idx,
            "nodes4unique_edges": nodes4unique_edges,
            "elements4boundary_edges": elements4boundary_edges,
            "nodes4inner_edges": nodes4inner_edges,
            "elements4inner_edges": elements4inner_edges,
            "nodes_idx4boundary_edges": nodes_idx4boundary_edges,
            "inner_edges_length": inner_edges_length,
            "normal4inner_edges": normal4inner_edges,
            "elements_diameter": elements_diameter,
            "nodes4boundary": nodes4boundary,
        }

        return edges_parameters

    def get_edges_idx(self, nodes4elements, nodes4unique_edges):
        # 1. Obtener los 3 edges de cada triángulo
        i0 = nodes4elements[..., 0]
        i1 = nodes4elements[..., 1]
        i2 = nodes4elements[..., 2]

        # Cada edge como par ordenado (min, max)
        tri_edges = torch.stack(
            [
                torch.stack([torch.min(i0, i1), torch.max(i0, i1)], dim=1),
                torch.stack([torch.min(i1, i2), torch.max(i1, i2)], dim=1),
                torch.stack([torch.min(i2, i0), torch.max(i2, i0)], dim=1),
            ],
            dim=1,
        )  # (n_triangles, 3, 2)

        # 2. Convertimos cada par (a,b) en una clave única: a * M + b
        M = nodes4elements.max().item() + 1  # M debe ser mayor al número de nodos
        tri_keys = tri_edges[:, :, 0] * M + tri_edges[:, :, 1]  # (n_triangles, 3)

        # 3. Hacer lo mismo con edges únicos
        edge_keys = (
            nodes4unique_edges.min(dim=-1).values * M
            + nodes4unique_edges.max(dim=-1).values
        )  # (n_unique_edges,)

        # 4. Crear tabla de búsqueda
        sorted_keys, sorted_idx = torch.sort(edge_keys)  # Necesario para searchsorted
        flat_tri_keys = tri_keys.flatten()  # (n_triangles * 3,)

        # 5. Buscar cada key en sorted_keys
        edge_pos = torch.searchsorted(sorted_keys, flat_tri_keys)
        edge_indices = sorted_idx[edge_pos].reshape(tri_keys.shape)  # (n_triangles, 3)

        return edge_indices


class Element_Fracture(Abstract_Element):
    def __init__(self, P_order, int_order):

        self.barycentric_grad = torch.tensor([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])

        self.reference_element_area = 0.5

        super().__init__(P_order, int_order)

    def compute_barycentric_coordinates(self, x):

        return torch.stack(
            [1.0 - x[..., [0]] - x[..., [1]], x[..., [0]], x[..., [1]]], dim=-2
        )

    def compute_integral_values(
        self,
        coords4elements: torch.Tensor,
        fractures_map,
        fractures_map_jacobian_inv,
        det_fractures_map_jacobian,
    ):

        bar_coords = self.compute_barycentric_coordinates(self.gaussian_nodes)

        gaussian_map_nodes, det_map_jacobian, self.inv_map_jacobian = self.compute_map(
            coords4elements, bar_coords
        )

        gaussian_map_3D_nodes = fractures_map(gaussian_map_nodes)

        v, v_grad = self.compute_shape_functions(
            bar_coords,
            self.inv_map_jacobian,
            fractures_map_jacobian_inv.unsqueeze(-3).unsqueeze(-3),
        )

        integration_points = torch.split(gaussian_map_3D_nodes, 1, dim=-1)

        dx = (
            self.reference_element_area
            * self.gaussian_weights
            * det_map_jacobian
            * det_fractures_map_jacobian.reshape(-1, 1, 1, 1, 1)
        )

        return v, v_grad, integration_points, dx

    def compute_map(self, coords4elements: torch.Tensor, bar_coords: torch.Tensor):

        mapp = bar_coords.mT @ coords4elements.unsqueeze(-3)

        map_jacobian = coords4elements.mT @ self.barycentric_grad

        det_map_jacobian, inv_map_jacobian = self.compute_det_and_inv_map(map_jacobian)

        return mapp, det_map_jacobian, inv_map_jacobian

    def compute_shape_functions(
        self, bar_coords, inv_map_jacobian: torch.Tensor, fractures_map_jacobian_inv
    ):

        v, v_grad = self.shape_functions_value_and_grad(
            bar_coords, inv_map_jacobian, fractures_map_jacobian_inv
        )

        return v, v_grad

    @staticmethod
    def compute_inverse_map(
        first_node: torch.Tensor,
        integration_points: torch.Tensor,
        inv_map_jacobian: torch.Tensor,
    ):

        integration_points = torch.concat(integration_points, dim=-1)

        inv_map = (integration_points - first_node) @ inv_map_jacobian.mT

        return inv_map

    def shape_functions_value_and_grad(
        self,
        bar_coords: torch.Tensor,
        inv_map_jacobian: torch.Tensor,
        fractures_map_jacobian_inv,
    ):

        v = bar_coords

        v_grad = self.barycentric_grad @ inv_map_jacobian @ fractures_map_jacobian_inv

        return v, v_grad

    def compute_gauss_values(self):

        if self.int_order == 1:

            gaussian_nodes = torch.tensor([[1 / 3, 1 / 3]])

            gaussian_weights = torch.tensor([[[1.0]]])

        if self.int_order == 2:

            gaussian_nodes = torch.tensor(
                [[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]]
            )

            gaussian_weights = torch.tensor([[[1 / 3]], [[1 / 3]], [[1 / 3]]])

        if self.int_order == 3:

            gaussian_nodes = torch.tensor(
                [[1 / 3, 1 / 3], [0.6, 0.2], [0.2, 0.6], [0.2, 0.2]]
            )

            gaussian_weights = torch.tensor(
                [[[-9 / 16]], [[25 / 48]], [[25 / 48]], [[25 / 48]]]
            )

        if self.int_order == 4:

            gaussian_nodes = torch.tensor(
                [
                    [0.816847572980459, 0.091576213509771],
                    [0.091576213509771, 0.816847572980459],
                    [0.091576213509771, 0.091576213509771],
                    [0.108103018168070, 0.445948490915965],
                    [0.445948490915965, 0.108103018168070],
                    [0.445948490915965, 0.445948490915965],
                ]
            )

            gaussian_weights = torch.tensor(
                [
                    [[0.109951743655322]],
                    [[0.109951743655322]],
                    [[0.109951743655322]],
                    [[0.223381589678011]],
                    [[0.223381589678011]],
                    [[0.223381589678011]],
                ]
            )

        return gaussian_nodes, gaussian_weights

    @staticmethod
    def compute_det_and_inv_map(map_jacobian: torch.Tensor):

        ab, cd = torch.split(map_jacobian, 1, dim=-2)

        a, b = torch.split(ab, 1, dim=-1)
        c, d = torch.split(cd, 1, dim=-1)

        det_map_jacobian = (a * d - b * c).unsqueeze(-3)

        inv_map_jacobian = (1 / det_map_jacobian) * torch.stack(
            [torch.concat([d, -b], dim=-1), torch.concat([-c, a], dim=-1)], dim=-2
        )

        return det_map_jacobian, inv_map_jacobian


class Fracture_Element_Line(Element_Fracture):
    def __init__(self, P_order: int = 1, int_order: int = 2):

        self.compute_barycentric_coordinates = lambda x: torch.concat(
            [0.5 * (1.0 - x), 0.5 * (1.0 + x)], dim=-1
        )

        self.barycentric_grad = torch.tensor([[-0.5], [0.5]])

        self.reference_element_area = 2.0

        self.P_order = P_order
        self.int_order = int_order

        self.gaussian_nodes, self.gaussian_weights = self.compute_gauss_values()

    def compute_gauss_values(self):

        if self.int_order == 1:
            gaussian_nodes = torch.tensor([[-1.0], [1.0]])

            gaussian_weights = torch.tensor([[[1.0]], [[1.0]]])

        if self.int_order == 2:

            nodes = 1.0 / torch.sqrt(torch.tensor(3.0))

            gaussian_nodes = torch.tensor([[-nodes], [nodes]])

            gaussian_weights = torch.tensor([[[0.5]], [[0.5]]])

        if self.int_order == 3:

            nodes = torch.sqrt(torch.tensor(3 / 5))

            gaussian_nodes = torch.tensor([[0], [-nodes], [nodes]])

            gaussian_weights = torch.tensor([[[8 / 18]], [[5 / 18]], [[5 / 18]]])

        return gaussian_nodes, gaussian_weights.unsqueeze(0)

    def compute_det_and_inv_map(self, map_jacobian):

        det_map_jacobian = torch.linalg.norm(map_jacobian, dim=-2, keepdim=True)

        inv_map_jacobian = 1.0 / det_map_jacobian

        return det_map_jacobian.unsqueeze(-1), inv_map_jacobian

    def shape_functions_value_and_grad(
        self,
        bar_coords: torch.Tensor,
        inv_map_jacobian: torch.Tensor,
        fractures_map_jacobian_inv,
    ):

        v = bar_coords

        # v_grad = self.barycentric_grad @ inv_map_jacobian @ fractures_map_jacobian_inv

        v_grad = self.barycentric_grad @ inv_map_jacobian

        return v, v_grad


class Fracture_Basis(Abstract_Basis):
    def __init__(self, mesh: Abstract_Mesh, elements: Abstract_Element):

        self.elements = elements
        self.mesh = mesh

        self.global_triangulation = self.build_global_triangulation(
            self.mesh.local_triangulations
        )

        self.coords4elements = self.mesh.local_triangulations["coords4triangles"]

        # self.coords4elements = self.global_triangulation["vertices_2D"][self.global_triangulation["triangles"]].reshape(mesh.mesh_parameters["nb_fractures"], -1, 3, 2)

        (
            self.v,
            self.v_grad,
            self.integration_points,
            self.dx,
        ) = elements.compute_integral_values(
            self.coords4elements,
            mesh.local_triangulations["fractures_map_int"],
            mesh.local_triangulations["fractures_map_jacobian_inv"],
            mesh.local_triangulations["det_fractures_map_jacobian"],
        )

        self.coords4global_dofs, self.global_dofs4elements, self.nodes4boundary_dofs = (
            self.compute_dofs(self.global_triangulation)
        )

        self.basis_parameters = self.compute_basis_parameters(
            self.coords4global_dofs, self.global_dofs4elements, self.nodes4boundary_dofs
        )

    def build_global_triangulation(self, local_triangulations):

        nb_fractures, nb_vertices, nb_dim = local_triangulations["vertices"].shape

        nb_edges = local_triangulations["edges"].shape[-2]

        local_triangulation_3D_coords = local_triangulations["vertices_3D"].reshape(
            -1, 3
        )

        global_vertices_3D, global2local_idx, vertex_counts = torch.unique(
            local_triangulation_3D_coords,
            dim=0,
            return_inverse=True,
            return_counts=True,
        )

        nb_global_vertices = global_vertices_3D.shape[-2]

        traces__global_vertices_idx = torch.nonzero(vertex_counts > 1, as_tuple=True)[0]

        local2global_idx = torch.full(
            (nb_global_vertices,), (nb_fractures * nb_vertices) + 1, dtype=torch.int64
        )

        local2global_idx.scatter_reduce_(
            0,
            global2local_idx,
            torch.arange(nb_fractures * nb_vertices),
            reduce="amin",
            include_self=True,
        )

        global_vertices_2D = local_triangulations["vertices"].reshape(-1, 2)[
            local2global_idx
        ]

        vertices_offset = torch.arange(nb_fractures)[:, None, None] * nb_vertices

        global_triangles = global2local_idx[
            local_triangulations["triangles"] + vertices_offset
        ].reshape(-1, 3)

        local_edges_2_global = global2local_idx[
            local_triangulations["edges"] + vertices_offset
        ].reshape(-1, 2)

        global_edges, global2local_edges_idx, edges_counts = torch.unique(
            local_edges_2_global.reshape(-1, 2),
            dim=0,
            return_inverse=True,
            return_counts=True,
        )

        edge_offset = torch.arange(nb_fractures)[:, None] * nb_edges

        traces_global_edges_idx = torch.nonzero(edges_counts > 1, as_tuple=True)[0]

        traces_local_edges_idx = (
            torch.nonzero(
                torch.isin(global2local_edges_idx, traces_global_edges_idx),
                as_tuple=True,
            )[0].reshape(nb_fractures, -1)
            - edge_offset
        )

        nb_global_edges = global_edges.shape[-2]

        local2global_edges_idx = torch.full(
            (nb_global_edges,), (nb_fractures * nb_edges) + 1, dtype=torch.int64
        )

        local2global_edges_idx.scatter_reduce_(
            0,
            global2local_edges_idx,
            torch.arange(nb_fractures * nb_edges),
            reduce="amin",
            include_self=True,
        )

        global_vertices_marker = local_triangulations["vertex_markers"].reshape(-1)[
            local2global_idx
        ]
        global_edges_marker = local_triangulations["edge_markers"].reshape(-1)[
            local2global_edges_idx
        ]

        global_triangulation = td.TensorDict(
            vertices_3D=global_vertices_3D,
            vertices_2D=global_vertices_2D,
            vertex_markers=global_vertices_marker,
            triangles=global_triangles,
            edges=global_edges,
            edge_markers=global_edges_marker,
            global2local_idx=global2local_idx,
            local2global_idx=local2global_idx,
            traces__global_vertices_idx=traces__global_vertices_idx,
            traces_global_edges_idx=traces_global_edges_idx,
            traces_local_edges_idx=traces_local_edges_idx,
        )

        return global_triangulation

    def compute_dofs(self, global_triangulation):

        coords4global_dofs = global_triangulation["vertices_2D"]
        global_dofs4elements = global_triangulation["triangles"]
        nodes4boundary_dofs = torch.nonzero(
            global_triangulation["vertex_markers"] == 1
        )[:, 0]

        return coords4global_dofs, global_dofs4elements, nodes4boundary_dofs

    def compute_basis_parameters(
        self, coords4global_dofs, global_dofs4elements, nodes4boundary_dofs
    ):

        nb_global_dofs = self.global_triangulation["vertices_2D"].shape[-2]
        nb_local_dofs = self.global_triangulation["triangles"].shape[-1]

        inner_dofs = torch.arange(nb_global_dofs)[
            ~torch.isin(torch.arange(nb_global_dofs), nodes4boundary_dofs)
        ]

        rows_idx = (
            self.global_triangulation["triangles"]
            .repeat(1, 1, nb_local_dofs)
            .reshape(-1)
        )
        cols_idx = (
            self.global_triangulation["triangles"]
            .repeat_interleave(nb_local_dofs)
            .reshape(-1)
        )

        form_idx = self.global_triangulation["triangles"].reshape(-1)

        basis_parameters = {
            "bilinear_form_shape": (nb_global_dofs, nb_global_dofs),
            "bilinear_form_idx": (rows_idx, cols_idx),
            "linear_form_shape": (nb_global_dofs, 1),
            "linear_form_idx": (form_idx,),
            "inner_dofs": inner_dofs,
            "nb_dofs": nb_global_dofs,
        }

        return basis_parameters

    def reduce(self, tensor):
        idx = self.basis_parameters["inner_dofs"]
        if tensor.shape[-1] != 1:
            return tensor[idx, :][:, idx]
        else:
            return tensor[idx]

    def apply_BC(self, A: torch.tensor, b: torch.tensor, g, dofs_idx=None):

        if dofs_idx == None:
            dofs_idx = self.nodes4boundary_dofs

        coords4boundary_dofs = self.coords4global_dofs[dofs_idx]

        A[dofs_idx, :] = 0

        A[dofs_idx, dofs_idx] = 1

        b[dofs_idx, :] = g(*torch.split(coords4boundary_dofs, 1, dim=-1))

        return A, b

    # def interpolate(self, tensor, is_array: bool = True):

    #     dofs_idx = self.global_triangulation["triangles"].reshape(self.mesh.mesh_parameters["nb_fractures"], -1, 1, 3)
    #     v = self.v
    #     v_grad = self.v_grad

    #     if is_array:

    #         interpolation = (tensor[dofs_idx] * v).sum(-2, keepdim = True)

    #         interpolation_grad = (tensor[dofs_idx] * v_grad).sum(-2, keepdim = True)

    #     else:

    #         nodes = torch.split(self.global_triangulation["vertices_3D"], 1 , dim = -1)

    #         interpolation =  (tensor(*nodes)[dofs_idx] * v).sum(-2, keepdim = True)

    #         interpolation_grad = (tensor(*nodes)[dofs_idx] * v_grad).sum(-2, keepdim = True)

    #     return interpolation, interpolation_grad

    def interpolate(self, basis, tensor=None):

        if basis == self:
            dofs_idx = self.global_triangulation["triangles"].reshape(
                self.mesh.mesh_parameters["nb_fractures"], -1, 1, 3
            )
            v = self.v
            v_grad = self.v_grad

        # else:

        #     elements_mask = self.mesh.map_fine_mesh(basis.mesh)

        #     dofs_idx = self.global_dofs4elements[elements_mask]

        #     coords4elements_first_node = self.coords4elements[..., [0], :][elements_mask]

        #     inv_map_jacobian = self.elements.inv_map_jacobian[elements_mask]

        #     new_integrations_points = self.elements.compute_inverse_map(coords4elements_first_node,
        #                                                                 basis.integration_points,
        #                                                                 inv_map_jacobian)

        #     _, v, v_grad = self.elements.compute_shape_functions(new_integrations_points.squeeze(-2), inv_map_jacobian)

        if basis.__class__ == Interior_Facet_Fracture_Basis:

            elements_mask = basis.mesh.edges_parameters["elements4inner_edges"]

            nodes4elements = self.global_triangulation["triangles"].reshape(
                self.mesh.mesh_parameters["nb_fractures"], -1, 1, 3
            )

            dofs_idx = nodes4elements[
                torch.arange(nodes4elements.shape[0])[:, None, None], elements_mask
            ]

            coords4elements_first_node = self.mesh.local_triangulations[
                "stack_coords_3D4triangles"
            ][..., [0], :]

            coords4elements_first_node = coords4elements_first_node[
                torch.arange(coords4elements_first_node.shape[0])[:, None, None],
                elements_mask,
            ].unsqueeze(-3)

            inv_map_jacobian = self.elements.inv_map_jacobian

            inv_map_jacobian = inv_map_jacobian[
                torch.arange(inv_map_jacobian.shape[0])[:, None, None], elements_mask
            ]

            fractures_map_jacobian_inv = (
                self.mesh.local_triangulations["fractures_map_jacobian_inv"]
                .unsqueeze(-3)
                .unsqueeze(-3)
                .unsqueeze(-3)
            )

            inv_map_jac = inv_map_jacobian @ fractures_map_jacobian_inv

            integration_points = torch.split(
                torch.cat(basis.integration_points, dim=-1).unsqueeze(-4), 1, dim=-1
            )

            new_integrations_points = self.elements.compute_inverse_map(
                coords4elements_first_node, integration_points, inv_map_jac
            ).squeeze(-3)

            bar_coords = self.elements.compute_barycentric_coordinates(
                new_integrations_points
            )

            v, v_grad = self.elements.compute_shape_functions(
                bar_coords, inv_map_jacobian, fractures_map_jacobian_inv
            )

        if tensor != None:

            interpolation = (tensor[dofs_idx] * v).sum(-2, keepdim=True)

            interpolation_grad = (tensor[dofs_idx] * v_grad).sum(-2, keepdim=True)

            return interpolation, interpolation_grad

        else:

            nodes = torch.split(self.global_triangulation["vertices_3D"], 1, dim=-1)

            interpolator = lambda function: (function(*nodes)[dofs_idx] * v).sum(
                -2, keepdim=True
            )

            interpolator_grad = lambda function: (
                function(*nodes)[dofs_idx] * v_grad
            ).sum(-2, keepdim=True)

            return interpolator, interpolator_grad


class Interior_Facet_Fracture_Basis(Abstract_Basis):
    def __init__(self, mesh: Abstract_Mesh, elements: Fracture_Element_Line()):

        self.elements = elements
        self.mesh = mesh

        nodes4elements = mesh.edges_parameters["nodes4inner_edges"]
        coords4nodes = mesh.local_triangulations["vertices"]
        coords4elements = coords4nodes[
            torch.arange(coords4nodes.shape[0])[:, None, None], nodes4elements
        ]

        self.v, self.v_grad, self.integration_points, self.dx = (
            elements.compute_integral_values(
                coords4elements,
                mesh.local_triangulations["fractures_map_int"],
                mesh.local_triangulations["fractures_map_jacobian_inv"],
                mesh.local_triangulations["det_fractures_map_jacobian"],
            )
        )

        self.coords4global_dofs, self.global_dofs4elements, self.nodes4boundary_dofs = (
            self.compute_dofs(
                coords4nodes,
                nodes4elements,
                torch.tensor([-1]),
                mesh.mesh_parameters,
                mesh.edges_parameters,
                elements.P_order,
            )
        )

        self.coords4elements = self.mesh.local_triangulations["coords4triangles"]

        self.basis_parameters = self.compute_basis_parameters(
            self.coords4global_dofs, self.global_dofs4elements, self.nodes4boundary_dofs
        )

        # self.coords4global_dofs, self.global_dofs4elements, self.nodes4boundary_dofs = self.compute_dofs(self.meshl.local_triangulations)

        # self.basis_parameters = self.compute_basis_parameters(self.coords4global_dofs,
        #                                                       self.global_dofs4elements,
        #                                                       self.nodes4boundary_dofs)

    def compute_dofs(
        self,
        coords4nodes,
        nodes4elements,
        nodes4boundary,
        mesh_parameters,
        edges_parameters,
        P_order,
    ):

        if P_order == 1:
            coords4global_dofs = coords4nodes
            global_dofs4elements = nodes4elements
            nodes4boundary_dofs = nodes4boundary

        return coords4global_dofs, global_dofs4elements, nodes4boundary_dofs

    def compute_basis_parameters(
        self, coords4global_dofs, global_dofs4elements, nodes4boundary_dofs
    ):

        nb_global_dofs = coords4global_dofs.shape[-2]
        nb_local_dofs = global_dofs4elements.shape[-1]

        inner_dofs = torch.arange(nb_global_dofs)[
            ~torch.isin(torch.arange(nb_global_dofs), nodes4boundary_dofs)
        ]

        rows_idx = global_dofs4elements.repeat(1, 1, nb_local_dofs).reshape(-1)
        cols_idx = global_dofs4elements.repeat_interleave(nb_local_dofs).reshape(-1)

        form_idx = global_dofs4elements.reshape(-1)

        basis_parameters = {
            "bilinear_form_shape": (nb_global_dofs, nb_global_dofs),
            "bilinear_form_idx": (rows_idx, cols_idx),
            "linear_form_shape": (nb_global_dofs, 1),
            "linear_form_idx": (form_idx,),
            "inner_dofs": (inner_dofs),
        }

        return basis_parameters
