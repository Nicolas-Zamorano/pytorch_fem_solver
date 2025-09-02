"""Finite Element Method (FEM) module for mesh, element, and basis representation."""

import abc
import torch
import tensordict

torch.set_default_dtype(torch.float64)


class AbstractMesh(abc.ABC):
    """Abstract class for mesh representation"""

    def __init__(self, triangulation: dict):

        self._triangulation = triangulation

        # self.mesh = self.triangle_to_tensordict(triangulation)

        self.mesh_parameters = self.compute_mesh_parameters(self._triangulation)

        self.elements_diameter, self.nodes4boundary, self.edges_parameters = (
            self.compute_edges_values(self._triangulation)
        )

    def __getitem__(self, key):
        return self._triangulation[key]

    def __setitem__(self, key, value):
        self._triangulation[key] = value

    def __iter__(self):
        return iter(self._triangulation)

    def __len__(self):
        return len(self._triangulation)

    @staticmethod
    def triangle_to_tensordict(mesh_dict: dict):
        """Convert a mesh dictionary from 'triangle' library to a TensorDict"""
        key_map = {
            "vertices": ("vertices", "coordinates"),
            "vertex_markers": ("vertices", "markers"),
            "triangles": ("triangles", "indices"),
            "neighbors": ("triangles", "neighbors"),
            "edges": ("edges", "indices"),
            "edge_markers": ("edges", "markers"),
        }

        sub_dictionaries = {
            "coords": {},
            "triangles": {},
            "edges": {},
        }

        for key, value in mesh_dict.items():
            subname, new_key = key_map[key]
            sub_dictionaries[subname][new_key] = value

        td = tensordict.TensorDict(
            {
                name: (
                    tensordict.TensorDict(
                        content, batch_size=[len(next(iter(content.values())))]
                    )
                    if content
                    else tensordict.TensorDict({}, batch_size=[0])
                )
                for name, content in sub_dictionaries.items()
            },
            batch_size=[],
        )
        return td

    def compute_edges_values(self, mesh: dict):
        """Compute edge-related parameters for the mesh."""
        nodes4elements = mesh["triangles"]

        coords4nodes = mesh["vertices"]

        coords4elements = self.compute_coords4nodes(coords4nodes, nodes4elements)

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

        nodes4unique_edges = mesh["edges"]
        boundary_mask = mesh["edge_markers"].squeeze(-1)

        edges_idx = self.get_edges_idx(nodes4elements, nodes4unique_edges)

        nodes4boundary_edges = nodes4unique_edges[boundary_mask == 1]
        nodes4inner_edges = nodes4unique_edges[boundary_mask != 1]
        nodes4boundary = torch.nonzero(mesh["vertex_markers"])[:, [0]]

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
        )[1].reshape(-1, coords4nodes.shape[-1])

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

    @staticmethod
    def compute_coords4nodes(coords4nodes, nodes4elements):
        """Compute the coordinates of the nodes in the mesh."""
        return coords4nodes[nodes4elements]

    @staticmethod
    @abc.abstractmethod
    def compute_mesh_parameters(mesh: dict):
        """Compute basic mesh parameters related to number of nodes, elements and dimensions."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def edges_permutations(self):
        """Return the local node indices defining each edge of the element.
        the convection is the i-th node share numbering with the edge opposite to it."""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_edges_idx(nodes4elements, nodes4unique_edges):
        """Return the indices of the edges for each element in the mesh."""
        raise NotImplementedError


class MeshTri(AbstractMesh):
    """Class for triangular mesh representation"""

    @property
    def edges_permutations(self):
        return torch.tensor([[0, 1], [1, 2], [0, 2]])

    @staticmethod
    def compute_mesh_parameters(mesh: dict):

        nb_nodes, nb_dimensions = mesh["vertices"].shape
        nb_simplex = mesh["triangles"].shape[-2]

        return {
            "nb_nodes": nb_nodes,
            "nb_dimensions": nb_dimensions,
            "nb_simplex": nb_simplex,
        }

    @staticmethod
    def map_fine_mesh(c4e_coarser_mesh: torch.Tensor, c4e_finer_mesh: torch.Tensor):
        """Map each element of a finer mesh to an element of the current coarser mesh."""
        centroids_h = c4e_finer_mesh.mean(dim=-2)  # (n_elem_h, 2)

        p = centroids_h[:, None, :]  # (n_elem_h, 1, 2)
        a = c4e_coarser_mesh[:, 0, 0, :][None, :, :]  # (1, n_elem_H, 2)
        b = c4e_coarser_mesh[:, 0, 1, :][None, :, :]
        c = c4e_coarser_mesh[:, 0, 2, :][None, :, :]

        v0 = c - a  # (1, n_elem_H, 2)
        v1 = b - a
        v2 = p - a  # (n_elem_h, n_elem_H, 2)

        dot00 = (v0 * v0).sum(dim=-1)  # (1, n_elem_H)
        dot01 = (v0 * v1).sum(dim=-1)
        dot11 = (v1 * v1).sum(dim=-1)

        dot02 = (v0 * v2).sum(dim=-1)  # (n_elem_h, n_elem_H)
        dot12 = (v1 * v2).sum(dim=-1)

        denom = dot00 * dot11 - dot01 * dot01  # (1, n_elem_H)
        denom = denom.clamp(min=1e-14)

        u = (dot11 * dot02 - dot01 * dot12) / denom  # (n_elem_h, n_elem_H)
        v = (dot00 * dot12 - dot01 * dot02) / denom

        inside = (u >= 0) & (v >= 0) & (u + v <= 1)  # (n_elem_h, n_elem_H)

        mapping = torch.full((c4e_finer_mesh.shape[0],), -1, dtype=torch.long)

        candidates = inside.nonzero(as_tuple=False)  # shape (n_matches, 2)

        seen = torch.zeros(c4e_finer_mesh.shape[0], dtype=torch.bool)
        for i in range(candidates.shape[0]):
            idx_finer_mesh, idx_coarser_mesh = candidates[i]
            if not seen[idx_finer_mesh]:
                mapping[idx_finer_mesh] = idx_coarser_mesh
                seen[idx_finer_mesh] = True

        return mapping

    @staticmethod
    def get_edges_idx(nodes4elements, nodes4unique_edges):
        i0 = nodes4elements[..., 0]
        i1 = nodes4elements[..., 1]
        i2 = nodes4elements[..., 2]

        tri_edges = torch.stack(
            [
                torch.stack([torch.min(i0, i1), torch.max(i0, i1)], dim=1),
                torch.stack([torch.min(i1, i2), torch.max(i1, i2)], dim=1),
                torch.stack([torch.min(i2, i0), torch.max(i2, i0)], dim=1),
            ],
            dim=1,
        )  # (n_triangles, 3, 2)

        m = nodes4elements.max().item() + 1
        tri_keys = tri_edges[:, :, 0] * m + tri_edges[:, :, 1]  # (n_triangles, 3)

        edge_keys = (
            nodes4unique_edges.min(dim=-1).values * m
            + nodes4unique_edges.max(dim=-1).values
        )  # (n_unique_edges,)

        sorted_keys, sorted_idx = torch.sort(edge_keys)
        flat_tri_keys = tri_keys.flatten()  # (n_triangles * 3,)

        edge_pos = torch.searchsorted(sorted_keys, flat_tri_keys)
        return sorted_idx[edge_pos].reshape(tri_keys.shape)


class Fractures(AbstractMesh):
    """Class for handling multiple fractures represented as triangular meshes"""

    def __init__(self, triangulations: list, fractures_3d_data: torch.Tensor):

        self.fractures_3d_data = fractures_3d_data

        self.local_triangulations, self.edges_parameters = self.stack_triangulations(
            triangulations, fractures_3d_data
        )

        super().__init__(self.local_triangulations)

    @property
    def edges_permutations(self):
        return torch.tensor([[0, 1], [1, 2], [0, 2]])

    @staticmethod
    def compute_mesh_parameters(mesh: dict):

        nb_fractures, nb_nodes, nb_dimensions = mesh["vertices"].shape
        _, nb_simplex, nb_size4simplex = mesh["triangles"].shape

        mesh_parameters = {
            "nb_fractures": nb_fractures,
            "nb_nodes": nb_nodes,
            "nb_dimensions": nb_dimensions,
            "nb_simplex": nb_simplex,
            "nb_size4simplex": nb_size4simplex,
        }

        return mesh_parameters

    def stack_triangulations(self, fracture_triangulations: list, fractures_3d_data):
        """Stack multiple fracture triangulations into a single TensorDict
        and compute mapping to 3D space."""
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

        stack_coords4triangles = stack_vertices[
            torch.arange(stack_vertices.shape[0])[:, None, None], stack_triangles
        ]

        stack_edge_parameters = torch.stack(
            [
                tensordict.TensorDict(self.compute_edges_values(triangulation))
                for triangulation in fracture_triangulations
            ],
            dim=0,
        )

        fractures_2d_vertices = stack_vertices[:, :3, :]

        fractures_3d_vertices = fractures_3d_data[:, :3, :].mT

        hat_v = torch.cat(
            [
                fractures_2d_vertices.mT,
                torch.ones_like(fractures_3d_vertices[:, [-1], :]),
            ],
            dim=-2,
        )

        linear_equation = fractures_3d_vertices @ torch.inverse(hat_v)

        # Split A and b
        fractures_map_jacobian = linear_equation[..., :2]
        b = linear_equation[..., [-1]]

        fractures_map_jacobian_int = fractures_map_jacobian.unsqueeze(-3).unsqueeze(-3)
        b_int = b.unsqueeze(-3).unsqueeze(-3)

        def fractures_map(x):
            return (fractures_map_jacobian @ x.mT + b).mT

        def fractures_map_int(x):
            return (fractures_map_jacobian_int @ x.mT + b_int).mT

        det_fractures_map_jacobian = torch.norm(
            torch.cross(*torch.split(fractures_map_jacobian, 1, dim=-1), dim=-2),
            dim=-2,
            keepdim=True,
        )

        fractures_map_jacobian_inv = (
            torch.inverse(fractures_map_jacobian.mT @ fractures_map_jacobian)
            @ fractures_map_jacobian.mT
        )

        stack_vertices_3d = fractures_map(stack_vertices)

        stack_coords_3d4triangles = stack_vertices_3d[
            torch.arange(stack_vertices_3d.shape[0])[:, None, None], stack_triangles
        ]

        stack_normal4inner_edges_3d = fractures_map_int(
            stack_edge_parameters["normal4inner_edges"].unsqueeze(-2)
        )

        stack_triangulation = tensordict.TensorDict(
            vertices=stack_vertices,
            vertices_3D=stack_vertices_3d,
            vertex_markers=stack_vertex_markers,
            triangles=stack_triangles,
            edges=stack_edges,
            edge_markers=stack_edge_markers,
            fractures_map=fractures_map,
            coords4triangles=stack_coords4triangles,
            stack_coords_3d4triangles=stack_coords_3d4triangles,
            fractures_map_jacobian=fractures_map_jacobian,
            det_fractures_map_jacobian=det_fractures_map_jacobian,
            fractures_map_jacobian_inv=fractures_map_jacobian_inv,
            fractures_map_int=fractures_map_int,
            normal4inner_edges_3D=stack_normal4inner_edges_3d,
        )

        return stack_triangulation, stack_edge_parameters

    @staticmethod
    def get_edges_idx(nodes4elements, nodes4unique_edges):
        i0 = nodes4elements[..., 0]
        i1 = nodes4elements[..., 1]
        i2 = nodes4elements[..., 2]

        tri_edges = torch.stack(
            [
                torch.stack([torch.min(i0, i1), torch.max(i0, i1)], dim=1),
                torch.stack([torch.min(i1, i2), torch.max(i1, i2)], dim=1),
                torch.stack([torch.min(i2, i0), torch.max(i2, i0)], dim=1),
            ],
            dim=1,
        )

        m = nodes4elements.max().item() + 1
        tri_keys = tri_edges[:, :, 0] * m + tri_edges[:, :, 1]

        edge_keys = (
            nodes4unique_edges.min(dim=-1).values * m
            + nodes4unique_edges.max(dim=-1).values
        )

        sorted_keys, sorted_idx = torch.sort(edge_keys)
        flat_tri_keys = tri_keys.flatten()

        edge_pos = torch.searchsorted(sorted_keys, flat_tri_keys)
        edge_indices = sorted_idx[edge_pos].reshape(tri_keys.shape)

        return edge_indices

    @staticmethod
    def compute_coords4nodes(coords4nodes, nodes4elements):
        """Compute the coordinates of the nodes in the mesh."""
        return coords4nodes[
            torch.arange(coords4nodes.shape[0])[:, None, None], nodes4elements
        ]


class AbstractElement(abc.ABC):
    """Abstract class for element representation"""

    def __init__(self, polynomial_order: int, integration_order: int):

        self.polynomial_order = polynomial_order
        self.integration_order = integration_order
        self.inv_map_jacobian = None

        self.gaussian_nodes, self.gaussian_weights = self.compute_gauss_values()

    def compute_inverse_map(
        self,
        first_node: torch.Tensor,
        integration_points: torch.Tensor,
        inv_map_jacobian: torch.Tensor,
    ):
        """Compute the inverse map from physical coordinates to reference coordinates"""
        integration_points = torch.concat(integration_points, dim=-1)

        return (integration_points - first_node) @ inv_map_jacobian.mT

    @abc.abstractmethod
    def compute_shape_functions(self, bar_coords, inv_map_jacobian):
        """Compute the shape functions and their gradients at given barycentric coordinates"""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_gauss_values(self):
        """Compute the Gaussian integration points and weights"""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_barycentric_coordinates(self, x):
        """Compute the barycentric coordinates of given points x"""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_det_and_inv_map(self, map_jacobian):
        """Compute the determinant and inverse of the mapping Jacobian"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def reference_element_area(self):
        """Return the area (or length) of the reference element"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def barycentric_grad(self):
        """Return the gradients of the barycentric coordinates in the reference element"""
        raise NotImplementedError


class ElementLine(AbstractElement):
    """Class for 1D line element representation"""

    @property
    def barycentric_grad(self):
        return torch.tensor([[-0.5], [0.5]])

    @property
    def reference_element_area(self):
        return 2.0

    def compute_barycentric_coordinates(self, x):
        return torch.concat([0.5 * (1.0 - x), 0.5 * (1.0 + x)], dim=-1)

    def compute_gauss_values(self):

        if self.integration_order == 2:

            nodes = 1.0 / torch.sqrt(torch.tensor(3.0))

            gaussian_nodes = torch.tensor([[-nodes], [nodes]])

            gaussian_weights = torch.tensor([[[0.5]], [[0.5]]])

        elif self.integration_order == 3:

            nodes = torch.sqrt(torch.tensor(3 / 5))

            gaussian_nodes = torch.tensor([[0], [-nodes], [nodes]])

            gaussian_weights = torch.tensor([[[8 / 18]], [[5 / 18]], [[5 / 18]]])

        else:

            raise NotImplementedError("Integration order not implemented")

        return gaussian_nodes, gaussian_weights.unsqueeze(0)

    def compute_shape_functions(
        self, bar_coords: torch.Tensor, inv_map_jacobian: torch.Tensor
    ):

        if self.polynomial_order == 1:

            v = bar_coords

            v_grad = self.barycentric_grad @ inv_map_jacobian

        else:

            raise NotImplementedError("Polynomial order not implemented")

        return v, v_grad

    def compute_det_and_inv_map(self, map_jacobian):

        det_map_jacobian = torch.norm(map_jacobian, dim=-2, keepdim=True)

        inv_map_jacobian = 1.0 / det_map_jacobian

        return det_map_jacobian.unsqueeze(-1), inv_map_jacobian


class ElementTri(AbstractElement):
    """Class for 2D triangular element representation"""

    @property
    def barycentric_grad(self):
        return torch.tensor([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])

    @property
    def reference_element_area(self):
        return 0.5

    @property
    def outward_normal(self):
        """Return the outward normal vectors of the reference triangle edges."""
        return torch.tensor([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

    def compute_barycentric_coordinates(self, x):
        return torch.stack(
            [1.0 - x[..., [0]] - x[..., [1]], x[..., [0]], x[..., [1]]], dim=-2
        )

    def compute_shape_functions(
        self, bar_coords: torch.Tensor, inv_map_jacobian: torch.Tensor
    ):
        lambda_1, lambda_2, lambda_3 = torch.split(bar_coords, 1, dim=-2)

        grad_lambda_1, grad_lambda_2, grad_lambda_3 = torch.split(
            self.barycentric_grad, 1, dim=-2
        )

        if self.polynomial_order == 1:

            v = bar_coords

            v_grad = self.barycentric_grad @ inv_map_jacobian

        elif self.polynomial_order == 2:

            v = torch.concat(
                [
                    lambda_1 * (2 * lambda_1 - 1),
                    lambda_2 * (2 * lambda_2 - 1),
                    lambda_3 * (2 * lambda_3 - 1),
                    4 * lambda_1 * lambda_2,
                    4 * lambda_2 * lambda_3,
                    4 * lambda_3 * lambda_1,
                ],
                dim=-2,
            )

            v_grad = (
                torch.concat(
                    [
                        (4 * lambda_1 - 1) * grad_lambda_1,
                        (4 * lambda_2 - 1) * grad_lambda_2,
                        (4 * lambda_3 - 1) * grad_lambda_3,
                        4 * (lambda_2 * grad_lambda_1 + lambda_1 * grad_lambda_2),
                        4 * (lambda_3 * grad_lambda_2 + lambda_2 * grad_lambda_3),
                        4 * (lambda_1 * grad_lambda_3 + lambda_3 * grad_lambda_1),
                    ],
                    dim=-2,
                )
                @ inv_map_jacobian
            )
        else:

            raise NotImplementedError("Polynomial order not implemented")

        return v, v_grad

    def compute_gauss_values(self):

        if self.integration_order == 1:

            gaussian_nodes = torch.tensor([[1 / 3, 1 / 3]])

            gaussian_weights = torch.tensor([[[1.0]]])

        elif self.integration_order == 2:

            gaussian_nodes = torch.tensor(
                [[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]]
            )

            gaussian_weights = torch.tensor([[[1 / 3]], [[1 / 3]], [[1 / 3]]])

        elif self.integration_order == 3:

            gaussian_nodes = torch.tensor(
                [[1 / 3, 1 / 3], [0.6, 0.2], [0.2, 0.6], [0.2, 0.2]]
            )

            gaussian_weights = torch.tensor(
                [[[-9 / 16]], [[25 / 48]], [[25 / 48]], [[25 / 48]]]
            )

        elif self.integration_order == 4:

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
        else:

            raise NotImplementedError("Integration order not implemented")

        return gaussian_nodes, gaussian_weights

    def compute_det_and_inv_map(self, map_jacobian: torch.Tensor):

        ab, cd = torch.split(map_jacobian, 1, dim=-2)

        a, b = torch.split(ab, 1, dim=-1)
        c, d = torch.split(cd, 1, dim=-1)

        det_map_jacobian = (a * d - b * c).unsqueeze(-3)

        inv_map_jacobian = (1 / det_map_jacobian) * torch.stack(
            [torch.concat([d, -b], dim=-1), torch.concat([-c, a], dim=-1)], dim=-2
        )

        return det_map_jacobian, inv_map_jacobian


class AbstractBasis(abc.ABC):
    """Abstract class for basis representation"""

    def __init__(self, mesh: AbstractMesh, element: AbstractElement):

        self.element = element
        self.mesh = mesh

        (
            self.v,
            self.v_grad,
            self.integration_points,
            self.dx,
            self.inv_map_jacobian,
        ) = self.compute_integral_values(mesh, element)

        (
            self.coords4global_dofs,
            self.global_dofs4elements,
            self.nodes4boundary_dofs,
            self.coords4elements,
        ) = self.compute_dofs(
            mesh,
            element,
        )

        self.basis_parameters = self.compute_basis_parameters(
            self.coords4global_dofs, self.global_dofs4elements, self.nodes4boundary_dofs
        )

    @abc.abstractmethod
    def compute_integral_values(
        self,
        mesh: AbstractMesh,
        element: AbstractElement,
    ):
        """Compute the integral values needed for the basis functions"""
        raise NotImplementedError

    def integrate_functional(self, function, *args, **kwargs):
        """Integrate a given functional over the mesh elements"""
        return (function(self, *args, **kwargs) * self.dx).sum(-3).sum(-2)

    def integrate_bilinear_form(self, function, *args, **kwargs):
        """Integrate a given bilinear form over the mesh elements"""
        global_matrix = torch.zeros(self.basis_parameters["bilinear_form_shape"])

        local_matrix = (function(self, *args, **kwargs) * self.dx).sum(-3)

        global_matrix.index_put_(
            self.basis_parameters["bilinear_form_idx"],
            local_matrix.reshape(-1),
            accumulate=True,
        )

        return global_matrix

    def integrate_linear_form(self, function, *args, **kwargs):
        """Integrate a given linear form over the mesh elements"""
        integral_value = torch.zeros(self.basis_parameters["linear_form_shape"])

        integrand_value = (function(self, *args, **kwargs) * self.dx).sum(-3)

        integral_value.index_put_(
            self.basis_parameters["linear_form_idx"],
            integrand_value.reshape(-1, 1),
            accumulate=True,
        )

        return integral_value

    def reduce(self, tensor):
        """Reduce a tensor to only include inner degrees of freedom"""
        idx = self.basis_parameters["inner_dofs"]
        return tensor[idx, :][:, idx] if tensor.shape[-1] != 1 else tensor[idx]

    def interpolate(self, basis, tensor=None):
        """Interpolate a tensor from the current basis to another basis."""
        if basis == self:
            dofs_idx = self.global_dofs4elements.unsqueeze(-2)

            v = self.v
            v_grad = self.v_grad

        # else:

        #     elements_mask = self.mesh.map_fine_mesh(basis.mesh)

        #     dofs_idx = self.global_dofs4elements[elements_mask]

        #     coords4elements_first_node = self.coords4elements[..., [0], :][
        #         elements_mask
        #     ]

        #     inv_map_jacobian = self.elements.inv_map_jacobian[elements_mask]

        #     new_integrations_points = self.elements.compute_inverse_map(
        #         coords4elements_first_node, basis.integration_points, inv_map_jacobian
        #     )

        #     _, v, v_grad = self.elements.compute_shape_functions(
        #         new_integrations_points.squeeze(-2), inv_map_jacobian
        #     )

        if basis.__class__ == InteriorFacetBasis:

            elements_mask = basis.mesh.edges_parameters["elements4inner_edges"]

            dofs_idx = basis.mesh.nodes4elements[elements_mask].unsqueeze(-2)

            coords4elements_first_node = self.coords4elements[..., [0], :][
                elements_mask
            ].unsqueeze(-3)

            inv_map_jacobian = self.inv_map_jacobian[elements_mask]

            integration_points = torch.split(
                torch.cat(basis.integration_points, dim=-1).unsqueeze(-4), 1, dim=-1
            )

            new_integrations_points = self.element.compute_inverse_map(
                coords4elements_first_node, integration_points, inv_map_jacobian
            )

            bar_coords = self.element.compute_barycentric_coordinates(
                new_integrations_points.squeeze(-3)
            )

            v, v_grad = self.element.compute_shape_functions(
                bar_coords, inv_map_jacobian
            )

        else:
            raise NotImplementedError(
                "Interpolation between different basis not implemented"
            )

        if tensor is not None:

            interpolation = (tensor[dofs_idx] * v).sum(-2, keepdim=True)

            interpolation_grad = (tensor[dofs_idx] * v_grad).sum(-2, keepdim=True)

            return interpolation, interpolation_grad

        else:

            nodes = torch.split(self.coords4global_dofs, 1, dim=-1)

            def interpolator(function):
                return (function(*nodes)[dofs_idx] * v).sum(-2, keepdim=True)

            def interpolator_grad(function):
                return (function(*nodes)[dofs_idx] * v_grad).sum(-2, keepdim=True)

            return interpolator, interpolator_grad

    @abc.abstractmethod
    def compute_dofs(
        self,
        mesh: AbstractMesh,
        element: AbstractElement,
    ):
        """Compute the degrees of freedom for the basis functions"""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_basis_parameters(
        self, coords4global_dofs, global_dofs4elements, nodes4boundary_dofs
    ):
        """Compute parameters related to the basis functions"""
        raise NotImplementedError


class Basis(AbstractBasis):
    """Class for standard basis representation"""

    def compute_integral_values(
        self,
        mesh: AbstractMesh,
        element: AbstractElement,
    ):

        map_jacobian = mesh.coords4elements.mT @ element.barycentric_grad

        det_map_jacobian, inv_map_jacobian = element.compute_det_and_inv_map(
            map_jacobian
        )

        bar_coords = element.compute_barycentric_coordinates(element.gaussian_nodes)

        v, v_grad = element.compute_shape_functions(bar_coords, inv_map_jacobian)

        integration_points = torch.split(
            bar_coords.mT @ mesh.coords4elements.unsqueeze(-3), 1, dim=-1
        )

        dx = (
            element.reference_element_area * element.gaussian_weights * det_map_jacobian
        )

        return v, v_grad, integration_points, dx, inv_map_jacobian

    def compute_dofs(
        self,
        mesh: AbstractMesh,
        element: AbstractElement,
    ):

        if element.polynomial_order == 1:

            coords4global_dofs = mesh.coords4nodes
            global_dofs4elements = mesh.nodes4elements
            nodes4boundary_dofs = mesh.nodes4boundary

        elif element.polynomial_order == 2:

            new_coords4dofs = (
                mesh.coords4nodes[mesh.edges_parameters["nodes4unique_edges"]]
            ).mean(-2)
            new_nodes4dofs = (
                mesh.edges_parameters["edges_idx"].reshape(
                    mesh.mesh_parameters["nb_simplex"], 3
                )
                + mesh.mesh_parameters["nb_nodes"]
            )
            new_nodes4boundary_dofs = (
                mesh.edges_parameters["nodes_idx4boundary_edges"]
                + mesh.mesh_parameters["nb_nodes"]
            )

            coords4global_dofs = torch.cat([mesh.coords4nodes, new_coords4dofs], dim=-2)
            global_dofs4elements = torch.cat(
                [mesh.nodes4elements, new_nodes4dofs], dim=-1
            )
            nodes4boundary_dofs = torch.cat(
                [mesh.nodes4boundary, new_nodes4boundary_dofs], dim=-1
            )
        else:
            raise NotImplementedError("Polynomial order not implemented")

        coords4elements = coords4global_dofs[global_dofs4elements]

        return (
            coords4global_dofs,
            global_dofs4elements,
            nodes4boundary_dofs,
            coords4elements,
        )

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

        return {
            "bilinear_form_shape": (nb_global_dofs, nb_global_dofs),
            "bilinear_form_idx": (rows_idx, cols_idx),
            "linear_form_shape": (nb_global_dofs, 1),
            "linear_form_idx": (form_idx,),
            "inner_dofs": inner_dofs,
            "nb_dofs": nb_global_dofs,
        }


class FractureBasis(AbstractBasis):
    """Class for basis representation on fractures"""

    def __init__(self, mesh: AbstractMesh, element: AbstractElement):
        self.global_triangulation = self.build_global_triangulation(
            mesh.local_triangulations
        )

        super().__init__(mesh, element)

    def build_global_triangulation(self, local_triangulations):
        """Build a global triangulation from local triangulations of multiple fractures."""
        nb_fractures, nb_vertices, _ = local_triangulations["vertices"].shape

        nb_edges = local_triangulations["edges"].shape[-2]

        local_triangulation_3d_coords = local_triangulations["vertices_3D"].reshape(
            -1, 3
        )

        global_vertices_3d, global2local_idx, vertex_counts = torch.unique(
            local_triangulation_3d_coords,
            dim=0,
            return_inverse=True,
            return_counts=True,
        )

        nb_global_vertices = global_vertices_3d.shape[-2]

        traces_global_vertices_idx = torch.nonzero(vertex_counts > 1, as_tuple=True)[0]

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

        global_vertices_2d = local_triangulations["vertices"].reshape(-1, 2)[
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

        global_triangulation = tensordict.TensorDict(
            vertices_3D=global_vertices_3d,
            vertices_2D=global_vertices_2d,
            vertex_markers=global_vertices_marker,
            triangles=global_triangles,
            edges=global_edges,
            edge_markers=global_edges_marker,
            global2local_idx=global2local_idx,
            local2global_idx=local2global_idx,
            traces__global_vertices_idx=traces_global_vertices_idx,
            traces_global_edges_idx=traces_global_edges_idx,
            traces_local_edges_idx=traces_local_edges_idx,
        )

        return global_triangulation

    def compute_integral_values(self, mesh: AbstractMesh, element: AbstractElement):

        map_jacobian = mesh.coords4elements.mT @ element.barycentric_grad

        det_map_jacobian, inv_map_jacobian = element.compute_det_and_inv_map(
            map_jacobian
        )

        inv_map_jacobian_3d = (
            inv_map_jacobian @ mesh.local_triangulations["fractures_map_jacobian_inv"]
        )

        bar_coords = element.compute_barycentric_coordinates(element.gaussian_nodes)

        v, v_grad = element.compute_shape_functions(bar_coords, inv_map_jacobian_3d)

        mapped_gaussian_nodes = bar_coords.mT @ mesh.coords4elements.unsqueeze(-3)

        mapped_3d_gaussian_nodes = mesh.local_triangulations["fractures_map_int"](
            mapped_gaussian_nodes
        )

        integration_points = torch.split(mapped_3d_gaussian_nodes, 1, dim=-1)

        dx = (
            element.reference_element_area
            * element.gaussian_weights
            * det_map_jacobian
            * mesh.local_triangulations["det_fractures_map_jacobian"]
        )

        return v, v_grad, integration_points, dx, inv_map_jacobian

    def compute_dofs(self, mesh, element):

        if element.polynomial_order == 1:

            coords4global_dofs = self.global_triangulation["vertices_2D"]
            global_dofs4elements = self.global_triangulation["triangles"]
            nodes4boundary_dofs = torch.nonzero(
                self.global_triangulation["vertex_markers"] == 1
            )[:, 0]

        else:
            raise NotImplementedError("Polynomial order not implemented")

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


class InteriorFacetBasis(AbstractBasis):
    """Class for basis representation on interior facets"""

    def compute_integral_values(self, mesh, element):

        coords4inner_facet = mesh.coords4nodes[
            mesh.edges_parameters["nodes4inner_edges"]
        ]

        map_jacobian = coords4inner_facet.mT @ element.barycentric_grad

        det_map_jacobian, inv_map_jacobian = element.compute_det_and_inv_map(
            map_jacobian
        )

        bar_coords = element.compute_barycentric_coordinates(element.gaussian_nodes)

        v, v_grad = element.compute_shape_functions(bar_coords, inv_map_jacobian)

        integration_points = torch.split(
            bar_coords.mT @ coords4inner_facet.unsqueeze(-3), 1, dim=-1
        )

        dx = (
            element.reference_element_area * element.gaussian_weights * det_map_jacobian
        )

        return v, v_grad, integration_points, dx, inv_map_jacobian

    def compute_dofs(
        self,
        mesh: AbstractMesh,
        element: AbstractElement,
    ):

        if element.polynomial_order == 1:
            coords4global_dofs = mesh.coords4nodes
            global_dofs4elements = mesh.nodes4elements
            nodes4boundary_dofs = mesh.nodes4boundary

        else:
            raise NotImplementedError("Polynomial order not implemented")

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

        return {
            "bilinear_form_shape": (nb_global_dofs, nb_global_dofs),
            "bilinear_form_idx": (rows_idx, cols_idx),
            "linear_form_shape": (nb_global_dofs, 1),
            "linear_form_idx": (form_idx,),
            "inner_dofs": (inner_dofs),
        }


class InteriorFacetFractureBasis(AbstractBasis):
    """Class for basis representation on interior facets of fractures"""

    def compute_integral_values(self, mesh: AbstractMesh, element: AbstractElement):

        nodes4elements = mesh.edges_parameters["nodes4inner_edges"]
        coords4nodes = mesh.local_triangulations["vertices"]
        coords4elements = coords4nodes[
            torch.arange(coords4nodes.shape[0])[:, None, None], nodes4elements
        ]

        map_jacobian = coords4elements.mT @ element.barycentric_grad

        det_map_jacobian, inv_map_jacobian = element.compute_det_and_inv_map(
            map_jacobian
        )

        inv_map_jacobian_3d = (
            inv_map_jacobian @ mesh.local_triangulations["fractures_map_jacobian_inv"]
        )

        bar_coords = element.compute_barycentric_coordinates(element.gaussian_nodes)

        v, v_grad = element.compute_shape_functions(bar_coords, inv_map_jacobian_3d)

        mapped_gaussian_nodes = bar_coords.mT @ coords4elements.unsqueeze(-3)

        mapped_3d_gaussian_nodes = mesh.local_triangulations["fractures_map_int"](
            mapped_gaussian_nodes
        )

        integration_points = torch.split(mapped_3d_gaussian_nodes, 1, dim=-1)

        dx = (
            element.reference_element_area
            * element.gaussian_weights
            * det_map_jacobian
            * mesh.local_triangulations["det_fractures_map_jacobian"]
        )

        return v, v_grad, integration_points, dx, inv_map_jacobian

    def compute_dofs(
        self,
        mesh: AbstractMesh,
        element: AbstractElement,
    ):

        if element.polynomial_order == 1:
            coords4global_dofs = mesh.coords4nodes
            global_dofs4elements = mesh.nodes4elements
            nodes4boundary_dofs = mesh.nodes4boundary

        else:
            raise NotImplementedError("Polynomial order not implemented")

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
