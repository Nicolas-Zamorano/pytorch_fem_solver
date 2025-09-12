"""Finite Element Method (FEM) module for mesh, element, and basis representation."""

import abc
import torch
import tensordict

torch.set_default_dtype(torch.float64)


class AbstractMesh(abc.ABC):
    """Abstract class for mesh representation"""

    def __init__(self, triangulation: dict):

        triangulation_tensordict = self._triangle_to_tensordict(triangulation)

        self._triangulation = self._build_optional_parameters(triangulation_tensordict)

    def __getitem__(self, key: str):
        return self._triangulation[key]

    def __setitem__(self, key: str, value):
        self._triangulation[key] = value

    @staticmethod
    def _triangle_to_tensordict(mesh_dict: dict):
        """Convert a mesh dictionary from 'triangle' library to a TensorDict"""
        key_map = {
            "vertices": ("vertices", "coordinates"),
            "vertex_markers": ("vertices", "markers"),
            "triangles": ("cells", "vertices"),
            "neighbors": ("cells", "neighbors"),
            "edges": ("edges", "vertices"),
            "edge_markers": ("edges", "markers"),
        }

        sub_dictionaries = {
            "vertices": {},
            "cells": {},
            "edges": {},
            "boundary_edges": {},
            "interior_edges": {},
        }

        for key, value in mesh_dict.items():
            if key in key_map:
                subname, new_key = key_map[key]
                sub_dictionaries[subname][new_key] = value

        mesh_tensordict = tensordict.TensorDict(
            {
                name: (
                    tensordict.TensorDict(
                        content, batch_size=[len(next(iter(content.values())))]
                    )
                    if content
                    else tensordict.TensorDict({})
                )
                for name, content in sub_dictionaries.items()
            },
            batch_size=[],
        )

        return mesh_tensordict.auto_batch_size_()

    def _build_optional_parameters(self, triangulation: tensordict.TensorDict):
        """Compute parameters that are not in mesh dict."""

        triangulation["cells"]["coordinates"] = self.compute_coordinates_4_cells(
            triangulation["vertices"]["coordinates"], triangulation["cells"]["vertices"]
        )

        if "vertices" not in triangulation["edges"]:
            vertices_4_unique_edges, boundary_mask = self._compute_edges_indices(
                triangulation,
            )
            triangulation["edges"]["vertices"] = vertices_4_unique_edges
            triangulation["edges"]["markers"] = boundary_mask

        interior_edges, boundary_edges = self._compute_interior_and_boundary_edges(
            triangulation
        )

        cells_length = self._compute_cells_min_length(triangulation)

        triangulation["interior_edges"] = interior_edges
        triangulation["boundary_edges"] = boundary_edges
        triangulation["cells"]["length"] = cells_length

        return triangulation

    def _compute_interior_and_boundary_edges(
        self, triangulation: tensordict.TensorDict
    ):
        """Compute interior and boundary edges."""

        indices_4_edges = triangulation["edges"]["vertices"]
        markers_4_edges = triangulation["edges"]["markers"].squeeze(-1)

        indices_4_boundary_edges = indices_4_edges[markers_4_edges == 1]
        indices_4_interior_edges = indices_4_edges[markers_4_edges != 1]

        if "neighbors" in triangulation["cells"]:
            neighbors = triangulation["cells"]["neighbors"]

            # number_cells, dimension_cells = triangulation["cells"]["vertices"].size()

            # interior_edges = []
            # boundary_edges = []

            # for t in range(number_cells):
            #     for i in range(dimension_cells):
            #         t_neigh = neighbors[t, i].item()
            #         if t_neigh != -1:
            #             if (
            #                 t < t_neigh
            #             ):  # to avoid append the same edge but seems for the
            #                 # other triangle perspective (ex: [1,2] and [2,1])
            #                 interior_edges.append([t, t_neigh])
            #         else:
            #             boundary_edges.append([t])

            # cells_4_boundary_edges = torch.tensor(boundary_edges, dtype=torch.long)
            # cells_4_interior_edges = torch.tensor(interior_edges, dtype=torch.long)

            number_neighbors = neighbors.size(-2)

            cells_idx = torch.arange(number_neighbors).repeat_interleave(
                triangulation["cells"]["vertices"].size(-1)
            )

            neigh_flat = neighbors.reshape(-1)

            mask_inner = neigh_flat != -1
            mask_boundary = neigh_flat == -1

            tri1 = cells_idx[mask_inner]
            tri2 = neigh_flat[mask_inner]

            pair = torch.stack(
                [torch.minimum(tri1, tri2), torch.maximum(tri1, tri2)], dim=1
            )

            cells_4_interior_edges = torch.unique(pair, dim=0)

            cells_4_boundary_edges = cells_idx[mask_boundary].unsqueeze(1)

        else:
            raise NotImplementedError
            # WARNING!!! DOESN'T WORK AS INTENDED
            # indices_4_cells = triangulation["cells"]["vertices"]

            # cells_4_boundary_edges = (
            #     (
            #         indices_4_boundary_edges.unsqueeze(-2).unsqueeze(-2)
            #         == indices_4_cells.unsqueeze(-1).unsqueeze(-4)
            #     )
            #     .any(dim=-2)
            #     .all(dim=-1)
            #     .float()
            #     .argmax(dim=-1, keepdim=True)
            # )
            # cells_4_interior_edges = torch.nonzero(
            #     (
            #         indices_4_interior_edges.unsqueeze(-2).unsqueeze(-2)
            #         == indices_4_cells.unsqueeze(-1).unsqueeze(-4)
            #     )
            #     .any(dim=-2)
            #     .all(dim=-1),
            #     as_tuple=True,
            # )[1].reshape(-1, triangulation["vertices"]["coordinates"].size(-1))

        coordinates_4_interior_edges = self.compute_coordinates_4_cells(
            triangulation["vertices"]["coordinates"], indices_4_interior_edges
        )

        coordinates_4_boundary_edges = self.compute_coordinates_4_cells(
            triangulation["vertices"]["coordinates"], indices_4_boundary_edges
        )

        (
            coordinates_4_interior_edges_first_vertex,
            coordinates_4_interior_edges_second_vertex,
        ) = torch.split(coordinates_4_interior_edges, 1, dim=-2)

        interior_edges_vector = (
            coordinates_4_interior_edges_second_vertex
            - coordinates_4_interior_edges_first_vertex
        )

        interior_edges_length = torch.norm(interior_edges_vector, dim=-1, keepdim=True)

        normal_4_interior_edges = (
            interior_edges_vector[..., [1, 0]]
            * torch.tensor([-1.0, 1.0])
            / interior_edges_length
        )

        # Fix orientation

        centroid_4_interior_edges = self.compute_coordinates_4_cells(
            triangulation["cells"]["coordinates"], cells_4_interior_edges
        ).mean(dim=-2)

        (
            centroid_4_interior_edges_first_coordinate,
            centroid_4_interior_edges_second_coordinate,
        ) = torch.split(centroid_4_interior_edges, 1, dim=-2)

        normal_direction_4_interior_edges = (
            normal_4_interior_edges
            * (
                centroid_4_interior_edges_second_coordinate
                - centroid_4_interior_edges_first_coordinate
            )
        ).sum(dim=-1)

        normal_4_interior_edges[normal_direction_4_interior_edges < 0] *= -1

        boundary_edges = tensordict.TensorDict(
            {
                "cells": cells_4_boundary_edges,
                "vertices": indices_4_boundary_edges,
                "coordinates": coordinates_4_boundary_edges,
            }
        ).auto_batch_size_()
        interior_edges = tensordict.TensorDict(
            {
                "cells": cells_4_interior_edges,
                "vertices": indices_4_interior_edges,
                "coordinates": coordinates_4_interior_edges,
                "length": interior_edges_length,
                "normals": normal_4_interior_edges,
            }
        ).auto_batch_size_()

        return interior_edges, boundary_edges

    @staticmethod
    def compute_coordinates_4_cells(
        coordinates_4_vertices: torch.Tensor, vertices_4_cells: torch.Tensor
    ):
        """Compute the coordinates of the cells in the mesh."""
        return coordinates_4_vertices[vertices_4_cells]

    def _compute_edges_indices(self, triangulation: tensordict.TensorDict):
        """Compute vertices for unique edges."""

        indices_4_edges = triangulation["cells"]["vertices"][self._edges_permutations]

        indices_4_unique_edges, _, boundary_mask = torch.unique(
            indices_4_edges.reshape(-1, 2).mT,
            return_inverse=True,
            sorted=False,
            return_counts=True,
            dim=-1,
        )

        return indices_4_unique_edges, boundary_mask

    def _compute_cells_min_length(self, triangulation: tensordict.TensorDict):
        """For each cells, compute the smaller length of the edges."""
        indices_4_edges, _ = torch.sort(
            triangulation["cells"]["vertices"][..., self._edges_permutations], dim=-1
        )

        coordinates_4_edges = self.compute_coordinates_4_cells(
            triangulation["vertices"]["coordinates"], indices_4_edges
        )

        coordinates_4_edges_first_vertex, coordinates_4_edges_second_vertex = (
            torch.split(coordinates_4_edges, 1, dim=-2)
        )

        diameter_4_cells = torch.min(
            torch.norm(
                coordinates_4_edges_second_vertex - coordinates_4_edges_first_vertex,
                dim=-1,
                keepdim=True,
            ),
            dim=-2,
            keepdim=True,
        )[0]

        return diameter_4_cells

    @property
    @abc.abstractmethod
    def _edges_permutations(self):
        """Return the local node vertices defining each edge of the element.
        the convection is the i-th node share numbering with the edge opposite to it."""
        raise NotImplementedError


class MeshTri(AbstractMesh):
    """Class for triangular mesh representation"""

    @property
    def _edges_permutations(self):
        return torch.tensor([[0, 1], [1, 2], [0, 2]])

    @staticmethod
    def _map_fine_mesh(c4e_coarser_mesh: torch.Tensor, c4e_finer_mesh: torch.Tensor):
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

        mapping = torch.full((c4e_finer_mesh.size(0),), -1, dtype=torch.long)

        candidates = inside.nonzero(as_tuple=False)  # (n_matches, 2)

        seen = torch.zeros(c4e_finer_mesh.size(0), dtype=torch.bool)
        for i in range(candidates.size(0)):
            idx_finer_mesh, idx_coarser_mesh = candidates[i]
            if not seen[idx_finer_mesh]:
                mapping[idx_finer_mesh] = idx_coarser_mesh
                seen[idx_finer_mesh] = True

        return mapping


class FracturesTri(MeshTri):
    """Class for handling multiple fractures represented as triangular meshes"""

    def __init__(self, triangulations: list, fractures_3d_data: torch.Tensor):

        triangulations = self._stack_triangulations(triangulations)

        super().__init__(triangulations)

        self._compute_fracture_map(fractures_3d_data)

        self["vertices"]["coordinates_3d"] = self._fracture_map(
            self["vertices"]["coordinates"]
        )

    @property
    def _edges_permutations(self):
        return torch.tensor([[0, 1], [1, 2], [0, 2]])

    def _stack_triangulations(self, fracture_triangulations: list):
        """Stack multiple fracture triangulations into a single TensorDict"""

        fracture_triangulations_tensordict = [
            tensordict.TensorDict(fracture_triangulation)
            for fracture_triangulation in fracture_triangulations
        ]

        stacked_fractured_triangulations = torch.cat(
            fracture_triangulations_tensordict, dim=0
        )

        return stacked_fractured_triangulations

    def _compute_fracture_map(self, fractures_3d_data: torch.Tensor):
        """compute mapping for each fracture from the 2D space to 3D."""
        vertices_2d = self["vertices"]["coordinates"][:, :3, :]

        vertices_3d = fractures_3d_data[:, :3, :]

        extended_vertices_2d = torch.cat(
            [vertices_2d.mT, torch.ones_like(vertices_3d[:, [-1], :])], dim=-2
        )

        linear_equation = vertices_3d @ torch.inverse(extended_vertices_2d)

        jacobian_fracture_map = linear_equation[..., :2]
        translation_vector = linear_equation[..., [-1]]

        det_jacobian_fracture_map = torch.norm(
            torch.cross(*torch.split(jacobian_fracture_map, 1, dim=-1), dim=-2),
            dim=-2,
            keepdim=True,
        )

        inv_jacobian_fracture_map = (
            torch.inverse(jacobian_fracture_map.mT @ jacobian_fracture_map)
            @ jacobian_fracture_map.mT
        )

        self["jacobian_fracture_map"] = jacobian_fracture_map
        self["inv_jacobian_fracture_map"] = inv_jacobian_fracture_map
        self["det_jacobian_fracture_map"] = det_jacobian_fracture_map
        self["translation_vector"] = translation_vector

    @staticmethod
    def compute_coordinates_4_cells(
        coordinates_4_vertices: torch.Tensor, vertices_4_cells: torch.Tensor
    ):
        """Compute the coordinates of the nodes in the mesh."""
        return coordinates_4_vertices[
            torch.arange(coordinates_4_vertices.size(0))[:, None, None],
            vertices_4_cells,
        ]

    def _fracture_map(self, coordinate_2d: torch.Tensor):
        """For each fracture, compute the map from the 2D
        coordinate to his position of in 3D space"""
        return (
            self["jacobian_fracture_map"] @ coordinate_2d.mT
            + self["translation_vector"]
        ).mT


class AbstractElement(abc.ABC):
    """Abstract class for element representation"""

    def __init__(self, polynomial_order: int, integration_order: int):

        self.polynomial_order = polynomial_order
        self.integration_order = integration_order
        self.inv_map_jacobian = None

        self.gaussian_nodes, self.gaussian_weights = self._compute_gauss_values()

    def compute_inverse_map(
        self,
        first_node: torch.Tensor,
        integration_points: torch.Tensor,
        inv_map_jacobian: torch.Tensor,
    ):
        """Compute the inverse map from physical coordinates to reference coordinates"""

        return (integration_points - first_node) @ inv_map_jacobian.mT

    @abc.abstractmethod
    def compute_shape_functions(
        self, bar_coords: torch.Tensor, inv_map_jacobian: torch.Tensor
    ):
        """Compute the shape functions and their gradients at given barycentric coordinates"""
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_gauss_values(self):
        """Compute the Gaussian integration points and weights"""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_barycentric_coordinates(self, x: torch.Tensor):
        """Compute the barycentric coordinates of given points x"""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_det_and_inv_map(self, map_jacobian: torch.Tensor):
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

    def compute_barycentric_coordinates(self, x: torch.Tensor):
        return torch.concat([0.5 * (1.0 - x), 0.5 * (1.0 + x)], dim=-1)

    def _compute_gauss_values(self):

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

    def compute_det_and_inv_map(self, map_jacobian: torch.Tensor):

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

    def compute_barycentric_coordinates(self, x: torch.Tensor):
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

    def _compute_gauss_values(self):

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
        ) = self._compute_integral_values(mesh, element)

        (
            self.coords4global_dofs,
            self.global_dofs4elements,
            self.nodes4boundary_dofs,
            self.coords4elements,
        ) = self._compute_dofs(
            mesh,
            element,
        )

        self.basis_parameters = self._compute_basis_parameters(
            self.coords4global_dofs, self.global_dofs4elements, self.nodes4boundary_dofs
        )

    def _compute_integral_values(
        self,
        mesh: AbstractMesh,
        element: AbstractElement,
    ):
        """Compute the values for the numerical integration"""

        map_jacobian = self._compute_jacobian_map(mesh, element)

        det_map_jacobian, inv_map_jacobian = element.compute_det_and_inv_map(
            map_jacobian
        )

        bar_coords = element.compute_barycentric_coordinates(element.gaussian_nodes)

        v, v_grad = element.compute_shape_functions(bar_coords, inv_map_jacobian)

        integration_points = self._compute_integration_points(mesh, bar_coords)

        dx = self._compute_integral_weights(element, det_map_jacobian)

        return v, v_grad, integration_points, dx, inv_map_jacobian

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

    def reduce(self, tensor: torch.Tensor):
        """Reduce a tensor to only include inner degrees of freedom"""
        idx = self.basis_parameters["inner_dofs"]
        return tensor[idx, :][:, idx] if tensor.size(-1) != 1 else tensor[idx]

    def interpolate(self, basis: "AbstractBasis", tensor: torch.Tensor = None):
        """Interpolate a tensor from the current basis to another basis."""
        if basis is self:
            indices_4_cells_4_interior_edges = self.global_dofs4elements.unsqueeze(-2)

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

        elif basis.__class__ == InteriorEdgesBasis:

            # elements_mask = basis.mesh.edges_parameters["elements4inner_edges"]

            cells_4_interior_edges = basis.mesh["interior_edges"]["cells"]

            # dofs_idx = basis.mesh.nodes4elements[elements_mask].unsqueeze(-2)

            indices_4_cells_4_interior_edges = basis.mesh.compute_coordinates_4_cells(
                basis.mesh["cells"]["vertices"], cells_4_interior_edges
            ).unsqueeze(-2)

            coordinates_4_cells_first_vertex = basis.mesh.compute_coordinates_4_cells(
                self.mesh["cells"]["coordinates"][..., [0], :], cells_4_interior_edges
            ).unsqueeze(-3)

            inv_map_jacobian = basis.mesh.compute_coordinates_4_cells(
                self.inv_map_jacobian, cells_4_interior_edges
            )

            integration_points = basis.integration_points.unsqueeze(-3)

            # For computing the inverse mapping of the integrations points of the interior edges,
            # is necessary that tensor are in the size (N_T, q_T, q_E, N_f, N_d).

            new_integrations_points = self.element.compute_inverse_map(
                coordinates_4_cells_first_vertex, integration_points, inv_map_jacobian
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

            interpolation = (tensor[indices_4_cells_4_interior_edges] * v).sum(
                -2, keepdim=True
            )

            interpolation_grad = (
                tensor[indices_4_cells_4_interior_edges] * v_grad
            ).sum(-2, keepdim=True)

            return interpolation, interpolation_grad

        else:

            nodes = self.coords4global_dofs

            def interpolator(function):
                return (function(nodes)[indices_4_cells_4_interior_edges] * v).sum(
                    -2, keepdim=True
                )

            def interpolator_grad(function):
                return (function(nodes)[indices_4_cells_4_interior_edges] * v_grad).sum(
                    -2, keepdim=True
                )

            return interpolator, interpolator_grad

    @abc.abstractmethod
    def _compute_dofs(
        self,
        mesh: AbstractMesh,
        element: AbstractElement,
    ):
        """Compute the degrees of freedom for the basis functions"""
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_basis_parameters(
        self, coords4global_dofs, global_dofs4elements, nodes4boundary_dofs
    ):
        """Compute parameters related to the basis functions"""
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_jacobian_map(self, mesh: AbstractMesh, element: AbstractElement):
        """Compute the jacobian of the map that maps the local element to the physical one."""
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_integration_points(self, mesh: AbstractMesh, bar_coords: torch.Tensor):
        """Compute the integration points, applying to map to the local quadrature points
        to obtain the physical integration points for each element"""
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_integral_weights(
        self, element: AbstractMesh, det_map_jacobian: torch.Tensor
    ):
        """Compute the integration weight, composing of the quadrature weights, area of the
        reference element, the determent of the jacobian of the map, as well other quantities
        """
        raise NotImplementedError


class Basis(AbstractBasis):
    """Class for standard basis representation"""

    def _compute_dofs(
        self,
        mesh: AbstractMesh,
        element: AbstractElement,
    ):

        if element.polynomial_order == 1:

            coords_4_global_dofs = mesh["vertices"]["coordinates"]
            global_dofs_4_elements = mesh["cells"]["vertices"]
            nodes4boundary_dofs = mesh["vertices"]["markers"]

        # need to be refactored to account for deprecation of get_edges_idx function
        # elif element.polynomial_order == 2:

        #     new_coords4dofs = (
        #         mesh.coords4nodes[mesh.edges_parameters["nodes4unique_edges"]]
        #     ).mean(-2)
        #     new_nodes4dofs = (
        #         mesh.edges_parameters["edges_idx"].reshape(
        #             mesh.mesh_parameters["nb_simplex"], 3
        #         )
        #         + mesh.mesh_parameters["nb_nodes"]
        #     )
        #     new_nodes4boundary_dofs = (
        #         mesh.edges_parameters["nodes_idx4boundary_edges"]
        #         + mesh.mesh_parameters["nb_nodes"]
        #     )

        #     coords4global_dofs = torch.cat([mesh.coords4nodes, new_coords4dofs], dim=-2)
        #     global_dofs4elements = torch.cat(
        #         [mesh.nodes4elements, new_nodes4dofs], dim=-1
        #     )
        #     nodes4boundary_dofs = torch.cat(
        #         [mesh.nodes4boundary, new_nodes4boundary_dofs], dim=-1
        #     )
        else:
            raise NotImplementedError("Polynomial order not implemented")

        coords4elements = mesh.compute_coordinates_4_cells(
            coords_4_global_dofs, global_dofs_4_elements
        )

        return (
            coords_4_global_dofs,
            global_dofs_4_elements,
            nodes4boundary_dofs,
            coords4elements,
        )

    def _compute_basis_parameters(
        self, coords4global_dofs, global_dofs4elements, nodes4boundary_dofs
    ):

        nb_global_dofs = coords4global_dofs.size(-2)
        nb_local_dofs = global_dofs4elements.size(-1)

        inner_dofs = torch.nonzero(nodes4boundary_dofs != 1, as_tuple=True)[-2]

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

    def _compute_jacobian_map(self, mesh, element):
        return mesh["cells"]["coordinates"].mT @ element.barycentric_grad

    def _compute_integration_points(self, mesh, bar_coords):
        return bar_coords.mT @ mesh["cells"]["coordinates"].unsqueeze(-3)

    def _compute_integral_weights(self, element, det_map_jacobian):
        return (
            element.reference_element_area * element.gaussian_weights * det_map_jacobian
        )


class InteriorEdgesBasis(AbstractBasis):
    """Class for basis representation on interior edges"""

    def _compute_dofs(
        self,
        mesh: AbstractMesh,
        element: AbstractElement,
    ):

        if element.polynomial_order == 1:
            ## WARNING !!!! THIS IS NOT CORRECT, NEED TO FIX
            coords_4_global_dofs = mesh["vertices"]["coordinates"]
            global_dofs_4_elements = mesh["cells"]["vertices"]
            nodes4boundary_dofs = mesh["vertices"]["markers"]
        else:
            raise NotImplementedError("Polynomial order not implemented")

        coords4elements = mesh.compute_coordinates_4_cells(
            coords_4_global_dofs, global_dofs_4_elements
        )

        return (
            coords_4_global_dofs,
            global_dofs_4_elements,
            nodes4boundary_dofs,
            coords4elements,
        )

    def _compute_basis_parameters(
        self, coords4global_dofs, global_dofs4elements, nodes4boundary_dofs
    ):

        nb_global_dofs = coords4global_dofs.size(-2)
        nb_local_dofs = global_dofs4elements.size(-1)

        inner_dofs = torch.nonzero(nodes4boundary_dofs != 1, as_tuple=True)[-2]

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

    def _compute_jacobian_map(self, mesh, element):
        return mesh["interior_edges"]["coordinates"].mT @ element.barycentric_grad

    def _compute_integration_points(self, mesh, bar_coords):
        return bar_coords.mT @ mesh["interior_edges"]["coordinates"].unsqueeze(-3)

    def _compute_integral_weights(self, element, det_map_jacobian):
        return (
            element.reference_element_area * element.gaussian_weights * det_map_jacobian
        )


class FractureBasis(AbstractBasis):
    """Class for basis representation on fractures"""

    def __init__(self, mesh: AbstractMesh, element: AbstractElement):
        self.global_triangulation = self._build_global_triangulation(mesh)

        super().__init__(mesh, element)

    def _build_global_triangulation(self, mesh):
        """Build a global triangulation from local triangulations of multiple fractures."""
        nb_fractures, nb_vertices, _ = mesh["vertices"]["coordinates"].size()

        nb_edges = mesh["edges"]["vertices"].size(-2)

        local_triangulation_3d_coords = mesh["vertices_3D"].reshape(-1, 3)

        global_vertices_3d, global2local_idx, vertex_counts = torch.unique(
            local_triangulation_3d_coords,
            dim=0,
            return_inverse=True,
            return_counts=True,
        )

        nb_global_vertices = global_vertices_3d.size(-2)

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

        global_vertices_2d = mesh["vertices"]["coordinates"].reshape(-1, 2)[
            local2global_idx
        ]

        vertices_offset = torch.arange(nb_fractures)[:, None, None] * nb_vertices

        global_triangles = global2local_idx[
            mesh["triangles"] + vertices_offset
        ].reshape(-1, 3)

        local_edges_2_global = global2local_idx[
            mesh["edges"] + vertices_offset
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

        nb_global_edges = global_edges.size(-2)

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

        global_vertices_marker = mesh["vertices"]["markers"].reshape(-1)[
            local2global_idx
        ]
        global_edges_marker = mesh["edges"]["markers"].reshape(-1)[
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

    def _compute_integral_values(self, mesh: AbstractMesh, element: AbstractElement):

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

        integration_points = mapped_3d_gaussian_nodes

        dx = (
            element.reference_element_area
            * element.gaussian_weights
            * det_map_jacobian
            * mesh.local_triangulations["det_fractures_map_jacobian"]
        )

        return v, v_grad, integration_points, dx, inv_map_jacobian

    def _compute_dofs(self, mesh, element):

        if element.polynomial_order == 1:

            coords4global_dofs = self.global_triangulation["vertices_2D"]
            global_dofs4elements = self.global_triangulation["triangles"]
            nodes4boundary_dofs = torch.nonzero(
                self.global_triangulation["vertex_markers"] == 1
            )[:, 0]

        else:
            raise NotImplementedError("Polynomial order not implemented")

        return coords4global_dofs, global_dofs4elements, nodes4boundary_dofs

    def _compute_basis_parameters(
        self, coords4global_dofs, global_dofs4elements, nodes4boundary_dofs
    ):

        nb_global_dofs = self.global_triangulation["vertices_2D"].size(-2)
        nb_local_dofs = self.global_triangulation["triangles"].size(-1)

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


class InteriorEdgesFractureBasis(AbstractBasis):
    """Class for basis representation on interior edges of fractures"""

    def _compute_integral_values(self, mesh: AbstractMesh, element: AbstractElement):

        nodes4elements = mesh.edges_parameters["nodes4inner_edges"]
        coords4nodes = mesh.local_triangulations["vertices"]
        coords4elements = coords4nodes[
            torch.arange(coords4nodes.size(0))[:, None, None], nodes4elements
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

        integration_points = mapped_3d_gaussian_nodes

        dx = (
            element.reference_element_area
            * element.gaussian_weights
            * det_map_jacobian
            * mesh.local_triangulations["det_fractures_map_jacobian"]
        )

        return v, v_grad, integration_points, dx, inv_map_jacobian

    def _compute_dofs(
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

    def _compute_basis_parameters(
        self, coords4global_dofs, global_dofs4elements, nodes4boundary_dofs
    ):

        nb_global_dofs = coords4global_dofs.size(-2)
        nb_local_dofs = global_dofs4elements.size(-1)

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
