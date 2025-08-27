from abc import ABC, abstractmethod
import torch

torch.set_default_dtype(torch.float64)


class AbstractMesh(ABC):

    def __init__(self, triangulation: dict):
        self.coords4nodes = triangulation["vertices"]
        self.nodes4elements = triangulation["triangles"]

        self.coords4elements = self.coords4nodes[self.nodes4elements]

        self.mesh_parameters = self.compute_mesh_parameters(
            self.coords4nodes, self.nodes4elements
        )

        self.elements_diameter, self.nodes4boundary, self.edges_parameters = (
            self.compute_edges_values(
                self.coords4nodes,
                self.nodes4elements,
                self.mesh_parameters,
                triangulation,
            )
        )

    def compute_edges_values(
        self,
        coords4nodes: torch.Tensor,
        nodes4elements: torch.Tensor,
        mesh_parameters: torch.Tensor,
        triangulation: dict,
    ):

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
        )[1].reshape(-1, mesh_parameters["nb_dimensions"])

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

        inner_elements_centroid = self.coords4elements[elements4inner_edges].mean(
            dim=-2
        )

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
        }

        return elements_diameter, nodes4boundary, edges_parameters

    @abstractmethod
    def compute_mesh_parameters(
        self, coords4nodes: torch.Tensor, nodes4elements: torch.Tensor
    ):
        raise NotImplementedError

    @property
    @abstractmethod
    def edges_permutations(self):
        raise NotImplementedError

    @abstractmethod
    def get_edges_idx(self, nodes4elements, nodes4unique_edges):
        raise NotImplementedError


class MeshTri(AbstractMesh):

    @property
    def edges_permutations(self):
        return torch.tensor([[0, 1], [1, 2], [0, 2]])

    def compute_mesh_parameters(
        self, coords4nodes: torch.Tensor, nodes4elements: torch.Tensor
    ):

        nb_nodes, nb_dimensions = coords4nodes.shape
        nb_simplex = nodes4elements.shape[-2]

        return {
            "nb_nodes": nb_nodes,
            "nb_dimensions": nb_dimensions,
            "nb_simplex": nb_simplex,
        }

    def map_fine_mesh(self, finer_mesh: torch.Tensor):
        c4e_finer_mesh = finer_mesh.coords4elements  # (n_elem_h, 3, 2)
        c4e_coarser_mesh = self.coords4elements  # (n_elem_H, 3, 2)
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

    def get_edges_idx(self, nodes4elements, nodes4unique_edges):
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


class AbstractElement(ABC):
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

        integration_points = torch.concat(integration_points, dim=-1)

        return (integration_points - first_node) @ inv_map_jacobian.mT

    @abstractmethod
    def shape_functions_value_and_grad(self, bar_coords, inv_map_jacobian):
        raise NotImplementedError

    @abstractmethod
    def compute_gauss_values(self):
        raise NotImplementedError

    @abstractmethod
    def compute_barycentric_coordinates(self, x):
        raise NotImplementedError

    @abstractmethod
    def compute_det_and_inv_map(self, map_jacobian):
        raise NotImplementedError

    @property
    @abstractmethod
    def reference_element_area(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def barycentric_grad(self):
        raise NotImplementedError


class ElementLine(AbstractElement):

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

    def shape_functions_value_and_grad(
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

    @property
    def barycentric_grad(self):
        return torch.tensor([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])

    @property
    def reference_element_area(self):
        return 0.5

    def compute_barycentric_coordinates(self, x):
        return torch.stack(
            [1.0 - x[..., [0]] - x[..., [1]], x[..., [0]], x[..., [1]]], dim=-2
        )

    def shape_functions_value_and_grad(
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


class AbstractBasis(ABC):
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

        self.coords4global_dofs, self.global_dofs4elements, self.nodes4boundary_dofs = (
            self.compute_dofs(
                mesh,
                element,
            )
        )

        self.coords4elements = self.coords4global_dofs[self.global_dofs4elements]

        self.basis_parameters = self.compute_basis_parameters(
            self.coords4global_dofs, self.global_dofs4elements, self.nodes4boundary_dofs
        )

    @abstractmethod
    def compute_integral_values(
        self,
        mesh: AbstractMesh,
        element: AbstractElement,
    ):
        raise NotImplementedError

    def integrate_functional(self, function, *args, **kwargs):

        return (function(self, *args, **kwargs) * self.dx).sum(-3).sum(-2)

    def integrate_bilinear_form(self, function, *args, **kwargs):

        global_matrix = torch.zeros(self.basis_parameters["bilinear_form_shape"])

        local_matrix = (function(self, *args, **kwargs) * self.dx).sum(-3)

        global_matrix.index_put_(
            self.basis_parameters["bilinear_form_idx"],
            local_matrix.reshape(-1),
            accumulate=True,
        )

        return global_matrix

    def integrate_linear_form(self, function, *args, **kwargs):

        integral_value = torch.zeros(self.basis_parameters["linear_form_shape"])

        integrand_value = (function(self, *args, **kwargs) * self.dx).sum(-3)

        integral_value.index_put_(
            self.basis_parameters["linear_form_idx"],
            integrand_value.reshape(-1, 1),
            accumulate=True,
        )

        return integral_value

    @abstractmethod
    def compute_dofs(
        self,
        mesh: AbstractMesh,
        element: AbstractElement,
    ):
        raise NotImplementedError

    @abstractmethod
    def compute_basis_parameters(
        self, coords4global_dofs, global_dofs4elements, nodes4boundary_dofs
    ):
        raise NotImplementedError


class Basis(AbstractBasis):

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

        v, v_grad = element.shape_functions_value_and_grad(bar_coords, inv_map_jacobian)

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
            "inner_dofs": inner_dofs,
            "nb_dofs": nb_global_dofs,
        }

    def reduce(self, tensor):
        idx = self.basis_parameters["inner_dofs"]
        return tensor[idx, :][:, idx] if tensor.shape[-1] != 1 else tensor[idx]

    def interpolate(self, basis, tensor=None):

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

            v, v_grad = self.element.shape_functions_value_and_grad(
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


class InteriorFacetBasis(AbstractBasis):

    def compute_integral_values(self, mesh, element):

        coords4inner_facet = mesh.coords4nodes[
            mesh.edges_parameters["nodes4inner_edges"]
        ]

        map_jacobian = coords4inner_facet.mT @ element.barycentric_grad

        det_map_jacobian, inv_map_jacobian = element.compute_det_and_inv_map(
            map_jacobian
        )

        bar_coords = element.compute_barycentric_coordinates(element.gaussian_nodes)

        v, v_grad = element.shape_functions_value_and_grad(bar_coords, inv_map_jacobian)

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
