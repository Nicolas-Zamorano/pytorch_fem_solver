"""Class for basis representation on fractures"""

import torch
import tensordict
from ..mesh.abstract_mesh import AbstractMesh
from ..element.abstract_element import AbstractElement
from .interior_edges_fracture_basis import InteriorEdgesFractureBasis
from .abstract_basis import AbstractBasis


class FractureBasis(AbstractBasis):
    """Class for basis representation on fractures"""

    def __init__(self, mesh: AbstractMesh, element: AbstractElement):
        self.global_triangulation = self._build_global_triangulation(mesh)

        super().__init__(mesh, element)

        self.v_grad = self.v_grad @ mesh["inv_jacobian_fracture_map"].unsqueeze(
            -3
        ).unsqueeze(-3)

        self.inv_map_jacobian = self.inv_map_jacobian @ mesh[
            "inv_jacobian_fracture_map"
        ].unsqueeze(-3).unsqueeze(-3)

    def _build_global_triangulation(self, mesh: AbstractMesh):
        """Build a global triangulation from local triangulations of multiple fractures."""
        nb_fractures, nb_vertices, _ = mesh["vertices", "coordinates"].size()

        nb_edges = mesh["edges", "vertices"].size(-2)

        local_triangulation_3d_coords = mesh["vertices", "coordinates_3d"].reshape(
            -1, 3
        )

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

        global_vertices_2d = mesh["vertices", "coordinates"].reshape(-1, 2)[
            local2global_idx
        ]

        vertices_offset = torch.arange(nb_fractures)[:, None, None] * nb_vertices

        global_triangles = global2local_idx[
            mesh["cells", "vertices"] + vertices_offset
        ].reshape(-1, 3)

        local_edges_2_global = global2local_idx[
            mesh["edges", "vertices"] + vertices_offset
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

        global_vertices_marker = mesh["vertices", "markers"].reshape(-1)[
            local2global_idx
        ]
        global_edges_marker = mesh["edges", "markers"].reshape(-1)[
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

    def _compute_dofs(self, mesh: AbstractMesh, element: AbstractElement):

        if element.polynomial_order == 1:

            coords_4_global_dofs = self.global_triangulation["vertices_2D"]
            global_dofs_4_elements = self.global_triangulation["triangles"]
            nodes_4_boundary_dofs = torch.nonzero(
                self.global_triangulation["vertex_markers"] == 1
            )[:, 0]

        else:
            raise NotImplementedError("Polynomial order not implemented")

        coords_4_elements = coords_4_global_dofs[global_dofs_4_elements]

        return (
            coords_4_global_dofs,
            global_dofs_4_elements,
            nodes_4_boundary_dofs,
            coords_4_elements,
        )

    def _compute_basis_parameters(
        self,
        coords4global_dofs: torch.Tensor,
        global_dofs4elements: torch.Tensor,
        nodes4boundary_dofs: torch.Tensor,
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

    def _compute_integral_weights(
        self, element: AbstractElement, det_map_jacobian: torch.Tensor
    ):
        return (
            element.reference_element_area
            * element.gaussian_weights
            * det_map_jacobian
            * self.mesh["det_jacobian_fracture_map"].unsqueeze(-1).unsqueeze(-1)
        )

    def _compute_integration_points(self, mesh: AbstractMesh, bar_coords: torch.Tensor):
        mapped_integration_points_2d = bar_coords.mT @ mesh[
            "cells", "coordinates"
        ].unsqueeze(-3)
        return (
            mesh["jacobian_fracture_map"].unsqueeze(-3).unsqueeze(-3)
            @ mapped_integration_points_2d.mT
            + mesh["translation_vector"].unsqueeze(-3).unsqueeze(-3)
        ).mT

    def _compute_jacobian_map(self, mesh, element):
        return mesh["cells", "coordinates"].mT @ element.barycentric_grad

    def interpolate(self, basis: AbstractBasis, tensor: torch.Tensor = None):
        """Interpolate a tensor from the current basis to another basis."""
        if basis is self:
            nb_fractures = self.mesh.batch_size()[0]
            nb_vertices_4_cells = self.mesh["cells"].batch_size[-1]
            # vertices_4_cells_4_interior_edges = self.global_dofs4elements.unsqueeze(-2)
            vertices_4_cells_4_interior_edges = self.global_dofs4elements.reshape(
                nb_fractures, -1, 1, nb_vertices_4_cells
            )

            v = self.v
            v_grad = self.v_grad

        elif basis.__class__ == InteriorEdgesFractureBasis:

            cells_4_interior_edges = basis.mesh["interior_edges", "cells"]

            vertices_4_cells_4_interior_edges = basis.mesh.compute_coordinates_4_cells(
                basis.mesh["cells", "vertices"], cells_4_interior_edges
            ).unsqueeze(-2)

            coordinates_4_cells_first_vertex = basis.mesh.compute_coordinates_4_cells(
                self.mesh["cells", "coordinates_3d"][..., [0], :],
                cells_4_interior_edges,
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
                "Interpolation to {basis.__class__} not implemented"
            )

        if tensor is not None:

            interpolation = (tensor[vertices_4_cells_4_interior_edges] * v).sum(
                -2, keepdim=True
            )

            interpolation_grad = (
                tensor[vertices_4_cells_4_interior_edges] * v_grad
            ).sum(-2, keepdim=True)

            return interpolation, interpolation_grad

        else:

            nodes = self.coords4global_dofs

            def interpolator(function):
                return (function(nodes)[vertices_4_cells_4_interior_edges] * v).sum(
                    -2, keepdim=True
                )

            def interpolator_grad(function):
                return (
                    function(nodes)[vertices_4_cells_4_interior_edges] * v_grad
                ).sum(-2, keepdim=True)

            return interpolator, interpolator_grad
