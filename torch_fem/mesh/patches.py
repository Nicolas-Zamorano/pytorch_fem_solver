"""Class for representing triangular-based Patches"""

import torch
from .meshes_tri import MeshesTri


class Patches(MeshesTri):
    """Class for representing triangular-based Patches"""

    def __init__(self, centers: torch.Tensor, radius: torch.Tensor):

        self.centers = centers
        self.radius = radius

        triangulations = self._compute_patches(centers, radius)

        super().__init__(triangulations)

    def _compute_patches(self, centers: torch.Tensor, radius: torch.Tensor):
        coordinates_4_vertices_4_patches = centers.unsqueeze(
            -2
        ) + self.signs_4_vertices * radius.unsqueeze(-2)

        angle = torch.tensor(torch.pi / 4)
        self.rotation_matrix = torch.tensor(
            [
                [torch.cos(angle), -torch.sin(angle)],
                [torch.sin(angle), torch.cos(angle)],
            ]
        )

        self.rotated_signs = (
            self.rotation_matrix
            @ self.signs_4_vertices.to(torch.get_default_dtype()).mT
        ).mT

        triangulations = []

        for coordinates_4_vertices_4_patch in coordinates_4_vertices_4_patches:
            triangulation = {
                "vertices": coordinates_4_vertices_4_patch,
                "triangles": self.vertices_4_cells_4_patch,
                "vertex_markers": self.markers_4_vertices,
            }
            triangulations.append(triangulation)

        return triangulations

    def refine_patches(
        self, refine_idx: torch.Tensor, maintain_old_patches: bool = False
    ):
        """Refine patches mark for refine_idx"""
        new_radius = 0.5 * self.radius[refine_idx]

        new_centers = self.centers[refine_idx, :].unsqueeze(-2) + self.signs_4_vertices[
            :-1, :
        ] * new_radius.unsqueeze(
            -2
        )  # WARNING!!! this swaps the order of the centers, now there are order
        # in clockwise order (originally, we markers in counter-clockwise order)

        new_coordinates_4_vertices = new_centers.unsqueeze(-2) + (
            self.signs_4_vertices * new_radius.unsqueeze(-2)
        ).unsqueeze(-3)

        rotated_radius = 2 * new_radius / torch.sqrt(torch.tensor(2.0))

        rotated_centers = self.centers[refine_idx, :]

        rotated_coordinates_4_vertices = rotated_centers.unsqueeze(
            -2
        ) + self.rotated_signs.unsqueeze(0) * rotated_radius.unsqueeze(-1)

        if maintain_old_patches is False:

            refined_radius = torch.concat(
                [self.radius[~refine_idx], new_radius.repeat(4, 1), rotated_radius],
                dim=0,
            )

            refined_centers = torch.concat(
                [
                    self.centers[~refine_idx, :],
                    new_centers.view(-1, 2),
                    rotated_centers,
                ],
                dim=0,
            )

            refined_coordinates_4_vertices = torch.concat(
                [
                    self["vertices", "coordinates"][~refine_idx, ...],
                    new_coordinates_4_vertices.view(-1, 4, 2),
                    rotated_coordinates_4_vertices,
                ],
                dim=0,
            )

        else:

            refined_radius = torch.concat(
                [self.radius, new_radius.repeat(4, 1), rotated_radius], dim=0
            )

            refined_centers = torch.concat(
                [
                    self.centers,
                    new_centers.view(-1, 2),
                    rotated_centers,
                ],
                dim=0,
            )

            refined_coordinates_4_vertices = torch.concat(
                [
                    self["vertices", "coordinates"],
                    new_coordinates_4_vertices.view(-1, 4, 2),
                    rotated_coordinates_4_vertices,
                ],
                dim=0,
            )

        return refined_centers, refined_radius, refined_coordinates_4_vertices

    def uniform_refine(self, nb_refinements: int = 1):
        """compute a uniform refine over every patch."""
        new_centers = self.centers
        new_radius = self.radius
        new_coords4nodes = None
        for _ in range(nb_refinements):
            new_centers, new_radius, new_coords4nodes = self.refine_patches(
                torch.tensor([True] * self.batch_size()[0])
            )

        return new_centers, new_radius, new_coords4nodes

    @property
    def signs_4_vertices(self):
        """signs necessary to compute vertices en the correct position"""
        return torch.tensor(
            [[-1, -1], [1, -1], [1, 1], [-1, 1], [0, 0]], dtype=torch.int64
        )

    @property
    def vertices_4_cells_4_patch(self):
        """vertices for cells of the each patch."""
        return torch.tensor(
            [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=torch.int64
        )

    @property
    def markers_4_vertices(self):
        """Markers for vertices, 1 is boundary and 0 is interior"""
        return torch.tensor([[1], [1], [1], [1], [0]])
