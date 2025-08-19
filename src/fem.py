from abc import ABC, abstractmethod
import torch

torch.set_default_dtype(torch.float64)


class AbstractMesh(ABC):
    """
    Abstract base class for mesh representation in finite element methods.
    This class provides the structure and essential methods for handling mesh data,
    including nodes, elements, edges, and boundary information. It is designed to be
    subclassed with specific implementations for mesh parameter computations.
    Attributes:
        coords4nodes (torch.Tensor): Coordinates of mesh nodes.
        nodes4elements (torch.Tensor): Node indices for each mesh element.
        coords4elements (torch.Tensor): Coordinates for each mesh element.
        mesh_parameters (dict): Computed mesh parameters.
        elements_diameter (torch.Tensor): Diameter of each element.
        nodes4boundary (torch.Tensor): Indices of boundary nodes.
        edges_parameters (dict): Parameters related to mesh edges.
    Args:
        triangulation (dict): Dictionary containing mesh data, including vertices,
            triangles, edges, and markers.
    Methods:
        compute_edges_values(coords4nodes, nodes4elements, mesh_parameters, triangulation):
            Computes edge-related values such as edge indices, boundary nodes, edge normals,
            and other edge parameters.
        compute_mesh_parameters(coords4nodes, nodes4elements):
            Abstract method to compute mesh parameters. Must be implemented by subclasses.
    """

    def __init__(self, triangulation: dict):
        """
        Initializes the FEM solver with the provided triangulation data.
        Args:
            triangulation (dict): A dictionary containing mesh information with at least the
            following keys:
                - "vertices": Array-like structure of node coordinates.
                - "triangles": Array-like structure of element connectivity (indices of nodes forming each element).
        Attributes:
            coords4nodes (array-like): Coordinates of the mesh nodes.
            nodes4elements (array-like): Node indices for each element in the mesh.
            coords4elements (array-like): Coordinates of the nodes for each element.
            mesh_parameters (Any): Mesh parameters computed from node coordinates and element connectivity.
            elements_diameter (Any): Diameter or characteristic size of each element.
            nodes4boundary (Any): Indices or identifiers of boundary nodes.
            edges_parameters (Any): Parameters associated with mesh edges, as computed by `compute_edges_values`.
        Raises:
            KeyError: If required keys are missing from the triangulation dictionary.
        """

        self.coords4nodes = triangulation["vertices"]
        self.nodes4elements = triangulation["triangles"]

        self.coords4elements = self.coords4nodes[self.nodes4elements]

        self.mesh_parameters = self.compute_mesh_parameters(self.coords4nodes, self.nodes4elements)

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
        """
        Computes various edge-related quantities for a finite element mesh.
        Args:
            coords4nodes (torch.Tensor): Tensor containing the coordinates of the mesh nodes.
            nodes4elements (torch.Tensor): Tensor mapping elements to their node indices.
            mesh_parameters (torch.Tensor): Tensor or dict containing mesh parameters such as number of dimensions.
            triangulation (dict): Dictionary containing triangulation information, including:
                - "edges": Tensor of unique edge node indices.
                - "edge_markers": Tensor indicating boundary (1) or inner (0) edges.
                - "vertex_markers": Tensor indicating boundary nodes.
        Returns:
            elements_diameter (torch.Tensor): Minimum edge length for each element.
            nodes4boundary (torch.Tensor): Indices of nodes located on the boundary.
            edges_parameters (dict): Dictionary containing computed edge-related quantities:
                - "nodes4edges": Node indices for each edge in each element.
                - "edges_idx": Indices mapping element edges to unique edges.
                - "nodes4unique_edges": Node indices for all unique edges.
                - "elements4boundary_edges": Element indices for boundary edges.
                - "nodes4inner_edges": Node indices for inner edges.
                - "elements4inner_edges": Element indices for inner edges.
                - "nodes_idx4boundary_edges": Indices of boundary edges among unique edges.
                - "inner_edges_length": Lengths of inner edges.
                - "normal4inner_edges": Normal vectors for inner edges, oriented consistently.
        """

        nodes4edges, _ = torch.sort(nodes4elements[..., self.edges_permutations], dim=-1)

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

        coords4inner_edges = coords4nodes[nodes4inner_edges]

        coords4inner_edges_1, coords4inner_edges_2 = torch.split(coords4inner_edges, 1, dim=-2)

        inner_edges_vector = coords4inner_edges_2 - coords4inner_edges_1

        inner_edges_length = torch.norm(inner_edges_vector, dim=-1, keepdim=True)

        normal4inner_edges = (
            inner_edges_vector[..., [1, 0]] * torch.tensor([-1.0, 1.0]) / inner_edges_length
        )

        inner_elements_centroid = self.coords4elements[elements4inner_edges].mean(dim=-2)

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
    def compute_mesh_parameters(self, coords4nodes: torch.Tensor, nodes4elements: torch.Tensor):
        """
        Computes mesh-related parameters based on node coordinates and element connectivity.

        Args:
            coords4nodes (torch.Tensor): A tensor containing the coordinates of each node in the mesh.
            nodes4elements (torch.Tensor): A tensor specifying the node indices that make up each element.

        Raises:
            NotImplementedError: This method should be implemented in a subclass.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    @property
    def edges_permutations(self):
        """
        Returns the permutations of node indices for edges in the mesh.
        This property should be defined in subclasses to specify how edges are represented.
        Returns:
            torch.Tensor: A tensor containing the permutations of node indices for edges.
        """
        raise NotImplementedError

    @abstractmethod
    def get_edges_idx(self, nodes4elements, nodes4unique_edges):
        """
        Given the node indices for each element and the unique edges in the mesh,
        returns the indices of the unique edges corresponding to each edge of every element.
        Args:
            nodes4elements (torch.Tensor): Tensor of shape (..., 3) containing node indices for each triangle element.
            nodes4unique_edges (torch.Tensor): Tensor of shape (n_unique_edges, 2) containing node indices for each unique edge.
        Returns:
            torch.Tensor: Tensor of shape (..., 3) with the indices of the unique edges for each triangle's edges.
        """
        raise NotImplementedError


class MeshTri(AbstractMesh):
    """Mesh representation for triangular elements in finite element methods."""

    def __init__(self, triangulation: dict):

        self.edges_permutations = torch.tensor([[0, 1], [1, 2], [0, 2]])

        super().__init__(triangulation)

    def compute_mesh_parameters(self, coords4nodes: torch.Tensor, nodes4elements: torch.Tensor):

        nb_nodes, nb_dimensions = coords4nodes.shape
        nb_simplex = nodes4elements.shape[-2]

        mesh_parameters = {
            "nb_nodes": nb_nodes,
            "nb_dimensions": nb_dimensions,
            "nb_simplex": nb_simplex,
        }

        return mesh_parameters

    def map_fine_mesh(self, fine_mesh: torch.Tensor):
        """
        Maps each triangle (element) of a fine mesh to the corresponding triangle in a coarse mesh.
        Given a fine mesh and the current (coarse) mesh, this method returns a tensor of shape
        (n_elements_fine,) where each entry i indicates the index of the triangle in the coarse mesh
        that contains the centroid of the i-th triangle in the fine mesh. If a fine mesh triangle's
        centroid is not contained in any coarse mesh triangle, its entry is set to -1.
        Args:
            fine_mesh (torch.Tensor): The fine mesh object, expected to have an attribute
                `coords4elements` of shape (n_elem_fine, 3, 2), representing the coordinates of the
                vertices of each triangle.
        Returns:
            torch.LongTensor: A tensor of shape (n_elements_fine,) where each entry is the index of the
            containing coarse mesh triangle, or -1 if not found.
        """
        c4e_finer_mesh = fine_mesh.coords4elements  # (n_elem_h, 3, 2)
        c4e_coarser_mesh = self.coords4elements  # (n_elem_H, 3, 2)
        centroids_finer_mesh = c4e_finer_mesh.mean(dim=-2)  # (n_elem_h, 2)

        # Expandimos para broadcasting
        P = centroids_finer_mesh[:, None, :]  # (n_elem_h, 1, 2)
        A = c4e_coarser_mesh[:, 0, 0, :][None, :, :]  # (1, n_elem_H, 2)
        B = c4e_coarser_mesh[:, 0, 1, :][None, :, :]
        C = c4e_coarser_mesh[:, 0, 2, :][None, :, :]

        v0 = C - A  # (1, n_elem_H, 2)
        v1 = B - A
        v2 = P - A  # (n_elem_h, n_elem_H, 2)

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

        # Inicializar con -1
        mapping = torch.full((c4e_finer_mesh.shape[0],), -1, dtype=torch.long)

        # Para cada triángulo fino, buscamos el primer triángulo grueso que lo contiene
        candidates = inside.nonzero(as_tuple=False)  # shape (n_matches, 2)

        seen = torch.zeros(c4e_finer_mesh.shape[0], dtype=torch.bool)
        for i in range(candidates.shape[0]):
            idx_finer_mesh, idx_coarser_mesh = candidates[i]
            if not seen[idx_finer_mesh]:
                mapping[idx_finer_mesh] = idx_coarser_mesh
                seen[idx_finer_mesh] = True

        return mapping

    def get_edges_idx(self, nodes4elements, nodes4unique_edges):
        """
        Given the node indices for each element (triangle) and the unique edges in the mesh,
        returns the indices of the unique edges corresponding to each edge of every triangle.
        Args:
            nodes4elements (torch.Tensor): Tensor of shape (..., 3) containing node indices for each triangle element.
            nodes4unique_edges (torch.Tensor): Tensor of shape (n_unique_edges, 2) containing node indices for each unique edge.
        Returns:
            torch.Tensor: Tensor of shape (..., 3) with the indices of the unique edges for each triangle's edges.
        """
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
            nodes4unique_edges.min(dim=-1).values * M + nodes4unique_edges.max(dim=-1).values
        )  # (n_unique_edges,)

        # 4. Crear tabla de búsqueda
        sorted_keys, sorted_idx = torch.sort(edge_keys)  # Necesario para searchsorted
        flat_tri_keys = tri_keys.flatten()  # (n_triangles * 3,)

        # 5. Buscar cada key en sorted_keys
        edge_pos = torch.searchsorted(sorted_keys, flat_tri_keys)
        edge_indices = sorted_idx[edge_pos].reshape(tri_keys.shape)  # (n_triangles, 3)

        return edge_indices


class AbstractElement(ABC):
    """
    Abstract base class for finite element types used in FEM solvers.
    This class defines the interface and common functionality for finite elements,
    including the computation of Gaussian quadrature nodes and weights, mapping
    between reference and physical elements, and evaluation of shape functions and
    their gradients.
    Attributes:
        polynomial_order (int): Polynomial order of the element.
        integration_order (int): Integration order for Gaussian quadrature.
        gaussian_nodes (torch.Tensor): Gaussian quadrature nodes for the element.
        gaussian_weights (torch.Tensor): Gaussian quadrature weights for the element.
    Methods:
        compute_integral_values(coords4elements):
            Computes shape function values, gradients, integration points, and
            integration weights for the given element coordinates.
        compute_map(coords4elements):
            Computes the Jacobian of the mapping from reference to physical element,
            its determinant, and its inverse.
        compute_shape_functions(gaussian_nodes, inv_map_jacobian):
            Computes barycentric coordinates, shape function values, and gradients
            at the given Gaussian nodes.
        compute_inverse_map(first_node, integration_points, inv_map_jacobian):
            Computes the inverse mapping from physical to reference coordinates.
        compute_gauss_values():
            Abstract method to compute Gaussian quadrature nodes and weights.
            Must be implemented by subclasses.
        shape_functions_value_and_grad(bar_coords, inv_map_jacobian):
            Abstract method to compute shape function values and gradients.
            Must be implemented by subclasses.
        compute_det_and_inv_map(map_jacobian):
            Abstract method to compute the determinant and inverse of the mapping
            Jacobian. Must be implemented by subclasses.
    """

    def __init__(self, polynomial_order: int, integration_order: int):

        self.polynomial_order = polynomial_order
        self.integration_order = integration_order

        self.gaussian_nodes, self.gaussian_weights = self.compute_gauss_values(polynomial_order)

    def compute_integral_values(self, coords4elements: torch.Tensor):
        """
        Computes the values required for numerical integration over finite elements.
        This method calculates the determinant and inverse of the mapping Jacobian,
        evaluates the shape functions and their gradients at Gaussian quadrature nodes,
        computes the physical coordinates of the integration points, and determines
        the integration weights for each element.
        Args:
            coords4elements (torch.Tensor): Tensor of shape (..., n_nodes, dim) containing
                the coordinates of the nodes for each element.
        Returns:
            v (torch.Tensor): Values of the shape functions at the integration points.
            v_grad (torch.Tensor): Gradients of the shape functions at the integration points.
            integration_points (tuple of torch.Tensor): Physical coordinates of the integration points.
            dx (torch.Tensor): Integration weights for each element.
        """
        det_map_jacobian, self.inv_map_jacobian = self.compute_map(coords4elements)

        bar_coords, v, v_grad = self.compute_shape_functions(
            self.gaussian_nodes, self.inv_map_jacobian, self.polynomial_order
        )

        integration_points = torch.split(bar_coords.mT @ coords4elements.unsqueeze(-3), 1, dim=-1)

        dx = self.reference_element_area * self.gaussian_weights * det_map_jacobian

        return v, v_grad, integration_points, dx

    def compute_map(self, coords4elements: torch.Tensor):
        """
        Computes the Jacobian matrix, its determinant, and its inverse for the mapping from reference to physical elements.
        Args:
            coords4elements (torch.Tensor): Tensor of shape (..., n_nodes, dim) containing the coordinates of the nodes for each element.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - det_map_jacobian: Determinant of the Jacobian matrix for each element.
                - inv_map_jacobian: Inverse of the Jacobian matrix for each element.
        """

        map_jacobian = coords4elements.mT @ self.barycentric_grad

        det_map_jacobian, inv_map_jacobian = self.compute_det_and_inv_map(map_jacobian)

        return det_map_jacobian, inv_map_jacobian

    def compute_shape_functions(
        self,
        gaussian_nodes: torch.Tensor,
        inv_map_jacobian: torch.Tensor,
        polynomial_order: int,
    ):
        """
        Computes the barycentric coordinates, shape function values, and their gradients at given Gaussian nodes.
        Args:
            gaussian_nodes (torch.Tensor): Tensor containing the coordinates of Gaussian quadrature nodes in the reference element.
            inv_map_jacobian (torch.Tensor): Tensor representing the inverse of the mapping Jacobian for the element.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - bar_coords: Barycentric coordinates of the Gaussian nodes.
                - v: Values of the shape functions evaluated at the barycentric coordinates.
                - v_grad: Gradients of the shape functions with respect to the physical coordinates.
        """

        bar_coords = self.compute_barycentric_coordinates(gaussian_nodes)

        v, v_grad = self.shape_functions_value_and_grad(
            bar_coords, inv_map_jacobian, polynomial_order
        )

        return bar_coords, v, v_grad

    @staticmethod
    def compute_inverse_map(
        first_node: torch.Tensor,
        integration_points: torch.Tensor,
        inv_map_jacobian: torch.Tensor,
    ):
        """
        Computes the inverse mapping of integration points from the reference element to the physical element.
        Args:
            first_node (torch.Tensor): The coordinates of the first node of the element, used as the origin for mapping.
            integration_points (torch.Tensor): The integration points in the reference element.
            inv_map_jacobian (torch.Tensor): The inverse of the mapping Jacobian matrix.
        Returns:
            torch.Tensor: The mapped integration points in the physical element coordinates.
        """
        integration_points = torch.concat(integration_points, dim=-1)

        inv_map = (integration_points - first_node) @ inv_map_jacobian.mT

        return inv_map

    @abstractmethod
    @property
    def reference_element_area(self) -> float:
        """
        Calculates and returns the area of the reference element.

        This method should be implemented by subclasses to provide the area
        of the reference element used in finite element computations.

        Returns:
            float: The area of the reference element.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    @property
    def barycentric_grad(self) -> float:
        """
        Calculates the gradient of the barycentric coordinates for a finite element.

        Returns:
            float: The computed gradient of the barycentric coordinates.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_gauss_values(self, integration_order: int):
        """
        Computes the values of the Gauss quadrature points and weights for a given integration order.

        Args:
            integration_order (int): The order of the Gauss quadrature to be used for numerical integration.

        Raises:
            NotImplementedError: This method should be implemented by subclasses to provide specific quadrature rules.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def shape_functions_value_and_grad(
        self,
        bar_coords: torch.Tensor,
        inv_map_jacobian: torch.Tensor,
        polynomial_order: int,
    ):
        """
        Computes the values and gradients of shape functions at given barycentric coordinates.

        Args:
            bar_coords (torch.Tensor): Barycentric coordinates at which to evaluate the shape functions.
            inv_map_jacobian (torch.Tensor): Inverse of the mapping Jacobian, used to transform gradients from reference to physical coordinates.
            polynomial_order (int): The polynomial order of the shape functions to be evaluated.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - Shape function values at the specified barycentric coordinates.
            - Gradients of the shape functions with respect to the physical coordinates.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_det_and_inv_map(self, map_jacobian: torch.Tensor):
        """
        Computes the determinant and inverse of the given Jacobian matrix.

        Parameters:
            map_jacobian (numpy.ndarray): The Jacobian matrix of the mapping, typically of shape (n, n),
                where n is the dimension of the space.

        Returns:
            tuple:
                det (float): The determinant of the Jacobian matrix.
                inv (numpy.ndarray): The inverse of the Jacobian matrix.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_barycentric_coordinates(self, x: torch.Tensor):
        """
        Computes the barycentric coordinates of the given points with respect to the elements in the mesh.

        Args:
            x (torch.Tensor): A tensor containing the coordinates of the points for which to compute barycentric coordinates.
                The shape and interpretation of this tensor depend on the mesh and element type.

        Returns:
            torch.Tensor: A tensor containing the barycentric coordinates of the input points with respect to the mesh elements.

        Raises:
            NotImplementedError: This method should be implemented in a subclass.
        """
        raise NotImplementedError


class ElementLine(AbstractElement):
    """
    ElementLine is a finite element class representing a 1D line element for use in FEM computations.
    Args:
        polynomial_order (int): The order of the polynomial basis functions used in the element.
        integration_order (int): The order of the Gaussian quadrature integration.
    Attributes:
        compute_barycentric_coordinates (Callable): Lambda function to compute barycentric coordinates for a given x.
        barycentric_grad (torch.Tensor): Gradient of barycentric coordinates with respect to the reference element.
        reference_element_area (float): The length of the reference element (default is 2.0).
    Methods:
        compute_gauss_values():
            Computes the Gaussian quadrature nodes and weights for the specified integration order.
            Returns:
                Tuple[torch.Tensor, torch.Tensor]: Gaussian nodes and weights.
        shape_functions_value_and_grad(bar_coords: torch.Tensor, inv_map_jacobian: torch.Tensor):
            Computes the values and gradients of the shape functions at given barycentric coordinates.
            Args:
                bar_coords (torch.Tensor): Barycentric coordinates at which to evaluate the shape functions.
                inv_map_jacobian (torch.Tensor): Inverse of the mapping Jacobian.
            Returns:
                Tuple[torch.Tensor, torch.Tensor]: Shape function values and their gradients.
        compute_det_and_inv_map(map_jacobian):
            Computes the determinant and inverse of the mapping Jacobian.
            Args:
                map_jacobian (torch.Tensor): The mapping Jacobian tensor.
            Returns:
                Tuple[torch.Tensor, torch.Tensor]: Determinant and inverse of the mapping Jacobian.
    """

    def __init__(self, polynomial_order: int, integration_order: int):

        self.compute_barycentric_coordinates = lambda x: torch.concat(
            [0.5 * (1.0 - x), 0.5 * (1.0 + x)], dim=-1
        )

        self.barycentric_grad = torch.tensor([[-0.5], [0.5]])

        self.reference_element_area = 2.0

        super().__init__(polynomial_order, integration_order)

    def compute_gauss_values(self, integration_order: int):
        """
        Computes Gaussian quadrature nodes and weights for a given integration order.
        Parameters:
            integration_order (int): The order of the Gaussian quadrature.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - gaussian_nodes: Tensor containing the positions of the Gaussian nodes.
                - gaussian_weights: Tensor containing the weights for each node, with an added batch dimension.
        Raises:
            ValueError: If the specified integration order is not supported.
        """
        if integration_order == 2:

            nodes = 1.0 / torch.sqrt(torch.tensor(3.0))

            gaussian_nodes = torch.tensor([[-nodes], [nodes]])

            gaussian_weights = torch.tensor([[[0.5]], [[0.5]]])

        if integration_order == 3:

            nodes = torch.sqrt(torch.tensor(3 / 5))

            gaussian_nodes = torch.tensor([[0], [-nodes], [nodes]])

            gaussian_weights = torch.tensor([[[8 / 18]], [[5 / 18]], [[5 / 18]]])

        else:
            raise ValueError(f"Integration order {integration_order} is not supported. ")

        return gaussian_nodes, gaussian_weights.unsqueeze(0)

    def shape_functions_value_and_grad(
        self,
        bar_coords: torch.Tensor,
        inv_map_jacobian: torch.Tensor,
        polynomial_order: int,
    ):

        if polynomial_order == 1:

            v = bar_coords

            v_grad = self.barycentric_grad @ inv_map_jacobian
        else:
            raise ValueError(
                f"Polynomial order {polynomial_order} is not supported for ElementLine."
            )

        return v, v_grad

    def compute_det_and_inv_map(self, map_jacobian):

        det_map_jacobian = torch.linalg.norm(map_jacobian, dim=-2, keepdim=True)

        inv_map_jacobian = 1.0 / det_map_jacobian

        return det_map_jacobian.unsqueeze(-1), inv_map_jacobian


class ElementTri(AbstractElement):
    """
    ElementTri(polynomial_order: int, integration_order: int)
    A finite element class for triangular elements supporting linear (P1) and quadratic (P2) shape functions.
    Implements barycentric coordinate computations, shape function evaluations, and Gaussian quadrature rules
    for integration over the reference triangle.
    Args:
        polynomial_order (int): The polynomial order of the shape functions (1 for linear, 2 for quadratic).
        integration_order (int): The order of Gaussian quadrature for numerical integration.
    Attributes:
        compute_barycentric_coordinates (Callable): Function to compute barycentric coordinates from reference coordinates.
        barycentric_grad (torch.Tensor): Gradients of barycentric coordinates with respect to reference coordinates.
        reference_element_area (float): Area of the reference triangle (default is 0.5).
    Methods:
        shape_functions_value_and_grad(bar_coords: torch.Tensor, inv_map_jacobian: torch.Tensor)
            Computes the values and gradients of shape functions at given barycentric coordinates.
            Args:
                bar_coords (torch.Tensor): Barycentric coordinates at which to evaluate shape functions.
                inv_map_jacobian (torch.Tensor): Inverse of the mapping Jacobian for coordinate transformation.
            Returns:
                Tuple[torch.Tensor, torch.Tensor]: Shape function values and their gradients.
        compute_gauss_values()
            Returns Gaussian quadrature nodes and weights for the reference triangle based on the integration order.
            Returns:
                Tuple[torch.Tensor, torch.Tensor]: Gaussian nodes and weights.
        compute_det_and_inv_map(map_jacobian: torch.Tensor)
            Static method to compute the determinant and inverse of the mapping Jacobian.
            Args:
                map_jacobian (torch.Tensor): The Jacobian matrix of the mapping.
            Returns:
                Tuple[torch.Tensor, torch.Tensor]: Determinant and inverse of the Jacobian.
    """

    def __init__(self, polynomial_order: int, integration_order: int):

        self.compute_barycentric_coordinates = lambda x: torch.stack(
            [1.0 - x[..., [0]] - x[..., [1]], x[..., [0]], x[..., [1]]], dim=-2
        )

        self.barycentric_grad = torch.tensor([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])

        self.reference_element_area = 0.5

        super().__init__(polynomial_order, integration_order)

    def shape_functions_value_and_grad(
        self,
        bar_coords: torch.Tensor,
        inv_map_jacobian: torch.Tensor,
        polynomial_order: int,
    ):

        if polynomial_order == 1:

            v = bar_coords

            v_grad = self.barycentric_grad @ inv_map_jacobian

            return v, v_grad

        lambda_1, lambda_2, lambda_3 = torch.split(bar_coords, 1, dim=-2)

        grad_lambda_1, grad_lambda_2, grad_lambda_3 = torch.split(self.barycentric_grad, 1, dim=-2)

        if self.polynomial_order == 2:

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
            raise ValueError(
                f"Polynomial order {self.polynomial_order} is not supported for ElementTri."
            )

        return v, v_grad

    def compute_gauss_values(self, integration_order: int):

        if self.integration_order == 1:

            gaussian_nodes = torch.tensor([[1 / 3, 1 / 3]])

            gaussian_weights = torch.tensor([[[1.0]]])

        if self.integration_order == 2:

            gaussian_nodes = torch.tensor([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])

            gaussian_weights = torch.tensor([[[1 / 3]], [[1 / 3]], [[1 / 3]]])

        if self.integration_order == 3:

            gaussian_nodes = torch.tensor([[1 / 3, 1 / 3], [0.6, 0.2], [0.2, 0.6], [0.2, 0.2]])

            gaussian_weights = torch.tensor([[[-9 / 16]], [[25 / 48]], [[25 / 48]], [[25 / 48]]])

        if self.integration_order == 4:

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
            raise ValueError(
                f"Integration order {integration_order} is not supported for ElementTri."
            )

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
    """
    AbstractBasis is an abstract base class for finite element basis functions.
    This class provides the core interface and functionality for defining basis functions
    over a finite element mesh, including integration of functionals, bilinear forms, and
    linear forms, as well as the computation of degrees of freedom (DOFs) and basis parameters.
    Attributes:
        elements (AbstractElement): The finite element type associated with the basis.
        mesh (AbstractMesh): The mesh on which the basis is defined.
        v (torch.Tensor): Basis function values at integration points.
        v_grad (torch.Tensor): Gradients of basis functions at integration points.
        integration_points (torch.Tensor): Integration points for numerical quadrature.
        dx (torch.Tensor): Integration weights for each element.
        coords4global_dofs (torch.Tensor): Coordinates of global degrees of freedom.
        global_dofs4elements (torch.Tensor): Mapping from elements to global DOFs.
        nodes4boundary_dofs (torch.Tensor): Mapping from boundary nodes to DOFs.
        coords4elements (torch.Tensor): Coordinates of DOFs for each element.
        basis_parameters (dict): Additional parameters required for assembling forms.
    Methods:
        integrate_functional(function, *args, **kwargs):
            Integrates a given functional over the domain using the basis functions.
        integrate_bilineal_form(function, *args, **kwargs):
            Assembles the global matrix for a bilinear form by integrating the provided function.
        integrate_lineal_form(function, *args, **kwargs):
            Assembles the global vector for a linear form by integrating the provided function.
        compute_dofs(coords4nodes, nodes4elements, nodes4boundary, mesh_parameters, edges_parameters, polynomial_order):
            Abstract method to compute the degrees of freedom for the basis.
        compute_basis_parameters(coords4global_dofs, global_dofs4elements, nodes4boundary_dofs):
            Abstract method to compute additional parameters required for basis assembly.
    Note:
        This class is intended to be subclassed, with concrete implementations provided
        for the abstract methods `compute_dofs` and `compute_basis_parameters`.
    """

    def __init__(self, mesh: AbstractMesh, elements: AbstractElement):

        self.elements = elements
        self.mesh = mesh

        (
            self.v,
            self.v_grad,
            self.integration_points,
            self.dx,
        ) = elements.compute_integral_values(mesh.coords4elements)

        self.coords4global_dofs, self.global_dofs4elements, self.nodes4boundary_dofs = (
            self.compute_dofs(
                mesh.coords4nodes,
                mesh.nodes4elements,
                mesh.nodes4boundary,
                mesh.mesh_parameters,
                mesh.edges_parameters,
                elements.polynomial_order,
            )
        )

        self.coords4elements = self.coords4global_dofs[self.global_dofs4elements]

        self.basis_parameters = self.compute_basis_parameters(
            self.coords4global_dofs, self.global_dofs4elements, self.nodes4boundary_dofs
        )

    def integrate_functional(self, function, *args, **kwargs):
        """
        Integrates a given functional over the domain represented by the object.
        Parameters:
            function (callable): A function that takes the current object as its first argument,
                followed by any additional positional and keyword arguments, and returns a tensor
                representing the functional to be integrated.
            *args: Additional positional arguments to pass to the function.
            **kwargs: Additional keyword arguments to pass to the function.
        Returns:
            torch.Tensor: The result of integrating the functional over the domain.
        """

        integral_value = (function(self, *args, **kwargs) * self.dx).sum(-3).sum(-2)

        return integral_value

    def integrate_bilineal_form(self, function, *args, **kwargs):
        """
        Assembles the global matrix for a bilinear form by integrating a user-provided function over all elements.
        Args:
            function (callable): A function that computes the local bilinear form matrix for each element.
                It should accept the current object (self) and any additional arguments.
            *args: Additional positional arguments to pass to the function.
            **kwargs: Additional keyword arguments to pass to the function.
        Returns:
            torch.Tensor: The assembled global matrix representing the bilinear form.
        Notes:
            - The function is expected to return a tensor representing the local matrix for each element.
            - The integration is performed using the element-wise differential `self.dx`.
            - The local matrices are accumulated into the global matrix using the indices specified in
              `self.basis_parameters["bilinear_form_idx"]`.
        """

        global_matrix = torch.zeros(self.basis_parameters["bilinear_form_shape"])

        local_matrix = (function(self, *args, **kwargs) * self.dx).sum(-3)

        global_matrix.index_put_(
            self.basis_parameters["bilinear_form_idx"],
            local_matrix.reshape(-1),
            accumulate=True,
        )

        return global_matrix

    def integrate_lineal_form(self, function, *args, **kwargs):
        """
        Integrates a given linear form (function) over the domain using the current basis and quadrature.
        Args:
            function (callable): A function representing the linear form to be integrated. It should accept the current object (self) and any additional arguments.
            *args: Additional positional arguments to pass to the function.
            **kwargs: Additional keyword arguments to pass to the function.
        Returns:
            torch.Tensor: The result of the integration, shaped according to 'linear_form_shape' in 'basis_parameters'.
        Notes:
            - The integration is performed using the quadrature weights (self.dx).
            - The result is accumulated at indices specified by 'linear_form_idx' in 'basis_parameters'.
        """

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
        coords4nodes,
        nodes4elements,
        nodes4boundary,
        mesh_parameters,
        edges_parameters,
        polynomial_order,
    ):
        """
        Computes the degrees of freedom (DOFs) for the finite element mesh.

        Parameters:
            coords4nodes (torch.Tensor or np.ndarray): Array of node coordinates, shape (num_nodes, dim).
            nodes4elements (torch.Tensor or np.ndarray): Connectivity array mapping elements to their nodes, shape (num_elements, nodes_per_element).
            nodes4boundary (torch.Tensor or np.ndarray): Array of boundary node indices or boundary condition information.
            mesh_parameters (dict): Dictionary containing mesh-related parameters (e.g., mesh size, element type).
            edges_parameters (dict): Dictionary containing edge-related parameters (e.g., edge connectivity, boundary edges).
            polynomial_order (int): The polynomial order of the finite element basis functions.

        Raises:
            NotImplementedError: This method must be implemented in a subclass.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def compute_basis_parameters(
        self, coords4global_dofs, global_dofs4elements, nodes4boundary_dofs
    ):
        """
        Computes the parameters required for the basis functions in the finite element method.

        Parameters:
            coords4global_dofs (np.ndarray): Array of coordinates for each global degree of freedom (DOF).
            global_dofs4elements (np.ndarray): Mapping from elements to their associated global DOFs.
            nodes4boundary_dofs (np.ndarray): Array indicating which nodes correspond to boundary DOFs.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.

        Returns:
            None
        """
        raise NotImplementedError


class Basis(AbstractBasis):
    """
    Basis class for finite element method (FEM) computations.
    Inherits from:
        AbstractBasis
    Args:
        mesh (AbstractMesh): The mesh object describing the computational domain.
        elements (AbstractElement): The element object describing the reference element and shape functions.
    Methods:
        compute_dofs(
            Computes the degrees of freedom (DOFs) for the basis functions, depending on the polynomial order.
            For polynomial_order == 1, uses node-based DOFs.
            For polynomial_order == 2, adds edge-based DOFs.
            Returns:
                coords4global_dofs (Tensor): Coordinates of all global DOFs.
                global_dofs4elements (Tensor): Mapping from elements to global DOFs.
                nodes4boundary_dofs (Tensor): Indices of boundary DOFs.
        compute_basis_parameters(
            coords4global_dofs,
            global_dofs4elements,
            nodes4boundary_dofs
            Computes and returns various parameters required for assembling FEM matrices and vectors.
            Returns:
                basis_parameters (dict): Dictionary containing shapes and indices for bilinear and linear forms,
                                        inner DOFs, and total number of DOFs.
        reduce(tensor):
            Reduces a tensor to the subspace of inner (non-boundary) DOFs.
            Args:
                tensor (Tensor): The tensor to be reduced.
            Returns:
                Tensor: The reduced tensor.
        interpolate(basis, tensor=None):
            Interpolates a function or tensor onto the current basis.
            Args:
                basis (Basis or InteriorFacetBasis): The basis to interpolate from.
                tensor (Tensor, optional): The tensor to interpolate. If None, returns interpolation functions.
            Returns:
                If tensor is provided:
                    interpolation (Tensor): Interpolated values.
                    interpolation_grad (Tensor): Gradients of the interpolated values.
                If tensor is None:
                    interpolator (callable): Function to interpolate a given function.
                    interpolator_grad (callable): Function to interpolate the gradient of a given function.
    """

    def __init__(self, mesh: AbstractMesh, elements: AbstractElement):

        super().__init__(mesh, elements)

    def compute_dofs(
        self,
        coords4nodes,
        nodes4elements,
        nodes4boundary,
        mesh_parameters,
        edges_parameters,
        polynomial_order,
    ):

        if polynomial_order == 1:

            coords4global_dofs = coords4nodes
            global_dofs4elements = nodes4elements
            nodes4boundary_dofs = nodes4boundary

        if polynomial_order == 2:

            new_coords4dofs = (coords4nodes[edges_parameters["nodes4unique_edges"]]).mean(-2)
            new_nodes4dofs = (
                edges_parameters["edges_idx"].reshape(mesh_parameters["nb_simplex"], 3)
                + mesh_parameters["nb_nodes"]
            )
            new_nodes4boundary_dofs = (
                edges_parameters["nodes_idx4boundary_edges"] + mesh_parameters["nb_nodes"]
            )

            coords4global_dofs = torch.cat([coords4nodes, new_coords4dofs], dim=-2)
            global_dofs4elements = torch.cat([nodes4elements, new_nodes4dofs], dim=-1)
            nodes4boundary_dofs = torch.cat([nodes4boundary, new_nodes4boundary_dofs], dim=-1)
        else:
            raise ValueError(f"Polynomial order {polynomial_order} is not supported for Basis.")

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
            "inner_dofs": inner_dofs,
            "nb_dofs": nb_global_dofs,
        }

        return basis_parameters

    def reduce(self, tensor):
        """
        Reduces the input tensor by selecting elements corresponding to the inner degrees of freedom (DOFs).

        If the tensor has more than one column (i.e., `tensor.shape[-1] != 1`), it returns the submatrix formed by selecting
        rows and columns indexed by `self.basis_parameters["inner_dofs"]`. Otherwise, it returns the elements of the tensor
        at the specified indices.

        Args:
            tensor (torch.Tensor): The input tensor to be reduced.

        Returns:
            torch.Tensor: The reduced tensor containing only the inner DOFs.
        """
        idx = self.basis_parameters["inner_dofs"]
        if tensor.shape[-1] != 1:
            return tensor[idx, :][:, idx]
        else:
            return tensor[idx]

    def interpolate(self, basis, tensor=None):
        """
        Interpolates a function or tensor over a finite element basis.
        Depending on the type of `basis` provided, this method computes the interpolation and its gradient
        at the integration points associated with the basis. If a tensor is provided, it performs the interpolation
        directly; otherwise, it returns callable interpolator functions.
        Parameters:
            basis: The finite element basis to interpolate onto. Can be `self` or an instance of `InteriorFacetBasis`.
            tensor (optional): A tensor containing degrees of freedom (DoFs) values to interpolate. If not provided,
                the method returns interpolator functions that accept a function to evaluate at the nodes.
        Returns:
            If `tensor` is provided:
                interpolation: The interpolated values at the integration points.
                interpolation_grad: The gradients of the interpolated values at the integration points.
            If `tensor` is None:
                interpolator: A function that takes a callable and returns its interpolation.
                interpolator_grad: A function that takes a callable and returns its interpolated gradient.
        Raises:
            ValueError: If the provided `basis` type is not supported for interpolation.
        """

        if basis == self:
            dofs_idx = self.global_dofs4elements.unsqueeze(-2)

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

        if basis.__class__ == InteriorFacetBasis:

            elements_mask = basis.mesh.edges_parameters["elements4inner_edges"]

            dofs_idx = basis.mesh.nodes4elements[elements_mask].unsqueeze(-2)

            coords4elements_first_node = self.coords4elements[..., [0], :][elements_mask].unsqueeze(
                -3
            )

            inv_map_jacobian = self.elements.inv_map_jacobian[elements_mask]

            integration_points = torch.split(
                torch.cat(basis.integration_points, dim=-1).unsqueeze(-4), 1, dim=-1
            )

            new_integrations_points = self.elements.compute_inverse_map(
                coords4elements_first_node, integration_points, inv_map_jacobian
            )

            _, v, v_grad = self.elements.compute_shape_functions(
                new_integrations_points.squeeze(-3), inv_map_jacobian
            )

        else:
            raise ValueError(
                f"Basis type {basis.__class__.__name__} is not supported for interpolation."
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
    """
    Basis class for interior facets (edges) in a finite element mesh.
    This class constructs basis functions and associated data structures for integration
    over interior facets (edges) of a mesh, typically used in discontinuous Galerkin or
    hybridized finite element methods.
    Args:
        mesh (AbstractMesh): The mesh object containing node coordinates and edge information.
        elements (ElementLine): The element object defining the polynomial order and integration rules.
    Attributes:
        elements (ElementLine): The finite element definition for the facets.
        mesh (AbstractMesh): The mesh associated with the basis.
        v (torch.Tensor): Basis function values at integration points.
        v_grad (torch.Tensor): Gradients of basis functions at integration points.
        integration_points (torch.Tensor): Integration points on the facets.
        dx (torch.Tensor): Integration weights for the facets.
        coords4global_dofs (torch.Tensor): Coordinates of global degrees of freedom.
        global_dofs4elements (torch.Tensor): Mapping from elements to global degrees of freedom.
        nodes4boundary_dofs (torch.Tensor): Indices of boundary degrees of freedom.
        coords4elements (torch.Tensor): Coordinates of degrees of freedom for each element.
        basis_parameters (dict): Dictionary containing parameters for assembling bilinear and linear forms.
    Methods:
        compute_dofs(coords4nodes, nodes4elements, nodes4boundary, mesh_parameters, edges_parameters, polynomial_order):
            Computes the global degrees of freedom, their coordinates, and identifies boundary dofs.
        compute_basis_parameters(coords4global_dofs, global_dofs4elements, nodes4boundary_dofs):
            Computes parameters required for assembling bilinear and linear forms, including index arrays and inner dofs.
    Raises:
        ValueError: If the polynomial order is not supported (currently only order 1 is implemented).
    """

    def __init__(self, mesh: AbstractMesh, elements: ElementLine):

        self.elements = elements
        self.mesh = mesh

        nodes4elements = mesh.edges_parameters["nodes4inner_edges"]
        coords4nodes = mesh.coords4nodes
        coords4elements = mesh.coords4nodes[nodes4elements]

        self.v, self.v_grad, self.integration_points, self.dx = elements.compute_integral_values(
            coords4elements
        )

        self.coords4global_dofs, self.global_dofs4elements, self.nodes4boundary_dofs = (
            self.compute_dofs(
                coords4nodes,
                nodes4elements,
                mesh.nodes4boundary,
                mesh.mesh_parameters,
                mesh.edges_parameters,
                elements.polynomial_order,
            )
        )

        self.coords4elements = self.coords4global_dofs[self.global_dofs4elements]

        self.basis_parameters = self.compute_basis_parameters(
            self.coords4global_dofs, self.global_dofs4elements, self.nodes4boundary_dofs
        )

    def compute_dofs(
        self,
        coords4nodes,
        nodes4elements,
        nodes4boundary,
        mesh_parameters,
        edges_parameters,
        polynomial_order,
    ):

        if polynomial_order == 1:
            coords4global_dofs = coords4nodes
            global_dofs4elements = nodes4elements
            nodes4boundary_dofs = nodes4boundary

        else:
            raise ValueError(
                f"Polynomial order {polynomial_order} is not supported for InteriorFacetBasis."
            )

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
