import torch


class Neural_Network(torch.nn.Module):
    """
    Neural Network class for constructing a deep neural network.

    Parameters:
    - input_dimension (int): Input dimension of the network.
    - output_dimension (int): Output dimension of the network.
    - deep_layers (int): Number of hidden layers.
    - hidden_layers_dimension (int): Dimension of each hidden layer.
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        deep_layers: int,
        hidden_layers_dimension: int,
    ):

        super().__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.layer_in = torch.nn.Linear(input_dimension, hidden_layers_dimension)
        self.layer_out = torch.nn.Linear(hidden_layers_dimension, output_dimension)
        self.middle_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(hidden_layers_dimension, hidden_layers_dimension)
                for _ in range(deep_layers)
            ]
        )
        self.activation_function = torch.nn.Tanh()

        torch.nn.init.xavier_uniform_(self.layer_in.weight)
        torch.nn.init.xavier_uniform_(self.layer_out.weight)
        for layer in self.middle_layers:
            torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Forward pass.

        Parameters:
        - x (torch.Tensor): values in the X axis.
        - y (torch.Tensor): values in the Y axis.

        Returns:
        - torch.Tensor: Output tensor after passing through the network.
        """
        Input = torch.cat([x, y], dim=-1)

        output = self.activation_function(self.layer_in(Input))

        for layer in self.middle_layers:
            output = self.activation_function(layer(output))

        output = self.layer_out(output)

        return output * (x - 1) * x * (y - 1) * y

        # return output


class Neural_Network_3D(torch.nn.Module):
    """
    Neural Network class for constructing a deep neural network.

    Parameters:
    - input_dimension (int): Input dimension of the network.
    - output_dimension (int): Output dimension of the network.
    - deep_layers (int): Number of hidden layers.
    - hidden_layers_dimension (int): Dimension of each hidden layer.
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        deep_layers: int,
        hidden_layers_dimension: int,
        activation_function=torch.nn.Tanh(),
    ):

        super().__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.layer_in = torch.nn.Linear(input_dimension, hidden_layers_dimension)
        self.layer_out = torch.nn.Linear(hidden_layers_dimension, output_dimension)
        self.middle_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(hidden_layers_dimension, hidden_layers_dimension)
                for _ in range(deep_layers)
            ]
        )
        self.activation_function = activation_function

        torch.nn.init.xavier_uniform_(self.layer_in.weight)
        torch.nn.init.xavier_uniform_(self.layer_out.weight)
        for layer in self.middle_layers:
            torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        """
        Forward pass.

        Parameters:
        - x (torch.Tensor): values in the X axis.
        - y (torch.Tensor): values in the Y axis.

        Returns:
        - torch.Tensor: Output tensor after passing through the network.
        """
        Input = torch.stack([x, y, z], dim=-1)

        output = self.activation_function(self.layer_in(Input))

        for layer in self.middle_layers:
            output = self.activation_function(layer(output))

        output = self.layer_out(output).squeeze(-1)

        return output * (x - 1) * (x + 1) * (y - 1) * (y + 0) * (z + 1) * (z - 1)
        # return output
