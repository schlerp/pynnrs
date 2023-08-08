class Layer:
    weights: list[list[float]]
    biases: list[float]
    outputs: list[float]
    deltas: list[float]

    def __init__(self, width: int, input_size: int): ...
    def forward(self, x: list[float]) -> list[float]: ...

class Network:
    layers: list[Layers]
    learning_rate: float

    def __init__(self, layer_sizes: list[int]): ...
    def forward(self, x: list[float]) -> list[float]: ...

def relu(x: list[float]) -> list[float]: ...
def train(
    network: Network, x: list[float], y: list[float], epochs: int
) -> list[float]: ...
