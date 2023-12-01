from typing import List, NoReturn, Union, Dict, Tuple
import numpy as np

def not_gate(x) -> np.ndarray:
    return np.logical_not(*x.T).reshape(-1, 1)

def and_gate(x) -> np.ndarray:
    return np.logical_and(*x.T).reshape(-1, 1)

def or_gate(x) -> np.ndarray:
    return np.logical_or(*x.T).reshape(-1, 1)

def nand_gate(x) -> np.ndarray:
    return not_gate(and_gate(x))

def nor_gate(x) -> np.ndarray:
    return not_gate(or_gate(x))

def xor_gate(x) -> np.ndarray:
    return np.logical_xor(*x.T).reshape(-1, 1)

GATES = {
    "not": not_gate,
    "and": and_gate,
    "or": or_gate,
    "nand": nand_gate,
    "nor": nor_gate,
    "xor": xor_gate,
}

class LogicGate:
    def __init__(self, indices: Union[List[int], Tuple[int, ...]] = (0, 0)):

        self.indices = indices
        self.operator = np.random.choice(list(GATES.values()))

        self.input = None
        self.output = None

    def compute(self, inputs: np.ndarray) -> bool | np.ndarray[bool]:

        self.input = inputs[:, self.indices]
        self.output = self.operator(self.input)
        # print(self.operator)
        # print(self.input.shape, self.output.shape)
        # print(self.output)
        # print(f'\n\n')
        return self.output

    def set_indices(self, indices: Union[List[int], Tuple[int, ...]]) -> NoReturn:

        self.indices = indices

    def set_operator(self, operator: str) -> NoReturn:

        # if not operator or operator not in GATES:
        #     raise ValueError("Operator must be valid")
        self.operator = GATES[operator]

    # def __str__(self):
    #     return (
    #         f"{self.__class__.__name__}("
    #         f"operator:{self.operator.__name__}, "
    #         f"idx:{self.indices})"
    #     )
    #
    # def __repr__(self):
    #     return self.__str__()


class Layer:
    def __init__(self, n_inputs, n_gates: Union[int, List[LogicGate], Dict[str, int]]):
        self.gates = []
        self.n_inputs = n_inputs

        self._initialize_gates(n_gates)
        # print([gate.operator for gate in self.gates])
        self.n_outputs = len(self.gates)

    def _initialize_gates(
        self, n_outputs: Union[int, List[LogicGate], Dict[str, int]]
    ) -> NoReturn:

        if isinstance(n_outputs, int):
            for _ in range(n_outputs):
                self.gates.append(self._initialize_gate())
        # elif isinstance(n_outputs, list):
        #     for operator in n_outputs:
        #         self.gates.append(self._initialize_gate(operator))
        # elif isinstance(n_outputs, dict):
        #     for operator, n in n_outputs.items():
        #         for _ in range(n):
        #             self.gates.append(self._initialize_gate(operator))
        # else:
        #     raise TypeError("n_outputs must be int, list or dict")

    def _initialize_gate(self) -> LogicGate:
        idx = np.random.choice(self.n_inputs, size=2, replace=False)
        return LogicGate(idx)

    def forward(self, x: np.ndarray) -> np.ndarray:
        samples, length = x.shape

        if length != self.n_inputs:
            raise ValueError(
                f"Input size must be equal to input size of layer. Expected {self.n_inputs}, got {length}"
            )

        out = np.zeros((samples, self.n_outputs), dtype=x.dtype)

        for j, gate in enumerate(self.gates):
            out[:, j] = gate.compute(x).reshape(-1)

        return out

    # def __str__(self):
    #     output_string = (
    #         f"Layer(n_inputs:{self.n_inputs}, n_outputs:{self.n_outputs}, \n\t gates:"
    #     )
    #     for gate in self.gates:
    #         output_string += f"\n\t\t{gate}"
    #     output_string += "\n)"
    #     return output_string
    #
    # def __repr__(self):
    #     return self.__str__()


class LogicGateNetwork:
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []

    def add_layer(
        self, n_outputs: Union[int, List[LogicGate], Dict[str, int]]) -> NoReturn:
        if len(self.layers) == 0:
            self.layers.append(Layer(self.input_size, n_outputs))
        else:
            self.layers.append(Layer(self.layers[-1].n_outputs, n_outputs))

    # def _check_prerequisites(self, x) -> NoReturn:
    #     if x.shape[1] != self.input_size:
    #         raise ValueError(f"Input size must be {self.input_size}")
    #
    #     if len(self.layers) == 0:
    #         raise ValueError("No layers added to network")
    #
    #     if self.layers[-1].n_outputs != self.output_size:
    #         raise ValueError(f"Last layer output size must match {self.output_size}")

    def predict(self, x: np.ndarray) -> np.ndarray:

        # self._check_prerequisites(x)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    # def __str__(self):
    #     out_string = (
    #         self.__class__.__name__
    #         + f"(input_size:{self.input_size}, output_size:{self.output_size}"
    #         + ", layers:"
    #     )
    #
    #     for layer in self.layers:
    #         out_string += "\n\t" + "\n\t".join(str(layer).split("\n"))
    #
    #     out_string += "\n)"
    #
    #     return out_string
    #
    # def __repr__(self):
    #     return self.__str__()


logic_gate_network = LogicGateNetwork(input_size=20, output_size=5)

logic_gate_network.add_layer(n_outputs=15)
logic_gate_network.add_layer(n_outputs=10)
logic_gate_network.add_layer(n_outputs=5)

random_data = (np.random.random((10000, 20)) > 0.5).astype(np.byte)
output = logic_gate_network.predict(random_data)