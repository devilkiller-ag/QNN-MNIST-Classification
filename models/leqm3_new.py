"""_summary_
Linear Entanglement Quantum Model for MNIST Data Classification with three linear entanglement layers of RXX, RYY, and RZZ.
"""

from classiq import create_model, QFunc, QParam, QArray, QBit, Output, allocate, RX, RY, RZ, RZZ, RXX, RYY, CZ, repeat

@QFunc
def encoding(qubits: QArray[QBit]) -> None:
    """
    This function encodes the input data into the qubits. This input data is a 4x4 image pixel values
    converted into angle for rotation gates (RX, RY, RZ, RX) in form of a 16x1 vector.
    We encode 4 pixels per qubit.

    Args:
        q (QArray[QBit]): Array of four Qubits to encode the input data into.
    """
    def encode_qubit(index: QParam[int]):
        input_count = 0
        RX(theta=f"input_{input_count}", target=qubits[index])
        input_count += 1
        RY(theta=f"input_{input_count}", target=qubits[index])
        input_count += 1
        RZ(theta=f"input_{input_count}", target=qubits[index])
        input_count += 1
        RX(theta=f"input_{input_count}", target=qubits[index])
        input_count += 1

    repeat(
        count = qubits.len() - 1,
        iteration = encode_qubit,
    )


@QFunc
def mixing(qubits: QArray[QBit], rotation_layers=None) -> None:
    """
    This function performs the mixing operation on the qubits.
    This is done by applying a series of RZZ, RXX, RYY gates to form a
    ring connection.

    Args:
        qubits (QArray[QBit]): Array of four Qubits to apply the mixing operation on.
        rotation_layers: Array of multi-qubit rotation gates ["rzz", "rxx", "ryy"], for which you want linear layers. Defaults to ["rxx"].
    """

    if rotation_layers is None:
        rotation_layers = ["rxx"]

    rotation_gates = {
        "rxx": RXX,
        "ryy": RYY,
        "rzz": RZZ
    }

    weight_count = 0

    for rot_gates in rotation_layers:
      repeat(
          count = qubits.len() - 1,
          iteration = lambda index: rotation_gates[rot_gates](theta=f"weight_{weight_count}", target=qubits[index : index + 2]),
      )

@QFunc
def cz_block(qubits: QArray[QBit]) -> None:
    """
    This function applies CZ gates between each qubit.

    Args:
        q (QArray[QBit]): Array of four Qubits to apply the entanglement operation on.
    """
    repeat(
        count = qubits.len() - 1,
        iteration = lambda index: CZ(control=qubits[index], target=qubits[index + 1])
    )


@QFunc
def main(res: Output[QArray[QBit]]) -> None:
    """
    This is the main function from which model will be created.
    It calls the other functions to perform the encoding, mixing and entanglement.

    Args:
        res (Output[QArray[QBit]]): Output QArray of QBits from which the model will be created.
    """
    allocate(4, res)
    encoding(qubits=res)
    mixing(qubits=res)
    cz_block(qubits=res)

def linear_entanglement_r3_quantum_model():
    model = create_model(main)
    return model