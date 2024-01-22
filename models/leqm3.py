"""_summary_
Linear Entanglement Quantum Model for MNIST Data Classification with three linear entanglement layers of RXX, RYY, and RZZ.
"""

from classiq import create_model, QFunc, QArray, QBit, Output, allocate, RX, RY, RZ, RZZ, RXX, RYY, CZ

@QFunc
def encoding(q: QArray[QBit]) -> None:
    """
    This function encodes the input data into the qubits. This input data is a 4x4 image pixel values 
    converted into angle for rotation gates (RX, RY, RZ, RX) in form of a 16x1 vector. 
    We encode 4 pixels per qubit.
    
    Args:
        q (QArray[QBit]): Array of four Qubits to encode the input data into.
    """
    RX(theta="input_0", target=q[0]) # Pixel 0 on Qubit 0
    RY(theta="input_1", target=q[0]) # Pixel 1 on Qubit 0
    RZ(theta="input_2", target=q[0]) # Pixel 2 on Qubit 0
    RX(theta="input_3", target=q[0]) # Pixel 3 on Qubit 0
    
    RX(theta="input_4", target=q[1]) # Pixel 4 on Qubit 1
    RY(theta="input_5", target=q[1]) # Pixel 5 on Qubit 1
    RZ(theta="input_6", target=q[1]) # Pixel 6 on Qubit 1
    RX(theta="input_7", target=q[1]) # Pixel 7 on Qubit 1
    
    RX(theta="input_8", target=q[2]) # Pixel 8 on Qubit 2
    RY(theta="input_9", target=q[2]) # Pixel 9 on Qubit 2
    RZ(theta="input_10", target=q[2]) # Pixel 10 on Qubit 2
    RX(theta="input_11", target=q[2]) # Pixel 11 on Qubit 2
    
    RX(theta="input_12", target=q[3]) # Pixel 12 on Qubit 3
    RY(theta="input_13", target=q[3]) # Pixel 13 on Qubit 3
    RZ(theta="input_14", target=q[3]) # Pixel 14 on Qubit 3
    RX(theta="input_15", target=q[3]) # Pixel 15 on Qubit 3

@QFunc
def mixing(q: QArray[QBit]) -> None:
    """
    This function performs the mixing operation on the qubits. 
    This is done by applying a series of RZZ, RXX, RYY gates to form a
    ring connection.
    
    Args:
        q (QArray[QBit]): Array of four Qubits to apply the mixing operation on.
    """
    RXX(theta="weight_0", target=q[0:2])
    RXX(theta="weight_1", target=q[1:3])
    RXX(theta="weight_2", target=q[2:4])

    # RZZ(theta="weight_3", target=q[0:2])
    # RZZ(theta="weight_4", target=q[1:3])
    # RZZ(theta="weight_5", target=q[2:4])
    
    # RYY(theta="weight_6", target=q[0:2])
    # RYY(theta="weight_7", target=q[1:3])
    # RYY(theta="weight_8", target=q[2:4])

@QFunc
def cz_block(q: QArray[QBit]) -> None:
    """
    This function applies CZ gates between each qubit.
    
    Args:
        q (QArray[QBit]): Array of four Qubits to apply the entanglement operation on.
    """
    CZ(control=q[0], target=q[1])
    CZ(control=q[1], target=q[2])
    CZ(control=q[2], target=q[3])

@QFunc
def main(res: Output[QArray[QBit]]) -> None:
    """
    This is the main function from which model will be created. 
    It calls the other functions to perform the encoding, mixing and entanglement.
    
    Args:
        res (Output[QArray[QBit]]): Output QArray of QBits from which the model will be created.
    """
    allocate(4, res)
    encoding(q=res)
    mixing(q=res)
    cz_block(q=res)

def linear_entanglement_r3_quantum_model():
    model = create_model(main)
    return model