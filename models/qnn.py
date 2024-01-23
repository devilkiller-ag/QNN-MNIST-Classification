import torch

from classiq.execution import execute_qnn
from classiq.synthesis import SerializedQuantumProgram
from classiq.applications.qnn import QLayer
from classiq.applications.qnn.types import (
    MultipleArguments,
    SavedResult,
    ResultsCollection,
)

def execute_fn(quantum_program: SerializedQuantumProgram, arguments: MultipleArguments) -> ResultsCollection:
    return execute_qnn(quantum_program, arguments)

def post_process_fn(result: SavedResult) -> torch.Tensor:
    counts: dict = result.value.counts
    
    # Calculate logits from counts
    logits: float = torch.zeros(16)
    for key, value in counts.items():
        logits[int(key, 2)] = value
    
    # Trim the logits from length 16 to length 10 since we have only 10 labels
    trimmed_logits = logits[:10]
    
    # Calculate prediction probabilities from logits by normalizing it
    pred_probs = torch.nn.functional.normalize(trimmed_logits, dim=0)
    
    # Convert the prediction probabilities into prediction labels
    # pred_labels = torch.argmax(pred_probs)
    
    ### WRITE COUNTS, OUTPUT LOGITS, PRED PROBS, PRED LABELS to a file
    # output_file = open("post_process_output.txt", "a")
    # print("----------------------------------------------------------------------------------------------------------------------------------------------", file=output_file)
    # print(f"COUNTS:: \n {counts} \n", file=output_file)
    # print(f"LOGITS:: \n {logits} \n", file=output_file)
    # print(f"TRIMMED LOGITS:: \n {trimmed_logits} \n", file=output_file)
    # print(f"PREDICTION PROBABILITIES:: \n {pred_probs} \n", file=output_file)
    # print(f"PREDICTION LABELS:: \n {pred_labels} \n", file=output_file)
    # output_file.close()
    
    return pred_probs.clone().detach()

def post_process_bin_fn(result: SavedResult) -> torch.Tensor:
    counts: dict = result.value.counts
    
    # Calculate logits from counts
    logits: float = torch.zeros(16)
    for key, value in counts.items():
        logits[int(key, 2)] = value
    
    # Trim the logits from length 16 to length 2 since we have only 2 labels
    trimmed_logits = logits[:2]
    
    # Calculate prediction probabilities from logits by normalizing it
    pred_probs = torch.nn.functional.normalize(trimmed_logits, dim=0)
    
    # Convert the prediction probabilities into prediction labels
    # pred_labels = torch.argmax(pred_probs)
    
    ### WRITE COUNTS, OUTPUT LOGITS, PRED PROBS, PRED LABELS to a file
    # output_file = open("post_process_output.txt", "a")
    # print("----------------------------------------------------------------------------------------------------------------------------------------------", file=output_file)
    # print(f"COUNTS:: \n {counts} \n", file=output_file)
    # print(f"LOGITS:: \n {logits} \n", file=output_file)
    # print(f"TRIMMED LOGITS:: \n {trimmed_logits} \n", file=output_file)
    # print(f"PREDICTION PROBABILITIES:: \n {pred_probs} \n", file=output_file)
    # print(f"PREDICTION LABELS:: \n {pred_labels} \n", file=output_file)
    # output_file.close()
    
    return pred_probs.clone().detach()

class QNN(torch.nn.Module):
    def __init__(self, quantum_program, execute, post_process, *args, **kwargs) -> None:
        super().__init__()
        self.qlayer = QLayer(
            quantum_program,
            execute=execute,
            post_process=post_process,
            *args,
            **kwargs
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.qlayer(x)
        return x