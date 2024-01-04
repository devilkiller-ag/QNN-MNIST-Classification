import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def test(
    model: nn.Module,
    data_loader: DataLoader,
    atol=0,
    device: str = 'cpu',
) -> float:
    num_correct = 0
    total = 0
    
    # Put the model in eval mode
    model.eval()
    
    # Turn on inference mode context manager
    with torch.inference_mode():
        for data, labels in data_loader:
            # Send data to GPU
            data, labels = data.to(device), labels.to(device)
            
            # 1. Forward pass: Let the model predict
            predictions = model(data)
            
            # Get a tensor of booleans, indicating if each label is close to the real label
            is_prediction_correct = torch.isclose(predictions.argmax(dim=1), labels.argmax(dim=1), atol=atol)
            output_file = open("test_loop_output.txt", "a")
            print("----------------------------------------------------------------------------------------------------------------------------------------------", file=output_file)
            print(f"LABELS:: \n {labels} \n", file=output_file)
            print(f"PREDICTIONS:: \n {predictions} \n", file=output_file)
            print(f"IS PREDICTIONS CORRECT:: \n {is_prediction_correct} \n", file=output_file)
            output_file.close()
            
            # Count the amount of `True` predictions
            num_correct += is_prediction_correct.sum().item()
            
            # Count the total evaluations
            #   the first dimension of `labels` is `batch_size`
            total += labels.size(0)
    
    # Calculate the accuracy
    accuracy = float(num_correct) / float(total)
    print(f"Test Accuracy of the model: {accuracy * 100:.2f}%")
    return accuracy * 100