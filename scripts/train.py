import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from typing import Dict, List
from tqdm.auto import tqdm

def train(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.modules.loss._Loss,
    optimizer: optim.Optimizer,
    writer: torch.utils.tensorboard.writer.SummaryWriter,
    epochs: int = 20,
    device: str = 'cpu',
) -> Dict[str, List]:
    model.to(device)
    
    # Setup train loss value
    train_loss = 0
    
    # Create empty results dictionary
    results = {
        "train_loss": [],
    }
    
    # Loop through training steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n----------")
        for batch, (data, label) in enumerate(data_loader):
            # Send data to device (GPU or CPU)
            data, label = data.to(device), label.to(device)
            
            # 1. Forward pass
            output = model(data).to(device)
            
            # 2. Calculate loss
            loss = loss_fn(output, label)
            train_loss += loss
            
            # 3. Optimizer zero grad
            optimizer.zero_grad()
            
            # 4. Loss backward
            loss.backward()
            
            # 5. Optimizer step
            optimizer.step()
        
        # Calculate loss per epoch and print out what's happening
        train_loss /= len(data_loader)
        
        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"Train loss: {train_loss:.5f}"
        )
        
        # Update results dictionary
        results["train_loss"].append(train_loss.detach().item())
        
        ### Experiment Tracking ###
        # See if there's a writer, if so, log to it
        if writer:
            # Add loss results to SummaryWriter
            writer.add_scalars(
                main_tag="Loss", 
                tag_scalar_dict={"train_loss": train_loss,},
                global_step=epoch
            )
            
            # Close the writer
            writer.close()
    
    # Return the filled results at the end of the epochs
    return results