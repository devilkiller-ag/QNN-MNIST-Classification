import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import pandas as pd

def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/leqm3/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                                model_name="leqm3",
                                extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/leqm3/5_epochs/")
    """
    
    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)

def write_train_results(
    experiment_name,
    model_name,
    epochs,
    results,
):
    output_dir = "outputs/train_results/"
    
    # Check if the directory exists, and create it if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file_name = f"{experiment_name}_{model_name}_epochs_{epochs}.csv"
    file_path = f"outputs/train_results/{output_file_name}"
    
    data = {
        'Epoch': list(range(1, len(results['train_loss']) + 1)),
        'Train Loss': results['train_loss']
    }
    
    df = pd.DataFrame(data)
    
    if os.path.exists(file_path):
        # If it exists, append data
        df.to_csv(file_path, mode='a', index=False, header=False)
    else:
        df.to_csv(file_path, mode='w', index=False, header=True)