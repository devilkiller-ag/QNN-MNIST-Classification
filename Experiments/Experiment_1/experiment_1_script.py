import torch
import classiq

import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

import os
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.abspath(os.path.join(current_directory, "../../"))
sys.path.append(root_directory) # Add the parent directory to the sys.path list

from models.leqm3 import linear_entanglement_r3_quantum_model
from models.qnn import execute_fn, post_process_fn, QNN

from scripts.helper import create_writer, write_train_results
from scripts.data_setup import create_mnist_dataloaders
from scripts.data_transforms import input_transform, target_transform
from scripts.train import train
from scripts.test import test
from scripts.save_model import save_model

## Authenticate Classiq
# classiq.authenticate()

## For setting up device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'

## HYPER PARAMETERS
_LEARNING_RATE = 1.0
BATCH_SIZE = 32
EPOCHS = 20

## Create a Linear Entanglement Quantum Model for MNIST Data Classification with three linear entanglement layers of RXX, RYY, and RZZ.
quantum_model = linear_entanglement_r3_quantum_model()
quantum_program = classiq.synthesize(quantum_model)

# View Quantum Program on Classiq Platform
# classiq.show(quantum_program)

qnn = QNN(
    quantum_program=quantum_program,
    execute=execute_fn,
    post_process=post_process_fn,
)

# summary(model=qnn, input_size=(32, 16), verbose=0, col_names=["input_size", "output_size", "num_params", "trainable"], col_width=20, row_settings=["var_names"])

# choosing our loss function
loss_fn = nn.L1Loss()
# choosing our optimizer
optimizer = optim.SGD(qnn.parameters(), lr=_LEARNING_RATE)

train_dataloader, test_dataloader, class_names = create_mnist_dataloaders(
    root="../../data",
    transform=input_transform,
    target_transform=target_transform,
    batch_size=BATCH_SIZE,
    create_subset=True,
    subset_size=512
)

# Let's check out what we've created
print(f"Dataloaders: {train_dataloader, test_dataloader}") 
print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")
print(f"Our Dataset have following classes: {class_names}")

data, label = next(iter(train_dataloader))

print(f"Image shape: {data.shape} -> [batch_size, pixel_angle]")
print(f"Label shape: {label.shape} -> [batch_size, label_value]")

# Create a writer for tracking our experiment
writer = create_writer(experiment_name="data_0.1_percent", model_name="linear_entanglement_r3", extra=f"{EPOCHS}_epochs")

train_results = train(
    model = qnn, 
    data_loader = train_dataloader, 
    loss_fn = loss_fn, 
    optimizer = optimizer, 
    writer = writer, 
    epochs = EPOCHS,
    device = device
)

# Check out the model results
print(train_results)

write_train_results(experiment_name="data_0.1_percent", model_name="linear_entanglement_r3", epochs=EPOCHS, results=train_results)

# %load_ext tensorboard
# %tensorboard --logdir runs

save_model(
    model=qnn,
    target_dir='outputs/saved_models',
    model_name='exp_1_leqmr3_subset512'
)

test_results = test(
    model = qnn, 
    data_loader = test_dataloader,
    device = device
)

print(test_results)