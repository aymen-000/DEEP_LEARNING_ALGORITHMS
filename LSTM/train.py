import torch
import torch.nn as nn
from torch.optim import Adam
from LSTM import CustomLSTM
# Define the CustomLSTM class as previously corrected

# Usage Example
input_size = 10  # Example input size
hidden_size = 20  # Example hidden size
sequence_length = 5  # Length of each sequence
batch_size = 16  # Number of sequences in a batch

# Initialize model, criterion, and optimizer
model = CustomLSTM(input_size=input_size, hidden_size=hidden_size)
criterion = nn.MSELoss()
optimizer = model.optim()

# Generate random data
inputs = torch.randn(batch_size, sequence_length, input_size)
targets = torch.randn(batch_size, hidden_size)

# Wrap inputs and targets in a batch
batch = (inputs, targets)

# Example training loop
for epoch in range(10):  # Assume 10 epochs
    for batch_index in range(100):  # Assume 100 batches per epoch
        loss = model.train_step(batch, batch_index, criterion, optimizer)

    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")