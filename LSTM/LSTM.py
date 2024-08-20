import torch
import torch.nn as nn
from torch.optim import Adam

class CustomLSTM(nn.Module):
    # INIT 
    def __init__(self, input_size, hidden_size): 
        super(CustomLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # INIT WEIGHT 
        self.wlr1 = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.wlr2 = nn.Parameter(torch.randn(input_size, hidden_size))
        self.blr1 = nn.Parameter(torch.zeros(hidden_size))
        
        self.wpr1 = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.wpr2 = nn.Parameter(torch.randn(input_size, hidden_size))
        self.bpr1 = nn.Parameter(torch.zeros(hidden_size))
        
        self.wp1 = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.wp2 = nn.Parameter(torch.randn(input_size, hidden_size))
        self.bp1 = nn.Parameter(torch.zeros(hidden_size))
        
        self.wo1 = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.wo2 = nn.Parameter(torch.randn(input_size, hidden_size))
        self.bo1 = nn.Parameter(torch.zeros(hidden_size))
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    # CREATE LSTM UNITS 
    def lstm_unit(self, input_value, long_memory, short_memory): 
        long_term_percent = self.sigmoid((short_memory @ self.wlr1) + 
                                         (input_value @ self.wlr2) + 
                                         self.blr1)
        
        potential_percent_to_remember = self.sigmoid((short_memory @ self.wpr1) + 
                                                     (input_value @ self.wpr2) + 
                                                     self.bpr1)
        
        potential_memory = self.tanh((short_memory @ self.wp1) + 
                                     (input_value @ self.wp2) + 
                                     self.bp1)
        
        new_long_memory = (long_memory * long_term_percent + 
                           potential_memory * potential_percent_to_remember)
        
        output_percent = self.sigmoid((short_memory @ self.wo1) + 
                                      (input_value @ self.wo2) + 
                                      self.bo1)
        
        new_short_memory = self.tanh(new_long_memory) * output_percent 
        
        return new_long_memory, new_short_memory
    
    # CREATE FORWARD 
    def forward(self, input): 
        long_memory = torch.zeros(input.size(0), self.hidden_size)
        short_memory = torch.zeros(input.size(0), self.hidden_size)
        
        for i in range(input.size(1)):  # Iterate over sequence length
            long_memory, short_memory = self.lstm_unit(input[:, i, :], long_memory, short_memory)
        
        return short_memory  # Return the final short memory
    
    # OPTIM 
    def optim(self): 
        return Adam(self.parameters())
    
    # TRAIN 
    def train_step(self, batch, batch_index, criterion, optimizer):
        # Set the model to training mode
        self.train()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        inputs, targets = batch
        outputs = self(inputs)

        # Compute the loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Optionally, print or log the loss
        if batch_index % 10 == 0:
            print(f"Batch {batch_index}, Loss: {loss.item()}")

        return loss.item()

    