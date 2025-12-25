"""
Neural network model for animal decision making
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNN(nn.Module):
    """Neural network for animal decision making"""
    def __init__(self, config):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.rnn = nn.GRU(4, 16, batch_first=True)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)

    def forward(self, animal_input: torch.Tensor, 
                visible_animals_input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        animal_output = F.relu(self.fc1(animal_input))
        
        if visible_animals_input.size(1) > 0:
            _, rnn_output = self.rnn(visible_animals_input)
            rnn_output = rnn_output.squeeze(0)
        else:
            rnn_output = torch.zeros(animal_input.size(0), 16)
        
        combined_output = torch.cat((animal_output, rnn_output), dim=-1)
        hidden = F.relu(self.fc2(combined_output))
        output = self.fc3(hidden)
        
        return F.softmax(output, dim=-1)
