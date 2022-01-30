import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        size_input = 32*32*1
        dev = 12
        number_perc = int(size_input/dev)
        self.fc1 = nn.Linear(size_input, number_perc)
        self.fc2 = nn.Linear(number_perc, number_perc)
        self.fc3 = nn.Linear(number_perc, number_perc)
        self.fc4 = nn.Linear(number_perc, size_input)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x