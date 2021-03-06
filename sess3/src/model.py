import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # For MNIST classification
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        # For sum calculation
        self.fc3 = nn.Linear(1, 16)
        self.fc4 = nn.Linear(144, 16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, x, num):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        y = self.fc3(num.unsqueeze(dim=1))
        y = F.relu(y)
        y = torch.cat((x, y), dim=1)
        y = self.fc4(y)
        y = F.relu(y)
        y = self.fc5(y)

        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x, y