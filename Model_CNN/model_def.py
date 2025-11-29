import torch
import torch.nn as nn
import torch.nn.functional as F

class  (nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(4, 32, kernel_size=50, stride=6)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=8, stride=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=8, stride=2)

        self.dropout = nn.Dropout(0.3)

        # compute the output size after conv layers for FC layer
        self._dummy = torch.zeros(1, 4, 7680)
        out = self._forward_features(self._dummy)
        self.flat_size = out.shape[1]

        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5)  # 5 sleep stages
        )

    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.fc(x)
        return x