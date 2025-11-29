import torch
import torch.nn as nn

class SleepStageMLP(nn.module):
    def __init__(self, input_dim, num_of_classes, dropout):
        super(SleepStageMLP, self).__init__()

        # Define layers
        self.layer1 = nn.Linear(input_dim, 1024)
        
        self.layer2 = nn.Linear(1024, 512)

        self.layer3 = nn.Linear(512, 128)

        self.out = nn.Linear(128, num_of_classes)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)

        self.dropout2 = nn.Dropout(dropout)

        self.dropout3 = nn.Dropout(dropout * 0.5)

        # --- Weight Initialization ---
        nn.init.kaiming_normal_(self.layer1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer3.weight, nonlinearity='relu')
        
    def  forward(self, x):

        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)

        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)

        x = torch.relu(self.layer3(x))
        x = self.dropout3(x)

        return self.out(x)
        