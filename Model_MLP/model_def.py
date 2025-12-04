import torch
import torch.nn as nn

class SleepStageMLP_ReLU(nn.Module):
    def __init__(self, input_dim, num_of_classes, dropout):
        super(SleepStageMLP_ReLU, self).__init__()

        # Define layers
        self.layer1 = nn.Linear(input_dim, 1024)
        
        self.layer2 = nn.Linear(1024, 512)

        self.layer3 = nn.Linear(512, 128)

        self.out = nn.Linear(128, num_of_classes)

        # Dropout layers
        self.dropout1 = nn.Dropout(0.2)

        self.dropout2 = nn.Dropout(0.3)

        self.dropout3 = nn.Dropout(0.5)

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
        

# LeakyReLU
class SleepStageMLP_LeakyReLU(nn.Module):

    def __init__(self, input_dim, num_of_classes, dropout, negative_slope=0.01):
        super(SleepStageMLP_LeakyReLU, self).__init__()
        self.layer1 = nn.Linear(input_dim, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, num_of_classes)

        # LeakyReLU is used here with a specified negative_slope
        self.act = nn.LeakyReLU(negative_slope=negative_slope)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.5)

        nn.init.kaiming_normal_(self.layer1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.layer2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.layer3.weight, nonlinearity='leaky_relu')
        
    def forward(self, x):
        x = self.act(self.layer1(x))
        x = self.dropout1(x)
        x = self.act(self.layer2(x))
        x = self.dropout2(x)
        x = self.act(self.layer3(x))
        x = self.dropout3(x)
        return self.out(x)


#Tanh Model (Hyperbolic Tangent)
class SleepStageMLP_Tanh(nn.Module):
    def __init__(self, input_dim, num_of_classes, dropout):
        super(SleepStageMLP_Tanh, self).__init__()
        self.layer1 = nn.Linear(input_dim, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, num_of_classes)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.5)

        # Xavier/Glorot initialization is standard for Tanh
        nn.init.xavier_normal_(self.layer1.weight)
        nn.init.xavier_normal_(self.layer2.weight)
        nn.init.xavier_normal_(self.layer3.weight)
        
    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = self.dropout1(x)
        x = torch.tanh(self.layer2(x))
        x = self.dropout2(x)
        x = torch.tanh(self.layer3(x))
        x = self.dropout3(x)
        return self.out(x)
