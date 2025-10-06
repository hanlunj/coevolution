import torch
import torch.nn as nn

class OneLayerFCRegressionRELU(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=10, dropout_p=0.1):
        super(OneLayerFCRegressionRELU, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_p)
    def forward(self, x, x_r=None):
        if x_r is not None:
            x = torch.hstack((x_r, x))
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return x

class TwoLayerFCRegressionRELU(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=10, dropout_p=0.1):
        super(TwoLayerFCRegressionRELU, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_p)
    def forward(self, x, x_r=None):
        if x_r is not None:
            x = torch.hstack((x_r, x))
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return x

class ThreeLayerFCRegressionRELU(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=10, dropout_p=0.1):
        super(ThreeLayerFCRegressionRELU, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_p)
    def forward(self, x, x_r=None):
        if x_r is not None:
            x = torch.hstack((x_r, x))
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        return x

class FourLayerFCRegressionRELU(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=10, dropout_p=0.1):
        super(FourLayerFCRegressionRELU, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_p)
    def forward(self, x, x_r=None):
        if x_r is not None:
            x = torch.hstack((x_r, x))
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout(x)
        x = torch.relu(self.fc5(x))
        return x

