import torch
import torch.nn as nn

class FullyConnectedNN(nn.Module):
    """
    Fully-connected Neural Network (FCNN) for EEG denoising.
    Input: vector of length N (number of data points in segment).
    Output: vector of length N (denoised EEG segment).
    """
    def __init__(self, datanum: int):
        super(FullyConnectedNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(datanum, datanum),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(datanum, datanum),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(datanum, datanum),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(datanum, datanum)
        )

    def forward(self, x):
        # x: tensor of shape (batch_size, datanum)
        return self.model(x)

class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network for EEG denoising.
    Four Conv1d layers (64 filters each) followed by flatten and a fully-connected output layer.
    Input: 1 x N signal, Output: 1 x N signal.
    """
    def __init__(self, datanum: int):
        super(SimpleCNN, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # After 4 conv layers (with padding), output has shape (batch, 64, N)
        # Flatten and linear layer to map 64*N features to N outputs.
        self.fc = nn.Linear(64 * datanum, datanum)

    def forward(self, x):
        # x: tensor of shape (batch_size, 1, datanum)
        out = self.conv_net(x)              # shape: (batch_size, 64, N)
        out = out.view(out.size(0), -1)     # flatten to (batch_size, 64*N)
        out = self.fc(out)                  # shape: (batch_size, N)
        return out
