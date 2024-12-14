
import torch
import torch.nn as nn

class CNN_LSTM_Model(nn.Module):
    def __init__(self):
        super(CNN_LSTM_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=45, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=45, hidden_size=400, num_layers=1, batch_first=True)
        self.fc = nn.Linear(400, 1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        output, _ = self.lstm(x)
        prediction = self.fc(output[:, -1, :])
        return prediction