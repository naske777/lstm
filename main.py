import torch
import torch.nn as nn
from data_generator import generate_sequence_data
from visualizacion import plot_predictions
from trainer import train_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN_LSTM_Model(nn.Module):
    def __init__(self):
        super(CNN_LSTM_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=45, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=45, hidden_size=400, num_layers=1, batch_first=True)
        self.fc = nn.Linear(400, 1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, features, sequence_length)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # Reshape back to (batch_size, sequence_length, features)
        output, _ = self.lstm(x)
        prediction = self.fc(output[:, -1, :])
        return prediction

model = CNN_LSTM_Model().to(device)

# Generar datos y entrenar
X_train, y_train, full_sequence, min_val, max_val = generate_sequence_data(device)
train_model(model, X_train, y_train, min_val, max_val)
 
# Realizar predicciones
normalized_predictions = []
denormalized_predictions = []
with torch.no_grad():
    for i in range(len(X_train)):
        seq = X_train[i].unsqueeze(0)
        pred = model(seq)  # Use the updated model for prediction
        normalized_pred = pred.cpu().item()
        normalized_predictions.append(normalized_pred)
        denormalized_pred = normalized_pred * (max_val - min_val) + min_val
        denormalized_predictions.append(denormalized_pred)

# Mostrar resultados finales
plot_predictions(full_sequence, denormalized_predictions)
print("Siguiente predicción (desnormalizada):", denormalized_predictions[-1])
print("Valor real:", full_sequence[len(normalized_predictions) + 4])