import torch
import numpy as np
from models import CNN_LSTM_Model
from data_generator import generate_sequence_data_by_index,normalize_data
from visualizacion import plot_predictions_vs_real

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar el modelo entrenado
model = CNN_LSTM_Model().to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

# Generar datos y entrenar
close_prices,close_prices_end = generate_sequence_data_by_index(device)
# Realizar predicciones sobre los datos existentes
normalized_predictions = []
denormalized_predictions = []
       
X = []
for i in range(10):
    normalized_sequence, min_val, max_val = normalize_data(close_prices)
    
    sequence_length = 72  # Longitud de la secuencia (3 días de datos)
    for j in range(len(normalized_sequence) - sequence_length):
        X.append(normalized_sequence[j:j + sequence_length])
        
    
    
    X_train = torch.FloatTensor(X).unsqueeze(-1).to(device)
    X_train = X_train[-24:-23]
    if(i == 0):
        print(X_train)
    
    with torch.no_grad():
        seq = X_train[0].unsqueeze(0)
        pred = model(seq)
        normalized_pred = pred.cpu().item()
        normalized_predictions.append(normalized_pred)
        denormalized_pred = normalized_pred * (max_val - min_val) + min_val
        denormalized_predictions.append(denormalized_pred)
        close_prices = close_prices + [denormalized_pred]

plot_predictions_vs_real(denormalized_predictions,denormalized_predictions)