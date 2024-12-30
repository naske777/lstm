import torch
import numpy as np
from models import CNN_LSTM_Model
from data_generator import denormalize_data, generate_sequence_data_by_index,normalize_data
from visualizacion import plot_predictions_vs_real

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar el modelo entrenado
model = CNN_LSTM_Model().to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

# Configuracion de longitud de prediccion
last_prediction_count = 50
first_prediction_count = 72

def predict_next_values():
    """Predice los próximos valores basados en los datos más recientes"""
    
    # Obtener datos recientes
    close_prices, _ = generate_sequence_data_by_index(device)
    close_prices = close_prices[-500:-50]
    denormalized_predictions = []
    
    # Realizar predicciones
    normalized_sequence, min_val, max_val = normalize_data(close_prices)
    for i in range(last_prediction_count):
        
        # Preparar secuencia de entrada
        X = [normalized_sequence[-72:]]
        X_train = torch.FloatTensor(X).unsqueeze(-1).to(device)

        # print("first: ", X_train[0][:5])
        print("last: ", X_train[0][-1:])
        print("last1: ", normalized_sequence[-1:])
        # Realizar predicción
        with torch.no_grad():
            pred = model(X_train[0].unsqueeze(0))
            normalized_pred = pred.cpu().item()
            normalized_sequence.append(normalized_pred)
            print("Predicción normalizada: ", normalized_pred)
            denormalized_pred = denormalize_data([normalized_pred], min_val, max_val)[0]
            denormalized_predictions.append(denormalized_pred)
            print("Predicción: ", denormalized_pred)
            close_prices.append(denormalized_pred)
    
    plot_predictions_vs_real(close_prices[-(first_prediction_count + last_prediction_count):],last_prediction_count)
    return denormalized_predictions

if __name__ == "__main__":
    predict_next_values()