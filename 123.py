import torch
from models import CNN_LSTM_Model
from data_generator import generate_sequence_data
from visualizacion import plot_predictions_vs_real

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar el modelo entrenado
model = CNN_LSTM_Model().to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

# Generar datos y entrenar
X_train, y_train, full_sequence, min_val, max_val = generate_sequence_data(device)
X_train = X_train[-11:]
# Realizar predicciones
normalized_predictions = []
denormalized_predictions = []
print(X_train[0])
with torch.no_grad():
    for i in range(len(X_train)):
        seq = X_train[i].unsqueeze(0)
        pred = model(seq)  # Use the updated model for prediction
        normalized_pred = pred.cpu().item()
        normalized_predictions.append(normalized_pred)
        denormalized_pred = normalized_pred * (max_val - min_val) + min_val
        denormalized_predictions.append(denormalized_pred)
        # if(i == len(X_train) -1):
        #     print(seq)
        #     print(denormalized_pred)

# Mostrar resultados finales
plot_predictions_vs_real(denormalized_predictions,full_sequence[-11:])