import torch
from models import CNN_LSTM_Model
from data_generator import generate_sequence_data
from visualizacion import plot_predictions
from trainer import train_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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