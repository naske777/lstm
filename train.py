import torch
from models import CNN_LSTM_Model
from data_generator import generate_sequence_data
from visualizacion import plot_predictions
from trainer import train_model
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_LSTM_Model().to(device)

# Obtener la longitud total de los datos
with open('btc_price_1h.json', 'r') as file:
    data = json.load(file)
total_segments = len(data) // 500  # Cada segmento de 500 elementos

print(f"Total de segmentos a entrenar: {total_segments}")
best_loss = float('inf')

# Entrenamiento iterativo por segmentos
for segment in range(total_segments):
    print(f"\nEntrenando segmento {segment + 1}/{total_segments}")
    
    # Define sequence_length before calling generate_sequence_data
    sequence_length = 72
    
    X_train, y_train, full_sequence, min_val, max_val = generate_sequence_data(device, segment, total_segments)
    
    if len(X_train) == 0:
        print(f"Saltando segmento {segment + 1} por falta de datos")
        continue
    
    # Cargar el mejor modelo si existe
    try:
        model.load_state_dict(torch.load('model.pth'))
    except:
        print("No se encontró modelo previo, comenzando desde cero")
    
    # Entrenar el modelo con el segmento actual
    loss, mse, mape = train_model(model, X_train, y_train, min_val, max_val)
    
    # Guardar el modelo si mejora
    if loss < best_loss:
        best_loss = loss
        print(f"Nuevo mejor modelo encontrado en segmento {segment + 1}")
        torch.save(model.state_dict(), 'model.pth')

# Cargar el mejor modelo para las predicciones
model.load_state_dict(torch.load('model.pth'))

# Obtener último segmento para evaluación final
X_train, y_train, full_sequence, min_val, max_val = generate_sequence_data(device, total_segments-1, total_segments)

# Realizar predicciones
normalized_predictions = []
denormalized_predictions = []
with torch.no_grad():
    for i in range(len(X_train)):
        seq = X_train[i].unsqueeze(0)
        pred = model(seq)
        normalized_pred = pred.cpu().item()
        normalized_predictions.append(normalized_pred)
        denormalized_pred = normalized_pred * (max_val - min_val) + min_val
        denormalized_predictions.append(denormalized_pred)

# Mostrar resultados finales
plot_predictions(full_sequence, denormalized_predictions)
print("Siguiente predicción (desnormalizada):", denormalized_predictions[-1])
print("Valor real:", full_sequence[len(normalized_predictions) + 4])