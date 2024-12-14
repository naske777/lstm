import torch
import json
import math

def normalize_data(data):
    min_val = min(data)
    max_val = max(data)
    normalized = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized, min_val, max_val

def denormalize_data(normalized_data, min_val, max_val):
    return [x * (max_val - min_val) + min_val for x in normalized_data]

def generate_sequence_data(device):
    # Leer el archivo JSON
    with open('btc_price_1h.json', 'r') as file:
        data = json.load(file)
    
    sample_size = math.ceil(len(data) * 0.1)
    
    # Extraer solo los últimos 10% de datos
    sample_data = data[-sample_size:]
    
    # Extraer solo los precios de cierre (close)
    close_prices = [float(entry.get('close', 0)) for entry in sample_data if 'close' in entry]
    
    normalized_sequence, min_val, max_val = normalize_data(close_prices)
    
    X, y = [], []
    sequence_length = 72  # Longitud de la secuencia (3 días de datos)
    for i in range(len(normalized_sequence) - sequence_length):
        X.append(normalized_sequence[i:i + sequence_length])
        y.append(normalized_sequence[i + sequence_length])
    
    return torch.FloatTensor(X).unsqueeze(-1).to(device), torch.FloatTensor(y).to(device), close_prices, min_val, max_val