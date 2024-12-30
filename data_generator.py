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

def generate_sequence_data(device, segment=0, total_segments=10):
    # Leer el archivo JSON
    with open('btc_price_1h.json', 'r') as file:
        data = json.load(file)
    
    segment_size = 500
    start_idx = segment * segment_size
    end_idx = min(start_idx + segment_size, len(data))
    
    # Asegurar que tengamos suficientes datos para el entrenamiento
    if end_idx - start_idx < 72:  # 72 es sequence_length
        start_idx = max(0, end_idx - 500)  # Asegurar al menos 500 elementos
    
    # Extraer el segmento actual de datos
    segment_data = data[start_idx:end_idx]
    
    # Extraer solo los precios de cierre (close)
    close_prices = [float(entry.get('close', 0)) for entry in segment_data if 'close' in entry]
    
    normalized_sequence, min_val, max_val = normalize_data(close_prices)
    
    # Define sequence_length before it is used
    sequence_length = 72
    
    # Asegurar que tenemos suficientes datos para la secuencia
    if len(normalized_sequence) <= sequence_length:
        print(f"Advertencia: Segmento {segment} tiene muy pocos datos. Usando datos adicionales.")
        start_idx = max(0, end_idx - 500)
        segment_data = data[start_idx:end_idx]
        close_prices = [float(entry.get('close', 0)) for entry in segment_data if 'close' in entry]
        normalized_sequence, min_val, max_val = normalize_data(close_prices)
    
    X, y = [], []
    sequence_length = 72  # Ensure sequence_length is defined before use
    for i in range(len(normalized_sequence) - sequence_length):
        X.append(normalized_sequence[i:i + sequence_length])
        y.append(normalized_sequence[i + sequence_length])
    
    x_train = torch.FloatTensor(X).unsqueeze(-1).to(device)
    return x_train, torch.FloatTensor(y).to(device), close_prices, min_val, max_val

def generate_sequence_data_by_index(device, sequence_index=1):
     # Leer el archivo JSON
    with open('btc_price_1h.json', 'r') as file:
        data = json.load(file)
    
    sample_size = math.ceil(len(data) * 0.1)
    
    # Extraer solo los últimos 10% de datos
    sample_data = data[-sample_size:]
    
    # Extraer solo los precios de cierre (close)
    close_prices = [float(entry.get('close', 0)) for entry in sample_data if 'close' in entry]    
    close_prices_end = [float(entry.get('close', 0)) for entry in sample_data if 'close' in entry][-5:]    
    
   
    
    
    return close_prices,close_prices_end

def generate_sequence_from_new_data(device, hours_back=100):
    """Genera secuencias de entrenamiento usando solo los datos más recientes"""
    with open('btc_price_1h.json', 'r') as file:
        data = json.load(file)
    
    # Tomar solo los últimos datos según hours_back
    recent_data = data[-hours_back:]
    
    # Extraer precios de cierre
    close_prices = [float(entry.get('close', 0)) for entry in recent_data]
    
    normalized_sequence, min_val, max_val = normalize_data(close_prices)
    
    X, y = [], []
    sequence_length = 72
    for i in range(len(normalized_sequence) - sequence_length):
        X.append(normalized_sequence[i:i + sequence_length])
        y.append(normalized_sequence[i + sequence_length])
    
    x_train = torch.FloatTensor(X).unsqueeze(-1).to(device)
    return x_train, torch.FloatTensor(y).to(device), close_prices, min_val, max_val