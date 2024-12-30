
import pandas as pd
from datetime import datetime
import json
import torch
import logging
from btc_data import fetch_historical_data
from models import CNN_LSTM_Model
from data_generator import generate_sequence_from_new_data
from trainer import train_model

def get_missing_hours(json_file):
    """Verifica cuántas horas de datos faltan por actualizar"""
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    df = pd.DataFrame(data)
    last_timestamp = df.iloc[-1]['timestamp']
    current_timestamp = int(datetime.now().timestamp() * 1000)
    time_diff_hours = (current_timestamp - last_timestamp) / (1000 * 3600)
    
    return int(time_diff_hours) if time_diff_hours > 1 else 0

def fetch_and_format_new_data(start_time, end_time, symbol="BTCUSDT", interval="1h"):
    """Obtiene y formatea nuevos datos de Binance"""
    new_data = fetch_historical_data(symbol, interval, start_time, end_time)
    if not new_data:
        return None
        
    return [{
        "timestamp": kline[0],
        "open": kline[1],
        "high": kline[2],
        "low": kline[3],
        "close": kline[4],
        "volume": kline[5]
    } for kline in new_data]

def update_json_data(json_file, new_data):
    """Actualiza el archivo JSON con nuevos datos"""
    with open(json_file, 'r') as file:
        existing_data = json.load(file)
    
    if new_data and existing_data[-1]['timestamp'] == new_data[0]['timestamp']:
        new_data = new_data[1:]
    
    existing_data.extend(new_data)
    
    with open(json_file, 'w') as f:
        json.dump(existing_data, f, indent=4)
    
    return len(new_data)

def setup_model(model_path):
    """Configura y carga el modelo"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN_LSTM_Model().to(device)
    
    try:
        model.load_state_dict(torch.load(model_path))
    except:
        pass
        
    return model, device

