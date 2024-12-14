import numpy as np
import torch

def calculate_metrics(true_values, predictions):
    """Calcula múltiples métricas de evaluación"""
    true_values = np.array(true_values)
    predictions = np.array(predictions)
    
    mse = np.mean((true_values - predictions) ** 2)
    mape = np.mean(np.abs((true_values - predictions) / true_values)) * 100
    
    return mse, mape

def combined_loss(pred, target, mse, mape):
    """Combina diferentes métricas en una única función de pérdida"""
    w1, w2, w3 = 0.7, 0.15, 0.15
    
    main_loss = torch.nn.functional.mse_loss(pred, target)
    
    mse_tensor = torch.tensor(mse / 1000.0, device=pred.device)
    mape_tensor = torch.tensor(mape / 100.0, device=pred.device)
    
    total_loss = w1 * main_loss + w2 * mse_tensor + w3 * mape_tensor
    
    return total_loss

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def create_sequences(data, sequence_length):
    """
    Crear secuencias de entrada/salida para el modelo LSTM
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)