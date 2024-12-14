import numpy as np
import torch

from data_generator import denormalize_data
from utils import calculate_metrics, combined_loss, EarlyStopping

def train_model(model, X, y, min_val, max_val, X_val=None, y_val=None, epochs=700, max_grad_norm=1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.99, patience=100, verbose=True)
    early_stopping = EarlyStopping(patience=100)
    
    best_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        optimizer.zero_grad()
        output, _ = model[0](X)
        prediction = model[1](output[:, -1, :])
        
        train_mse, train_mape = calculate_metrics(
            denormalize_data(y.cpu().detach().numpy(), min_val, max_val),
            denormalize_data(prediction.squeeze().cpu().detach().numpy(), min_val, max_val)
        )
        
        loss = combined_loss(prediction.squeeze(), y, train_mse, train_mape)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        # Guardar mejor modelo basado en loss de entrenamiento si no hay validación
        if X_val is None or y_val is None:
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model_state = {key: val.cpu().clone() for key, val in model.state_dict().items()}
                print(f'\nNuevo mejor modelo en época {epoch+1}:')
                print(f'  Train Loss: {loss.item():.4f}')
                print(f'  Train MSE: {train_mse:.4f}')
                print(f'  Train MAPE: {train_mape:.2f}%')
            scheduler.step(loss)
        else:
            # Validación
            model.eval()
            with torch.no_grad():
                val_output, _ = model[0](X_val)
                val_prediction = model[1](val_output[:, -1, :])
                
                val_mse, val_mape = calculate_metrics(
                    denormalize_data(y_val.cpu().detach().numpy(), min_val, max_val),
                    denormalize_data(val_prediction.squeeze().cpu().detach().numpy(), min_val, max_val)
                )
                
                val_loss = combined_loss(val_prediction.squeeze(), y_val, val_mse, val_mape)
                scheduler.step(val_loss)
                early_stopping(val_loss.item())
                
                if val_loss.item() < best_loss:
                    best_loss = val_loss.item()
                    best_model_state = {key: val.cpu().clone() for key, val in model.state_dict().items()}
                    print(f'\nNuevo mejor modelo en época {epoch+1}:')
                    print(f'  Val Loss: {val_loss.item():.4f}')
                    print(f'  Val MSE: {val_mse:.4f}')
                    print(f'  Val MAPE: {val_mape:.2f}%')
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
        if (epoch + 1) % 10 == 0:
            print(f'\nÉpoca {epoch+1}:')
            print(f'  Train Loss: {loss.item():.4f}')
            if X_val is not None:
                print(f'  Val Loss: {val_loss.item():.4f}')
    
    if best_model_state is None:
        print("Warning: No best model state was saved. Using final model state.")
        best_model_state = {key: val.cpu().clone() for key, val in model.state_dict().items()}
        
    model.load_state_dict(best_model_state)
    return (best_loss, train_mse, train_mape) if X_val is None else (best_loss, val_mse, val_mape)

