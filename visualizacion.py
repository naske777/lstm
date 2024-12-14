import matplotlib.pyplot as plt

def plot_predictions(full_sequence, denormalized_predictions):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Gráfico completo
    ax1.plot(range(5, len(full_sequence)), full_sequence[5:], 'b-o', label='Datos reales')
    ax1.plot(range(5, len(denormalized_predictions) + 5), denormalized_predictions, 'r--o', label='Predicciones')
    ax1.grid(True)
    ax1.legend()
    ax1.set_title('Secuencia y Predicciones LSTM - Vista Completa')
    ax1.set_xlabel('Posición en la secuencia')
    ax1.set_ylabel('Valor')

    # Gráfico últimos 100 datos
    last_100_real = full_sequence[-100:]
    last_100_pred = denormalized_predictions[-100:]
    x_real = range(len(full_sequence)-100, len(full_sequence))
    x_pred = range(len(full_sequence)-100, len(full_sequence)-100+len(last_100_pred))
    
    ax2.plot(x_real, last_100_real, 'b-o', label='Datos reales')
    ax2.plot(x_pred, last_100_pred, 'r--o', label='Predicciones')
    ax2.grid(True)
    ax2.legend()
    ax2.set_title('Secuencia y Predicciones LSTM - Últimos 100 datos')
    ax2.set_xlabel('Posición en la secuencia')
    ax2.set_ylabel('Valor')

    plt.tight_layout()
    plt.savefig('prediccion_lstm.png')
    plt.close()