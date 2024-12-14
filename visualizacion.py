
import matplotlib.pyplot as plt

def plot_predictions(full_sequence, denormalized_predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(range(5, len(full_sequence)), full_sequence[5:], 'b-o', label='Datos reales')
    plt.plot(range(5, len(denormalized_predictions) + 5), denormalized_predictions, 'r--o', label='Predicciones')
    plt.grid(True)
    plt.legend()
    plt.title('Secuencia y Predicciones LSTM')
    plt.xlabel('Posición en la secuencia')
    plt.ylabel('Valor')
    plt.savefig('prediccion_lstm.png')
    plt.close()