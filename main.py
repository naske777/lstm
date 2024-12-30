import json
import logging
from datetime import datetime

from functions import (
    get_missing_hours,
    fetch_and_format_new_data,
    update_json_data,
    setup_model,
)
from predict import predict_next_values

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuraciones
json_file = 'btc_price_1h.json'
model_path = 'model.pth'

# 1. Verificar datos faltantes
missing_hours = get_missing_hours(json_file)
if missing_hours == 0:
    logging.info("Datos actualizados. No se requiere entrenamiento.")
    
else:
    # 2. Obtener nuevos datos
    with open(json_file, 'r') as f:
        last_timestamp = json.load(f)[-1]['timestamp']

    start_time = datetime.fromtimestamp(last_timestamp/1000)
    end_time = datetime.now()
    new_data = fetch_and_format_new_data(start_time, end_time)
    print(len(new_data))
    if not new_data:
        logging.error("No se pudieron obtener nuevos datos")
        exit()

    #3. Actualizar JSON
    updated_count = update_json_data(json_file, new_data)
    logging.info(f"Actualizados {updated_count} registros nuevos")

# 4. Preparar modelo
model, device = setup_model(model_path)


# 5. Realizar predicciones
predictions = predict_next_values()

logging.info(predictions[:-10])