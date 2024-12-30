import requests
import pandas as pd
import datetime

def fetch_historical_data(symbol, interval, start_time, end_time):
    base_url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(start_time.timestamp() * 1000),
        "endTime": int(end_time.timestamp() * 1000),
        "limit": 1000
    }

    all_data = []
    while True:
        response = requests.get(base_url, params=params)
        print("code: ",response.status_code)
        data = response.json()

        if not data:
            break

        all_data.extend(data)

        # Update start_time to fetch the next batch
        params["startTime"] = data[-1][0] + 1

        if len(data) < 1000:
            break

    return all_data

if __name__ == "__main__":

    # Parámetros del script
    symbol = "BTCUSDT"  # Par de trading BTC/USDT
    interval = "1h"      # Intervalo de 1 hora
    
    # Fechas de inicio y fin (último año)
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(days=1095)
    
    # Obtener datos
    historical_data = fetch_historical_data(symbol, interval, start_time, end_time)
    
    # Convertir a JSON
    data_list = [
        {
            "timestamp": kline[0],
            "open": kline[1],
            "high": kline[2],
            "low": kline[3],
            "close": kline[4],
            "volume": kline[5]
        } for kline in historical_data
    ]
    
    # Guardar en un archivo JSON
    import json
    with open("btc_price_1h_full.json", "w") as f:
        json.dump(historical_data, f, indent=4)
    
    print("Datos guardados en btc_price_1h.json")
    
