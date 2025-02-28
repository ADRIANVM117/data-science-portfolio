import requests
import pandas as pd
import time

# 🔹 Configura tu API Key de Alpha Vantage
API_KEY = "#######"

# 🔹 Función para descargar datos según las opciones del usuario
def get_stock_data(symbol, function="TIME_SERIES_DAILY_ADJUSTED", interval="30min", outputsize="compact"):
    """
    Descarga datos de Alpha Vantage según los parámetros seleccionados.
    
    Parámetros:
    - symbol (str): Símbolo de la acción (ej. "AAPL", "MSFT").
    - function (str): Tipo de datos a obtener (Intraday, Daily, Weekly, Monthly).
    - interval (str, opcional): Intervalo de tiempo (solo para intradía). Opciones: "1min", "5min", "15min", "30min", "60min".
    - outputsize (str, opcional): Tamaño de la salida ("compact" = últimos 100 datos, "full" = todos los datos disponibles).
    
    Retorna:
    - DataFrame con los datos históricos de la acción.
    """
    function_map = {
        "Intraday": "TIME_SERIES_INTRADAY",
        "Daily": "TIME_SERIES_DAILY",
        "Daily Adjusted": "TIME_SERIES_DAILY_ADJUSTED",
        "Weekly": "TIME_SERIES_WEEKLY",
        "Weekly Adjusted": "TIME_SERIES_WEEKLY_ADJUSTED",
        "Monthly": "TIME_SERIES_MONTHLY",
        "Monthly Adjusted": "TIME_SERIES_MONTHLY_ADJUSTED",
    }

    if function not in function_map.values():
        raise ValueError("Función no válida. Opciones: Intraday, Daily, Weekly, Monthly.")

    url = f"https://www.alphavantage.co/query"
    params = {
        "function": function,
        "symbol": symbol,
        "outputsize": outputsize,
        "apikey": API_KEY
    }
    
    # Agregar parámetro de intervalo si se usa "Intraday"
    if function == "TIME_SERIES_INTRADAY":
        params["interval"] = interval

    response = requests.get(url, params=params)
    data = response.json()
    time.sleep(12)  # Evitar restricciones de la API

    # Definir las claves según el tipo de función
    key_map = {
        "TIME_SERIES_INTRADAY": f"Time Series ({interval})",
        "TIME_SERIES_DAILY": "Time Series (Daily)",
        "TIME_SERIES_DAILY_ADJUSTED": "Time Series (Daily)",
        "TIME_SERIES_WEEKLY": "Weekly Time Series",
        "TIME_SERIES_WEEKLY_ADJUSTED": "Weekly Adjusted Time Series",
        "TIME_SERIES_MONTHLY": "Monthly Time Series",
        "TIME_SERIES_MONTHLY_ADJUSTED": "Monthly Adjusted Time Series",
    }

    if key_map[function] in data:
        df = pd.DataFrame.from_dict(data[key_map[function]], orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={
            "1. open": "Open", "2. high": "High", "3. low": "Low",
            "4. close": "Close", "5. adjusted close": "Adjusted Close",
            "6. volume": "Volume", "7. dividend amount": "Dividend Amount",
            "8. split coefficient": "Split Coefficient"
        })
        df = df.astype(float)  # Convertir a valores numéricos
        return df
    else:
        print(f"⚠️ Error obteniendo datos para {symbol} con la función {function}")
        return None

# 🔹 Ejemplo de Uso
symbol = "AMZN"  # Acción a consultar
function = "TIME_SERIES_DAILY_ADJUSTED"  # Tipo de datos
interval = "60min"  # Solo para datos intradía
outputsize = "full"  # "compact" para menos datos, "full" para todos los datos históricos

df = get_stock_data(symbol, function, interval, outputsize)

# 🔹 Mostrar los primeros datos obtenidos
if df is not None:
    print(df.head())
