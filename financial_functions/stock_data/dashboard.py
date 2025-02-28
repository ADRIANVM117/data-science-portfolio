import sys
import os
import streamlit as st

# Agregar el directorio donde está `get_stock_data.py` al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_data_utils import get_stock_data  # Importamos la función correctamente

# 🔹 Configurar la Interfaz del Dashboard
st.title("📊 Dashboard de Descarga de Datos de Alpha Vantage")

# 🔹 Selección de Acción
symbol = st.text_input("🔹 Ingresa el símbolo de la acción:", "AAPL")

# 🔹 Selección del Tipo de Datos
function = st.selectbox("🔹 Selecciona el intervalo de tiempo:", 
                        ["TIME_SERIES_INTRADAY", "TIME_SERIES_DAILY", "TIME_SERIES_DAILY_ADJUSTED", 
                         "TIME_SERIES_WEEKLY", "TIME_SERIES_WEEKLY_ADJUSTED", "TIME_SERIES_MONTHLY", 
                         "TIME_SERIES_MONTHLY_ADJUSTED"])

# 🔹 Selección del Intervalo (Solo para datos Intradía)
interval = "30min"
if function == "TIME_SERIES_INTRADAY":
    interval = st.selectbox("🔹 Selecciona el intervalo de tiempo:", ["1min", "5min", "15min", "30min", "60min"])

# 🔹 Selección del Tamaño de Datos
outputsize = st.radio("🔹 Tamaño del dataset:", ["compact", "full"])

# 🔹 Botón para descargar datos
if st.button("📥 Descargar Datos"):
    df = get_stock_data(symbol, function, interval, outputsize)
    
    if df is not None and not df.empty:
        st.success(f"✅ Datos descargados para {symbol}")
        st.dataframe(df.head())  # Mostrar tabla con datos
        st.line_chart(df["Close"])  # Mostrar gráfico de cierre

        # Botón de descarga en CSV
        csv = df.to_csv().encode('utf-8')
        st.download_button("💾 Descargar CSV", csv, f"{symbol}_data.csv", "text/csv")
    else:
        st.error(f"⚠️ No se encontraron datos para {symbol}. Verifica el ticker o la API Key.")
