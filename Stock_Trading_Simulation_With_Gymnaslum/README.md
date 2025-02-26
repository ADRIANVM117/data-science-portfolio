# Stock Trading Simulation with Gymnasium

Este proyecto implementa un entorno de simulación para el trading de acciones utilizando la biblioteca Gymnasium. Se emplea la API de Alpha Vantage para obtener datos históricos de activos financieros como Amazon (AMZN), Apple (AAPL) y General Electric (GE).

📌 Características del Proyecto

Simulación de estrategias de trading en un entorno de refuerzo.

Integración con Alpha Vantage para datos de mercado en tiempo real.

Implementación de un entorno personalizado de Gymnasium para entrenar modelos de aprendizaje por refuerzo.

Soporte para múltiples activos financieros (AMZN, AAPL, GE).

## 📂 Estructura del Proyecto
📁 stock_trading_simulation_with_Gymnasium
1. 📂 data/                # Datos de precios de acciones descargados desde Alpha Vantage2.
2.  📂 environment/         # Definición del entorno de Gymnasium
3. 📂 models/              # Modelos de aprendizaje por refuerzo
4. 📂 notebooks/           # Jupyter Notebooks con pruebas y análisis
5. 📜 requirements.txt     # Dependencias del proyecto
6. 📜 Documentación.md            # Documentación del proyecto
7.  📜 train.py             # Script principal para entrenar el agente
8.  📜 evaluate.py          # Evaluación del rendimiento del agente

## 🔧 Instalación
### 1️⃣ Clonar el repositorio

### 2️⃣ Crear un entorno virtual (Recomendado)
`python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate`


  # En Windows: venv\Scripts\activate

### 3️⃣ Instalar dependencias
`pip install -r requirements.txt`

## 📊 Uso del Proyecto

### 🔹 Obtener datos de Alpha Vantage
Debes generar una API Key de Alpha Vantage y almacenarla en un archivo .env:}
`API_KEY=tu_api_key
ticker=AMZN,AAPL,GE`

Ejecuta el script para descargar los datos:
python fetch_data.py

🔹 Entrenar el agente
python train.py

🔹 Evaluar el modelo
python evaluate.py


### 📌 Dependencias

1. Python 3.8+

2. gymnasium

3. pandas

4. numpy

5. matplotlib

6. stable-baselines3

7. alpha_vantage

8. dotenv

🚀 Mejoras futuras

Soporte para más activos financieros.

Implementación de estrategias más avanzadas.

Uso de modelos de Deep Reinforcement Learning (DRL).

