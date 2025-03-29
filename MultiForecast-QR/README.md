# MultiForecast-QR

- **Multi:** Representa el aspecto multivariado y múltiples pasos de predicción
- **Forecast:** Indica el objetivo de predicción de series temporales
- **QR:** Hace referencia a la regresión cuantil (Quantile Regression)

## Objetivo del Proyecto

Desarrollar un marco de trabajo para la regresión de series temporales que:

- Implementa predicción probabilística de múltiples pasos
- Utiliza redes neuronales sequence-to-sequence (estructuras recurrentes y convolucionales)
- Incorpora regresión cuantil no paramétrica
- Optimiza predicciones directas de múltiples horizontes

### Especificaciones Técnicas

- Horizonte de predicción: 24 horas
- Ventana de observación: 168 horas (1 semana)
- Tamaño de muestra: 100 muestras por casa
- Variables adicionales a crear: 3