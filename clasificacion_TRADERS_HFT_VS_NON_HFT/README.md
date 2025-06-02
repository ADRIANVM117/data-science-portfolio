#  Clasificación de Traders HFT vs NON-HFT con LDA

Este proyecto aplica Machine Learning para predecir si un trader es **High-Frequency Trader (HFT)** o **NON-HFT** usando un dataset financiero real.
## <b> Puedes revisar el codigo en GoogleColab </b>
[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ADRIANVM117/data-science-portfolio/blob/main/clasificacion_TRADERS_HFT_VS_NON_HFT/clasificacion_TRADERS.ipynb)

## 📂 Pasos del proyecto

1. **Exploración de Datos (EDA)**  
   - Se analizaron variables con Sweetviz y AutoViz.
   - Se detectaron outliers, nulos y variables poco útiles.

2. **Limpieza y Preprocesamiento**  
   - Imputación de valores faltantes con mediana.
   - Capping y log-transform en variables con outliers.
   - Codificación de variables categóricas (`Trader`).
   - Reducción de dimensionalidad con **PCA** (80% de varianza explicada).

3. **Modelado con LDA**  
   - Entrenamiento del modelo LDA (Linear Discriminant Analysis).
   - Evaluación con accuracy, F1-score, matriz de confusión y curva ROC.

##  Resultados

- **Accuracy:** 94%
- **AUC (ROC):** 0.988
- Buen equilibrio entre precisión y recall para ambas clases (HFT y NON-HFT).

##  ¿Por qué LDA?

Elegimos LDA por ser un modelo interpretativo, eficiente y adecuado para datos linealmente separables.  
Si se necesitara más potencia, podríamos usar Random Forest o XGBoost.
