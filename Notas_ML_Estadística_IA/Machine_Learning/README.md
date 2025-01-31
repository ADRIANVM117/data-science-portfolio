# Notas y Recursos de Machine Learning

¡Bienvenido a la carpeta de **Machine Learning**! Aquí encontrarás notas completas, tutoriales, ejemplos de código y recursos relacionados con el campo de **Machine Learning (ML)**. Este repositorio está diseñado para proporcionar tanto comprensión teórica como implementación práctica de los conceptos de Machine Learning.

## Índice

1. [Introducción al Machine Learning](#introducción-al-machine-learning)
2. [Aprendizaje Supervisado](#aprendizaje-supervisado)
    - [Regresión Lineal](#regresión-lineal)
    - [Regresión Logística](#regresión-logística)
    - [Árboles de Decisión](#árboles-de-decisión)
    - [Bosques Aleatorios](#bosques-aleatorios)
    - [Máquinas de Soporte Vectorial (SVM)](#máquinas-de-soporte-vectorial-svm)
    - [K-Vecinos Más Cercanos (KNN)](#k-vecinos-más-cercanos-knn)
3. [Aprendizaje No Supervisado](#aprendizaje-no-supervisado)
    - [Clustering](#clustering)
    - [Reducción de Dimensionalidad](#reducción-de-dimensionalidad)
4. [Evaluación de Modelos](#evaluación-de-modelos)
    - [Validación Cruzada](#validación-cruzada)
    - [Métricas de Rendimiento](#métricas-de-rendimiento)
5. [Temas Avanzados](#temas-avanzados)
    - [Redes Neuronales](#redes-neuronales)
    - [Deep Learning](#deep-learning)
    - [Reinforcement Learning](#reinforcement-learning)
6. [Proyectos Prácticos de ML](#proyectos-prácticos-de-ml)
7. [Referencias y Lecturas Adicionales](#referencias-y-lecturas-adicionales)

## Introducción al Machine Learning

Machine Learning es un subcampo de la Inteligencia Artificial (IA) que se enfoca en construir algoritmos que pueden aprender de los datos y hacer predicciones. A diferencia de los algoritmos tradicionales, los modelos de ML mejoran su desempeño con el tiempo sin una programación explícita. Esta carpeta cubre técnicas de **aprendizaje supervisado** y **no supervisado**, junto con guías prácticas sobre cómo implementarlas.

## Aprendizaje Supervisado

El aprendizaje supervisado es un tipo de Machine Learning en el cual el algoritmo se entrena con datos etiquetados, es decir, los datos de entrada están asociados a la salida correcta. El objetivo del modelo es aprender el mapeo entre la entrada y la salida para hacer predicciones sobre datos no vistos.

### Regresión Lineal
- **Objetivo**: Predecir una variable continua a partir de una o más características de entrada.
- **Concepto clave**: Relación lineal entre las variables de entrada y la salida.
- **Aplicaciones**: Predicción de precios de casas, precios de acciones, etc.

### Regresión Logística
- **Objetivo**: Predecir resultados binarios (0 o 1) a partir de características de entrada.
- **Concepto clave**: Usa la función logística para modelar probabilidades.
- **Aplicaciones**: Clasificación de correos electrónicos (spam o no spam), detección de enfermedades (sí o no).

### Árboles de Decisión
- **Objetivo**: Predecir el valor de una variable objetivo aprendiendo reglas de decisión simples.
- **Concepto clave**: Estructura tipo árbol con nodos de decisión y nodos hoja.
- **Aplicaciones**: Segmentación de clientes, aprobación de préstamos.

### Bosques Aleatorios
- **Objetivo**: Crear un conjunto de árboles de decisión para mejorar la precisión.
- **Concepto clave**: Agrega múltiples árboles de decisión para reducir el sobreajuste y aumentar la robustez.
- **Aplicaciones**: Tareas de clasificación y regresión con grandes volúmenes de datos.

### Máquinas de Soporte Vectorial (SVM)
- **Objetivo**: Clasificar datos encontrando el hiperplano óptimo que separa las distintas clases.
- **Concepto clave**: Maximiza el margen entre clases mientras minimiza los errores de clasificación.
- **Aplicaciones**: Reconocimiento de imágenes, clasificación de texto.

### K-Vecinos Más Cercanos (KNN)
- **Objetivo**: Clasificar nuevas instancias en base a la clase mayoritaria de sus vecinos más cercanos.
- **Concepto clave**: Aprendizaje basado en instancias; no se construye un modelo explícito.
- **Aplicaciones**: Sistemas de recomendación, reconocimiento de patrones.

## Aprendizaje No Supervisado

El aprendizaje no supervisado es un tipo de Machine Learning donde el algoritmo recibe datos sin etiquetas de salida. El objetivo es encontrar patrones o estructuras ocultas en los datos.

### Clustering
- **Objetivo**: Agrupar datos similares entre sí.
- **Concepto clave**: Algoritmos como K-means y DBSCAN identifican los clústeres dentro de los datos.
- **Aplicaciones**: Segmentación de mercado, detección de anomalías.

### Reducción de Dimensionalidad
- **Objetivo**: Reducir el número de características en un conjunto de datos mientras se preserva la información importante.
- **Concepto clave**: Métodos como PCA (Análisis de Componentes Principales) ayudan a encontrar las características más informativas.
- **Aplicaciones**: Visualización de datos, reducción de ruido, extracción de características.

## Evaluación de Modelos

Evaluar los modelos de Machine Learning es crucial para entender qué tan bien están funcionando y si están generalizando correctamente a nuevos datos.

### Validación Cruzada
- **Objetivo**: Evaluar el rendimiento del modelo particionando los datos en múltiples subconjuntos (pliegues).
- **Concepto clave**: Ayuda a prevenir el sobreajuste y garantiza una evaluación fiable del modelo.
- **Aplicaciones**: Ajuste de modelos, evaluación de rendimiento.

### Métricas de Rendimiento
- **Métricas clave**: 
    - Precisión
    - Precisión, Recall, F1-Score
    - Matriz de Confusión
    - Curva ROC y AUC
    - Error Cuadrático Medio (MSE), R-cuadrado para tareas de regresión.

## Temas Avanzados

A medida que Machine Learning continúa evolucionando, surgen nuevas técnicas y enfoques avanzados. Aquí se exploran algunas de las áreas más sofisticadas de ML.

### Redes Neuronales
- **Objetivo**: Construir modelos inspirados en el cerebro humano para capturar patrones complejos en los datos.
- **Concepto clave**: Redes de neuronas (nodos) conectadas por pesos.
- **Aplicaciones**: Reconocimiento de imágenes, procesamiento de lenguaje natural, sistemas de recomendación.

### Deep Learning
- **Objetivo**: Usar redes neuronales de múltiples capas para resolver tareas complejas que requieren extracción jerárquica de características.
- **Concepto clave**: Redes neuronales profundas (DNNs) para abstracciones de alto nivel.
- **Aplicaciones**: Coches autónomos, asistentes de voz, clasificación de imágenes.

### Reinforcement Learning
- **Objetivo**: Aprender estrategias para la toma de decisiones mediante ensayo y error, recibiendo recompensas o penalizaciones.
- **Concepto clave**: Los agentes interactúan con un entorno para maximizar la recompensa acumulada.
- **Aplicaciones**: Juego (p. ej., AlphaGo), robótica, vehículos autónomos.

## Proyectos Prácticos de ML

Esta sección incluye proyectos prácticos que te permitirán aplicar las técnicas de Machine Learning para resolver problemas del mundo real. Estos proyectos te ayudarán a reforzar los conceptos aprendidos en este repositorio y obtener experiencia práctica.

1. **Predicción de Precios de Viviendas**: Usando técnicas de regresión para predecir los precios de viviendas en función de varias características.
2. **Segmentación de Clientes**: Aplicando algoritmos de clustering para segmentar clientes en función de sus características.
3. **Análisis de Sentimientos**: Construyendo un modelo para analizar el sentimiento de un texto (positivo, negativo, neutral).
4. **Sistema de Recomendación**: Desarrollando un sistema para recomendar productos basados en las preferencias de los usuarios.

## Referencias y Lecturas Adicionales

Aquí tienes algunos libros clave, artículos y recursos en línea para profundizar en el aprendizaje de Machine Learning:

- **Libros**:
    - "Pattern Recognition and Machine Learning" de Christopher Bishop
    - "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" de Aurélien Géron
    - "Deep Learning" de Ian Goodfellow, Yoshua Bengio y Aaron Courville

- **Sitios web**:
    - [Kaggle](https://www.kaggle.com) - Plataforma de competiciones de datos y conjuntos de datos.
    - [Documentación de Scikit-learn](https://scikit-learn.org) - Documentación para Machine Learning en Python.
    - [Towards Data Science](https://towardsdatascience.com) - Blog con tutoriales, artículos y estudios de caso.

- **Cursos**:
    - [Coursera: Machine Learning por Andrew Ng](https://www.coursera.org/learn/machine-learning)
    - [Fast.ai: Practical Deep Learning for Coders](https://www.fast.ai)

## Conclusión

Este repositorio sirve como una guía de estudio y una referencia práctica para cualquiera que esté interesado en profundizar en **Machine Learning**. Ya seas principiante o busques mejorar tus habilidades, encontrarás una amplia gama de temas y recursos para dominar las técnicas de ML y aplicarlas a problemas del mundo real.

¡Gracias por visitar, y siéntete libre de contribuir, hacer preguntas o compartir tus pensamientos!

