# CSNN_self_driving


**Keywords: Convolutional Spiking Neural Networks (CSNN), Self-driving, Sustainable Neural Network Training**

Los avances en Inteligencia Artificial y Machine Learning están transformando diversos sectores de la economía, incluido el transporte, particularmente en la conducción autónoma. No obstante, el elevado consumo de energía requerido para el entrenamiento y procesamiento de muchos de estos algoritmos plantea desafíos significativos, especialmente en el contexto del calentamiento global y la sostenibilidad ambiental. Una posible solución radica en adoptar un enfoque biológicamente plausible inspirado en el cerebro, mediante el uso de Redes Neuronales de Impulsos (SNNs, por sus siglas en inglés), que ofrecen un consumo energético más eficiente y la capacidad de aprovechar la dimensión temporal, entre otros beneficios.

Este trabajo presenta el entrenamiento de una Red Neuronal Convolucional de Impulsos (CSNN, por sus siglas en inglés) utilizando técnicas de aprendizaje profundo para la conducción autónoma de vehículos, mientras se mide la ecoeficiencia del proceso de entrenamiento. Para ello, se emplean scripts de Python, las librerías snnTorch y CodeCarbon, entre otras, así como el simulador de conducción autónoma de Udacity. Se transforman los datos de conducción convencional en un formato compatible con CSNN mediante modulación delta, y se compara el rendimiento y la sostenibilidad de la CSNN con una CNN convencional.

Los resultados obtenidos contribuirán al conocimiento sobre la aplicación de SNNs en la conducción autónoma y su potencial para reducir el impacto ambiental de los algoritmos de aprendizaje profundo.

# Contenido

1. Introducción
2. Recursos
3. Objetivos
4. Fuente de datos 
5. Condificando datos convenicionales en impulsos con modulación Delta con la librería snnTorch
6. Implementando PilotNet en CSNN
7. Entrenando la CSNN
8. Midiendo la Ecoeficiencia del entrenamiento
9. Resultados
