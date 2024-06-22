# CSNN_self_driving


**Keywords: Convolutional Spiking Neural Networks (CSNN), Self-driving, Sustainable Neural Network Training**

Los avances en Inteligencia Artificial y Machine Learning están transformando diversos sectores de la economía, incluido el transporte, particularmente en la conducción autónoma. No obstante, el elevado consumo de energía requerido para el entrenamiento y procesamiento de muchos de estos algoritmos plantea desafíos significativos, especialmente en el contexto del calentamiento global y la sostenibilidad ambiental. Una posible solución radica en adoptar un enfoque biológicamente plausible inspirado en el cerebro, mediante el uso de Redes Neuronales de Impulsos (SNNs, por sus siglas en inglés), que ofrecen un consumo energético más eficiente y la capacidad de aprovechar la dimensión temporal, entre otros beneficios.

Este trabajo presenta el entrenamiento de una Red Neuronal Convolucional de Impulsos (CSNN, por sus siglas en inglés) utilizando técnicas de aprendizaje profundo para la conducción autónoma de vehículos, mientras se mide la ecoeficiencia del proceso de entrenamiento. Para ello, se emplean scripts de Python, las librerías snnTorch y CodeCarbon, entre otras, así como el simulador de conducción autónoma de Udacity. Se transforman los datos de conducción convencional en un formato compatible con CSNN mediante modulación delta, y se compara el rendimiento y la sostenibilidad de la CSNN con una CNN convencional.

Los resultados obtenidos contribuirán al conocimiento sobre la aplicación de SNNs en la conducción autónoma y su potencial para reducir el impacto ambiental de los algoritmos de aprendizaje profundo. 

# Contenido

1. Diagrama del proyecto
2. Recursos
3. Objetivo
4. Fuente de datos 
5. Condificando datos convenicionales en impulsos con modulación Delta con la librería snnTorch
6. Implementando PilotNet en CSNN
7. Entrenando la CSNN
8. Midiendo la Ecoeficiencia del entrenamiento
9. Resultados

## 1. Diagrama del proyecto

A continuación, se presenta un diagrama general del proyecto:

![Diagrama_proyecto](Images/Diagrama_proyecto.svg)

## 2. Recursos

El conjunto de imágenes que se utilizan como insumo para el proyecto, así como el conjunto de datos codificado en impulsos (Spikes) se almacenan en [Google Drive](https://drive.google.com) en formato [HDF5](https://docs.h5py.org/en/stable/index.html).

Este proyecto utiliza scripts de [Python](https://www.python.org/) que corren sobre [Google Colab](https://colab.research.google.com/) y para probar el modelo obtenido se realiza una conexión local con el simulador de conducción autónoma de [Udacity](https://github.com/udacity/self-driving-car-sim) creando un entorno de ejecución virtual utilizando el Prompt de [Anaconda](https://www.anaconda.com/).

La implementación de las redes neuronales de impulsos (SNNs) y particularmente de las CSNNs se realiza con la librería snnTorch [snnTorch](https://snntorch.readthedocs.io/en/latest/index.html#), en GitHub se encuentra disponible en este [enlace](https://github.com/jeshraghian/snntorch). El contenido publicado por los autores y su trabajo ha sido fundamental en este proyecto.

La arquitectura de la CSNNs corresponde a una adaptación de [PilotNet](https://github.com/lhzlhz/PilotNet).

La ecoeficiencia del proceso (energía consumida y emisiones generadas) se mide haciendo uso de la librería [CodeCarbon](https://codecarbon.io/).

[ChatGPT](https://chatgpt.com/) se utilizó como asistente en la generación de código con ejemplos, optimización, refactorización y depuración.

Por otra parte, también es importante mencionar el artículo que sirve como referencia e inspiración para este trabajo:  Martínez, F. S., Parada, R., & Casas-Roma, J. (2023). CO2 impact on convolutional network model training for autonomous driving through behavioral cloning. Advanced Engineering Informatics, 56, 101968. [https://doi.org/10.1016/j.aei.2023.101968](https://doi.org/10.1016/j.aei.2023.101968) 



