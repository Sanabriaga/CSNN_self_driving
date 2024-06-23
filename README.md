# CSNN_self_driving


**Keywords: Convolutional Spiking Neural Networks (CSNN), Self-driving, Sustainable Neural Network Training**

Los avances en Inteligencia Artificial y Machine Learning están transformando diversos sectores de la economía, incluido el transporte, particularmente en la conducción autónoma. No obstante, el elevado consumo de energía requerido para el entrenamiento y procesamiento de muchos de estos algoritmos plantea desafíos significativos, especialmente en el contexto del calentamiento global y la sostenibilidad ambiental. Una posible solución radica en adoptar un enfoque biológicamente plausible inspirado en el cerebro, mediante el uso de Redes Neuronales de Impulsos (SNNs, por sus siglas en inglés), que ofrecen un consumo energético más eficiente y la capacidad de aprovechar la dimensión temporal, entre otros beneficios.

Este trabajo presenta el entrenamiento de una Red Neuronal Convolucional de Impulsos (CSNN, por sus siglas en inglés) utilizando técnicas de aprendizaje profundo para la conducción autónoma de vehículos, mientras se mide la ecoeficiencia del proceso de entrenamiento. Para ello, se emplean scripts de Python, las librerías snnTorch y CodeCarbon, entre otras, así como el simulador de conducción autónoma de Udacity. Se transforman los datos de conducción convencional en un formato compatible con CSNN mediante modulación delta, y se compara el rendimiento y la sostenibilidad de la CSNN con una CNN convencional.

Los resultados obtenidos contribuirán al conocimiento sobre la aplicación de SNNs en la conducción autónoma y su potencial para reducir el impacto ambiental de los algoritmos de aprendizaje profundo. 

# Contenido

1. Diagrama del proyecto
2. Recursos
3. Objetivo
4. Condificando datos convenicionales en impulsos con modulación Delta con la librería snnTorch
5. Implementando y entrenando la CSNN (adaptación de PilotNet)
6. Midiendo la Ecoeficiencia del entrenamiento
7. Conectando el modelo con el simulador de Udacity
8. Resultados

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

## 3. Objetivo

Entrenar una red Red Neuronal Convolucional de Impulsos (CSNN) transformando un conjunto de datos convencionales de conducción autónoma y comparar los resultados en términos de rendimiento y sostenibilidad ambiental con los obtenidos en el entrenamiento de una Red Neuronal Convolucional (CNN) con el mismo conjunto de datos.

## 4. Condificando datos convenicionales en impulsos con modulación Delta con la librería snnTorch

El archivo *Optimization_Encode_Spikes_v12.ipynb* contiene el código por medio del cual se cargan, preprocesan, transforman, aumentan y codifican las imágenes en impulsos. Básicamente se cargan los datos generados por el simulador de Udacity en Google Drive y desde allí se utilizan como insumo. Al finalizar el proceso, en Google Drive se cargan dos archivos, que corresponden a los  datos de entrenamiento y validación codificados en impulsos y que se guardan en formato hdf5.

## 5. Implementando y entrenando la CSNN (adaptación de PilotNet)

El archivo *Optimization train CSNN v13.ipynb* contiene el código por medio del cual se define la arquitectura adaptada de PilotNet para construir la CSNN. Se adapta en varios aspectos, el primero es el uso de neuronas con modelo Leaky Integrate-and-Fire (LIF), el segundo es la eliminación de una capa convolucional, dejando cuatro capas convolucionales y cuatro capas completamente conectadas y finalmente la salida corresponde a 21 neuronas que codifican el valor del ángulo de giro del volante entre -1 y 1 en 21 pasos, es decir, la activación de la neurona 0 corresponde al ángulo -1, la activación de la neurona 10 al ángulo 0 y la activación de la neurona 20 al ángulo 1, con los posibles pasos intermedios entre estos valores.

## 6. Midiendo la Ecoeficiencia del entrenamiento

La librería CodeCarbon se integra con el proyecto a través de pocas líneas de código y permite estimar la cantidad de dióxido de carbono (CO2) producido por los recursos informáticos personales o en la nube utilizados para ejecutar el código.

```
tracker = EmissionsTracker(output_dir='/ubicacion_para_guardar_archivo_seguimiento', project_name=f"emissions_train_time}.csv")
tracker.start()
...Código a ejecutar
tracker.stop()
```

Y el resultado es un archivo csv con la información de la energía consumida y las emisiones generadas entre otros.
