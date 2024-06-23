import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn  # Importar la biblioteca snntorch
from snntorch import spikegen, surrogate
import socketio
import eventlet
from flask import Flask
from io import BytesIO
import base64
from PIL import Image
import numpy as np
import cv2

# Definir la arquitectura CSNN basada en PilotNet con 21 salidas
class CSNNPilotNet(nn.Module):
    def __init__(self, beta=0.9):
        super(CSNNPilotNet, self).__init__()
        # Definir capas convolucionales
        self.conv1 = nn.Conv2d(1, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        
        # Definir capas LIF con el parámetro beta
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.lif4 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        
        # Definir capas completamente conectadas
        self.fc1 = nn.Linear(self._get_conv_output((1, 1, 50, 160)), 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 21)

    # Función para obtener la salida de las capas convolucionales
    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.randn(*shape)
            output = self.conv1(input)
            output, _ = self.lif1(output, self.lif1.init_leaky())
            output = self.conv2(output)
            output, _ = self.lif2(output, self.lif2.init_leaky())
            output = self.conv3(output)
            output, _ = self.lif3(output, self.lif3.init_leaky())
            output = self.conv4(output)
            output, _ = self.lif4(output, self.lif4.init_leaky())
            output = output.view(output.size(0), -1)
            n_size = output.size(1)
        return n_size

    # Definir el paso adelante (forward) del modelo
    def forward(self, x, mem1, mem2, mem3, mem4):
        x = self.conv1(x)
        x, mem1 = self.lif1(x, mem1)
        x = self.conv2(x)
        x, mem2 = self.lif2(x, mem2)
        x = self.conv3(x)
        x, mem3 = self.lif3(x, mem3)
        x = self.conv4(x)
        x, mem4 = self.lif4(x, mem4)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x, mem1, mem2, mem3, mem4

# Función para preprocesar la imagen para el modelo CSNN
def img_preprocess(img):
    # Recortar la imagen para eliminar características innecesarias
    img = img[40:140, :, :]
    # Convertir la imagen a escala de grises
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Aplicar desenfoque gaussiano
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # Redimensionar la imagen
    img = cv2.resize(img, (160, 50), interpolation=cv2.INTER_AREA)
    # Normalizar la imagen
    img = (img - 128.) / 128.
    return img

# Codificación Delta usando snnTorch
def encode_delta(image1, image2):
    # Convertir las imágenes a tensores de PyTorch si no lo son ya
    image1 = torch.tensor(image1, dtype=torch.float32)
    image2 = torch.tensor(image2, dtype=torch.float32)
    # Aplicar la codificación delta
    delta_encoded = spikegen.delta(image1, image2)
    return delta_encoded

# Convertir el índice de clase en un valor entre -1 y 1
def convert_to_continuous(value, num_classes=21):
    step_size = 2 / (num_classes - 1)
    return -1 + value * step_size

# Predecir el ángulo de giro y convertir la clase a un valor continuo entre -1 y 1
def predict_continuous(model, delta_images):
    model.eval()
    with torch.no_grad():
        # Inicializar las memorias de las capas LIF
        mem1, mem2, mem3, mem4 = model.lif1.init_leaky(), model.lif2.init_leaky(), model.lif3.init_leaky(), model.lif4.init_leaky()
        # Pasar las imágenes codificadas delta a través del modelo
        outputs, mem1, mem2, mem3, mem4 = model(delta_images, mem1, mem2, mem3, mem4)
        # Calcular las probabilidades de cada clase
        probabilidades = F.softmax(outputs, dim=-1)
        # Obtener la clase con la mayor probabilidad
        pred_class = torch.argmax(probabilidades, dim=-1)
        # Convertir la clase a un valor continuo
        continuous_value = convert_to_continuous(pred_class)
    return continuous_value

# Inicializar el modelo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CSNNPilotNet(beta=0.9).to(device)
model.load_state_dict(torch.load('best_model_20240618_075633_256_lr-3_200ep.pth', map_location=device))
model.eval()

# Configuración del servidor Flask y SocketIO
sio = socketio.Server()
app = Flask(__name__)

previous_image = None

# Rango de velocidad deseado
MIN_SPEED = 10
MAX_SPEED = 30
speed = 0

# Función de telemetría, maneja los datos recibidos del simulador
@sio.on('telemetry')
def telemetry(sid, data):
    global previous_image, speed
    if data:
        # Obtener la imagen del simulador
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        
        # Preprocesar la imagen actual
        current_image = img_preprocess(image)

        if previous_image is not None:
            # Codificar la imagen actual y la imagen previa usando delta
            delta_encoded = encode_delta(previous_image, current_image)
            delta_encoded = delta_encoded.unsqueeze(0).unsqueeze(0).to(device)

            # Predecir el ángulo de giro
            predicted_value = predict_continuous(model, delta_encoded)
            steering_angle = predicted_value.item()

            # Obtener la velocidad actual del simulador
            speed = float(data["speed"])

            # Ajustar el acelerador para mantener la velocidad en el rango deseado
            throttle = 0.6 if speed < MIN_SPEED else 0.0 if speed > MAX_SPEED else 0.3

            # Enviar comandos al simulador
            send_control(steering_angle, throttle)

            # Imprimir en consola los comandos enviados
            print(f"Steering Angle: {steering_angle}, Throttle: {throttle}, Speed: {speed}")

        # Guardar la imagen actual como la imagen previa para la siguiente iteración
        previous_image = current_image

# Función para enviar los comandos de control al simulador
def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={'steering_angle': str(steering_angle), 'throttle': str(throttle)},
        skip_sid=True)

# Ejecutar la aplicación web
if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 4567)), app)
