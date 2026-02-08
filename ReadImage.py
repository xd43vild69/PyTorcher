import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image

# 1. Configurar el dispositivo (Aprovechando tu M4 Pro)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 2. Cargar el modelo pre-entrenado y las transformaciones necesarias
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights).to(device)
model.eval()  # Modo evaluación (importante)

# 3. Cargar y preparar tu imagen
preprocess = weights.transforms()
img = Image.open("tu_imagen.jpg").convert("RGB") # Cambia por tu archivo
img_transformed = preprocess(img).unsqueeze(0).to(device)

# 4. Inferencia
with torch.no_grad():
    prediction = model(img_transformed)

# 5. Ver resultados
print(prediction)