import torch
import torchvision
import matplotlib.pyplot as plt
import requests
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from io import BytesIO

def run_detector():
    # 1. Configuración del dispositivo (Optimizado para NVIDIA/CUDA)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"--- Ejecutando en GPU NVIDIA: {torch.cuda.get_device_name(0)} ---")
    else:
        device = torch.device("cpu")
        print("--- CUDA no disponible, ejecutando en CPU ---")

    # 2. Cargar modelo y pesos (Usando la API moderna de Weights)
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights).to(device)
    model.eval()

    # 3. Obtener imagen (URL de prueba)
    url = "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog2.jpg"
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        print("--- Imagen cargada correctamente ---")
    except Exception as e:
        print(f"Error cargando imagen: {e}")
        return

    # 4. Preprocesamiento (Asegurando que los tensores vayan al dispositivo correcto)
    preprocess = weights.transforms()
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    # 5. Inferencia (Sin cálculo de gradientes para ahorrar memoria de video)
    with torch.no_grad():
        prediction = model(img_tensor)[0]

    # 6. Post-procesamiento (Filtro de confianza > 0.8)
    # Movemos los resultados a CPU para poder visualizarlos con Matplotlib/PIL
    scores = prediction['scores'].cpu()
    mask = scores > 0.8
    
    boxes = prediction['boxes'].cpu()[mask]
    labels = [weights.meta["categories"][i] for i in prediction['labels'].cpu()[mask]]
    confidences = scores[mask]
    # --- AÑADE ESTO PARA VERLO EN CONSOLA ---
    if len(labels) > 0:
        for label, conf in zip(labels, confidences):
            print(f"¡Objeto detectado!: {label} (Confianza: {conf:.2%})")
    else:
        print("No se detectó ningún objeto con suficiente confianza.")
    # ----------------------------------------
    
    # 7. Visualización
    img_int = torchvision.transforms.functional.pil_to_tensor(img)
    
    # Dibujamos las cajas (draw_bounding_boxes espera tensores en CPU y tipo uint8)
    result_tensor = draw_bounding_boxes(
        img_int, 
        boxes=boxes, 
        labels=labels, 
        colors="lime", 
        width=3
    )

    # Mostrar resultado
    plt.figure(figsize=(12, 8))
    plt.imshow(result_tensor.permute(1, 2, 0))
    plt.axis("off")
    plt.title(f"Detección en NVIDIA - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    plt.show()
    
    print(f"--- Detección finalizada: {len(labels)} objetos encontrados ---")

if __name__ == "__main__":
    run_detector()