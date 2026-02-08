import torch
import torch.nn as nn
from src.models.simple_model import DeliveryNet

def train_plan():
    # 1. Configuración de Hardware (Validando tu GPU NVIDIA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Iniciando Plan de Trabajo en: {device} ---")

    # 2. Hiperparámetros Básicos
    input_size = 5  # Ejemplo: hora, distancia, clima, etc.
    learning_rate = 0.001
    
    # 3. Inicializar el Modelo
    model = DeliveryNet(input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss() # Para predecir tiempo (regresión)

    print("Modelo cargado y listo para el entrenamiento.")
    # TODO: Implementar el bucle de entrenamiento (Train Loop)

if __name__ == "__main__":
    train_plan()