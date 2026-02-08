import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.simple_model import DeliveryNet
from src.utils.dataset import DeliveryDataset

def train_lab():
    # 1. Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Ejecutando en: {device} ---")

    # 2. Cargar Datos
    dataset = DeliveryDataset("data/raw/delivery_data.csv")
    # El DataLoader maneja el barajado (shuffle) y grupos (batches)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 3. Inicializar Red (4 entradas: dist, hora, rush, clima)
    model = DeliveryNet(input_dim=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 4. Bucle de Entrenamiento Minimalista (10 épocas)
    model.train()
    for epoch in range(10):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        print(f"Época {epoch+1} | Error (Loss): {loss.item():.4f}")

if __name__ == "__main__":
    train_lab()