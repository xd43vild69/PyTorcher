import pandas as pd
import torch
from torch.utils.data import Dataset

class DeliveryDataset(Dataset):
    def __init__(self, csv_path):
        # Cargamos los datos
        df = pd.read_csv(csv_path)
        
        # Separamos las características (X) del objetivo (y)
        # Tomamos todas las columnas excepto la última (tiempo_entrega_min)
        self.X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
        self.y = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Test rápido
if __name__ == "__main__":
    dataset = DeliveryDataset("/home/d13/Lab/PyTorcher/data/raw/delivery_data.csv")
    print(f"Dataset cargado con {len(dataset)} ejemplos.")
    features, target = dataset[0]
    print(f"Ejemplo de entrada: {features} | Objetivo: {target}")