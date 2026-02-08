import torch.nn as nn

class DeliveryNet(nn.Module):
    def __init__(self, input_dim):
        super(DeliveryNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Salida única: tiempo de entrega
        )
        
    def forward(self, x):
        return self.net(x)