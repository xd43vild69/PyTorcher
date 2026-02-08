import torch
print(f"¿GPU disponible?: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Trabajando en: {torch.cuda.get_device_name(0)}")