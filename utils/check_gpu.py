import torch
print(f"Versão do PyTorch: {torch.__version__}")
print(f"CUDA está disponível? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Placa detectada: {torch.cuda.get_device_name(0)}")