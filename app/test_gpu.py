import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA disponível:", torch.cuda.is_available())
print("Device usado:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

x = torch.randn(3, 3).to(device)
print("Tensor está em:", x.device)
print(x)