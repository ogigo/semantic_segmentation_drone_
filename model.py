import torch

model=torch.load("Unet-Mobilenet.pt",map_location=torch.device("cpu"))