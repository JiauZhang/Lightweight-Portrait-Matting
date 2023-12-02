import torch
from models.model import LiPM

model = LiPM()
t = torch.randn(2, 3, 512, 512)
o = model(t)
print(o.shape)