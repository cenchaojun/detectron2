import torch

a = torch.arange(0,6)
a = a.view(2,3)
print(a.shape)
print(a)
a = a.unsqueeze(1)
print(a.shape)
print(a)

