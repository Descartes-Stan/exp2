import torch
x = torch.randn(2, 2, dtype=torch.double)
print(x)
x = torch.where(x > 0.2, x, 0.)
print(x)