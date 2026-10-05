import autoroot
import torch
from torch import nn

from src.models.VNet3D import VNet3D

net = VNet3D(1, 5, kernels=8)
net.init_weights()

x = torch.randn(2, 1, 36, 36, 32)
y = net(x)
assert y.shape == (2, 5, 36, 36, 32), y.shape
y.mean().backward()

# Odd sizes: downsampling floor-divides, upsampling corrects back via interpolate
y = net(torch.randn(1, 1, 33, 35, 31))
assert y.shape == (1, 5, 33, 35, 31), y.shape

# paper-original: no norm layer
net_nonorm = VNet3D(1, 5, kernels=8, norm_layer=nn.Identity)
net_nonorm.init_weights()
y = net_nonorm(torch.randn(1, 1, 36, 36, 32))
assert y.shape == (1, 5, 36, 36, 32), y.shape

print("OK", sum(p.numel() for p in net.parameters()), "params")
