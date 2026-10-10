import autoroot
import torch

from src.models.MedNeXt3D import MedNeXt3D, upkern_load_weights

net = MedNeXt3D(1, 5, kernels=8)
net.init_weights()

x = torch.randn(2, 1, 32, 48, 48)
y = net(x)
assert y.shape == (2, 5, 32, 48, 48), y.shape
y.mean().backward()

# Odd sizes: strided convs round up, upsampling corrects back via interpolate
y = net(torch.randn(1, 1, 33, 35, 31))
assert y.shape == (1, 5, 33, 35, 31), y.shape

# Anisotropic: no downsampling along z in the first step
net_aniso = MedNeXt3D(1, 5, kernels=8, pool_kernels=((1, 2, 2), (2, 2, 2), (2, 2, 2)))
net_aniso.init_weights()
y = net_aniso(torch.randn(1, 1, 16, 64, 64))
assert y.shape == (1, 5, 16, 64, 64), y.shape

# Deep supervision + gradient checkpointing: list in train mode, tensor in eval mode
net_ds = MedNeXt3D(1, 5, kernels=8, deep_supervision=True, grad_checkpoint=True)
net_ds.init_weights()
net_ds.train()
outs = net_ds(torch.randn(1, 1, 32, 48, 48))
assert isinstance(outs, list) and len(outs) == 3, len(outs)
assert outs[0].shape == (1, 5, 32, 48, 48), outs[0].shape
assert outs[1].shape == (1, 5, 16, 24, 24), outs[1].shape
assert outs[2].shape == (1, 5, 8, 12, 12), outs[2].shape
sum(o.mean() for o in outs).backward()

net_ds.eval()
with torch.no_grad():
    y = net_ds(torch.randn(1, 1, 32, 48, 48))
assert isinstance(y, torch.Tensor) and y.shape == (1, 5, 32, 48, 48), type(y)

# UpKern: kernel-3 weights -> kernel-5 model
net_k5 = MedNeXt3D(1, 5, kernels=8, kernel_size=5)
upkern_load_weights(net_k5, net.state_dict())
assert net_k5.encoders[0][0].dw_conv.weight.shape[2:] == (5, 5, 5)
assert torch.equal(net_k5.final.weight, net.final.weight)  # 1x1 convs copied as-is
y = net_k5(torch.randn(1, 1, 32, 48, 48))
assert y.shape == (1, 5, 32, 48, 48), y.shape

print("OK", sum(p.numel() for p in net.parameters()), "params")
