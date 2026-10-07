# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn.functional as F


""" 
3D lOSS Calculations
"""


class CrossEntropy:
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs["idk"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        log_p = (pred_softmax + 1e-10).log()
        target_class = weak_target.max(dim=1).indices
        weight = torch.zeros(pred_softmax.shape[1], device=pred_softmax.device)
        weight[self.idk] = 1.0
        return F.nll_loss(log_p, target_class, weight=weight)


class DiceLoss:
    def __init__(self, *, idk, is_3d=True, **kwargs):
        self.idk = idk
        self.dims = (2, 3, 4) if is_3d else (2, 3)
        print(f"Initialized {self.__class__.__name__} with idk={idk}, is_3d={is_3d}")

    def __call__(self, pred_softmax, weak_target):
        # assert pred_softmax.shape == weak_target.shape
        # assert simplex(pred_softmax)
        # assert sset(weak_target, [0, 1])

        p = pred_softmax[:, self.idk, ...]
        g = weak_target[:, self.idk, ...].float()

        intersection = (p * g).sum(dim=self.dims)
        union = p.sum(dim=self.dims) + g.sum(dim=self.dims)

        dice_score = (2 * intersection + 1e-10) / (union + 1e-10)
        loss = 1 - dice_score.mean()
        return loss


class CrossEntropyPlusDice:
    def __init__(self, *, ce_idk, dice_idk, is_3d=True, dice_weight=1.0):
        self.ce = CrossEntropy(idk=ce_idk)
        self.dice = DiceLoss(idk=dice_idk, is_3d=is_3d)
        self.dice_weight = dice_weight

    def __call__(self, pred_softmax, weak_target):
        return self.ce(pred_softmax, weak_target) + self.dice_weight * self.dice(
            pred_softmax, weak_target
        )
