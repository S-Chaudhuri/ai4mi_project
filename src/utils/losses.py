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

class TverskyLoss:
    def __init__(self, *, idk, is_3d=True, alpha=0.3, beta=0.7, smooth=1e-6, **kwargs):
        self.idk = idk
        self.dims = (2, 3, 4) if is_3d else (2, 3)
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        print(f"Initialized {self.__class__.__name__} with idk={idk}, is_3d={is_3d}")

    def __call__(self, pred_softmax, weak_target):
        p = pred_softmax[:, self.idk, ...].float()
        g = weak_target[:, self.idk, ...].float()

        tp = (p * g).sum(dim=self.dims)
        fp = (p * (1 - g)).sum(dim=self.dims)
        fn = ((1 - p) * g).sum(dim=self.dims)

        tversky_score = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        loss = 1 - tversky_score.mean()
        return loss

class DeepSupervisionLoss:
    def __init__(self, *, loss):
        self.loss = loss
        print(f"Initialized {self.__class__.__name__}")

    def __call__(self, pred_softmax, weak_target):
        if not isinstance(pred_softmax, (list, tuple)):
            return self.loss(pred_softmax, weak_target)

        weights = [0.5 ** i for i in range(len(pred_softmax))]
        loss = 0
        for p, weight in zip(pred_softmax, weights):
            g = weak_target
            if p.shape[2:] != g.shape[2:]:
                g = F.interpolate(
                    g.float(), size=p.shape[2:], mode="nearest"
                ).to(weak_target.dtype)
            loss = loss + weight * self.loss(p, g)
        return loss / sum(weights)
class SkeletonRecallLoss:
    def __init__(self, *args, **kwargs):
        print(f"Initialized dummy {self.__class__.__name__}")
    def __call__(self, *args, **kwargs):
        return torch.tensor(0.0)


class SkeletonRecallLoss:
    """Skeleton Recall loss (Kirchhoff et al., ECCV 2024) for thin tubular
    structures (esophagus, trachea).

    The GT centreline of each class is precomputed in the dataset (a slightly
    dilated skeleton, see utils.dataset.tubed_skeleton). The loss is
    1 - the fraction of that centreline the prediction covers:

        recall_k = sum(p_k * skel_k) / sum(skel_k)

    so a break anywhere along the tube is penalised, which Dice barely notices.
    Only the predicted probabilities are differentiated; the skeleton is a
    fixed target, so it costs almost nothing per step.
    """

    def __init__(self, *, classes=(1, 3)):
        self.classes = tuple(classes)
        print(f"Initialized {self.__class__.__name__} with classes={self.classes}")

    def __call__(self, pred_softmax, skel):
        """pred_softmax: (B, K, D, H, W) probabilities, the full-resolution output.
        skel: (B, D, H, W) integer map, voxel = class id on that class's
        centreline, 0 elsewhere."""
        if isinstance(pred_softmax, (list, tuple)):  # deep supervision: full res only
            pred_softmax = pred_softmax[0]
        dims = tuple(range(1, skel.ndim))
        losses = []
        for k in self.classes:
            s = (skel == k).float()
            n = s.sum(dim=dims)
            present = n > 0  # boxes without this organ's centreline don't count
            if present.any():
                covered = (pred_softmax[:, k] * s).sum(dim=dims)
                losses.append(1 - (covered[present] / n[present]).mean())
        if not losses:  # no centreline in the whole batch: zero loss, keeps the graph
            return pred_softmax.sum() * 0.0
        return torch.stack(losses).mean()