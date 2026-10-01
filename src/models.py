import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class GWNConv2d(nn.Conv2d):
    """Ghost Weight Normalization for Conv2d layers."""
    def __init__(self, *args, ghost_groups=2, eps=1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.ghost_groups = max(1, ghost_groups)
        self.eps = eps

    def forward(self, x):
        w = self.weight
        out_channels = w.shape[0]
        groups = min(self.ghost_groups, out_channels)

        pad = (-out_channels) % groups
        if pad:
            w = F.pad(w, (0, 0, 0, 0, 0, 0, 0, pad))

        shape = w.shape
        w = w.reshape(groups, shape[0] // groups, -1)
        w = w / torch.sqrt((w ** 2).sum(dim=-1, keepdim=True) + self.eps)
        w = w.reshape(shape)[:out_channels]

        return F.conv2d(
            x, w, self.bias, self.stride,
            self.padding, self.dilation, self.groups
        )


class GWNLinear(nn.Linear):
    """Ghost Weight Normalization for Linear layers."""
    def __init__(self, *args, ghost_groups=2, eps=1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.ghost_groups = max(1, ghost_groups)
        self.eps = eps

    def forward(self, x):
        w = self.weight
        out_features = w.shape[0]
        groups = min(self.ghost_groups, out_features)

        pad = (-out_features) % groups
        if pad:
            w = F.pad(w, (0, 0, 0, pad))

        shape = w.shape
        w = w.reshape(groups, shape[0] // groups, -1)
        w = w / torch.sqrt((w ** 2).sum(dim=-1, keepdim=True) + self.eps)
        w = w.reshape(shape)[:out_features]

        return F.linear(x, w, self.bias)


def apply_gwn(module, ghost_groups=2):
    """Recursively replaces Conv2d and Linear layers with GWN variants."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            new = GWNConv2d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
                ghost_groups=ghost_groups,
            )
            new.load_state_dict(child.state_dict())
            setattr(module, name, new)

        elif isinstance(child, nn.Linear):
            new = GWNLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                ghost_groups=ghost_groups,
            )
            new.load_state_dict(child.state_dict())
            setattr(module, name, new)

        else:
            apply_gwn(child, ghost_groups)

    return module


def build_model(num_classes=3, use_gwn=False, ghost_groups=2, dropout=0.5):
    """Builds ResNet50 baseline or GWN-enhanced model."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes)
    )

    if use_gwn:
        model = apply_gwn(model, ghost_groups)

    return model
