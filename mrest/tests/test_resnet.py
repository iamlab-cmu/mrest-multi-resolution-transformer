import numpy as np

import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models._utils import IntermediateLayerGetter


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


def main_intermediate():
    frozen_bn = True
    if frozen_bn:
        name = 'resnet101'
        dilation = False
        backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation], pretrained=True, norm_layer=FrozenBatchNorm2d
            )
    else:
        backbone = resnet101(weights=ResNet101_Weights)
    return_interm_layers = False
    train_backbone = True

    for name, parameter in backbone.named_parameters():
        if not train_backbone or "layer2" not in name and "layer3" not in name and "layer4" not in name:
            parameter.requires_grad_(False)
    if return_interm_layers:
        return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
    else:
        return_layers = {"layer4": 0}

    body = IntermediateLayerGetter(backbone, return_layers=return_layers)
    body = body.to('cuda')
    criterion = nn.MSELoss()

    bs = 64
    img_size = (3, 224, 224)
    for _ in range(100):
        inp = torch.rand((bs,) + img_size).to('cuda')
        out = body(inp)
        target = torch.ones_like(out[0]).to('cuda')
        
        loss = criterion(out[0], target)
        print(f'loss: {loss:.4f}')


def main():
    model = resnet101(weights=ResNet101_Weights)

    train_backbone = True
    for name, parameter in model.named_parameters():
        if not train_backbone or "layer2" not in name and "layer3" not in name and "layer4" not in name:
            parameter.requires_grad_(False)

    model = model.to('cuda')
    criterion = nn.MSELoss()
    
    # NOTE: no optimizer

    bs = 64
    img_size = (3, 224, 224)
    for _ in range(100):
        inp = torch.rand((bs,) + img_size).to('cuda')
        target = torch.ones((bs, 1000)).to('cuda')
        out = model(inp)
        
        loss = criterion(out, target)
        print(f'loss: {loss:.4f}')


if __name__ == '__main__':
    # main()
    main_intermediate()