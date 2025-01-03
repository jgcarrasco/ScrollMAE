# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import List
from timm.models import register_model

import os
import sys
# Add the parent directory of the script to sys.path (go up three levels)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Append the root directory (where dataset.py is located)
sys.path.append(root_dir)
from model import ResNet3D, generate_model


class YourConvNet(nn.Module):
    """
    This is a template for your custom ConvNet.
    It is required to implement the following three functions: `get_downsample_ratio`, `get_feature_map_channels`, `forward`.
    You can refer to the implementations in `pretrain\models\resnet.py` for an example.
    """
    
    def get_downsample_ratio(self) -> int:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).
        
        :return: the TOTAL downsample ratio of the ConvNet.
        E.g., for a ResNet-50, this should return 32.
        """
        return NotImplementedError
    
    def get_feature_map_channels(self) -> List[int]:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).
        
        :return: a list of the number of channels of each feature map.
        E.g., for a ResNet-50, this should return [256, 512, 1024, 2048].
        """
        raise NotImplementedError
    
    def forward(self, inp_bchw: torch.Tensor, hierarchical=False):
        """
        The forward with `hierarchical=True` would ONLY be used in `SparseEncoder.forward` (see `pretrain/encoder.py`).
        
        :param inp_bchw: input image tensor, shape: (batch_size, channels, height, width).
        :param hierarchical: return the logits (not hierarchical), or the feature maps (hierarchical).
        :return:
            - hierarchical == False: return the logits of the classification task, shape: (batch_size, num_classes).
            - hierarchical == True: return a list of all feature maps, which should have the same length as the return value of `get_feature_map_channels`.
              E.g., for a ResNet-50, it should return a list [1st_feat_map, 2nd_feat_map, 3rd_feat_map, 4th_feat_map].
                    for an input size of 224, the shapes are [(B, 256, 56, 56), (B, 512, 28, 28), (B, 1024, 14, 14), (B, 2048, 7, 7)]
        """
        raise NotImplementedError


@register_model
def your_convnet_small(pretrained=False, **kwargs):
    raise NotImplementedError
    return YourConvNet(**kwargs)


def get_downsample_ratio(self: ResNet3D) -> int:
    return 32

def get_feature_map_channels(self: ResNet3D) -> List[int]:
    # `self.feature_info` is maintained by `timm`
    return [256, 512, 1024, 2048]

def forward(self, x, hierarchical=False):
    if len(x.shape) < 5:
        x = x.unsqueeze(1)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    if not self.no_max_pool:
        x = self.maxpool(x)

    x1 = self.layer1(x)
    x2 = self.layer2(x1)
    x3 = self.layer3(x2)
    x4 = self.layer4(x3)
    if hierarchical:
        return [x1.max(dim=2)[0],x2.max(dim=2)[0],x3.max(dim=2)[0],x4.max(dim=2)[0]]
    else:
        x = self.avgpool(x4)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

ResNet3D.get_downsample_ratio = get_downsample_ratio
ResNet3D.get_feature_map_channels = get_feature_map_channels
ResNet3D.forward = forward

@register_model
def resnet_3d_50(pretrained=False, **kwargs):
    return generate_model(model_depth=50, n_input_channels=1, n_classes=1, **kwargs)

@torch.no_grad()
def convnet_test():
    from timm.models import create_model
    cnn = create_model('resnet_3d_50')
    print('get_downsample_ratio:', cnn.get_downsample_ratio())
    print('get_feature_map_channels:', cnn.get_feature_map_channels())
    
    downsample_ratio = cnn.get_downsample_ratio()
    feature_map_channels = cnn.get_feature_map_channels()
    
    # check the forward function
    B, C, H, W = 4, 3, 224, 224
    inp = torch.rand(B, C, H, W)
    feats = cnn(inp, hierarchical=True)
    assert isinstance(feats, list)
    assert len(feats) == len(feature_map_channels)
    print([tuple(t.shape) for t in feats])
    
    # check the downsample ratio
    feats = cnn(inp, hierarchical=True)
    assert feats[-1].shape[-2] == H // downsample_ratio
    assert feats[-1].shape[-1] == W // downsample_ratio
    
    # check the channel number
    for feat, ch in zip(feats, feature_map_channels):
        assert feat.ndim == 4
        assert feat.shape[1] == ch


if __name__ == '__main__':
    convnet_test()
