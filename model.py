import torch.nn as nn 
import torch
import time
import loss 
from legacy import model as legacy_model

'''
    B = residual block & # of repeats
    ["C", chan_out, kernel, stride]
'''

CONFIG_BACKBONE = [
    ["C",32, 3, 1],
    ["C",64, 3, 2],
    ["B", 1],
    ["C",128, 3, 2],
    ["B", 2],
    ["C",256, 3, 2],
    ["B", 8],
    ["C",512, 3, 2],
    ["B", 8],
    ["C",1024, 3, 2],
    ["B", 4]
] 
CONFIG_NECK = [
    ["C",512, 1, 1],
    ["C",1024, 3, 1],
    ["S"],
    ["C",256, 1, 1],
    ["U"],
    ["C",256, 1, 1],
    ["C",512, 3, 1],
    ["S"],
    ["C",128, 1, 1],
    ["U"],
    ["C",128, 1, 1],
    ["C",256, 3, 1],
    ["S"]
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs) -> None:
        super(CNNBlock, self).__init__()
        self.bn_act = bn_act
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
    def forward(self, x):
        if self.bn_act:
            return self.leakyrelu(self.batchnorm(self.conv(x)))
        else:
            return self.conv(x)
        
class ResBlockA(nn.Module):
    def __init__(self, in_channel_1, use_residual=True, num_repeats=1) -> None:
        super(ResBlockA, self).__init__()
        self.use_residual = use_residual
        self.num_repeats  = num_repeats
        self.layers = self._make(in_channel_1)
        
    def _make(self, in_channel_1):
        layers = []
        for _ in range(self.num_repeats):
            layers.append(nn.Sequential(
                CNNBlock(in_channel_1, in_channel_1//2, kernel_size=1),
                CNNBlock(in_channel_1//2, in_channel_1, kernel_size=3, padding=1))
            )
        return torch.nn.Sequential(*layers)
        
    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)
        return x

class Head(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.convA = CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1, stride=1)
        self.convB = CNNBlock(2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1)
        
    def forward(self, x):
        x = self.convA(x)
        x = self.convB(x)
        x = x.reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
        x = x.permute(0, 1, 3, 4, 2)
        return x
        
class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.backbone = self._create_backbone()
        self.necks, self.heads = self._create_heads(1024)
        
    def forward(self, x):
        o = []
        x = self.backbone(x)
        for i, neck in enumerate(self.necks):
            x = neck(x)
            o.append(self.heads[i](x))
        return o
    
    def _create_backbone(self):
        layers = []
        in_chan = self.in_channels
        for l in CONFIG_BACKBONE:
            if l[0] == "C":
                # in_channels, out_channels, kernel_size, stride, padding
                layers.append(CNNBlock(in_chan, l[1], kernel_size=l[2], stride=l[3], padding=1))
                in_chan = l[1]
            elif l[0] == "B":
                layers.append(ResBlockA(in_chan, l[1]))
                in_chan = in_chan
            else:
                print("ERROR: Invalid layer type")
        return nn.Sequential(*layers)
    
    def _create_heads(self, in_chan):
        in_chan = in_chan
        tmp_layers = []
        layers, heads = nn.ModuleList(), nn.ModuleList()

        for l in CONFIG_NECK:
            if l[0] == "C":
                tmp_layers.append(CNNBlock(in_chan, l[1], kernel_size=l[2], stride=l[3], padding=1))
                in_chan = l[1]
            elif l[0] == "U":
                tmp_layers.append(nn.Upsample(scale_factor=2))
            elif l[0] == "S":
                layers.append(nn.Sequential(*tmp_layers))
                heads.append(Head(in_chan, self.num_classes))
                tmp_layers = []
            else:
                print("ERROR: Invalid layer type")
        return layers, heads

DEVICE = torch.device("cuda:1")
if __name__ == '__main__':
    model = YOLOv3(num_classes=10).to(DEVICE)
    model_leg = legacy_model.YOLOv3(num_classes=10).to(DEVICE)
    
    x = torch.randn((1, 3, 416, 416), device=DEVICE)
    outputs = model(x)
    
    print(outputs[0].shape)
    print(outputs[1].shape)
    print(outputs[2].shape)