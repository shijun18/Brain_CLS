import torch
from torch import nn


def BasicBlock(cin, cout, n):
    """
    Construct a VGG block with BatchNorm placed after each Conv2d
    :param cin: Num of input channels
    :param cout: Num of output channels
    :param n: Num of conv layers
    """
    layers = [
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU()
    ]
    for _ in range(n - 1):
        layers.append(nn.Conv2d(cout, cout, 3, padding=1))
        layers.append(nn.BatchNorm2d(cout))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class SimpleNet(nn.Module):

    def __init__(self,block,layers,num_features,num_classes=2,input_channels=1):
        """
        Construct a SimpleNet
        """
        super(SimpleNet, self).__init__()
        self.backbone = nn.Sequential(
            block(input_channels, num_features[0], layers[0]), 
            block(num_features[0], num_features[1], layers[1]),  
            block(num_features[1], num_features[2], layers[2]),  
            block(num_features[2], num_features[3], layers[3]),  
            block(num_features[3], num_features[4], layers[4]),  
        )
        self.bridge = nn.AdaptiveMaxPool2d((1, 1))
        self.cls = nn.Sequential(
            nn.Linear(num_features[4], num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.bridge(x)
        x = torch.flatten(x, 1)
        x = self.cls(x)
        return x


def simple_net(**kwargs):
    """Constructs a simplenet with 6 layers. 
    """
    model = SimpleNet(BasicBlock, [1,2,1,1,1], [64]*5, **kwargs)
    return model


def tiny_net(**kwargs):
    """Constructs a simplenet with 6 layers. 
    """
    model = SimpleNet(BasicBlock, [1,1,1,1,1], [16,32,64,64,128], **kwargs)
    return model