import torch
from torch import nn



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



def SEBasicBlock(cin, cout, n):
    """
    Construct a VGG block with BatchNorm placed after each Conv2d
    :param cin: Num of input channels
    :param cout: Num of output channels
    :param n: Num of conv layers
    """
    layers = [
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        SELayer(cout),
        nn.ReLU()
    ]
    for _ in range(n - 1):
        layers.append(nn.Conv2d(cout, cout, 3, padding=1))
        layers.append(nn.BatchNorm2d(cout))
        layers.append(SELayer(cout))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)



def BasicBlock(cin, cout, n):
    """
    Construct basic block with BatchNorm placed after each Conv2d
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

    def __init__(self, block, layers, num_features, num_classes=2, input_channels=1):
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
    model = SimpleNet(BasicBlock, [1, 2, 1, 1, 1], [64]*5, **kwargs)
    return model


def tiny_net(**kwargs):
    """Constructs a simplenet with 5 layers. 
       v5.0:[16,32,64,64,128] 
       v5.1:[32,32,64,64,128]
       v5.2:[64]*5
    """
    model = SimpleNet(BasicBlock, [1,1,1,1,1], [64]*5, **kwargs)
    return model



def se_tiny_net(**kwargs):
    """Constructs a simplenet with 5 layers.  
       v6.0:[32,32,64,64,128]
       v6.1:[64]*5
    """
    model = SimpleNet(SEBasicBlock, [1,1,1,1,1], [32,32,64,64,128], **kwargs)
    return model
