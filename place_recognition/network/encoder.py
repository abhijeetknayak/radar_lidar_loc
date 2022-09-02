import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class UNetConvLayer(nn.Module):
    def __init__(self, in_channel=1, out_channel=64, pooling=True):
        super(UNetConvLayer, self).__init__()

        self.pooling = pooling
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, stride=1, kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, stride=1, kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(out_channel)

        if self.pooling:
            self.mp = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))

        if self.pooling:
            out = self.mp(out)

        return out

class UNetEncoder(nn.Module):
    def __init__(self, in_channels=1, filter_start=64, depth=5):
        super(UNetEncoder, self).__init__()
        self.filter_start = filter_start
        self.in_channels = in_channels
        self.depth = depth
        self.net = nn.Sequential()

        for i in range(self.depth):
            pooling = True if i < depth - 1 else False
            layer = UNetConvLayer(in_channel=self.in_channels, out_channel=self.filter_start, pooling=pooling)
            self.in_channels = self.filter_start
            self.filter_start *= 2
            self.net.append(layer)

    def forward(self, X):
        return self.net(X)

if __name__ == '__main__':
    net = UNetEncoder(in_channels=1, filter_start=64, depth=5)
    print(summary(net.cuda(), (1, 600, 600)))

