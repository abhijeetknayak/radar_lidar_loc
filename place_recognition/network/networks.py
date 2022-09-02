import torch
import torch.nn as nn
import torch.nn.functional as F
from network.encoder import *
from network.necks import *

import ipdb

class PlaceRecogNet(nn.Module):
    def __init__(self):
        super(PlaceRecogNet, self).__init__()
        self.encoder_radar = UNetEncoder(in_channels=1, filter_start=32, depth=5)
        self.encoder_lidar = UNetEncoder(in_channels=1, filter_start=32, depth=5)
        self.neck_radar = NetVLAD(num_clusters=512, dim=512)
        self.neck_lidar = NetVLAD(num_clusters=512, dim=512)

    def forward(self, anchor, pos, neg):
        enc_anchor = self.encoder_radar(anchor)
        enc_pos = self.encoder_lidar(pos)
        enc_neg = self.encoder_lidar(neg)

        out_anchor = self.neck_radar(enc_anchor)
        out_pos = self.neck_lidar(enc_pos)
        out_neg = self.neck_lidar(enc_neg)

        return out_anchor, out_pos, out_neg

if __name__ == '__main__':
    model = PlaceRecogNet().cuda()
    ipdb.set_trace()
    summary(model, [(1, 512, 512), (1, 512, 512), (1, 512, 512)])