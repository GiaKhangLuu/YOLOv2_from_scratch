from torch import nn
import torch
from torchsummary import summary

def Conv(n_input, n_output, k_size=4, stride=2, padding=0, bn=False):
    return nn.Sequential(
        nn.Conv2d(
            n_input, n_output,
            kernel_size=k_size,
            stride=stride,
            padding=padding, bias=True),
        nn.BatchNorm2d(n_output),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(p=0.2, inplace=False))

class YOLOv2(nn.Module):
    def __init__(self, nc=32, S=13, num_anchor_boxes=5,
                 num_classes=3):
        super(YOLOv2, self).__init__()
        self.nc = nc
        self.S = S
        self.num_anchor_boxes = num_anchor_boxes
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Conv2d(3, nc, kernel_size=4, stride=2, padding=1,
                      bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            Conv(nc, nc, 3, 2, 1),

            Conv(nc, nc * 2, 3, 2, 1),

            Conv(nc * 2, nc * 4, 3, 2, 1),

            Conv(nc * 4, nc * 8, 3, 2, 1),

            Conv(nc * 8, nc * 16, 3, 1, 1),
            Conv(nc * 16, nc * 8, 3, 1, 1),

            Conv(nc * 8, num_anchor_boxes *(4 + 1 + num_classes), 3, 1, 1)
        )

    def forward(self, input):
        output_tensor = self.net(input)
        output_tensor = output_tensor.permute(0, 2, 3, 1)  # B, H, W, C
        W_grid, H_grid = self.S, self.S
        output_tensor = output_tensor.view(-1, H_grid, W_grid, 
                                           self.num_anchor_boxes, 
                                           4 + 1 + self.num_classes)
        return output_tensor
