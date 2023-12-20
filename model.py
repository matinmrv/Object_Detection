
import torch
import torch.nn as nn

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, split_size=7, num_boxes=2, num_classes=20):
        super(Yolov1, self).__init__()
        self.in_channels = in_channels
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.darknet = self._create_conv_layers()
        self.fcs = self._create_fcs()

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self):
        layers = []
        in_channels = self.in_channels

        for layer_config in architecture_config:
            if type(layer_config) == tuple:
                kernel_size, out_channels, stride, padding = layer_config
                layers.append(CNNBlock(in_channels, out_channels, kernel_size, stride, padding))
                in_channels = out_channels

            elif type(layer_config) == str:
                layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

            elif type(layer_config) == list:
                conv1, conv2, num_repeats = layer_config
                for _ in range(num_repeats):
                    layers.append(CNNBlock(in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3]))
                    layers.append(CNNBlock(conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3]))
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.split_size * self.split_size, 4096),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, self.split_size * self.split_size * (self.num_classes + self.num_boxes * 5)),
        )