import torch
import torch.nn as nn


class SematicEmbbedBlock(nn.Module):
    def __init__(self, high_in_plane, low_in_plane, out_plane):
        super(SematicEmbbedBlock, self).__init__()
        self.conv3x3 = nn.Conv2d(high_in_plane, out_plane, 3, 1, 1)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1x1 = nn.Conv2d(low_in_plane, out_plane, 1)

    def forward(self, high_x, low_x):
        high_x = self.upsample(self.conv3x3(high_x))
        low_x = self.conv1x1(low_x)
        return high_x * low_x


class KeyPointModel(nn.Module):
    """
    downsample ratio=2
    """

    def __init__(self, num_class=2):
        super(KeyPointModel, self).__init__()
        self.num_class = num_class
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(True)
        self.maxpool3 = nn.MaxPool2d((2, 2))

        self.conv4 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(True)

        self.seb1 = SematicEmbbedBlock(512, 256, 256)
        self.seb2 = SematicEmbbedBlock(256, 128, 128)
        self.seb3 = SematicEmbbedBlock(128, 64, 64)

        self.heatmap = nn.Conv2d(64, self.num_class, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        m1 = self.maxpool1(x1)

        x2 = self.conv2(m1)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)

        m2 = self.maxpool2(x2)

        x3 = self.conv3(m2)
        x3 = self.bn3(x3)
        x3 = self.relu3(x3)

        m3 = self.maxpool3(x3)

        x4 = self.conv4(m3)
        x4 = self.bn4(x4)
        x4 = self.relu4(x4)

        up1 = self.seb1(x4, x3)
        up2 = self.seb2(up1, x2)
        up3 = self.seb3(up2, x1)
        # print(up3.shape)

        out = self.heatmap(up3)
        return out


if __name__ == "__main__":
    model = KeyPointModel()

    input_tensor = torch.zeros((4, 3, 128, 64))

    output_tensor = model(input_tensor)

    print(output_tensor.shape)
