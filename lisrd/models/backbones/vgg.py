import torch


class VGGLikeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._relu = torch.nn.ReLU(inplace=True)
        self._pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self._conv1_1 = torch.nn.Conv2d(3, 64, kernel_size=3,
                                        stride=1, padding=1)
        self._bn1_1 = torch.nn.BatchNorm2d(64)
        self._conv1_2 = torch.nn.Conv2d(64, 64, kernel_size=3,
                                        stride=1, padding=1)
        self._bn1_2 = torch.nn.BatchNorm2d(64)
        self._conv2_1 = torch.nn.Conv2d(64, 64, kernel_size=3,
                                        stride=1, padding=1)
        self._bn2_1 = torch.nn.BatchNorm2d(64)
        self._conv2_2 = torch.nn.Conv2d(64, 64, kernel_size=3,
                                        stride=1, padding=1)
        self._bn2_2 = torch.nn.BatchNorm2d(64)
        self._conv3_1 = torch.nn.Conv2d(64, 128, kernel_size=3,
                                        stride=1, padding=1)
        self._bn3_1 = torch.nn.BatchNorm2d(128)
        self._conv3_2 = torch.nn.Conv2d(128, 128, kernel_size=3,
                                        stride=1, padding=1)
        self._bn3_2 = torch.nn.BatchNorm2d(128)
        self._conv4_1 = torch.nn.Conv2d(128, 256, kernel_size=3,
                                        stride=1, padding=1)
        self._bn4_1 = torch.nn.BatchNorm2d(256)
        self._conv4_2 = torch.nn.Conv2d(256, 256, kernel_size=3,
                                        stride=1, padding=1)
        self._bn4_2 = torch.nn.BatchNorm2d(256)

    def forward(self, inputs):
        x = self._bn1_1(self._relu(self._conv1_1(inputs)))
        x = self._bn1_2(self._relu(self._conv1_2(x)))
        x = self._pool(x)
        x = self._bn2_1(self._relu(self._conv2_1(x)))
        x = self._bn2_2(self._relu(self._conv2_2(x)))
        x = self._pool(x)
        x = self._bn3_1(self._relu(self._conv3_1(x)))
        x = self._bn3_2(self._relu(self._conv3_2(x)))
        x = self._pool(x)
        x = self._bn4_1(self._relu(self._conv4_1(x)))
        x = self._bn4_2(self._relu(self._conv4_2(x)))
        return x
