import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=13, gray=False):
        super(SimpleCNN, self).__init__()

        if gray:
            self.input_channels = 1
        else:
            self.input_channels = 3

        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
