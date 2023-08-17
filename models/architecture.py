import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=13, input_size=(224, 224), gray=False):
        super(SimpleCNN, self).__init__()

        self.input_size = input_size
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
        # Calculate the final flattened feature size based on input size
        self.final_feature_size = self.calculate_final_feature_size()

        self.classifier = nn.Sequential(
            nn.Linear(32 * self.final_feature_size * self.final_feature_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def calculate_final_feature_size(self):
        # Calculate the size of the features after passing through the convolutional layers
        # Adjust this calculation based on your conv layers and strides
        dummy_input = torch.zeros(1, self.input_channels, self.input_size[0], self.input_size[1])
        dummy_output = self.features(dummy_input)
        final_feature_size = dummy_output.size(2)  # Assuming square feature maps

        return final_feature_size
