import torch
import torch.nn as nn
import torchvision.models as models


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=13, input_size=(227, 227), gray=False):
        super(SimpleCNN, self).__init__()

        self.input_size = input_size
        if gray:
            self.input_channels = 1
        else:
            self.input_channels = 3

        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Calculate the final flattened feature size based on input size
        self.final_feature_size = self.calculate_final_feature_size()

        self.classifier = nn.Sequential(
            nn.Linear(32 * self.final_feature_size * self.final_feature_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
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


class ResNetCustom(nn.Module):
    def __init__(self, num_classes=13, input_size=(227, 227), gray=False, pretrained=None, freeze_layers=False):
        super(ResNetCustom, self).__init__()

        if gray:
            input_channels = 1
        else:
            input_channels = 3

        # Load a pre-trained ResNet model
        if pretrained:
            self.resnet = models.resnet18(weights=pretrained)

        # Freeze the layers in the pre-trained model (if needed)
        # for param in self.resnet.parameters():
        #     param.requires_grad = False

        # Modify the final classification layer to match your output size
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            # nn.Dropout(0.5), #TODO
            nn.Linear(in_features, num_classes)
        )

        # Adjust the first convolutional layer to accept the specified number of channels
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7),
                                      stride=(2, 2), padding=(3, 3), bias=False)

        if freeze_layers:
            # Freeze or fine-tune specific layers
            for name, param in self.resnet.named_parameters():
                if 'fc' in name or 'layer4' in name:
                    param.requires_grad = True  # Fine-tune these layers
                else:
                    param.requires_grad = False  # Freeze all other layers

    def forward(self, x):
        return self.resnet(x)


class MobileNetCustom(nn.Module):
    def __init__(self, num_classes=13, input_size=(227, 227), gray=False, pretrained=None):
        super(MobileNetCustom, self).__init__()

        if gray:
            input_channels = 1
        else:
            input_channels = 3

        # Load a pre-trained MobileNetV2 model
        self.mobilenet = models.mobilenet_v2(weights=pretrained)

        # Modify the final classification layer to match your output size
        in_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

        # Adjust the first convolutional layer to accept the specified number of channels
        self.mobilenet.features[0][0] = nn.Conv2d(input_channels, 64, kernel_size=(7, 7),
                                                  stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        return self.mobilenet(x)


class ShuffleNetCustom(nn.Module):
    def __init__(self, num_classes=13, input_size=(227, 227), gray=False, pretrained=None):
        super(ShuffleNetCustom, self).__init__()

        if gray:
            input_channels = 1
        else:
            input_channels = 3

        # Load a pre-trained ShuffleNet model
        self.shufflenet = models.shufflenet_v2_x1_0(weights=pretrained)

        # Modify the final classification layer to match your output size
        in_features = self.shufflenet.fc.in_features
        self.shufflenet.fc = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

        # Adjust the first convolutional layer to accept the specified number of channels
        self.shufflenet.conv1 = nn.Conv2d(input_channels, 24, kernel_size=(3, 3),
                                          stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        return self.shufflenet(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())