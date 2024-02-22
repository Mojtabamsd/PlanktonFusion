import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=16, input_size=(227, 227), gray=False, encoder_mode=False):
        super(ConvAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.encoder_mode = encoder_mode
        if gray:
            self.input_channels = 1
        else:
            self.input_channels = 3

        self.num_ch1 = 64       #32
        self.num_ch2 = 192      #64
        self.num_ch3 = 384      #192
        self.num_ch4 = 256

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(self.input_channels, self.num_ch1, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(self.num_ch1),
            nn.ReLU(inplace=True),
        )

        self.encoder_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(self.num_ch1, self.num_ch2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.num_ch2),
            nn.ReLU(inplace=True),
        )

        self.encoder_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True)

        self.encoder3 = nn.Sequential(
            nn.Conv2d(self.num_ch2, self.num_ch3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_ch3),
            nn.ReLU(inplace=True),
        )

        self.encoder4 = nn.Sequential(
            nn.Conv2d(self.num_ch3, self.num_ch4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_ch4),
            nn.ReLU(inplace=True),
        )

        # Linear Transformation
        self.linear_en = nn.Sequential(
            nn.Linear(self.calculate_flatten_size().numel(), self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.ReLU(inplace=True),

        )

        self.linear_de = nn.Sequential(
            nn.Linear(self.latent_dim, self.calculate_flatten_size().numel()),
            nn.BatchNorm1d(self.calculate_flatten_size().numel()),
            nn.ReLU(inplace=True),

        )

        # Decoder
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(self.num_ch4, self.num_ch3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_ch3),
            nn.ReLU(inplace=True),
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(self.num_ch3, self.num_ch2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.num_ch2),
            nn.ReLU(inplace=True),
        )

        self.decoder_un_pool2 = nn.MaxUnpool2d(kernel_size=3, stride=2)
        self.decoder_up_sample2 = nn.UpsamplingBilinear2d(size=(28, 28))

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(self.num_ch2, self.num_ch1, kernel_size=5, padding=2),
            nn.BatchNorm2d(self.num_ch1),
            nn.ReLU(inplace=True),
        )

        self.decoder_un_pool1 = nn.MaxUnpool2d(kernel_size=3, stride=2)
        self.decoder_up_sample1 = nn.UpsamplingBilinear2d(size=(113, 113))

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(self.num_ch1, self.input_channels, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # nn.Sigmoid(),
        )

    def calculate_flatten_size(self, device='cpu'):
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, self.input_size[0], self.input_size[1]).to(device)
            x = self.encoder1(dummy_input)
            x, _ = self.encoder_pool1(x)
            x = self.encoder2(x)
            x, _ = self.encoder_pool2(x)
            x = self.encoder3(x)
            encoder_output = self.encoder4(x)
            # return encoder_output.view(encoder_output.size(0), -1).shape[1]
            return encoder_output.size()

    def forward(self, x):
        x = self.encoder1(x)
        x, idx_mp_1 = self.encoder_pool1(x)
        x = self.encoder2(x)
        x, idx_mp_2 = self.encoder_pool2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        x = x.view(x.size(0), -1)  # Flatten for linear transformation
        latent = self.linear_en(x)

        if self.encoder_mode:
            return latent

        x = self.linear_de(latent)
        size_out = self.calculate_flatten_size(x.device)
        x = x.view(x.size(0), size_out[1], size_out[2], size_out[3])
        x = self.decoder4(x)
        x = self.decoder3(x)
        x = self.decoder_un_pool2(x, idx_mp_2)
        x = self.decoder2(x)
        x = self.decoder_un_pool1(x, idx_mp_1, output_size=(56, 56))
        # x = self.decoder_un_pool1(x, idx_mp_1)
        # x = F.pad(x, (0, 1, 0, 1))

        x = self.decoder1(x)

        # x = F.pad(x, (0, self.input_size[1] - x.size(3), 0, self.input_size[0] - x.size(2)))
        return x, latent


class ResNetCustom(nn.Module):
    def __init__(self, num_classes=13, latent_dim=16, gray=False, pretrained=None, encoder_mode=False):
        super(ResNetCustom, self).__init__()

        if gray:
            input_channels = 1
        else:
            input_channels = 3

        self.encoder_mode = encoder_mode

        # Load a pre-trained ResNet18 model
        resnet = models.resnet18(pretrained=pretrained)

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7),
                               stride=(2, 2), padding=(3, 3), bias=False)

        # Remove the original classification head
        self.features = nn.Sequential(self.conv1, *list(resnet.children())[1:-1])

        # Add a latent layer
        self.latent_layer = nn.Linear(resnet.fc.in_features, latent_dim)

        # Add a new classification head
        self.classification_head = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        features = self.features(x)
        features = torch.flatten(features, 1)
        latent_output = self.latent_layer(features)

        if self.encoder_mode:
            return latent_output

        classification_output = self.classification_head(latent_output)
        return classification_output, latent_output


class Encoder(nn.Module):
    def __init__(self, latent_dim, input_channels=1):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.latent_dim = latent_dim
        self.flatten = nn.Flatten()
        # self.fc = nn.Linear(512*8*8, latent_dim)
        self.fc = nn.Sequential(
            nn.Linear(512*8*8, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x, idx1 = self.max_pool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x4 = self.avg_pool(x4)

        x5 = self.flatten(x4)
        x5 = self.fc(x5)

        return x0, idx1, x1, x2, x3, x4, x5


class Decoder(nn.Module):
    def __init__(self, latent_dim=16, input_channels=1, input_size=(227, 227)):
        super(Decoder, self).__init__()
        # self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 8 * 8),
            nn.BatchNorm1d(512 * 8 * 8),
            nn.ReLU(inplace=True),)
        self.input_size = input_size
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, input_channels, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False)
        # self.decoder_un_pool = nn.MaxUnpool2d(kernel_size=3, stride=2)
        self.deconv4_un_pool = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x0, idx1, x1, x2, x3, x4, x5):
        x = self.fc(x5)
        x = x.view(-1, 512,  8, 8)
        # x = torch.cat((x, x4), dim=1)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = F.interpolate(x, size=(x3.size(2), x3.size(3)), mode='bilinear', align_corners=False)
        # x = torch.cat((x, x3), dim=1)  # Concatenate with x3

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = F.interpolate(x, size=(x2.size(2), x2.size(3)), mode='bilinear', align_corners=False)
        # x = torch.cat((x, x2), dim=1)  # Concatenate with x2

        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = F.interpolate(x, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=False)
        # x = torch.cat((x, x1), dim=1)  # Concatenate with x1
        x = self.deconv4(x)

        x = F.interpolate(x, size=(idx1.size(2), idx1.size(3)), mode='bilinear', align_corners=False)
        # x = self.decoder_un_pool(x, idx1)
        x = self.deconv4_un_pool(x)

        x = F.interpolate(x, size=(x0.size(2), x0.size(3)), mode='bilinear', align_corners=False)
        # x = torch.cat((x, x0), dim=1)  # Concatenate with x1
        x = self.deconv5(x)

        x = self.tanh(x)
        x = F.interpolate(x, size=(self.input_size[0], self.input_size[1]), mode='bilinear', align_corners=False)
        return x


class ResNetAutoencoder(nn.Module):
    def __init__(self, latent_dim=16, input_size=(227, 227), gray=False, encoder_mode=False):
        super(ResNetAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.encoder_mode = encoder_mode
        if gray:
            input_channels = 1
        else:
            input_channels = 3
        self.encoder = Encoder(latent_dim, input_channels)
        self.decoder = Decoder(latent_dim, input_channels, input_size)

    def forward(self, x):
        x0, idx1, x1, x2, x3, x4, x5 = self.encoder(x)
        if self.encoder_mode:
            return x5
        reconstructed = self.decoder(x0, idx1, x1, x2, x3, x4, x5)
        return reconstructed, x5
