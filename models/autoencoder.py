import torch
import torch.nn as nn
import torchvision.models as models


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=16, input_size=(227, 227), gray=False):
        super(ConvAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size
        if gray:
            self.input_channels = 1
        else:
            self.input_channels = 3

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # nn.MaxPool2d(kernel_size=2, stride=2),  # Added max pooling

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # nn.MaxPool2d(kernel_size=2, stride=2),  # Added max pooling

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Linear Transformation
        self.linear_en = nn.Sequential(
            nn.Linear(self.calculate_flatten_size().numel(), self.latent_dim),
            # nn.BatchNorm1d(4096),
            nn.ReLU(),

        )

        self.linear_de = nn.Sequential(
            nn.Linear(self.latent_dim, self.calculate_flatten_size().numel()),
            # nn.BatchNorm1d(self.latent_dim),
            nn.ReLU()

        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, self.input_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Sigmoid()

        )

    def calculate_flatten_size(self, device='cpu'):
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, self.input_size[0], self.input_size[1]).to(device)
            encoder_output = self.encoder(dummy_input)
            # return encoder_output.view(encoder_output.size(0), -1).shape[1]
            return encoder_output.size()

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten for linear transformation
        latent = self.linear_en(x)

        x = self.linear_de(latent)
        size_out = self.calculate_flatten_size(x.device)
        x = x.view(x.size(0), size_out[1], size_out[2], size_out[3])
        x = self.decoder(x)
        return x, latent
