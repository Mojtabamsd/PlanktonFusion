import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.num_ch1 = 32
        self.num_ch2 = 64
        self.num_ch3 = 192
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
        x = F.pad(x, (0, 1, 0, 1))

        x = self.decoder1(x)

        # x = F.pad(x, (0, self.input_size[1] - x.size(3), 0, self.input_size[0] - x.size(2)))
        return x, latent



