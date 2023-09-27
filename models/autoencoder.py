import torch
import torch.nn as nn
import torch.nn.functional as F


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
            nn.BatchNorm1d(self.latent_dim),
            nn.ReLU(),

        )

        self.linear_de = nn.Sequential(
            nn.Linear(self.latent_dim, self.calculate_flatten_size().numel()),
            nn.BatchNorm1d(self.calculate_flatten_size().numel()),
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
        x = F.pad(x, (0, self.input_size[1] - x.size(3), 0, self.input_size[0] - x.size(2)))
        return x, latent


class ConvAutoencoderAlex(nn.Module):

    def __init__(self, latent_dim=16, input_size=(227, 227), gray=False, encoder_mode=False):
        super(ConvAutoencoderAlex, self).__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.encoder_mode = encoder_mode
        if gray:
            self.input_channels = 1
        else:
            self.input_channels = 3

        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.encoder_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True)

        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        self.encoder_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True)

        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )

        self.encoder_conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.encoder_conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.encoder_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True)

        self.encoder_avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.encoder_fc = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(4096, self.latent_dim),
        )

        self.decoder_fc = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(self.latent_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(4096),
            # nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(4096),
            # nn.Dropout(),
            nn.Linear(4096, 256 * 6 * 6),
            nn.BatchNorm1d(256 * 6 * 6),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(256 * 6 * 6)
        )

        self.decoder_unpool3 = nn.MaxUnpool2d(kernel_size=3, stride=2)
        self.decoder_upsample3 = nn.UpsamplingBilinear2d()
        self.decoder_upsample3_ct = nn.ConvTranspose2d

        self.decoder_convtrans5 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.decoder_convtrans4 = nn.Sequential(
            nn.ConvTranspose2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )

        self.decoder_convtrans3 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        self.decoder_unpool2 = nn.MaxUnpool2d(kernel_size=3, stride=2)

        self.decoder_convtrans2 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.decoder_unpool1 = nn.MaxUnpool2d(kernel_size=3, stride=2)

        self.decoder_convtrans1 = nn.Sequential(
            nn.ConvTranspose2d(64, self.input_channels, kernel_size=11, stride=4, padding=2),
        )

    def encode(self, x, get_mid=True):
        # encoder
        x = self.encoder_conv1(x)  # (3, 227, 227 ->) 64, 56, 56
        x, idx_mp_1 = self.encoder_pool1(x)  # 64, 27, 27

        x = self.encoder_conv2(x)  # 192, 27, 27
        x, idx_mp_2 = self.encoder_pool2(x)  # 192, 13, 13

        x = self.encoder_conv3(x)  # 384, 13, 13

        x = self.encoder_conv4(x)  # 384, 13, 13

        x = self.encoder_conv5(x)  # 256, 13, 13

        x, idx_mp_3 = self.encoder_pool3(x)  # 256, 6, 6

        x = self.encoder_avgpool(x)  # 256, 6, 6

        x = x.view(x.size(0), 256 * 6 * 6)  # 9216

        x = self.encoder_fc(x)  # 4096

        # x = self.encoder_fc_last(x)

        return x, idx_mp_1, idx_mp_2, idx_mp_3

    # decode with unpool
    def decode(self, x, idx_mp_1, idx_mp_2, idx_mp_3):

        x = self.decoder_fc(x)  # (4096 ->) 9216

        x = x.view(x.size(0), 256, 6, 6)  # 256, 6, 6

        x = self.decoder_unpool3(x, idx_mp_3)  # 256, 13, 13

        x = self.decoder_convtrans5(x)  # 384, 13, 13
        x = self.decoder_convtrans4(x)  # 384, 13, 13
        x = self.decoder_convtrans3(x)  # 192, 13, 13

        x = self.decoder_unpool2(x, idx_mp_2)  # 192, 27, 27
        x = self.decoder_convtrans2(x)  # 64, 27, 27

        x = self.decoder_unpool1(
            x, idx_mp_1, output_size=(56, 56)
        )  # 64, 56, 56, if without output_size option 64, 55, 55
        x = self.decoder_convtrans1(x)  # 3, 227, 227

        # x = self.decoder_adjust_size(x) # 3, 227, 227

        return x

    def forward(self, x):
        #
        list_layers = [
            self.encoder_conv1,
            self.encoder_conv2,
            self.encoder_conv3,
            self.encoder_conv4,
            self.encoder_conv5,
            self.decoder_convtrans1,
            self.decoder_convtrans2,
            self.decoder_convtrans3,
            self.decoder_convtrans4,
            self.decoder_convtrans5,
        ]
        for layer in list_layers:
            for param in layer.parameters():
                param.requires_grad = True

        h, idx_mp_1, idx_mp_2, idx_mp_3 = self.encode(x)

        if self.encoder_mode:
            return h

        x = self.decode(h, idx_mp_1, idx_mp_2, idx_mp_3)

        return x, h
