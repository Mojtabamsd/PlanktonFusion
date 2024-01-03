import torch
import torch.nn as nn


class ViT(nn.Module):
    def __init__(self, input_size, patch_size, num_classes, dim, depth, heads, mlp_dim, gray=False, dropout=0.1):
        super(ViT, self).__init__()

        assert input_size % patch_size == 0, f'patch size {patch_size} not divisible'
        num_patches = (input_size // patch_size) ** 2

        if gray:
            input_channels = 1
        else:
            input_channels = 3

        self.patch_embedding = nn.Conv2d(in_channels=input_channels, out_channels=dim, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout),
            num_layers=depth
        )
        self.fc = nn.Linear(dim, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embedding(x)  # (B, dim, L, L)
        x = x.permute(0, 2, 3, 1)  # (B, L, L, dim)
        x = x.reshape(B, -1, x.size(-1))  # (B, num_patches, dim)

        # Add positional embeddings
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat([cls_token, x], dim=1)  # (B, num_patches + 1, dim)
        x += self.positional_embedding

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Fully connected layer for classification
        x = self.fc(x)

        return x


