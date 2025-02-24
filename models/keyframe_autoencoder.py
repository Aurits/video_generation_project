import torch
import torch.nn as nn

class KeyframeAutoencoder(nn.Module):
    def __init__(self, in_channels=3, base_filters=32):
        super(KeyframeAutoencoder, self).__init__()
        # Encoder: using depthwise separable convolutions can be added here for extra efficiency
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # Bottleneck
        self.bottleneck = nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1)
        # Decoder with skip connections added in forward if needed
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_filters, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Assuming input images are normalized between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        bottleneck = self.bottleneck(encoded)
        decoded = self.decoder(bottleneck)
        return decoded

if __name__ == '__main__':
    # Quick test of the model
    model = KeyframeAutoencoder()
    dummy_input = torch.randn(1, 3, 64, 64)
    output = model(dummy_input)
    print("Keyframe Autoencoder output shape:", output.shape)
