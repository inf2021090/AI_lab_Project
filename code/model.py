import torch
import torch.nn as nn
from torchsummary import summary

class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(Conv_Block, self).__init__()
        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        # Second convolution layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.down1 = Conv_Block(3, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2 = Conv_Block(16, 32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down3 = Conv_Block(32, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down4 = Conv_Block(64, 128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = Conv_Block(128, 256)

    def forward(self, x):
        # Encoding path
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        # Bottleneck
        bn = self.bottleneck(p4)

        return bn, [d1, d2, d3, d4]

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = Conv_Block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = Conv_Block(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = Conv_Block(64, 32)
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv_up4 = Conv_Block(32, 16)

    def forward(self, bn, enc_features):
        d1, d2, d3, d4 = enc_features

        # Decoding path
        u1 = self.up1(bn)
        u1 = torch.cat([u1, d4], dim=1)
        u1 = self.conv_up1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d3], dim=1)
        u2 = self.conv_up2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d2], dim=1)
        u3 = self.conv_up3(u3)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, d1], dim=1)
        u4 = self.conv_up4(u4)

        return u4

class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.final = nn.Conv2d(16, n_classes, kernel_size=1)  # Assuming multi-class segmentation

    def forward(self, x):
        bn, enc_features = self.encoder(x)
        dec_out = self.decoder(bn, enc_features)
        out = self.final(dec_out)
        return out

# Test
if __name__ == "__main__":
    num_classes = 6
    model = UNet(num_classes)
    x = torch.randn(1, 3, 256, 256)  # batch size 1, 3 color ch, 256x256 image
    output = model(x)
    print(output.shape)  # result [1, 6, 256, 256]

 



