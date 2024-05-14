import torch
import torch.nn as nn


class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_Block, self).__init__()
        #first covolition layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn1 = nn.BathNormal2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        #2nd convolution layer
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BathNormal2d(out_channels)
        self.relu2 = nn.ReLu(inplace = True)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
        return x


    def forward(self, x):
        pass

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()


    def forward(self, x):
        pass

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()


    def forward(self, x):
        pass

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Down part of U-Net
        self.down1 = Conv_Block(3, 64)
        self.down2 = Conv_Block(64, 128)
        self.down3 = Conv_Block(128, 256)
        self.down4 = Conv_Block(256, 512)

        # Up part of U-Net
        self.up1 = Conv_Block(512 + 256, 256)
        self.up2 = Conv_Block(256 + 128, 128)
        self.up3 = Conv_Block(128 + 64, 64)

        # Final part to reduce channels to the number of classes
        self.final = nn.Conv2d(64, 1, kernel_size=1)  # Assuming binary classification

    def forward(self, x):
        # Downsample
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        # Upsample and concatenate
        u1 = self.up1(torch.cat([d4, d3], dim=1))
        u2 = self.up2(torch.cat([u1, d2], dim=1))
        u3 = self.up3(torch.cat([u2, d1], dim=1))

        # Final convolution
        out = self.final(u3)
        return out
