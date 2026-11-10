import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class AttentionBlock(nn.Module):

    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention_map = self.sigmoid(self.conv1(x) * self.conv2(x))
        return x * attention_map

class UNetWithAttentionResNet34(nn.Module):

    def __init__(self, in_channels=3, out_channels=1):
        super(UNetWithAttentionResNet34, self).__init__()
        self.encoder = smp.encoders.get_encoder('resnet34', in_channels=in_channels, depth=5, weights='imagenet')
        self.unet = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=in_channels, classes=out_channels)
        self.attention1 = AttentionBlock(in_channels=16)
        self.attention2 = AttentionBlock(in_channels=16)
        self.attention3 = AttentionBlock(in_channels=16)
        self.attention4 = AttentionBlock(in_channels=16)
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.unet.encoder(x)
        x = self.unet.decoder(x1)
        x = self.attention1(x)
        x = self.attention2(x)
        x = self.attention3(x)
        x = self.attention4(x)
        x = self.final_conv(x)
        return x
