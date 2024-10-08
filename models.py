import torch
import torch.nn as nn
import torch.nn.functional as F


# This function initializes the weights in an intelligent way
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Contracting path
        self.enc_conv1 = self.double_conv(20, 64)
        self.enc_conv2 = self.double_conv(64, 128)
        self.enc_conv3 = self.double_conv(128, 256)
        self.enc_conv4 = self.double_conv(256, 512)
        self.enc_conv5 = self.double_conv(512, 1024)
        
        # Expansive path
        self.up_trans1 = self.up_conv(1024, 512)
        self.dec_conv1 = self.double_conv(1024, 512)
        self.up_trans2 = self.up_conv(512, 256)
        self.dec_conv2 = self.double_conv(512, 256)
        self.up_trans3 = self.up_conv(256, 128)
        self.dec_conv3 = self.double_conv(256, 128)
        self.up_trans4 = self.up_conv(128, 64)
        self.dec_conv4 = self.double_conv(128, 64)
        
        # Final output
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

        initialize_weights(self)
    
    def forward(self, x):
        # x has shape (batch_size, depth, height, width). depth is used as in_channels 
        # Encoder
        x1 = self.enc_conv1(x)
        x2 = self.enc_conv2(F.max_pool2d(x1, kernel_size=2))
        x3 = self.enc_conv3(F.max_pool2d(x2, kernel_size=2))
        x4 = self.enc_conv4(F.max_pool2d(x3, kernel_size=2))
        x5 = self.enc_conv5(F.max_pool2d(x4, kernel_size=2))
        
        # Decoder
        x = self.up_trans1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.dec_conv1(x)
        
        x = self.up_trans2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec_conv2(x)
        
        x = self.up_trans3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec_conv3(x)
        
        x = self.up_trans4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec_conv4(x)
        
        x = self.out_conv(x) # (batch_size, 1, h, w)
        
        return x[:, 0, :, :]
    
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


