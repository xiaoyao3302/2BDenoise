from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, args):
        super(UNet, self).__init__()

        self.encoder = UNet_Encoder(args)
        self.cls = UNet_cls(args)

    def forward(self, x):

        feature = self.encoder(x)
        pred = self.cls(x, feature)

        return feature, pred


class UNet_Encoder(nn.Module):
    def __init__(self, args):
        super(UNet_Encoder, self).__init__()
        self.bilinear = args.bilinear

        self.inc = (DoubleConv(3, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if self.bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, self.bilinear))
        self.up2 = (Up(512, 256 // factor, self.bilinear))
        self.up3 = (Up(256, 128 // factor, self.bilinear))
        self.up4 = (Up(128, 64, self.bilinear)) 

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x
    

class UNet_cls(nn.Module):
    def __init__(self, args):
        super(UNet_cls, self).__init__()

        self.outc = (OutConv(64, args.num_class))

    def forward(self, x, feature):

        pred = self.outc(feature)
        return pred
    