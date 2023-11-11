""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, config, bilinear=False):
        
        n_channels = config['Unet']['n_channels']
        n_classes = config['Unet']['n_classes']
        bilinear = config['Unet']['bilinear']
        hiden_channels = config['Unet']['hiden_channels']
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = (DoubleConv(n_channels, hiden_channels[0]))
        self.down1 = (Down(hiden_channels[0], hiden_channels[1]))
        self.down2 = (Down(hiden_channels[1], hiden_channels[2]))
        self.down3 = (Down(hiden_channels[2], hiden_channels[3]))
        factor = 2 if bilinear else 1
        self.down4 = (Down(hiden_channels[3], hiden_channels[3] * 2 // factor))
        self.up1 = (Up(hiden_channels[3] * 2, hiden_channels[3] // factor, bilinear))
        self.up2 = (Up(hiden_channels[3], hiden_channels[2] // factor, bilinear))
        self.up3 = (Up(hiden_channels[2], hiden_channels[1] // factor, bilinear))
        self.up4 = (Up(hiden_channels[1], hiden_channels[0], bilinear))
        self.outc = (OutConv(hiden_channels[0], n_classes))
        
        self.sigmoid = torch.nn.ReLU()# torch.nn.Sigmoid()
        self.mode = 'train'

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
        logits = self.outc(x)
        
        # if self.mode != 'train':
        logits = self.heatmap(logits)
        return logits

    def heatmap(self, x, eps = 1e-7):
        x -= torch.min(x, dim=0)[0]
        x = torch.div(x, eps + torch.max(x, dim=0)[0])
        # print(x.sum().item())
        return x
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)