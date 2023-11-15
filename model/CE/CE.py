
import torch.nn as nn
import torch

class CE(nn.Module):
    def __init__(self, config):
        assert config['model_name'] == 'CE'
        
        super(CE, self).__init__()

        # [b, 3, 512, 512] -> [b, 8, 256, 256]
        self.encoder1 = nn.Sequential(
            # [b, 3, 512, 512]
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=8, stride=2, padding=3, padding_mode='reflect'),
            nn.BatchNorm2d(num_features=4),
            nn.ReLU(),
            # [b, 4, 256, 256]
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU()
            # [b, 8, 256, 256]
        )

        # [b, 8, 256, 256] -> [b, 32, 64, 64]
        self.encoder2 = nn.Sequential(
            # [b, 8, 256, 256]
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=4, padding_mode='reflect',
                      dilation=3),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            # [b, 16, 128, 128]
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=7, padding_mode='reflect',
                      dilation=5),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
            # [b, 32, 64, 64]
        )

        # [b, 32, 64, 64] -> [b, 128, 16, 16]
        self.encoder3 = nn.Sequential(
            # [b, 32, 64, 64]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            # [b, 64, 32, 32]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
            # [b, 128, 16, 16]
        )

        # [b, 128, 16, 16] -> [b, 256, 4, 4]
        self.encoder4 = nn.Sequential(
            # [b, 128, 16, 16]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            # [b, 256, 8, 8]
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
            # [b, 256, 4, 4]
        )

        # [b, 256, 4, 4] -> [b, 512, 1, 1]
        self.encoder5 = nn.Sequential(
            # [b, 256, 4, 4]
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU()
            # [b, 512, 1, 1]
        )

        # [b, 512, 1, 1] -> [b, 256, 4, 4]
        self.decoder5 = nn.Sequential(
            # [b, 512, 1, 1]
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
            # [b, 256, 4, 4]
        )

        # [b, 256, 4, 4] -> [b, 128, 16, 16]
        self.decoder4 = nn.Sequential(
            # [b, 256, 4, 4]
            nn.Upsample(scale_factor=2),
            # [b, 256, 8, 8]
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
            # [b, 128, 16, 16]
        )

        # [b, 256, 16, 16] -> [b, 32, 64, 64]
        self.decoder3 = nn.Sequential(
            # [b, 128, 16, 16] + [b, 128, 16, 16]
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            # [b, 128, 16, 16]
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            # [b, 64, 32, 32]
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
            # [b, 32, 64, 64]
        )

        # [b, 64, 64, 64] -> [b, 8, 256, 256]
        self.decoder2 = nn.Sequential(
            # [b, 32, 64, 64] + [b, 32, 64, 64]
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            # [b, 32, 64, 64]
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=7, dilation=5),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            # [b, 16, 128, 128]
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=4, dilation=3),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU()
            # [b, 8, 256, 256]
        )

        # [b, 16, 256, 256] -> [b, 4, 512, 512]
        self.decoder1 = nn.Sequential(
            # [b, 8, 256, 256] + [b, 8, 256, 256]
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            # [b, 8, 256, 256]
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=4, stride=2, padding=4, dilation=3),
            nn.BatchNorm2d(num_features=4),
            nn.ReLU()
            # [b, 4, 512, 512]
        )

        # [b, 4, 512, 512] -> [b, 1, 512, 512]
        self.decoder0 = nn.Sequential(
            # [b, 4, 512, 512]
            nn.Conv2d(in_channels=7, out_channels=1, kernel_size=3, padding=1),
            # nn.Conv2d(in_channels=7, out_channels=1, kernel_size=1, padding=0),
            nn.Tanh(),
            nn.ReLU(),
            # [b, 1, 512, 512]
        )

    def forward(self, x):
        """

        :param x: [b, 3, 1080, 1920]
        :return: []
        """

        # encoder
        # [b, 3, 512, 512] -> [b, 8, 256, 256]
        h1 = self.encoder1(x)
        # print("h1 shape:{}".format(h1.shape))
        # [b, 8, 256, 256] -> [b, 32, 64, 64]
        h2 = self.encoder2(h1)
        # [b, 32, 64, 64] -> [b, 128, 16, 16]
        h3 = self.encoder3(h2)
        # print("h3 shape:{}".format(h3.shape))
        # [b, 128, 16, 16]
        h4 = self.encoder4(h3)
        # print("h4 shape:{}".format(h4.shape))
        # [b, 256, 4, 4]
        h5 = self.encoder5(h4)
        # print("h5 shape:{}".format(h5.shape))
        # [b, 512, 1, 1]

        # decoder
        # [b, 512, 1, 1]
        d4 = self.decoder5(h5)
        # [b, 256, 4, 4]
        d3 = self.decoder4(d4)
        # print("d3 shape:{}".format(d3.shape))
        # [b, 128, 16, 16] + [b, 128, 16, 16]
        d2 = self.decoder3(torch.cat([h3, d3], dim=1))
        # [b, 32, 64, 64] + [b, 32, 64, 64]
        d1 = self.decoder2(torch.cat([h2, d2], dim=1))
        # [b, 8, 256, 256] + [b, 8, 256, 256]
        d = self.decoder1(torch.cat([h1, d1], dim=1))
        # [b, 4, 512, 512]
        # y = self.decoder0(d)
        y = self.decoder0(torch.cat([d, x], dim=1))
        # [b, 1, 512, 512]
        
        return y