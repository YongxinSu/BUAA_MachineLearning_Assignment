import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# Saliency map predictor network
class SalPreNet(nn.Module):
    def __init__(self, config):
        assert config['model_name'] == 'SalPreNet'
        
        super(SalPreNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=11, stride=1, padding=5)
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.conv9 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=13, stride=1, padding=6)

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.lrn = nn.LocalResponseNorm(5)

        self.deconv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=8, stride=4, padding=2,
                                         output_padding=0, bias=False)
        self.init_weights()

    def forward(self, x):
        # print(x.shape)
        x = self.maxpool1(self.lrn(F.relu(self.conv1(x))))
        # print(x.shape)
        x = self.maxpool2(F.relu(self.conv2(x)))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.conv9(x)
        # print(x.shape)
        x = self.deconv(x)
        # print(x.shape)
        return nn.Sigmoid()(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight, 0.01, 0.00001)