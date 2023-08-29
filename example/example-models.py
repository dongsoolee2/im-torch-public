import torch
import torch.nn as nn

class BNCNN3L(nn.Module):
    def __init__(self, out):
        super(BNCNN3L, self).__init__()
        self.name = 'BNCNN3L'
        self.conv1 = nn.Conv2d(40, 8, kernel_size=13, bias=True)
        self.batch1 = nn.BatchNorm1d(8*20*20, eps=1e-3, momentum=.99)
        self.nonlinear1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 8, kernel_size=11, bias=True)
        self.batch2 = nn.BatchNorm1d(8*10*10, eps=1e-3, momentum=.99)
        self.nonlinear2 = nn.ReLU()
        self.conv3 = nn.Conv2d(8, out, kernel_size=10, bias=True)
        self.batch3 = nn.BatchNorm1d(out, eps=1e-3, momentum=.99)
        self.nonlinear3 = nn.Softplus()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x.view(x.size(0), -1))
        x = self.nonlinear1(x)
        x = self.conv2(x)
        x = self.batch2(x.view(x.size(0), -1))
        x = self.nonlinear2(x)
        x = self.conv3(x)
        x = self.batch3(x.view(x.size(0), -1))
        x = self.nonlinear3(x)
        return x

class CNN2L(nn.Module):
    def __init__(self, out):
        super(CNN2L, self).__init__()
        self.name = 'CNN2L'
        self.conv1 = nn.Conv2d(40, 8, kernel_size=16, bias=True)
        self.nonlinear1 = nn.Softplus()
        self.conv2 = nn.Conv2d(8, out, kernel_size=17, bias=True)
        self.nonlinear2 = nn.Softplus()

    def forward(self, x):
        x = self.conv1(x)
        x = self.nonlinear1(x)
        x = self.conv2(x)
        x = self.nonlinear2(x)
        return x

class LN(nn.Module):
    def __init__(self, out):
        super(LN, self).__init__()
        self.name = 'LN'
        self.conv = nn.Conv2d(40, out, kernel_size=32, bias=True)
        self.nonlinear = nn.Softplus()

    def forward(self, x):
        x = self.conv(x)
        x = self.nonlinear(x)
        return x
