import torch 

DEVICE = 'mps'


## TEST 1. ##
x = torch.randn((1, 3, 416, 416), device=DEVICE)
y = torch.randn((1, 3, 416, 416), device=DEVICE)
s = x*y
print(s.shape)

## TEST 2. ##
x = torch.randn((1, 3, 416, 416))
x = x.to(DEVICE)
y = torch.randn((1, 3, 416, 416), device=DEVICE)
s = x*y
print(s.shape)

## TEST 3. ##
l = torch.nn.Conv2d(3, 3, 3, 1, 1)
l = l.to(DEVICE)
x = torch.randn((1, 3, 416, 416)).to(DEVICE)
print(l(x).shape)

## TEST 4. ##
class testModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, 1, 1)
    def forward(self, x):
        return self.conv(x)
m = testModel()
x = torch.randn((1, 3, 416, 416)).to(DEVICE)
m = m.to(DEVICE)
print(m(x).shape)

## TEST 5. ##
class CNNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1) -> None:
        super(CNNBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.batchnorm = torch.nn.BatchNorm2d(out_channels)
        self.leakyrelu = torch.nn.LeakyReLU(0.1)
    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
        
class testModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = CNNBlock(3, 3, 3, 1, 1)
    def forward(self, x):
        return self.conv(x)
m = testModel()
x = torch.randn((1, 3, 416, 416)).to(DEVICE)
m = m.to(DEVICE)
print(m(x).shape)

## TEST 6. ##
class CNNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1) -> None:
        super(CNNBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.batchnorm = torch.nn.BatchNorm2d(out_channels)
        self.leakyrelu = torch.nn.LeakyReLU(0.1)
    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
        
class testModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = self._make()
    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        return x
    def _make(self):
        l1 = CNNBlock(3, 3, 3, 1, 1)
        l2 = CNNBlock(3, 3, 3, 1, 1)
        layers = [l1, l2]
        return torch.nn.Sequential(*layers)
    
m = testModel()
x = torch.randn((1, 3, 416, 416)).to(DEVICE)
m = m.to(DEVICE)
print(m(x).shape)

# ModuleList vs Sequential vs ModuleDict 
# Difference between ModuleList and Sequential is that ModuleList can contain any type of layers, while Sequential can only contain layers of the same type.