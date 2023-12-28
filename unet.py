import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf
import math

class EncoderBlock(nn.Module):
    def __init__(self, dim1, dim2, activation='relu', padding=False):
        super().__init__()
        if activation == 'leakyrelu':
            self.activ = nn.LeakyReLU(inplace=True)
        elif activation == 'gelu':
            self.activ = nn.GELU()
        else:
            self.activ = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2,2)
        if padding:
            self.conv1 = nn.Conv2d(dim1, dim2, 3, 1, 1)
            self.conv2 = nn.Conv2d(dim2, dim2, 3, 1, 1)
        else:
            self.conv1 = nn.Conv2d(dim1, dim2, 3)
            self.conv2 = nn.Conv2d(dim2, dim2, 3)

    def forward(self, x):
        return self.maxpool(self.activ(self.conv2(self.activ(self.conv1(x)))))

class DecoderBlock(nn.Module):
    def __init__(self, dim1, dim2, activation='relu'):
        super().__init__()
        if activation == 'leakyrelu':
            self.activ = nn.LeakyReLU(inplace=True)
        elif activation == 'gelu':
            self.activ = nn.GELU()
        else:
            self.activ = nn.ReLU(inplace=True)
        self.upconv = nn.ConvTranspose2d(dim2, int(dim2//2), 2, 2)
        self.conv1 = nn.Conv2d(dim1, dim2, 3)
        self.conv2 = nn.Conv2d(dim2, dim2, 3)

    def forward(self, x):
        return self.upconv(self.activ(self.conv2(self.activ(self.conv1(x)))))

class UNET(nn.Module):
    def __init__(self, activation='relu'):
        super().__init__()
        if activation == 'leakyrelu':
            self.activ = nn.LeakyReLU(inplace=True)
        elif activation == 'gelu':
            self.activ = nn.GELU()
        else:
            self.activ = nn.ReLU(inplace=True)
        self.conv5_1 = nn.Conv2d(512, 1024, 3)
        self.conv5_2 = nn.Conv2d(1024, 1024, 3)
        self.out1 = nn.Conv2d(128, 64, 3)
        self.out2 = nn.Conv2d(64, 64, 3)
        self.out3 = nn.Conv2d(64, 1, 1)
        self.upconv1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.EncoderLayer1 = EncoderBlock(1, 64)
        self.EncoderLayer2 = EncoderBlock(64, 128)
        self.EncoderLayer3 = EncoderBlock(128, 256)
        self.EncoderLayer4 = EncoderBlock(256, 512)
        self.DecoderLayer1 = DecoderBlock(1024, 512)
        self.DecoderLayer2 = DecoderBlock(512, 256)
        self.DecoderLayer3 = DecoderBlock(256, 128)

    def forward(self, x):
        h, w = x.shape[2:]
        x = overlap_tiles(x, h, w)
        Layer1 = self.EncoderLayer1(x)
        Layer2 = self.EncoderLayer2(Layer1)
        Layer3 = self.EncoderLayer3(Layer2)
        Layer4 = self.EncoderLayer4(Layer3)
        x = self.upconv1(self.activ(self.conv5_2(self.activ(self.conv5_1(Layer4)))))
        x = torch.concat([tf.center_crop(Layer4, (x.shape[2], x.shape[3])), x], dim=1)
        x = self.DecoderLayer1(x)
        x = torch.concat([tf.center_crop(Layer3, (x.shape[2], x.shape[3])), x], dim=1)
        x = self.DecoderLayer2(x)
        x = torch.concat([tf.center_crop(Layer2, (x.shape[2], x.shape[3])), x], dim=1)
        x = self.DecoderLayer3(x)
        x = torch.concat([tf.center_crop(Layer1, (x.shape[2], x.shape[3])), x], dim=1)
        x = self.activ(self.out2(self.activ(self.out1(x))))
        x = self.out3(x)
        x = F.sigmoid(x)
        return pad(x, h, w)

def overlap_tiles(x, dimh, dimw):
    height, width = input_dim(dimh), input_dim(dimw)
    height, width = input_dim(dimh), input_dim(dimw)
    padleft, padtop = (height-dimh)//2, (width-dimw)//2
    padright = padleft if (height-dimh) % 2 == 0 else padleft+1
    padbottom = padtop if (width-dimw) % 2 == 0 else padtop+1
    x = F.pad(x, [padtop, padbottom, padleft, padright], mode='reflect')
    return x

def pad(x, h, w):
    height, width = x.shape[2:]
    padleft, padtop = (h-height)//2, (w-width)//2
    padright = padleft if (h-height) % 2 == 0 else padleft+1
    padbottom = padtop if (w-width) % 2 == 0 else padtop+1
    x = F.pad(x, [padtop, padbottom, padleft, padright], 'constant', 0)
    return x

def output_dim(input_dim, layers=4):
    for _ in range(layers):
        input_dim = (input_dim-4)//2
    for _ in range(layers):
        input_dim = (input_dim-4)*2
    return input_dim-4

def input_dim(output_dim, layers=4):
    output_dim = int((output_dim-4)/16)*16+4
    output_dim += 4
    for _ in range(layers):
        output_dim = math.ceil((output_dim//2))+4 
    for _ in range(layers):
        output_dim = (output_dim*2)+4
    return output_dim

def main():
    x = torch.zeros((1, 1, 1928,1928)).cuda()
    model = UNET().cuda()
    print(model(x).shape)

if __name__ == '__main__':
    main()
        

