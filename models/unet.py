import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf
import math

class EncoderBlock2d(nn.Module):
    def __init__(self, dim1, dim2, activation, norm):
        super().__init__()
        self.norm = norm
        self.activ = activation
        self.maxpool = nn.MaxPool2d(2,2)
        self.batchnorm = nn.BatchNorm2d(dim2)
        self.conv1 = nn.Conv2d(dim1, dim2, 3)
        self.conv2 = nn.Conv2d(dim2, dim2, 3)

    def forward(self, x):
        if self.norm:
            x = self.activ(self.batchnorm(self.conv2(self.activ(self.batchnorm(self.conv1(x))))))
        else:
            x = self.activ(self.conv2(self.activ(self.conv1(x))))
        return self.maxpool(x), x
    
class EncoderBlock3d(nn.Module):
    def __init__(self, dim1, dim2, dim3, activation, norm):
        super().__init__()
        self.norm = norm
        self.activ = activation
        self.maxpool = nn.MaxPool3d(2,2)
        self.batchnorm1 = nn.BatchNorm3d(dim2)
        self.batchnorm2 = nn.BatchNorm3d(dim3)
        self.conv1 = nn.Conv3d(dim1, dim2, 3)
        self.conv2 = nn.Conv3d(dim2, dim3, 3)

    def forward(self, x):
        if self.norm:
            x = self.activ(self.batchnorm2(self.conv2(self.activ(self.batchnorm1(self.conv1(x))))))
        else:
            x = self.activ(self.conv2(self.activ(self.conv1(x))))
        return self.maxpool(x), x

class DecoderBlock2d(nn.Module):
    def __init__(self, dim1, dim2, activation, norm):
        super().__init__()
        self.norm = norm
        self.activ = activation
        self.batchnorm = nn.BatchNorm2d(dim2)
        self.upconv = nn.ConvTranspose2d(dim2, int(dim2//2), 2, 2)
        self.conv1 = nn.Conv2d(dim1, dim2, 3)
        self.conv2 = nn.Conv2d(dim2, dim2, 3)

    def forward(self, x):
        if self.norm:
            return self.upconv(self.activ(self.batchnorm(self.conv2(self.activ(self.batchnorm(self.conv1(x)))))))
        else:
            return self.upconv(self.activ(self.conv2(self.activ(self.conv1(x)))))
        
class DecoderBlock3d(nn.Module):
    def __init__(self, dim1, dim2, dim3, activation, norm):
        super().__init__()
        self.norm = norm
        self.activ = activation
        self.batchnorm1 = nn.BatchNorm3d(dim2)
        self.batchnorm2 = nn.BatchNorm3d(dim3)
        self.upconv = nn.ConvTranspose3d(dim3, dim3, 2, 2)
        self.conv1 = nn.Conv3d(dim1, dim2, 3)
        self.conv2 = nn.Conv3d(dim2, dim3, 3)

    def forward(self, x):
        if self.norm:
            return self.upconv(self.activ(self.batchnorm2(self.conv2(self.activ(self.batchnorm1(self.conv1(x)))))))
        else:
            return self.upconv(self.activ(self.conv2(self.activ(self.conv1(x)))))

class UNET2D(nn.Module):
    def __init__(self, activation='relu', norm=True):
        super().__init__()
        if activation == 'leakyrelu':
            self.activ = nn.LeakyReLU(inplace=True)
        elif activation == 'gelu':
            self.activ = nn.GELU()
        else:
            self.activ = nn.ReLU(inplace=True)
        self.norm = norm
        self.batchnorm = nn.BatchNorm2d(64)
        self.EncoderLayer1 = EncoderBlock2d(1, 64, self.activ, norm)
        self.EncoderLayer2 = EncoderBlock2d(64, 128, self.activ, norm)
        self.EncoderLayer3 = EncoderBlock2d(128, 256, self.activ, norm)
        self.EncoderLayer4 = EncoderBlock2d(256, 512, self.activ, norm)
        self.DecoderLayer1 = DecoderBlock2d(512, 1024, self.activ, norm)
        self.DecoderLayer2 = DecoderBlock2d(1024, 512, self.activ, norm)
        self.DecoderLayer3 = DecoderBlock2d(512, 256, self.activ, norm)
        self.DecoderLayer4 = DecoderBlock2d(256, 128, self.activ, norm)
        self.out1 = nn.Conv2d(128, 64, 3)
        self.out2 = nn.Conv2d(64, 64, 3)
        self.out3 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        h, w = x.shape[2:]
        x = overlap_tiles(x, h, w, 4)
        x, Layer1 = self.EncoderLayer1(x)
        x, Layer2 = self.EncoderLayer2(x)
        x, Layer3 = self.EncoderLayer3(x)
        x, Layer4 = self.EncoderLayer4(x)
        x = self.DecoderLayer1(x)
        x = self.DecoderLayer2(torch.cat([tf.center_crop(Layer4, (x.shape[2], x.shape[3])), x], dim=1))
        x = self.DecoderLayer3(torch.cat([tf.center_crop(Layer3, (x.shape[2], x.shape[3])), x], dim=1))
        x = self.DecoderLayer4(torch.cat([tf.center_crop(Layer2, (x.shape[2], x.shape[3])), x], dim=1))
        x = torch.cat([tf.center_crop(Layer1, (x.shape[2], x.shape[3])), x], dim=1)
        if self.norm:
            x = self.activ(self.out2(self.activ(self.out1(x))))
        else:
            x = self.activ(self.batchnorm(self.out2(self.activ(self.batchnorm(self.out1(x))))))
        x = self.out3(x)
        x = F.sigmoid(x)
        return x

class UNET3D(nn.Module):
    def __init__(self, activation='relu', norm=True):
        super().__init__()
        if activation == 'leakyrelu':
            self.activ = nn.LeakyReLU(inplace=True)
        elif activation == 'gelu':
            self.activ = nn.GELU()
        else:
            self.activ = nn.ReLU(inplace=True)
        self.norm = norm
        self.batchnorm = nn.BatchNorm3d(64)
        self.EncoderLayer1 = EncoderBlock3d(1, 32, 64, self.activ, norm)
        self.EncoderLayer2 = EncoderBlock3d(64, 64, 128, self.activ, norm)
        self.EncoderLayer3 = EncoderBlock3d(128, 128, 256, self.activ, norm)
        self.DecoderLayer1 = DecoderBlock3d(256, 256, 512, self.activ, norm)
        self.DecoderLayer2 = DecoderBlock3d(768, 256, 256, self.activ, norm)
        self.DecoderLayer3 = DecoderBlock3d(384, 128, 128, self.activ, norm)
        self.out1 = nn.Conv3d(192, 64, 3)
        self.out2 = nn.Conv3d(64, 64, 3)
        self.out3 = nn.Conv3d(64, 1, 1)

    def forward(self, x):
        h, w = x.shape[3:]
        z = x.shape[2]
        x = overlap_tiles(x, h, w, z, 3)
        x, Layer1 = self.EncoderLayer1(x)
        x, Layer2 = self.EncoderLayer2(x)
        x, Layer3 = self.EncoderLayer3(x)
        x = self.DecoderLayer1(x)
        zcrop = int((Layer3.shape[2]-x.shape[2])//2)
        x = self.DecoderLayer2(torch.cat([tf.center_crop(Layer3[:,:,zcrop:(Layer3.shape[2]-zcrop),:,:], (x.shape[3], x.shape[4])), x], dim=1))
        zcrop = int((Layer2.shape[2]-x.shape[2])//2)
        x = self.DecoderLayer3(torch.cat([tf.center_crop(Layer2[:,:,zcrop:(Layer2.shape[2]-zcrop),:,:], (x.shape[3], x.shape[4])), x], dim=1))
        zcrop = int((Layer1.shape[2]-x.shape[2])//2)
        x = torch.cat([tf.center_crop(Layer1[:,:,zcrop:(Layer1.shape[2]-zcrop),:,:], (x.shape[3], x.shape[4])), x], dim=1)
        if self.norm:
            x = self.activ(self.out2(self.activ(self.out1(x))))
        else:
            x = self.activ(self.batchnorm(self.out2(self.activ(self.batchnorm(self.out1(x))))))
        x = self.out3(x)
        x = F.sigmoid(x)
        return x      

def overlap_tiles(x, dimh, dimw, dimz, layers):
    height, width, z = input_dim(dimh, layers), input_dim(dimw, layers), input_dim(dimz, layers)
    padleft, padtop = (height-dimh)//2, (width-dimw)//2
    padright = padleft if (height-dimh) % 2 == 0 else padleft+1
    padbottom = padtop if (width-dimw) % 2 == 0 else padtop+1
    if layers == 4:
        return F.pad(x, [padtop, padbottom, padleft, padright], mode='reflect')
    else:
        padfront = (height-dimz)//2
        padback = padfront if (padfront-dimz) % 2 == 0 else padfront+1
        return F.pad(x, [padtop, padbottom, padleft, padright, padback, padback], mode='reflect')

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
    output_dim = int((output_dim-4)/(2**layers))*(2**layers)+4
    output_dim += 4
    for _ in range(layers):
        output_dim = math.ceil((output_dim//2))+4 
    for _ in range(layers):
        output_dim = (output_dim*2)+4
    return output_dim

def main():
    x = torch.zeros((1, 1, 116, 132, 132)).cuda()
    model = UNET3D(activation='gelu', norm=True).cuda()
    print(model(x).shape)

if __name__ == '__main__':
    main()
        

