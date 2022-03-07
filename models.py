import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.nn.functional as F

from keras.activations import sigmoid
from keras.activations import relu
from keras.activations import tanh

from keras.initializers import RandomNormal

from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from tensorflow_addons.layers import InstanceNormalization

from keras.models import Input
from keras.models import Model

from keras.optimizers import adam_v2

def define_discriminator(image_shape, filters=64):
    
    # define weight values for initialisation
    init = RandomNormal(stddev=0.02)
    
    # wrap input shape into keras Input
    in_image = Input(shape=image_shape)

    # 5 convolutional networks that downsample the dimensions of the image
    d = Conv2D(filters, kernel_size=4, strides=2, padding='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(filters*2, kernel_size=4, strides=2, padding='same', kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = BatchNormalization(momentum=0.99)(d)
    
    d = Conv2D(filters*4, kernel_size=4, strides=2, padding='same', kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = BatchNormalization(momentum=0.99)(d)
    
    d = Conv2D(filters*8, kernel_size=4, strides=2, padding='same', kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = BatchNormalization(momentum=0.99)(d)
    
    d = Conv2D(filters*16, kernel_size=4, strides=2, padding='same', kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = BatchNormalization(momentum=0.99)(d)
    
    # flattens outputs into one final value
    d = Conv2D(1, kernel_size=2, strides=1, padding='valid', kernel_initializer=init)(d)
    # out > 0.5 = real
    # otherwise = false
    out = sigmoid(d)
    
    model = Model(in_image, out)
    
    model.compile(loss='mse', optimizer=adam_v2.Adam(learning_rate=0.0002, beta_1=0.5), loss_weights=[0.5])
    
    return model
    
    
    
# input_channels
class Discriminator(nn.Module):
    def __init__(self, input_channels, f = 64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential( 
            # 128 * 128 input shape
            nn.Conv2d(input_channels, f, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 * 64 input shape
            nn.Conv2d(f, f*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f*2),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 * 32 input shape
            nn.Conv2d(f*2, f*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f*4),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 * 16 input shape
            nn.Conv2d(f*4, f*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f*8),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 * 8 input shape
            nn.Conv2d(f*8, f*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f*16),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(f*16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            
        )
        
    def forward(self, x):
        return self.main(x)
    
    
norm_layer = nn.InstanceNorm2d

def resnet_block(f, input_layer):
    init = RandomNormal(stddev=0.02)
    
    g = Conv2D(f, (3,3), padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = relu(g)
    
    
    g = Conv2D(f, (3,3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    
    g = Concatenate()([g, input_layer])
    
    return g

def define_generator(image_shape, blocks, f):
    # create random noise for weights initialization
    init = RandomNormal(stddev=0.2)
    
    # wrap input image in keras Input
    in_image = Input(shape=image_shape)
    
    # 3 Convolutional layers shrinks height and width
    g = Conv2D(f, (7,7), padding='same', kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = relu(g)
    
    g = Conv2D(f*2, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = relu(g)
    
    g = Conv2D(f*4, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = relu(g)
    
    # resnet blocks help with disappearing gradients
    
    for _ in range(blocks):
        g= resnet_block(f*4, g)
        
    
    # upsample the resulting images into original dimensions of image
    g = Conv2D(f*4, (7,7), padding='same', kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = relu(g)
    
    g = Conv2D(f*2, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = relu(g)
    
    g = Conv2D(3, (7,7), strides=(1,1), padding='same', kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    out_image = tanh(g)
    
    # wrap in keras Model object
    model = Model(in_image, out_image)
    
    
    return model
    

class ResBlock(nn.Module):
    def __init__(self, f):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(f, f, 3, 1, 1), norm_layer(f), nn.ReLU(),
                                  nn.Conv2d(f, f, 3, 1, 1))
        self.norm = norm_layer(f)
    def forward(self, x):
        return F.relu(self.norm(self.conv(x)+x))
        
        
class Generator(nn.Module):
    def __init__(self, f=64, blocks=6):
        super(Generator, self).__init__()
        
        layers = [nn.ReflectionPad2d(3),
                  nn.Conv2d(  3,   f, 7, 1, 0), norm_layer(  f), nn.ReLU(True),
                  nn.Conv2d(  f, 2*f, 3, 2, 1), norm_layer(2*f), nn.ReLU(True),
                  nn.Conv2d(2*f, 4*f, 3, 2, 1), norm_layer(4*f), nn.ReLU(True)]
        
        for i in range(int(blocks)):
            layers.append(ResBlock(4*f))
            
        layers.extend([
                nn.ConvTranspose2d(4*f, 4*2*f, 3, 1, 1), nn.PixelShuffle(2), norm_layer(2*f), nn.ReLU(True),
                nn.ConvTranspose2d(2*f,   4*f, 3, 1, 1), nn.PixelShuffle(2), norm_layer(  f), nn.ReLU(True),
                nn.ReflectionPad2d(3), nn.Conv2d(f, 3, 7, 1, 0),
                nn.Tanh()])
        self.conv = nn.Sequential(*layers)
 
        
    def forward(self, x):
        return self.conv(x)
       


        

            
        