import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import numpy as np

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            #layers.append(nn.Dropout(p=0.2))
            return layers

        self.net = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=True),
            *block(128, 256),
            *block(256, 256),
            *block(256, 512),
            *block(512, 256),
            #*block(512, 256),
            nn.Linear(256, opt.vector_size),
            nn.Tanh()
        )
        self.mean_w = 0
        self.gamma_avg = opt.gamma_avg

    def forward(self, z):
        self.mean_w = self.gamma_avg * self.mean_w + (1-self.gamma_avg) * z.mean(dim=0, keepdim=True).detach()

        img = self.net(z)
        return img


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(opt.vector_size, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(p=0.2),
            #nn.Linear(128, 128),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        #img_flat = img.view(img.shape[0], -1)
        validity = self.net(img)
        return validity