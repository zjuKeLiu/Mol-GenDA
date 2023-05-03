
import torch
import numpy as np
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class Adaptor(nn.Module):
    def __init__(self, dim, beta=0.9):
        super(Adaptor, self).__init__()
        self.a = nn.Parameter(torch.randn(dim, requires_grad=True))
        self.b = nn.Parameter(torch.randn(dim, requires_grad=True))
        self.beta = beta
    def forward(self, x, mode, mean_w=None):
        #a_normalize = (self.a- torch.min(self.a))/(torch.max(self.a)-torch.min(self.a))
        #b_normalize = (self.b- torch.min(self.b))/(torch.max(self.b)-torch.min(self.b))
        #return a_normalize*x + b_normalize
        if mode == "generate":
            return self.a*x + self.b
            #return self.a * (mean_w + self.beta*(x-mean_w)) + self.b
            #return self.a*(self.beta*x+(1-self.beta)*torch.mean(x,0))+self.b
        else:
            return self.a*x + self.b
