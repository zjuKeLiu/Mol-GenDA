import numpy as np
import pandas as pd
from utils import Adaptor
import torch
from models import Generator, Discriminator
from utils import compute_gradient_penalty
from tqdm import tqdm
NOLOG = False
FREEZE_LAYER = 4
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class PipLine(object):
    """Solver for training and testing WGAN."""

    def __init__(self, opt):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.build_model(opt)
        self.latent_dim = opt.latent_dim
        self.n_epochs  = opt.n_epochs
        self.lambda_gp = opt.lambda_gp
        self.n_critic = opt.n_critic
        self.store_interval = opt.store_interval
        self.mode = opt.mode
        self.G_path = opt.G
        self.D_path = opt.D
        self.A_path = opt.A
        self.criterion = torch.nn.BCELoss()
    def build_model(self, opt):
        self.G = Generator(opt)
        self.D = Discriminator(opt)
        if opt.mode == "train":
            self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=opt.lr)#, betas=(opt.b1, opt.b2))
            self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=opt.lr)#, betas=(opt.b1, opt.b2))
        if opt.mode == "finetune":
            self.Adaptor = Adaptor(opt.latent_dim, opt.mode)
            self.G.load_state_dict(torch.load(opt.G))
            #self.D.load_state_dict(torch.load(opt.D))
            ####################################################
            model_dict = self.D.state_dict()
            save_model = torch.load(opt.D)
            count = 0
            state_dict = {}
            for k, v in save_model.items():
                state_dict[k]=v
            
            model_dict.update(save_model)
            #self.D.load_state_dict(torch.load(opt.D))
            self.D.load_state_dict(model_dict)
            for child in self.G.children():
                for param in child.parameters():
                    param.requires_grad = False
            ###################################################
            for i, child in enumerate(self.D.children()):
                for j, param in enumerate(child.parameters()):
                    #print(j, param)
                    #print(j,"###################################")
                    if j == 4:
                        #print("*************************3*******************")
                        break
                    param.requires_grad = False
            self.optimizer_D = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
            self.optimizer_G = torch.optim.Adam(self.Adaptor.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

            self.Adaptor.to(self.device)

        if opt.mode == "generate":
            self.G = Generator(opt)
            self.D = Discriminator(opt)
            self.Adaptor = Adaptor(opt.latent_dim)
            self.G.load_state_dict(torch.load(opt.G))
            self.D.load_state_dict(torch.load(opt.D))
            self.Adaptor.load_state_dict(torch.load(opt.A))
            self.Adaptor.to(self.device)

        if opt.mode == "generate_raw":
            self.G = Generator(opt)
            self.D = Discriminator(opt)
            self.G.load_state_dict(torch.load(opt.G))
            self.D.load_state_dict(torch.load(opt.D))

        self.G.to(self.device)
        self.D.to(self.device)

    
    def train(self, dataloader):
        real_label = 1.
        fake_label = 0.
        for epoch in range(self.n_epochs):
            if NOLOG:
                pbar = tqdm(enumerate(dataloader))
            else:
                pbar = enumerate(dataloader)
            for i, latent in pbar:

                # Configure input
                real_latents = latent.to(self.device).float()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Sample noise as generator input
                #print("************************real_latents.shape", real_latents.shape)
                z = np.random.normal(0, 1, size=(real_latents.shape[0], self.latent_dim))
                #print("************************z.shape", z.shape)
                z = torch.from_numpy(z).to(self.device).float()
                # Generate a batch of images
                if self.mode != "train":
                    k = self.Adaptor(z, self.mode, self.G.mean_w)
                    fake_latents = self.G(k)
                else:
                    fake_latents = self.G(z)

                # Real images
                real_validity = self.D(real_latents).view(-1)
                # Fake images
                fake_validity = self.D(fake_latents).view(-1)
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(self.D, real_latents.data, fake_latents.data)
                
                label_real = torch.full((real_latents.shape[0],), real_label, dtype=torch.float, device=self.device)
                label_fake = torch.full((fake_latents.shape[0],), fake_label, dtype=torch.float, device=self.device)
                errD_real = self.criterion(real_validity, label_real)
                errD_fake = self.criterion(fake_validity, label_fake)
                # Adversarial loss
                #d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty
                d_loss_pure = errD_fake + errD_real
                d_loss = errD_fake + errD_real + self.lambda_gp * gradient_penalty
                d_loss.backward()
                self.optimizer_D.step()

                self.optimizer_G.zero_grad()

                # Train the generator every n_critic steps
                
                if i % self.n_critic == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of images
                    if self.mode != "train":
                        k = self.Adaptor(z, self.mode, self.G.mean_w)
                        fake_latents = self.G(k)
                    else:
                        fake_latents = self.G(z)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = self.D(fake_latents).view(-1)
                    #g_loss = -torch.mean(fake_validity)
                    g_loss = self.criterion(fake_validity, label_real)
                    g_loss.backward()
                    self.optimizer_G.step()
                    '''
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, self.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                    )
                    '''
                if NOLOG:
                    pbar.set_description("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"%(epoch, self.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))
                else:
                    print(d_loss_pure.item(),",", g_loss.item(),",", gradient_penalty.item())

            if (epoch+1) % self.store_interval == 0:
                torch.save(self.G.state_dict(), self.G_path+str(epoch))
                torch.save(self.D.state_dict(), self.D_path+str(epoch))
                if self.mode == "finetune":
                    torch.save(self.Adaptor.state_dict(), self.A_path+str(epoch))
                if NOLOG:
                    print("Model Saved", self.G_path+str(epoch), self.D_path+str(epoch))


    def generate(self, num_sample, file_path):
        self.Adaptor.eval()
        self.D.eval()
        self.G.eval()
        z = np.random.normal(0, 1, (num_sample, self.latent_dim))
        z = torch.from_numpy(z).to(self.device).float()
        z = self.Adaptor(z, self.mode, self.G.mean_w)
        fake_latents = self.G(z)
        print(fake_latents.shape)
        df = pd.DataFrame(data=fake_latents.float().cpu().detach().numpy())
        df.to_csv(file_path)
        print("!!!generation done!!!")

    
    def generate_raw(self, num_sample, file_path):
        self.D.eval()
        self.G.eval()
        z = np.random.normal(0, 1, (num_sample, self.latent_dim))
        z = torch.from_numpy(z).to(self.device).float()
        fake_latents = self.G(z)
        print(fake_latents.shape)
        df = pd.DataFrame(data=fake_latents.float().cpu().detach().numpy())
        df.to_csv(file_path)
        print("!!!original GAN generation done!!!")