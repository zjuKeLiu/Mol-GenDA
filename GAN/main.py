import argparse
import os
import numpy as np
from data.loader import TrajDataset
import torch
from solver import NOLOG, PipLine

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def main(opt):
    train_dataset = TrajDataset(opt.train_dataset_dir)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
    )
    manager = PipLine(opt)
    if opt.mode=="train" or opt.mode=="finetune":
        manager.train(train_loader)
    elif opt.mode=="generate":
        manager.generate(opt.num_gen, opt.output_path)
    else:
        manager.generate_raw(opt.num_gen, opt.output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="SGD: learning rate")
    parser.add_argument("--b1", type=float, default=0.7, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--vector_size", type=int, default=56, help="size of each latent dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.001, help="lower and upper clip value for disc. weights")
    parser.add_argument("--store_interval", type=int, default=10, help="interval betwen model restore")
    parser.add_argument("--train_dataset_dir", type=str, default="/home/liuke/liuke/prj/StructureGeneration/GAN/data/X_JTVAE_250k_rndm_zinc.csv", help="training dataset")
    parser.add_argument("--mode", type=str, default="train", help="train or finetune or generate")
    parser.add_argument("--G", type=str, default="./ckpt/pretrain/G", help="pre-trained Generator path")
    parser.add_argument("--D", type=str, default="./ckpt/pretrain/D", help="pre-trained Discriminator path")
    parser.add_argument("--A", type=str, default="./ckpt/pretrain/A", help="pre-trained Discriminator path")
    parser.add_argument("--lambda_gp", type=float, default=10, help="penalty weight")
    parser.add_argument("--num_gen", type=int, default=1000, help="number of samples to generate")
    parser.add_argument("--output_path", type=str, default="./result/", help="number of samples to generate")
    parser.add_argument("--gamma_avg", type=float, default=0.9, help="average rate")

    opt = parser.parse_args()
    #print(opt)
    main(opt)
