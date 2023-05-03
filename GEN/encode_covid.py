import argparse
import os
import numpy as np
import pandas as pd
import torch
import tqdm
from jtvae import (Vocab,
                   JTNNVAE)

def load_model():
    vocab = [x.strip("\r\n ") for x in open("./jtvae/data/zinc/vocab.txt")]
    vocab = Vocab(vocab)
    hidden_size = 450
    latent_size = 56
    depth = 3
    model = JTNNVAE(vocab, hidden_size, latent_size, depth)
    model.load_state_dict(torch.load("./GEN/jtvae/molvae/MPNVAE-h450-L56-d3-beta0.005/model.iter-4"))
    return model.cuda()

mol_list = ["Nc3ncnn2c3ccc2C(C#N)(C1O)OC(C1O)CO[P](=O)(NC(C)C(=O)OCC(CC)CC)Oc4ccccc4",
"CC1([C@@H]2[C@H]1[C@H](N(C2)C(=O)[C@H](C(C)(C)C)NC(=O)C(F)(F)F)C(=O)N[C@@H](C[C@@H]3CCNC3=O)C#N)C",
"CCS(=O)(=O)N1CC(CC#N)(n2cc(-c3ncnc4[nH]ccc34)cn2)C1",
"COC1=CC(=CC(=C1OC)OC)C(=O)C2=CN=C(N2)C3=CNC4=CC=CC=C43",
"CC(C)C(=O)OC[C@H]1O[C@@H](n2ccc(NO)nc2=O)[C@H](O)[C@@H]1O"
]

model = load_model()
res = model.encode_latent_mean(mol_list)

for re in res:
    print(re)
