import argparse
import os

import numpy as np
import pandas as pd
from utils import MolecularMetrics, all_scores
from rdkit import Chem


def valid(args):
    valid_mol = list()
    train_mol = list()
    valid_temp = pd.read_csv(args.valid_data)
    valid_temp.dropna(axis=0, how='any', inplace=True)
    valid_smile = valid_temp['SMILES'].values
    valid_mol = [Chem.MolFromSmiles(r) for r in valid_smile]
    
    train_temp = pd.read_csv(args.train_data)
    train_temp.dropna(axis=0, how='any', inplace=True)
    train_smile = train_temp['smiles'].values
    train_mol = [Chem.MolFromSmiles(r) for r in train_smile]
    train_smiles = [r for r in train_smile]
    
    result = pd.DataFrame()
    result['smile']=valid_smile
    m0, m1 = all_scores(valid_mol, train_mol, train_smiles)
    for key in m0.keys():
        print(key, ":", np.array(m0[key]).mean(), "--", len(m0[key]))
        result[key] = m0[key]
    for key in m1.keys():
        print(key, ":", m1[key])
    result.to_csv(args.output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_data", default="/home/liuke/liuke/prj/Materil-GAN/StructureGeneration/GEN/result/3/exp1_decode")
    parser.add_argument("--train_data", default="/home/liuke/liuke/prj/Materil-GAN/StructureGeneration/GAN/data/mol_with_3_rings.csv")
    parser.add_argument("--output_dir", default="./result/rings/2/3rings.csv")

    args = parser.parse_args()
    valid(args)


if __name__ == "__main__":
    main()
