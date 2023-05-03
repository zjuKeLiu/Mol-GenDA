#!/bin/bash
python main.py --store_interval 20 --n_epochs 80 --mode finetune --G ./ckpt/rings/0/G199 --D ./ckpt/rings/0/D199 --A ./ckpt/rings/0/A --train_dataset_dir ./data/mol_with_0_rings.csv > logs/0.csv 
