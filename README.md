# StructureGeneration


## JT-VAE
- JT-VAE is the decoder in this work 
- From: https://github.com/wengong-jin/icml18-jtnn/tree/370c6d1a9fe6ed44841430de2febee36f694678d

## GAN
- Generator

## one-shot GenDA
- Domain adaption

## Usage

- In ./GAN

  - "sh train.sh" train the GAN with the whole dataset ZINC-250K

  - "sh finetune.sh" finetune the GAN with specific dataset so that it can generate materials similar to the target

  - "sh generate.sh" generate the desired material

- In ./GEN

  - "sh decode.sh" docode the material SMILEs from latent vector

- In ./Valid

  - "sh valid.sh" validate the results 

