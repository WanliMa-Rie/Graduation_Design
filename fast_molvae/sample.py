import sys
sys.path.append('../')
import torch
import torch.nn as nn

import math, random, sys
import argparse
from fast_jtnn import *
import rdkit

def load_model(vocab, model_path, hidden_size=256, latent_size=56, depthT=20, depthG=3):
    vocab = [x.strip("\r\n ") for x in open(vocab)] 
    vocab = Vocab(vocab)

    model = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
    dict_buffer = torch.load(model_path)
    print(dict_buffer)
    model.load_state_dict(dict_buffer)
    model = model.cuda()
    # model = model.cuda()

    torch.manual_seed(0)

    # print("Model Parameters:")
    # for name, param in model.named_parameters():
    #     print(f"Parameter name: {name}")
    #     print(f"Shape: {param.shape}")
    #     print(f"Values: {param}")  # 打印具体值（如果张量较大，可能需要截断）
    #     print("-" * 50)
    return model

def main_sample(vocab, output_file, model_path, nsample, hidden_size=256, latent_size=56, depthT=20, depthG=3):
    vocab = [x.strip("\r\n ") for x in open(vocab)] 
    vocab = Vocab(vocab)

    model = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
    dict_buffer = torch.load(model_path)
    model.load_state_dict(dict_buffer)
    model = model.cuda()

    torch.manual_seed(0)
    with open(output_file, 'w') as out_file:
        for i in range(nsample):
            out_file.write(str(model.sample_prior())+'\n')

if __name__ == '__main__':
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--nsample', type=int, required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--latent_size', type=int, default=56)
    parser.add_argument('--depthT', type=int, default=20)
    parser.add_argument('--depthG', type=int, default=3)

    args = parser.parse_args()
    
    main_sample(args.vocab, args.output_file, args.model, args.nsample, args.hidden_size, args.latent_size, args.depthT, args.depthG)