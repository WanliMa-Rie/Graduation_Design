import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math, random, sys
import numpy as np
import argparse
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from typing import List, Tuple

# 假设 fuseprop 模块已正确导入
from conditionVAE import *

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--rationale', default='c1ccc[c:1]c1', help='Path to rationale file or single SMILES (default: benzene ring)')
parser.add_argument('--train_molecules', type=str, default='data/ZINC/zinc.txt', help='Path to training set SMILES file')
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--model', required=True, help='Path to pretrained model checkpoint')
parser.add_argument('--num_decode', type=int, default=100, help='Number of molecules to generate per rationale')
parser.add_argument('--num_samples', type=int, default=1000, help='Total number of molecules to generate')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--embed_size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--latent_size', type=int, default=64)
parser.add_argument('--depth', type=int, default=10)
parser.add_argument('--diter', type=int, default=3)
args = parser.parse_args()

# 设置随机种子
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# 加载模型
model = AtomVGNN(args)#.cuda()
model_ckpt = torch.load(args.model, map_location=torch.device('cpu'))#, map_location='cuda')
if isinstance(model_ckpt, tuple):
    print('loading model with rationale distribution', file=sys.stderr)
    testdata = list(model_ckpt[0].keys())
    model.load_state_dict(model_ckpt[1])
else:
    print('loading pre-trained model', file=sys.stderr)
    if args.rationale.endswith('.smi') or args.rationale.endswith('.txt'):
        testdata = [line.split()[0] for line in open(args.rationale)] 
    else:
        testdata = [args.rationale]  # 单个 SMILES（例如，苯环）
    testdata = unique_rationales(testdata)
    model.load_state_dict(model_ckpt)

print('total # rationales:', len(testdata), file=sys.stderr)
model.eval()

# 加载训练集分子
train_molecules = []
i = 0
try:
    with open(args.train_molecules, "r") as f:
        for line in f:
            if line.strip() and i < 50000:
                train_molecules.append(line.strip())
                i += 1
        # train_molecules = [line.strip() for line in f if line.strip()]
    print(f'Loaded {len(train_molecules)} training molecules', file=sys.stderr)
except FileNotFoundError:
    print(f'Error: {args.train_molecules} not found', file=sys.stderr)
    sys.exit(1)

# 数据集和加载器
dataset = SubgraphDataset(testdata, args.atom_vocab, args.batch_size, args.num_decode)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])

# 评估函数
def compute_metrics(generated_smiles: List[str], train_molecules: List[str]) -> Tuple[float, float, float]:
    """
    计算生成分子的有效性、多样性和新颖性。
    Args:
        generated_smiles: 生成的分子 SMILES 列表
        train_molecules: 训练集分子 SMILES 列表
    Returns:
        Tuple: (validity, diversity, novelty)
    """
    valid_molecules = []
    valid_smiles = []

    # 计算有效性
    for smi in generated_smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol and Chem.SanitizeMol(mol, catchErrors=True) == 0:
                valid_molecules.append(mol)
                valid_smiles.append(smi)
        except:
            pass
    
    validity = len(valid_molecules) / len(generated_smiles) if generated_smiles else 0.0
    print(f"Validity: {validity:.4f} ({len(valid_molecules)}/{len(generated_smiles)} valid molecules)")

    if len(valid_molecules) < 2:
        print("Not enough valid molecules to compute diversity and novelty.", file=sys.stderr)
        return validity, 0.0, 0.0

    # 计算多样性
    def compute_tanimoto_similarity(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 2048)
        return rdkit.DataStructs.TanimotoSimilarity(fp1, fp2)

    total_similarity = 0.0
    pairs = 0
    for i in range(len(valid_molecules)):
        for j in range(i + 1, len(valid_molecules)):
            total_similarity += compute_tanimoto_similarity(valid_molecules[i], valid_molecules[j])
            pairs += 1
    
    avg_similarity = total_similarity / pairs if pairs > 0 else 0.0
    diversity = 1.0 - avg_similarity
    print(f"Diversity: {diversity:.4f}")

    # 计算新颖性
    def compute_nearest_neighbor_similarity(gen_mol: Chem.Mol, train_mols: List[Chem.Mol]) -> float:
        gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_mol, 2, 2048)
        similarities = [
            rdkit.DataStructs.TanimotoSimilarity(gen_fp, AllChem.GetMorganFingerprintAsBitVect(train_mol, 2, 2048))
            for train_mol in train_mols if train_mol is not None
        ]
        return min(similarities) if similarities else 1.0

    train_mols = [Chem.MolFromSmiles(smi) for smi in train_molecules if Chem.MolFromSmiles(smi)]
    
    novelty_count = 0
    for mol in tqdm(valid_molecules, desc="Computing novelty"):
        nn_similarity = compute_nearest_neighbor_similarity(mol, train_mols)
        if nn_similarity < 0.4:
            novelty_count += 1
    
    novelty = novelty_count / len(valid_molecules) if valid_molecules else 0.0
    print(f"Novelty: {novelty:.4f}")

    return validity, diversity, novelty

# 主逻辑
all_generated_smiles = []
with torch.no_grad():
    for init_smiles in tqdm(loader, desc="Generating molecules"):
        final_smiles = model.decode(init_smiles)
        for x, y in zip(init_smiles, final_smiles):
            print(x, y)  # 保持原有输出
            all_generated_smiles.append(y)

# 计算评估指标
print("\nComputing evaluation metrics...")
validity, diversity, novelty = compute_metrics(all_generated_smiles, train_molecules)

# 保存生成分子
with open("generated_molecules.smi", "w") as f:
    for smi in all_generated_smiles:
        f.write(smi + "\n")

# 打印最终结果
print("\nFinal Results:")
print(f"Validity: {validity:.4f}")
print(f"Diversity: {diversity:.4f}")
print(f"Novelty: {novelty:.4f}")