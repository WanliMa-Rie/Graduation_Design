import sys
sys.path.append('../')
import torch
import torch.nn as nn
from multiprocessing import Pool
import numpy as np
import os
from tqdm import tqdm
import math, random, sys
from optparse import OptionParser
import pickle
from fast_jtnn import *
import rdkit
from rdkit import Chem, RDLogger

# 禁用 RDKit 的警告日志
RDLogger.DisableLog('rdApp.*')

# 将 SMILES 字符串转化为分子数 MolTree 对象
def tensorize(smiles, assm=True):
    try:
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        if assm:
            mol_tree.assemble()
            for node in mol_tree.nodes:
                if node.label not in node.cands:
                    node.cands.append(node.label)
        del mol_tree.mol
        for node in mol_tree.nodes:
            del node.mol
        return mol_tree
    except Exception as e:
        print(f"Error processing SMILES: {smiles}, Exception: {e}")
        return None

# 检查 SMILES 是否有效（可被 RDKit 解析并支持 Kekulization）
def is_valid_smiles(smiles):
    """检查 SMILES 是否可以被 RDKit 处理并支持 Kekulization，包括子结构"""
    try:
        # 检查整体分子
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return False
        Chem.Kekulize(mol)

        # 模拟 MolTree 的分解过程
        from fast_jtnn.chemutils import get_clique_mol
        from fast_jtnn.mol_tree import tree_decomp

        # 分解为团
        cliques, _ = tree_decomp(mol)
        # 检查每个团是否可以 Kekulize
        for clique in cliques:
            cmol = get_clique_mol(mol, clique)
            if cmol is None:
                return False
        return True
    except Exception:
        return False

# 主处理函数，将输入的 SMILES 文件转化为张量并保存。
def convert(train_path, pool, num_splits, output_path):
    out_path = os.path.join(output_path, './')
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    
    # 读取输入文件
    with open(train_path) as f:
        data = [line.strip("\r\n ").split()[0] for line in f]
    print('Input File read')
    
    # 分析 SMILES 并过滤
    print('Analyzing SMILES...')
    valid_data = []
    invalid_data = []
    for smiles in tqdm(data, desc="Validating SMILES"):
        if is_valid_smiles(smiles):
            valid_data.append(smiles)
        else:
            invalid_data.append(smiles)
    
    # 计算并打印有效 SMILES 的占比
    total_smiles = len(data)
    valid_smiles = len(valid_data)
    valid_ratio = valid_smiles / total_smiles if total_smiles > 0 else 0
    print(f'Valid SMILES: {valid_smiles}/{total_smiles} ({valid_ratio:.2%})')
    
    # 保存有效 SMILES
    valid_smiles_file = os.path.join(output_path, 'valid_smiles.txt')
    with open(valid_smiles_file, 'w') as f:
        for smiles in valid_data:
            f.write(smiles + '\n')
    print(f'Valid SMILES saved to {valid_smiles_file}')
    
    # 保存无效 SMILES（便于调试）
    if invalid_data:
        invalid_smiles_file = os.path.join(output_path, 'invalid_smiles.txt')
        with open(invalid_smiles_file, 'w') as f:
            for smiles in invalid_data:
                f.write(smiles + '\n')
        print(f'Invalid SMILES saved to {invalid_smiles_file}')
    
    # 直接处理有效的 SMILES
    print('Tensorizing valid SMILES...')
    all_data = pool.map(tensorize, valid_data)
    
    # 过滤可能的 None 值（安全措施）
    valid_tensor_data = [x for x in all_data if x is not None]
    if len(valid_tensor_data) < len(valid_data):
        print(f"Warning: {len(valid_data) - len(valid_tensor_data)} SMILES failed during tensorization")
    
    all_data_split = np.array_split(valid_tensor_data, num_splits)
    print('Tensorizing Complete')
    
    # 保存分片
    print('Saving tensorized data...')
    for split_id in tqdm(range(num_splits), desc="Saving splits"):
        with open(os.path.join(output_path, 'tensors-%d.pkl' % split_id), 'wb') as f:
            pickle.dump(all_data_split[split_id], f)
    
    return True

# 封装预处理逻辑，提供默认参数。
def main_preprocess(train_path, output_path, num_splits=10, njobs=os.cpu_count()):
    pool = Pool(njobs)
    convert(train_path, pool, num_splits, output_path)
    return True

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train_path")
    parser.add_option("-n", "--split", dest="nsplits", default=10)
    parser.add_option("-j", "--jobs", dest="njobs", default=8)
    parser.add_option("-o", "--output", dest="output_path")
    
    opts, args = parser.parse_args()
    opts.njobs = int(opts.njobs)

    pool = Pool(opts.njobs)
    num_splits = int(opts.nsplits)
    convert(opts.train_path, pool, num_splits, opts.output_path)