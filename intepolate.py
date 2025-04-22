import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Contrib.SA_Score import sascorer # type: ignore
from model import VAE_Autoencoder
from dataset import tokenizer, Gene_Dataloader
import argparse

# 球形插值函数
def slerp(z1, z2, t):
    z1_norm = z1 / torch.norm(z1)
    z2_norm = z2 / torch.norm(z2)
    dot = torch.dot(z1_norm, z2_norm).clamp(-1, 1)  # 避免数值误差
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    if sin_theta < 1e-6:  # 当theta接近0时，使用线性插值
        return (1 - t) * z1 + t * z2
    return (torch.sin((1 - t) * theta) / sin_theta) * z1 + (torch.sin(t * theta) / sin_theta) * z2

# 插值并生成分子
def interpolate_and_generate(model, z1, z2, num_steps=11, device='cpu'):
    t_values = np.linspace(0, 1, num_steps)
    linear_z = []
    slerp_z = []
    
    for t in t_values:
        z_linear = (1 - t) * z1 + t * z2
        z_slerp = slerp(z1, z2, t)
        linear_z.append(z_linear)
        slerp_z.append(z_slerp)
    
    linear_z = torch.stack(linear_z).to(device)
    slerp_z = torch.stack(slerp_z).to(device)
    
    with torch.no_grad():
        linear_logits = model.decode(linear_z, model.max_len)
        slerp_logits = model.decode(slerp_z, model.max_len)
    
    linear_smiles = indices_to_smiles(linear_logits.argmax(dim=-1).cpu().numpy(), model.int_to_char)
    slerp_smiles = indices_to_smiles(slerp_logits.argmax(dim=-1).cpu().numpy(), model.int_to_char)
    
    return linear_smiles, slerp_smiles

# SMILES转换函数（基于您的tokenizer）
def indices_to_smiles(indices, int_to_char):
    smiles_list = []
    for seq in indices:
        chars = [int_to_char[i] for i in seq if i in int_to_char and int_to_char[i] not in ['^', '$', ' ']]
        smiles = ''.join(chars)
        smiles_list.append(smiles)
    return smiles_list

# 评估函数
def evaluate_interpolation(smiles_list):
    valid_smiles = [smi for smi in smiles_list if Chem.MolFromSmiles(smi) is not None]
    validity = len(valid_smiles) / len(smiles_list) if smiles_list else 0.0
    
    if len(valid_smiles) < 2:
        return validity, 0.0, 0.0, 0.0
    
    qed_values = [QED.qed(Chem.MolFromSmiles(smi)) for smi in valid_smiles]
    sas_values = [sascorer.calculateScore(Chem.MolFromSmiles(smi)) for smi in valid_smiles]
    
    qed_diffs = [abs(qed_values[i] - qed_values[i-1]) for i in range(1, len(qed_values))]
    sas_diffs = [abs(sas_values[i] - sas_values[i-1]) for i in range(1, len(sas_values))]
    
    fp_gen = GetMorganGenerator(radius=2)
    fps = [fp_gen.GetFingerprint(Chem.MolFromSmiles(smi)) for smi in valid_smiles]
    similarities = [DataStructs.TanimotoSimilarity(fps[i], fps[i-1]) for i in range(1, len(fps))]
    
    avg_qed_diff = np.mean(qed_diffs) if qed_diffs else 0.0
    avg_sas_diff = np.mean(sas_diffs) if sas_diffs else 0.0
    avg_similarity = np.mean(similarities) if similarities else 0.0
    
    return validity, avg_qed_diff, avg_sas_diff, avg_similarity

# 主函数
def main(args):
    # 初始化tokenizer和模型
    tk = tokenizer()
    model = VAE_Autoencoder(
        vocab_size=tk.vocab_size,
        embed_size=args.embedding_dim,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        max_len=args.max_len,
        model_type='VAE'
    ).to(args.device)
    model.load_state_dict(torch.load(f'model/VAE_model_{args.cell_name}.pth', map_location=args.device))
    model.eval()
    model.vocab = tk.char_to_int
    model.int_to_char = tk.int_to_char

    # 获取数据
    train_loader, _ = Gene_Dataloader(args.batch_size, args.path, args.cell_name, 1.0).get_dataloader()
    batch = next(iter(train_loader)).to(args.device)
    
    # 编码分子对
    with torch.no_grad():
        mu, _, _ = model.encode(batch)
        z1, z2 = mu[0], mu[1]  # 选择前两个分子
    
    # 插值并生成分子
    linear_smiles, slerp_smiles = interpolate_and_generate(model, z1, z2, num_steps=11, device=args.device)
    
    # 评估插值结果
    linear_validity, linear_qed_diff, linear_sas_diff, linear_sim = evaluate_interpolation(linear_smiles)
    slerp_validity, slerp_qed_diff, slerp_sas_diff, slerp_sim = evaluate_interpolation(slerp_smiles)
    
    # 输出结果
    print("=== Linear Interpolation Results ===")
    print(f"Validity: {linear_validity:.4f}")
    print(f"Avg QED Difference: {linear_qed_diff:.4f}")
    print(f"Avg SAS Difference: {linear_sas_diff:.4f}")
    print(f"Avg Tanimoto Similarity: {linear_sim:.4f}")
    print("\nSMILES Sequence:")
    for i, smi in enumerate(linear_smiles):
        print(f"t={i/10:.1f}: {smi}")
    
    print("\n=== Spherical Interpolation Results ===")
    print(f"Validity: {slerp_validity:.4f}")
    print(f"Avg QED Difference: {slerp_qed_diff:.4f}")
    print(f"Avg SAS Difference: {slerp_sas_diff:.4f}")
    print(f"Avg Tanimoto Similarity: {slerp_sim:.4f}")
    print("\nSMILES Sequence:")
    for i, smi in enumerate(slerp_smiles):
        print(f"t={i/10:.1f}: {smi}")
    
    # 判断潜空间几何
    if (linear_validity > slerp_validity and 
        linear_qed_diff < slerp_qed_diff and 
        linear_sas_diff < slerp_sas_diff and 
        linear_sim > slerp_sim):
        print("\n结论：线性插值表现更优，潜空间更接近欧式空间。")
    elif (slerp_validity > linear_validity and 
          slerp_qed_diff < linear_qed_diff and 
          slerp_sas_diff < linear_sas_diff and 
          slerp_sim > linear_sim):
        print("\n结论：球形插值表现更优，潜空间更接近球形空间。")
    else:
        print("\n结论：两种插值表现混合，需进一步分析潜空间结构。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--latent_size', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--path', type=str, default='data/')
    parser.add_argument('--cell_name', type=str, default='zinc')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    main(args)