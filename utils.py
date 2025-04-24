from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity  
import torch
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import QED
from model import VAE_Autoencoder
from rdkit.Contrib.SA_Score import sascorer # type: ignore
from dataset import *
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Define the model

def set_model(args):
    my_tokenizer = tokenizer()
    vocab_size = my_tokenizer.vocab_size
    model = VAE_Autoencoder(vocab_size, args.embedding_dim, args.hidden_size, args.latent_size, args.max_len, model_type=args.model)
    model.vocab = my_tokenizer.char_to_int
    model.to(args.device)
    model.load_state_dict(torch.load(f'model/{args.model}_model.pth', map_location=args.device))
    return model


def latent_visulization(args):
    train_latent_vectors = np.load(f'results/{args.model}_latent_representation.npy')

    model = set_model(args)
    



    # 4.1 使用 KDE 拟合训练集潜在向量
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(train_latent_vectors) #调整带宽
    # 4.2 从 KDE 中采样
    num_samples = args.num_samples
    z_samples = kde.sample(num_samples)
    z_samples = torch.tensor(z_samples, dtype=torch.float32).to(args.device) #

    # 4.3 解码采样点
    model.eval()
    with torch.no_grad():
        sampled_logits = model.decode(z_samples, args.max_len)
        sampled_indices = torch.argmax(sampled_logits, dim=-1).cpu().numpy()


    # 4.4 可视化
    # 生成网格点
    x_grid = np.linspace(train_latent_vectors[:, 0].min() - 1, train_latent_vectors[:, 0].max() + 1, 100) #根据数据范围
    y_grid = np.linspace(train_latent_vectors[:, 1].min() - 1, train_latent_vectors[:, 1].max() + 1, 100) #
    xx, yy = np.meshgrid(x_grid, y_grid)
    xy = np.vstack([xx.ravel(), yy.ravel()]).T

    # 计算每个网格点的概率密度
    log_dens = kde.score_samples(xy)
    dens = np.exp(log_dens).reshape(xx.shape)

    # Sample 5000 random indices from train_latent_vectors for visualization
    num_points_to_visualize = 0
    if len(train_latent_vectors) > num_points_to_visualize:
        indices = np.random.choice(len(train_latent_vectors), size=num_points_to_visualize, replace=False)
        train_latent_vectors_subset = train_latent_vectors[indices]
    else:
        train_latent_vectors_subset = train_latent_vectors


    # 绘制等高线图
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, dens, cmap='Blues')
    plt.scatter(train_latent_vectors_subset[:, 0], train_latent_vectors_subset[:, 1], c='red', s=10, label='Training Data')
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title(f"KDE of Latent Space ({args.model})")
    plt.colorbar(label="Probability Density")
    plt.legend()
    plt.savefig(f'latent_space_visualization_{args.model}.png')
    plt.show()


def indices_to_smiles(indices, int_to_char):
    smiles_list = []
    for seq in indices:
        smiles = "".join([int_to_char[i] for i in seq])
        # 去除 <sos>, <eos> 和 <pad>
        smiles = smiles.replace('^', '').replace('$', '').replace(' ', '')
        smiles_list.append(smiles)
    return smiles_list

# Validity
def check_validity(smiles_list):
    valid_smiles = []
    for smi in smiles_list:
        if rdkit.Chem.MolFromSmiles(smi) is not None:
            valid_smiles.append(smi)
        else:
            pass
    
    validity = len(valid_smiles) / len(smiles_list)
    return valid_smiles, validity

# Uniqueness
def check_uniqueness(valid_smiles):
    smiles = [rdkit.Chem.MolToSmiles(rdkit.Chem.MolFromSmiles(smi), canonical=True) for smi in valid_smiles]
    unique_smiles = set(smiles)
    uniqueness = len(unique_smiles) / len(valid_smiles)
    return unique_smiles, uniqueness

def check_novelty(unique_smiles, refer_smiles):
    refer_smiles_set = set(refer_smiles)
    novel_smiles = unique_smiles - refer_smiles_set
    novelty = len(novel_smiles) / len(unique_smiles)
    return novel_smiles, novelty

def check_diversity(valid_smiles):
    mols = [rdkit.Chem.MolFromSmiles(smi) for smi in valid_smiles]
    generator = GetMorganGenerator(2, fpSize=2048)
    fps = [generator.GetFingerprint(mol) for mol in mols]
    if len(fps) < 2:
        return 0
    similarities = []
    for i in range(len(fps)):
        for j in range(i+1, len(fps)):
            sim = rdkit.Chem.DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)
    avg_similarities = np.mean(similarities)
    diversity = 1 - avg_similarities
    return diversity

def evaluate(args,my_tokenizer):
    train_latent_vectors = np.load(f'results/{args.model}_latent_vectors_{args.latent_size}_{args.cell_name}.npy')

    vocab_size = my_tokenizer.vocab_size
    model = VAE_Autoencoder(vocab_size, args.embedding_dim, args.hidden_size, args.latent_size, args.max_len, model_type=args.model)
    model.vocab = my_tokenizer.char_to_int
    model.to(args.device)
    model.load_state_dict(torch.load(f'model/VAE_model_{args.cell_name}.pth', map_location=args.device))
    # 4.1 使用 KDE 拟合训练集潜在向量
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(train_latent_vectors) #调整带宽
    # 4.2 从 KDE 中采样
    num_samples = args.num_samples
    z_samples = kde.sample(num_samples)
    z_samples = torch.tensor(z_samples, dtype=torch.float32).to(args.device) #

    # 4.3 解码采样点
    model.eval()
    with torch.no_grad():
        sampled_logits = model.decode(z_samples, args.max_len)
        sampled_indices = torch.argmax(sampled_logits, dim=-1).cpu().numpy()
    my_tokenizer = tokenizer()
    sampled_smiles = indices_to_smiles(sampled_indices, my_tokenizer.int_to_char)

    # 5.1 过滤无效的 SMILES (与之前相同)
    valid_smiles = []
    for smiles in sampled_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles.append(smiles)

    # 5.2 计算 QED 和 SAS (与之前相同)
    qed_values = []
    sas_values = []
    for smiles in valid_smiles:
        mol = Chem.MolFromSmiles(smiles)
        qed_values.append(QED.qed(mol))
        sas_values.append(sascorer.calculateScore(mol))

    # 5.3 计算平均值 (与之前相同)
    avg_qed = np.mean(qed_values)
    avg_sas = np.mean(sas_values)

    print(f"有效 SMILES 数量: {len(valid_smiles)} / {len(sampled_smiles)}")
    print(f"平均 QED: {avg_qed:.4f}")
    print(f"平均 SAS: {avg_sas:.4f}")


def latent_sample(args):
    model = set_model(args)
    z_samples = np.random.randn(args.num_samples, args.latent_size)
    z_samples = torch.tensor(z_samples, dtype=torch.float32).to(args.device)

    model.eval()
    with torch.no_grad():
        sampled_logits = model.decode(z_samples, args.max_len)
        sampled_indices = torch.argmax(sampled_logits, dim=-1).cpu().numpy()

    my_tokenizer = tokenizer()
    sampled_smiles = indices_to_smiles(sampled_indices, my_tokenizer.int_to_char)

    for smile in sampled_smiles:
        print(smile)


# Calculate candidates for exploration
def generate_candidates_near(anchors, num_samples, latent_size, noise_std, lb=None, ub=None):
    """
    Generates candidate points by perturbing anchor points (e.g. Z_obs) for broader exploration.
    """
    num_anchors = anchors.shape[0]
    candidates = []
    indices = np.random.choice(num_anchors, size=num_samples,replace=True)
    selected_anchors = anchors[indices]
    noise = np.random.randn(num_samples, latent_size) * noise_std
    candidates = selected_anchors + noise

    if lb is not None and ub is not None:
        candidates = np.clip(candidates, lb, ub)
    
    return candidates

def generate_candidates_target(mu_target, std_target, num_samples, latent_size):
    """
    Generates candidates by sampling from the learned target Gaussian distribution.
    """
    std_target = np.maximum(std_target, 1e-6)
    return mu_target + np.random.randn(num_samples, latent_size) * std_target