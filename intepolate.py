import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import QED, RDConfig, Draw
from rdkit import RDLogger
import os
import sys
import pickle
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer  # type: ignore
from model import VAE_Autoencoder
from dataset import tokenizer  # Assumes tokenizer has char_to_int and int_to_char

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
Tokenizer = tokenizer()
vocab_size = len(Tokenizer.char_to_int)
embed_size = 128
hidden_size = 256
latent_size = 64
max_len = 120
model_type = 'VAE'
model_path = 'results/bo_idl_zinc_64d/VAE_model_zinc_final_optimized.pth'

# Load the model
model = VAE_Autoencoder(vocab_size, embed_size, hidden_size, latent_size, max_len, model_type).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Set the vocab for the model
model.vocab = Tokenizer.char_to_int

# Function to decode logits to SMILES
def logits_to_smiles(logits, int_to_char):
    indices = logits.argmax(dim=-1)  # (batch_size, seq_len)
    batch_smiles = []
    for seq in indices:
        chars = []
        for idx in seq:
            char = int_to_char[idx.item()]
            if char == '$':  # End token
                break
            if char != '^':  # Skip start token
                chars.append(char)
        smiles = ''.join(chars)
        batch_smiles.append(smiles)
    return batch_smiles

# Function to validate SMILES
def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        return mol is not None
    except:
        return False

# Function to compute SAS and QED
def compute_metrics(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        sas = sascorer.calculateScore(mol)
        qed = QED.qed(mol)
        return sas, qed
    except:
        return None, None

# Spherical linear interpolation (slerp)
def slerp(z1, z2, t):
    omega = torch.acos(torch.dot(z1 / torch.norm(z1), z2 / torch.norm(z2)))
    sin_omega = torch.sin(omega)
    if sin_omega == 0:
        return (1 - t) * z1 + t * z2  # Fallback to linear if parallel
    return torch.sin((1 - t) * omega) / sin_omega * z1 + torch.sin(t * omega) / sin_omega * z2

# Sample valid latent vectors
def sample_valid_z(n_samples=1):
    valid_z = []
    while len(valid_z) < n_samples:
        z = torch.randn(1, latent_size).to(device)  # Sample from standard normal
        with torch.no_grad():
            outputs = model.decode(z, max_len)  # (1, max_len, vocab_size)
        smiles = logits_to_smiles(outputs, Tokenizer.int_to_char)[0]
        if is_valid_smiles(smiles):
            valid_z.append(z.squeeze(0))
    return valid_z

# Sample two valid latent vectors
z1, z2 = sample_valid_z(2)
z1, z2 = z1.to(device), z2.to(device)

# Interpolate: 50 points + start and end (52 total)
n_points = 150
t = np.linspace(0, 1, n_points + 2)  # Includes start and end

# Euclidean interpolation
euclidean_z = []
for ti in t:
    z_t = (1 - ti) * z1 + ti * z2
    euclidean_z.append(z_t)
euclidean_z = torch.stack(euclidean_z)  # (52, latent_size)

# Spherical interpolation
spherical_z = []
for ti in t:
    z_t = slerp(z1, z2, torch.tensor(ti, device=device))
    spherical_z.append(z_t)
spherical_z = torch.stack(spherical_z)  # (52, latent_size)

# Decode interpolated points and store molecular images
def decode_and_evaluate(z_vectors, name):
    valid_z = []
    valid_smiles = []
    sas_scores = []
    qed_scores = []
    mols = []
    with torch.no_grad():
        outputs = model.decode(z_vectors, max_len)  # (52, max_len, vocab_size)
    smiles_list = logits_to_smiles(outputs, Tokenizer.int_to_char)
    for i, smiles in enumerate(smiles_list):
        if is_valid_smiles(smiles):
            sas, qed = compute_metrics(smiles)
            if sas is not None and qed is not None:
                valid_z.append(z_vectors[i].cpu().numpy())
                valid_smiles.append(smiles)
                sas_scores.append(sas)
                qed_scores.append(qed)
                mol = Chem.MolFromSmiles(smiles)
                mols.append(mol)
    return np.array(valid_z), valid_smiles, sas_scores, qed_scores, mols

# Process both interpolations
euclidean_valid_z, euclidean_smiles, euclidean_sas, euclidean_qed, euclidean_mols = decode_and_evaluate(euclidean_z, "Euclidean")
spherical_valid_z, spherical_smiles, spherical_sas, spherical_qed, spherical_mols = decode_and_evaluate(spherical_z, "Spherical")

# Visualization of molecular structures
def plot_molecules(mols, smiles_list, sas_scores, qed_scores, row, ax, cols, max_mols):
    for i, (mol, smiles, sas, qed) in enumerate(zip(mols, smiles_list, sas_scores, qed_scores)):
        if i >= max_mols:  # Limit to max_mols per row
            break
        if mol is None:
            continue
        col = i
        img = Draw.MolToImage(mol, size=(200, 200))
        ax[row, col].imshow(img)
        ax[row, col].set_title(f"SAS: {sas:.2f}, QED: {qed:.2f}", fontsize=10)
        ax[row, col].text(0.5, -0.1, smiles, ha='center', va='top', fontsize=8, wrap=True, transform=ax[row, col].transAxes)
        ax[row, col].axis('off')

# Create figure
cols = 10  # Fixed number of columns
rows = 2  # One row for Euclidean, one for Spherical
fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

# Clear all axes
for r in range(rows):
    for c in range(cols):
        ax[r, c].axis('off')

# Plot Euclidean molecules (first row)
if euclidean_mols:
    plot_molecules(euclidean_mols, euclidean_smiles, euclidean_sas, euclidean_qed, 0, ax, cols, max_mols=cols)
else:
    ax[0, 0].text(0.5, 0.5, "Euclidean: No valid molecules", fontsize=12, ha='center', va='center')
    ax[0, 0].axis('off')

# Plot Spherical molecules (second row)
if spherical_mols:
    plot_molecules(spherical_mols, spherical_smiles, spherical_sas, spherical_qed, 1, ax, cols, max_mols=cols)
else:
    ax[1, 0].text(0.5, 0.5, "Spherical: No valid molecules", fontsize=12, ha='center', va='center')
    ax[1, 0].axis('off')

# Add row titles
fig.suptitle("Euclidean Interpolation (Top Row), Spherical Interpolation (Bottom Row)", fontsize=14, y=1.05)

plt.tight_layout()
plt.savefig('interpolated_molecules.png', bbox_inches='tight')
plt.close()

# Save metrics
metrics = {
    'euclidean': {
        'smiles': euclidean_smiles,
        'sas': euclidean_sas,
        'qed': euclidean_qed
    },
    'spherical': {
        'smiles': spherical_smiles,
        'sas': spherical_sas,
        'qed': spherical_qed
    }
}
with open('interpolation_metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

print(f"Euclidean interpolation: {len(euclidean_smiles)} valid molecules")
print(f"Spherical interpolation: {len(spherical_smiles)} valid molecules")
print("Metrics saved to 'interpolation_metrics.pkl'")
print("Plot saved to 'interpolated_molecules.png'")