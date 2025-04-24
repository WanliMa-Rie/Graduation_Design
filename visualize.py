import torch
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import QED, RDConfig
import os
import sys
import argparse # For setting arguments like model path, etc.

# Assuming 'sascorer' is available in the path
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer # type: ignore

# Assuming your model definition and tokenizer are importable
from model import VAE_Autoencoder
from dataset import tokenizer
from bo_idl import logits_to_smiles # Use the same decoding helper

# Disable RDKit logs if desired
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

def evaluate_target_dist_sampling(args):
    """
    Loads a trained VAE model AND target distribution parameters (mu, std),
    samples from the target Gaussian distribution, decodes samples,
    calculates QED and SAS for valid molecules, and visualizes the results.
    """
    print("-" * 50)
    print(f"Evaluating VAE model by sampling from learned target distribution.")
    print(f"Model Path: {args.model_path}")
    print(f"Target Mu Path: {args.mu_path}")
    print(f"Target Std Path: {args.std_path}")
    print(f"Number of Samples: {args.num_samples}")
    print(f"Device: {args.device}")
    print("-" * 50)

    # --- 1. Initialization ---
    my_tokenizer = tokenizer()
    vocab_size = my_tokenizer.vocab_size
    device = torch.device(args.device)

    # --- 2. Load Model ---
    print("Loading model...")
    # Ensure these dimensions match the saved model!
    model = VAE_Autoencoder(
        vocab_size=vocab_size,
        embed_size=args.embedding_dim,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        max_len=args.max_len,
        model_type='VAE'
    ).to(device)
    model.vocab = my_tokenizer.char_to_int

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # --- 3. Load Target Distribution Parameters ---
    print("Loading target distribution parameters (mu, std)...")
    if not os.path.exists(args.mu_path):
        print(f"Error: Target mu file not found at {args.mu_path}")
        return
    if not os.path.exists(args.std_path):
        print(f"Error: Target std file not found at {args.std_path}")
        return

    mu_target_np = np.load(args.mu_path)
    std_target_np = np.load(args.std_path)

    # Basic shape check
    if mu_target_np.shape != (args.latent_size,) or std_target_np.shape != (args.latent_size,):
        print(f"Error: Shape mismatch for loaded mu/std. Expected ({args.latent_size},), got mu:{mu_target_np.shape}, std:{std_target_np.shape}")
        return
    print(f"Target mu shape: {mu_target_np.shape}, Target std shape: {std_target_np.shape}")

    # Ensure std dev is positive
    std_target_np = np.maximum(std_target_np, 1e-6)


    # --- 4. Sample from Target Distribution ---
    print(f"Sampling {args.num_samples} points from target Gaussian distribution...")
    # Sample using numpy: z = mu + N(0, I) * std
    noise = np.random.randn(args.num_samples, args.latent_size)
    z_samples_np = mu_target_np + noise * std_target_np
    # Convert to torch tensor for decoding
    z_samples = torch.tensor(z_samples_np, dtype=torch.float32, device=device)
    print(f"Generated z_samples shape: {z_samples.shape}")


    # --- 5. Decode Samples ---
    print("Decoding sampled latent vectors...")
    decoding_max_len = args.max_len
    with torch.no_grad():
        sampled_logits = model.decode(z_samples, decoding_max_len)
        sampled_smiles_raw = logits_to_smiles(sampled_logits, my_tokenizer)
    print(f"Generated {len(sampled_smiles_raw)} raw SMILES strings.")

    # --- 6. Calculate Properties (QED, SAS) for Valid SMILES ---
    print("Calculating QED and SAS for valid decoded SMILES...")
    valid_smiles = []
    qed_values = []
    sas_values = []
    validity_count = 0

    for smiles in sampled_smiles_raw:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            validity_count += 1
            valid_smiles.append(smiles)
            try:
                qed = QED.qed(mol)
                sas = sascorer.calculateScore(mol)
                qed_values.append(qed)
                sas_values.append(sas)
            except Exception as e:
                # print(f"Warning: Could not calculate properties for valid SMILES '{smiles}': {e}")
                pass # Skipping points where property calculation fails

    validity = validity_count / args.num_samples if args.num_samples > 0 else 0.0
    num_evaluated = len(qed_values)

    print(f"Validity: {validity:.4f} ({validity_count}/{args.num_samples})")
    if num_evaluated > 0:
        avg_qed = np.mean(qed_values)
        avg_sas = np.mean(sas_values)
        print(f"Evaluated properties for {num_evaluated} valid molecules.")
        print(f"Average QED: {avg_qed:.4f}")
        print(f"Average SAS: {avg_sas:.4f}")
    else:
        print("No valid molecules found or property calculation failed for all.")
        avg_qed, avg_sas = 0.0, 0.0

    # --- 7. Visualization ---
    if num_evaluated > 0:
        print("Generating visualization...")
        qed_array = np.array(qed_values)
        sas_array = np.array(sas_values)

        # --- Plotting setup (same as before) ---
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.01
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.2]
        rect_histy = [left + width + spacing, bottom, 0.2, height]

        fig = plt.figure(figsize=(8, 8))
        ax_scatter = plt.axes(rect_scatter)
        ax_histx = plt.axes(rect_histx)
        ax_histy = plt.axes(rect_histy)
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)
        fig.subplots_adjust(wspace=0, hspace=0)

        # --- Scatter plot (same as before) ---
        ax_scatter.scatter(sas_array, qed_array, alpha=0.5, s=20, c='green', edgecolors='w', linewidths=0.5) # Changed color to green for distinction
        ax_scatter.set_xlabel("SAS (Lower is Better)")
        ax_scatter.set_ylabel("QED (Higher is Better)")
        ax_scatter.grid(True, linestyle='--', alpha=0.6)

        # --- Histogram setup (same as before) ---
        binwidth_qed = 0.05
        binwidth_sas = 0.2
        xlim = (np.min(sas_array) - binwidth_sas, np.max(sas_array) + binwidth_sas)
        ylim = (np.min(qed_array) - binwidth_qed, np.max(qed_array) + binwidth_qed)
        ax_scatter.set_xlim(xlim)
        ax_scatter.set_ylim(ylim)
        bins_sas = np.arange(xlim[0], xlim[1] + binwidth_sas, binwidth_sas)
        bins_qed = np.arange(ylim[0], ylim[1] + binwidth_qed, binwidth_qed)

        # --- Histograms (same as before, maybe adjust colors) ---
        ax_histx.hist(sas_array, bins=bins_sas, color='skyblue', edgecolor='grey')
        ax_histy.hist(qed_array, bins=bins_qed, orientation='horizontal', color='lightcoral', edgecolor='grey')

        # --- Alignment and Titles (Adjust title) ---
        ax_histx.set_xlim(ax_scatter.get_xlim())
        ax_histy.set_ylim(ax_scatter.get_ylim())
        ax_histx.set_title(f"QED vs SAS Dist. from Target Sampling (N={args.num_samples})") # Modified title
        ax_histx.set_ylabel("Count")
        ax_histy.set_xlabel("Count")

        # --- Save Figure (same as before) ---
        save_path = args.output_plot_path
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        # plt.show()
        plt.close(fig)

    else:
        print("Skipping visualization as no properties could be evaluated.")

    print("-" * 50)
    print("Evaluation finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate VAE by sampling learned target distribution")

    # --- Model Architecture Arguments ---
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of character embeddings')
    parser.add_argument('--hidden_size', type=int, default=256, help='Size of RNN hidden states')
    parser.add_argument('--latent_size', type=int, default=64, help='Dimension of the latent space')
    parser.add_argument('--max_len', type=int, default=120, help='Max sequence length used in model definition/decoding')

    # --- Evaluation Arguments ---
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained .pth VAE model file')
    # <<< NEW: Arguments for target distribution parameters >>>
    parser.add_argument('--mu_path', type=str, default='results/bo_idl_zinc_64d/final_target_mu.npy', help='Path to the final_target_mu.npy file')
    parser.add_argument('--std_path', type=str, default='results/bo_idl_zinc_64d/final_target_std.npy', help='Path to the final_target_std.npy file')

    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to draw from the target distribution') # Increased default
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda/cpu)')
    parser.add_argument('--output_plot_path', type=str, default='results/sample/optimized_sample.png', help='Path to save the output plot')

    args = parser.parse_args()

    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_plot_path), exist_ok=True)

    evaluate_target_dist_sampling(args)