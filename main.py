
from dataset import *
from utils import *
from train import train
from bo_idl import optimize_molecules, SAS_AVAILABLE

import argparse
import torch 
import os

# =============== Params Settings ================
parser = argparse.ArgumentParser()
# ----- Task execution ------
parser.add_argument('--train_model', action='store_true', help='Train VAE')
parser.add_argument('--optimize', action='store_true', help='Run BO-IDL Multi-Objective Optimization')
parser.add_argument('--visualization', action='store_true', help='Latent space visualization (from utils.py)')
parser.add_argument('--evaluation', action='store_true', help='Evaluation ASA and QED (from utils.py)')
parser.add_argument('--latent_sample', action='store_true',help='Sample latent vector (from utils.py)')

# ----- Model params settings -----
parser.add_argument('--embedding_dim', type=int, default=128, help='The dimension of embedding')
parser.add_argument('--hidden_size',type=int, default=256, help='The dimension of NN hidden layer')
parser.add_argument('--latent_size', type=int, default=64, help='The dimension of latent representation')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate of Adam for VAE training')
parser.add_argument('--num_epochs', type=int, default=100, help='Training epochs for VAE')
parser.add_argument('--batch_size', type=int, default=64, help='Training batch')
parser.add_argument('--model', type=str, default='VAE', choices=['VAE', 'AE'], help='VAE or AE')
parser.add_argument('--train_rate', type=float, default=0.8)
parser.add_argument('--device', type=str, default='cpu', help="Device ('cpu', 'cuda', 'mps')")

# ----- Data info -----
parser.add_argument('--path', type=str, default='data/', help="Path to data directory")
parser.add_argument('--cell_name', type=str, default='zinc', help="Dataset name (e.g., 'zinc' without extension)")
parser.add_argument('--max_len', type=int, default=120, help="Maximum SMILES length for padding/generation")

# ----- Visualization and evaluation info -----
parser.add_argument('--num_samples', type=int, default=1000, help="Number of samples for visualization/evaluation")

# ----- BO-IDL specific parameters -----
parser.add_argument('--bo_idl_iterations', type=int, default=80, help='Number of BO-IDL iterations')
parser.add_argument('--bo_initial_points', type=int, default=3000, help='Number of initial points to evaluate before BO')
parser.add_argument('--bo_points_per_iter', type=int, default=100, help='Number of points selected by BO in each iteration')
parser.add_argument('--bo_acquisition_samples', type=int, default=1000, help='Number of candidates sampled for acquisition function')
parser.add_argument('--idl_epochs', type=int, default=60, help='Number of epochs for IDL update')
parser.add_argument('--idl_lr', type=float, default=1e-4, help='Learning rate for IDL optimizer')
parser.add_argument('--idl_kl_weight', type=float, default=0.01, help='Weight for KL term in IDL loss')


args = parser.parse_args()

# Set device
if args.device == 'mps':
    if not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = 'cpu'
elif args.device == 'cuda':
     if not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'


# Ensure results and model directories exist
os.makedirs("results/latent", exist_ok=True)
os.makedirs("model", exist_ok=True)

print(f"Using device: {args.device}")

def main(args):

    # =============== Train Models ================
    if args.train_model:
        print("--- Starting VAE Training ---")
        train(args)
        print("--- VAE Training Finished ---")

    # =============== Optimize Molecules ================
    if args.optimize:
        # Check if VAE model exists before starting optimization
        model_path = f'model/{args.model}_model_{args.cell_name}.pth'
        if not os.path.exists(model_path):
             print(f"Error: Pre-trained model not found at {model_path}")
             print("Please train the VAE model first using --train_model flag.")
        elif args.model != 'VAE':
             print("Error: Optimization currently only supports VAE model type.")
        else:
            print("--- Starting MOBO-IDL Optimization ---")
            if not SAS_AVAILABLE:
                 print("\nWARNING: sascorer not found. SAS objective will be approximated. Optimization quality might be affected.\n")
            optimize_molecules(args)
            print("--- MOBO-IDL Optimization Finished ---")


    # =============== Sample and Visualize (Optional post-hoc) ================
    # These might need adaptation if they rely on specific outputs not generated by optimize_molecules
    if args.visualization:
        print("--- Starting Latent Space Visualization ---")
        # Assuming latent_visulization function exists in utils.py
        # You might need to point it to the results from the optimization run
        try:
            latent_visulization(args) # Modify this function if needed
        except NameError:
            print("Warning: latent_visulization function not found in utils.py")
        except Exception as e:
            print(f"Error during visualization: {e}")
        print("--- Visualization Finished ---")


    # ============= Evaluate SAS and QED (Optional post-hoc) ===============
    if args.evaluation:
        print("--- Starting Property Evaluation ---")
        # Assuming evaluate function exists in utils.py
        # You likely want to evaluate the final Pareto SMILES generated by optimization
        try:
            evaluate(args, tokenizer()) # Modify this function if needed
        except NameError:
            print("Warning: evaluate function not found in utils.py")
        except Exception as e:
            print(f"Error during evaluation: {e}")
        print("--- Evaluation Finished ---")

    # ============= Latent Sampling (Optional post-hoc) ===============
    if args.latent_sample:
        print("--- Starting Latent Sampling ---")
         # Assuming latent_sample function exists in utils.py
         # You might want to sample from the final optimized model
        try:
            latent_sample(args) # Modify this function if needed
        except NameError:
            print("Warning: latent_sample function not found in utils.py")
        except Exception as e:
            print(f"Error during latent sampling: {e}")
        print("--- Latent Sampling Finished ---")


if __name__ == '__main__':
    main(args)
