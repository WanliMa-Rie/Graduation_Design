import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import QED, RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer #type: ignore
SAS_AVAILABLE = True
import GPy
from utils import *
from torch.nn.utils.rnn import pad_sequence

from dataset import tokenizer, Gene_Dataloader
from model import VAE_Autoencoder

# ====== functions ======
def logits_to_smiles(logits, tk):
    """Converts model output logits to SMILES strings using the tokenizer's decode method."""
    token_ids = logits.argmax(dim=-1).cpu().numpy()  # (batch_size, max_len)
    smiles_list = []
    pad_idx = tk.char_to_int.get(tk.pad, -1)
    start_idx = tk.char_to_int.get(tk.start, -1)
    end_idx = tk.char_to_int.get(tk.end, -1)

    for i in range(token_ids.shape[0]):
        ids_for_decode = []
        for token_id in token_ids[i]:
            if token_id == start_idx:
                continue
            if token_id == end_idx:
                break
            if token_id == pad_idx:
                 continue

            ids_for_decode.append(token_id)

        smiles = tk.decode(ids_for_decode)
        smiles_list.append(smiles)

    return smiles_list

def compute_targets(z, model, tk, max_len, device):
    """
    Generates molecules from latent vectors z and computes objectives (-SAS, QED).
    Handles invalid SMILES.
    Args:
        z (np.array): Batch of latent vectors (batch_size, latent_size).
        model (VAE_Autoencoder): The VAE model.
        tk (tokenizer): The SMILES tokenizer.
        max_len (int): Maximum generation length.
        device (torch.device): Computation device.

    Returns:
        np.array: Target values (batch_size, 2) where columns are [-SAS, QED].
                  Returns [-10.0, 0.0] for invalid SMILES.
        list: List of generated SMILES strings (including invalid ones).
        list: List of valid SMILES strings.
        np.array: The latent vectors corresponding to valid SMILES.
    """
    model.eval()
    targets = []
    all_smiles = []
    valid_smiles_list = []
    valid_z = []

    batch_size = z.shape[0]
    z_tensor = torch.tensor(z, dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = model.decode(z_tensor, max_len)  # (batch_size, max_len, vocab_size)
        generated_smiles = logits_to_smiles(logits, tk)

    for i, smiles in enumerate(generated_smiles):
        all_smiles.append(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:  # Check if molecule is valid
            try:
                qed = QED.qed(mol)
                sas = sascorer.calculateScore(mol)
                targets.append([-sas, qed])  # SAS is negated (maximization)
                valid_smiles_list.append(smiles)
                valid_z.append(z[i])
            except Exception as e:
                targets.append([-10.0, 0.0]) # Penalize errors during calculation too
        else:
            targets.append([-10.0, 0.0])  # Penalize invalid SMILES heavily

    return np.array(targets), all_smiles, valid_smiles_list, np.array(valid_z)


def get_pareto_front(Y):
    """
    Identifies the Pareto front from a set of observations Y.
    Assumes maximization for all objectives.

    Args:
        Y (np.array): Objective values (n_points, n_objectives).

    Returns:
        np.array: Indices of the points on the Pareto front.
    """
    is_pareto = np.ones(Y.shape[0], dtype=bool)
    for i, y in enumerate(Y):
        if is_pareto[i]:
            # Check if any other point dominates y
            is_pareto[i] = not np.any(np.all(Y[is_pareto] >= y, axis=1) & np.any(Y[is_pareto] > y, axis=1))
    return np.where(is_pareto)[0]


# ====== Bayesian Optimization ======

def run_bo(
    Z_obs, Y_obs, bounds,
    n_iter, latent_size, model, tk, max_len, device,
    main_iteration, total_iterations,
    mu_target_np, std_target_np,
    acquisition_samples=1000,
    noise_std_explore=0.5
    ):

    print(f"Running Bayesian Optimization (Main Iteration: {main_iteration+1}/{total_iterations})...")

    gp_models = []
    print("  Training GP models...")
    for i in range(Y_obs.shape[1]):
        kernel = GPy.kern.RBF(input_dim=latent_size, variance=1.0, lengthscale=1.0)
        Y_target_gp = Y_obs[:, i].reshape(-1, 1)
        gp = GPy.models.GPRegression(Z_obs, Y_target_gp, kernel)
        gp.optimize(messages=False, max_iters=200)
        gp.optimize_restarts(num_restarts=3, verbose=False)
        gp_models.append(gp)

    print("  Generating candidates...")
    lb, ub = bounds

    if main_iteration < total_iterations // 2:
        print(f"  Phase 1 (Exploration): Perturbing Z_obs (noise_std={noise_std_explore}).")
        Z_candidates = generate_candidates_near(
            Z_obs,
            acquisition_samples,
            latent_size,
            noise_std=noise_std_explore,
            lb=lb,
            ub=ub
        )
    else:
        print("  Phase 2 (Exploitation): Sampling from target distribution.")
        Z_candidates = generate_candidates_target(
            mu_target_np,
            std_target_np,
            acquisition_samples,
            latent_size
        )

    print(f"  Generated {Z_candidates.shape[0]} candidates using {'Phase 1' if main_iteration < total_iterations // 2 else 'Phase 2'} strategy.")

    print("  Predicting objectives for candidates...")
    Y_pred_mean = np.zeros((acquisition_samples, Y_obs.shape[1]))
    Y_pred_var = np.zeros((acquisition_samples, Y_obs.shape[1]))
    for i, gp in enumerate(gp_models):
            mean, var = gp.predict(Z_candidates)
            Y_pred_mean[:, i] = mean.flatten()
            Y_pred_var[:, i] = var.flatten()
            Y_pred_var[Y_pred_var < 1e-6] = 1e-6


    print("  Calculating acquisition scores...")
    acquisition_scores = []
    pareto_idx_obs = get_pareto_front(Y_obs)
    current_pareto_Y = Y_obs[pareto_idx_obs]

    if current_pareto_Y.shape[0] == 0:
        print("Warning: Current Pareto front is empty. Selecting based on predicted means.")
        acquisition_scores = np.sum(Y_pred_mean, axis=1)
    else:
        for k in range(acquisition_samples):
            y_pred_k = Y_pred_mean[k]
            potential_Y = np.vstack([current_pareto_Y, y_pred_k])
            potential_pareto_idx = get_pareto_front(potential_Y)
            new_pareto_Y = potential_Y[potential_pareto_idx]

            score = 0
            if new_pareto_Y.shape[0] > 1:
                 score = np.prod(np.max(new_pareto_Y, axis=0) - np.min(new_pareto_Y, axis=0))
            elif new_pareto_Y.shape[0] == 1:
                 score = np.prod(np.maximum(0, new_pareto_Y[0] - np.min(current_pareto_Y, axis=0))) if current_pareto_Y.shape[0] > 0 else np.prod(np.maximum(0, new_pareto_Y[0]))

            acquisition_scores.append(score)
    
    acquisition_scores = np.array(acquisition_scores)

    Z_new_list = []
    Y_new_list = []
    smiles_new_all_this_run = []
    smiles_new_valid_this_run = []

    if acquisition_scores is not None and len(acquisition_scores) > 0 and acquisition_scores.ndim > 0:
        num_to_select = min(n_iter, len(acquisition_scores))
        best_indices = np.argsort(acquisition_scores)[-num_to_select:][::-1]
    else:
        print("Warning: No valid acquisition scores calculated or array is empty/0-dim, cannot select points.")
        best_indices = []

    print(f"  Selecting {len(best_indices)} points based on acquisition scores...")
    for idx in tqdm(best_indices, desc="  BO Steps (Evaluating Selected)"):
        z_next = Z_candidates[idx:idx+1]
        y_next, smiles_next_all, smiles_next_valid, z_next_valid = compute_targets(z_next, model, tk, max_len, device)

        if len(smiles_next_valid) > 0:
            Z_new_list.extend(z_next_valid)
            Y_new_list.extend(y_next)
            smiles_new_all_this_run.extend(smiles_next_all)
            smiles_new_valid_this_run.extend(smiles_next_valid)
        else:
            pass

    Z_new_array = np.array(Z_new_list)
    Y_new_array = np.array(Y_new_list)

    if Z_new_array.shape[0] > 0:
        updated_Z_obs = np.vstack([Z_obs, Z_new_array])
        updated_Y_obs = np.vstack([Y_obs, Y_new_array])
    else:
        updated_Z_obs = Z_obs
        updated_Y_obs = Y_obs
        print("  No valid points added in this BO run.")

    print("  Updating GP models with all new points for next iteration...")
    for i, gp in enumerate(gp_models):
        if Z_new_array.shape[0] > 0:
            gp.set_XY(updated_Z_obs, updated_Y_obs[:, i].reshape(-1, 1))
            gp.optimize(messages=False, max_iters=100)
            gp.optimize_restarts(num_restarts=2, verbose=False, robust=True)
        else:
             print(f"  Skipping final GP Update for Objective {i+1} as no new valid points were added.")

    return updated_Z_obs, updated_Y_obs, Z_new_array, Y_new_array, smiles_new_all_this_run, smiles_new_valid_this_run


# --- Iterative Distribution Learning (IDL) ---

def update_decoder_idl(model, Z_elite, smiles_elite, tk, max_len, device, idl_epochs=5, idl_lr=1e-4, lambda_kl=0.01):
    """
    Updates the VAE decoder using IDL based on elite samples.

    Args:
        model (VAE_Autoencoder): The VAE model.
        Z_elite (np.array): Latent vectors of elite samples.
        smiles_elite (list): SMILES strings of elite samples.
        tk (tokenizer): The tokenizer.
        max_len (int): Maximum sequence length for padding/decoding.
        device (torch.device): Computation device.
        idl_epochs (int): Number of epochs for IDL training.
        idl_lr (float): Learning rate for IDL optimizer.
        lambda_kl (float): Weight for the KL divergence term.

    Returns:
        params of the fitted target Gaussian distribution.
    """
    mu_target_np, std_target_np = None, None
    print("Running Iterative Distribution Learning (IDL)...")
    if len(smiles_elite) == 0:
        print("  No elite samples found, skipping IDL update.")
        return mu_target_np, std_target_np

    model.train()

    # Target only decoder and latent mapping parameters
    optimizer = optim.Adam(
        list(model.decoder.parameters()) + list(model.latent_to_hidden.parameters()),
        lr=idl_lr
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tk.char_to_int[tk.pad], reduction='sum')

    # --- Target Latent Distribution ---
    # Fit a Gaussian to the elite latent points
    if Z_elite.shape[0] < 2:
        print("  Warning: Not enough elite samples (<2) to estimate covariance. Using diagonal.")
        if Z_elite.shape[0] > 0:
            mu_target_np = Z_elite.mean(axis=0)
            std_target_np = np.ones_like(mu_target_np) * 0.1
        else:
            mu_target_np, std_target_np = None, None
        mu_target = torch.tensor(mu_target_np, dtype=torch.float32, device=device)
        std_target = torch.tensor(std_target_np, dtype=torch.float32, device=device)
    else:
        mu_target_np = Z_elite.mean(axis=0)
        std_target_np = Z_elite.std(axis=0)
        
    if mu_target_np is not None and std_target_np is not None:
        mu_target = torch.tensor(mu_target_np, dtype=torch.float32, device=device)
        std_target = torch.tensor(std_target_np, dtype=torch.float32, device=device)
        target_dist = torch.distributions.Normal(mu_target, std_target)
    else:
        target_dist = None


    # --- Prepare Elite Data ---
    elite_tokens = [torch.tensor(tk.encode(smi), dtype=torch.long) for smi in smiles_elite]
    # Pad sequences
    X_elite_padded = pad_sequence(elite_tokens, batch_first=True, padding_value=tk.char_to_int[tk.pad]).to(device)
    elite_dataset = torch.utils.data.TensorDataset(X_elite_padded)
    # Use a small batch size for IDL update
    idl_batch_size = min(len(smiles_elite), 16)
    elite_loader = torch.utils.data.DataLoader(elite_dataset, batch_size=idl_batch_size)

    # --- IDL Training Loop ---
    for epoch in range(idl_epochs):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        for batch in elite_loader:
            x_batch = batch[0] # Get tensor from loader
            optimizer.zero_grad()

            # Encode elite samples using the *current* encoder
            mu_enc, logvar_enc, z_enc = model.encode(x_batch) # z_enc is reparameterized

            # Decode using the decoder we are training
            recon_logits = model.decode(z_enc, x_batch.size(1)) # Decode to original length

            # 1. Reconstruction Loss (Decoder Performance)
            # Shift for predicting next token
            target = x_batch[:, 1:]
            recon_logits_flat = recon_logits[:, :-1, :].reshape(-1, model.vocab_size)
            target_flat = target.reshape(-1)
            recon_loss = criterion(recon_logits_flat, target_flat)

            # 2. KL Divergence Loss (Latent Space Alignment)
            kl_loss = torch.tensor(0.0, device=device) # Default KL loss if target_dist is None
            if target_dist is not None:
                try:
                    q_dist = torch.distributions.Normal(mu_enc, torch.exp(0.5 * logvar_enc))
                    kl_loss = torch.distributions.kl_divergence(q_dist, target_dist).sum()
                    kl_loss = kl_loss / x_batch.size(0) # Average KL per sample in batch
                except Exception as e:
                    print(f"  Warning: KL divergence calculation failed in epoch {epoch+1}. Error: {e}")
                    kl_loss = torch.tensor(0.0, device=device) # Set KL to 0 if calc fails

            loss = recon_loss / x_batch.size(0) + lambda_kl * kl_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x_batch.size(0)
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += (lambda_kl * kl_loss.item()) * x_batch.size(0)


        avg_loss = epoch_loss / len(smiles_elite)
        avg_recon = epoch_recon_loss / len(smiles_elite)
        avg_kl = epoch_kl_loss / len(smiles_elite)
        print(f"  IDL Epoch {epoch+1}/{idl_epochs}, Avg Loss: {avg_loss:.4f}, Avg Recon: {avg_recon:.4f}, Avg KL: {avg_kl:.4f}")

    model.eval() 
    print(f"  IDL finished. Returning target mu: {mu_target_np is not None}, std: {std_target_np is not None}")
    return mu_target_np, std_target_np


# --- Main Optimization Loop ---

def optimize_molecules(args):
    """Main function to run the MOBO-IDL optimization."""
    print("Starting Molecular Optimization...")
    print(f"Arguments: {args}")

    # Initialization
    my_tokenizer = tokenizer()
    vocab_size = my_tokenizer.vocab_size
    device = torch.device(args.device)
    max_len = args.max_len # Use max_len from args for generation

    # Load Pre-trained VAE Model
    print(f"Loading pre-trained VAE model from: model/{args.model}_model_{args.cell_name}.pth")
    model = VAE_Autoencoder(vocab_size, args.embedding_dim, args.hidden_size, args.latent_size, max_len, model_type='VAE').to(device)
    model.vocab = my_tokenizer.char_to_int 
    model.load_state_dict(torch.load(f'model/{args.model}_model_{args.cell_name}.pth', map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully.")

    # Get Initial Latent Vectors
    print(f"Generating {args.bo_initial_points} initial latent points...")
    # Option 1: Encode subset of training data
    train_loader, _ = Gene_Dataloader(args.batch_size, args.path, args.cell_name, 1.0).get_dataloader()  # Load all data
    selected_samples_list = []
    print("  Selecting initial samples from training data...")
    # Iterate through the dataloader and collect raw samples first
    for batch in train_loader:
        # Assuming each batch is a tensor or can be iterated over
        for sample in batch:
            if len(selected_samples_list) < args.bo_initial_points:
                selected_samples_list.append(sample)
            else:
                break # Stop collecting samples once we have enough
        if len(selected_samples_list) >= args.bo_initial_points:
            break # Stop iterating through loader if we have enough samples

    print(f"  Collected {len(selected_samples_list)} initial samples.")

    # Stack the collected raw samples into a single tensor
    pad_idx = my_tokenizer.char_to_int.get(my_tokenizer.pad, 0) # Get pad index, default to 0 if not found
    selected_samples_tensor = pad_sequence(selected_samples_list, batch_first=True, padding_value=pad_idx).to(device)

    # Encode only the selected samples
    print("  Encoding selected initial samples...")
    with torch.no_grad():
        mu, _, _ = model.encode(selected_samples_tensor)
        Z_init = mu.cpu().numpy()
    # Option 2: Sample from prior N(0, I)
    # Z_init = np.random.randn(args.bo_initial_points, args.latent_size)

    print(f"Initial Z shape: {Z_init.shape}")

    # Compute initial objective values
    print("Evaluating initial points...")
    Y_init, smiles_init_all, smiles_init_valid, Z_init_valid = compute_targets(Z_init, model, my_tokenizer, max_len, device)
    print(f"Initial evaluation complete. Found {len(smiles_init_valid)} valid molecules.")

    # Use only valid points for the initial BO dataset
    Z_obs = Z_init_valid
    Y_obs = Y_init[np.all(Y_init != [-10.0, 0.0], axis=1)]

    num_initial_valid = Y_obs.shape[0]
    iteration_added = np.zeros(num_initial_valid, dtype=int)

    print(f"Starting BO with {Z_obs.shape[0]} valid points.")
    print(f"Initial Z_obs shape: {Z_obs.shape}, Initial Y_obs shape: {Y_obs.shape}")


    # Define latent space bounds (dynamic based on observed points)
    lb = Z_obs.min(axis=0) - 1.0
    ub = Z_obs.max(axis=0) + 1.0
    bounds = (lb, ub)


    # --- Main BO-IDL Loop ---
    all_smiles_generated = list(smiles_init_all)
    all_smiles_valid = list(smiles_init_valid) 
    hypervolumes = []
    pareto_fronts_Y = []
    mu_target_for_bo = None
    std_target_for_bo = None
    exploration_noise_std = 0.5
    for iteration in range(args.bo_idl_iterations):
        current_iter_label = iteration + 1
        print(f"\n{'='*20} Iteration {iteration + 1}/{args.bo_idl_iterations} {'='*20}")

        # 1. Bayesian Optimization Step
        Z_obs_before_bo = Z_obs
        Y_obs_before_bo = Y_obs
        iteration_added_before_bo = iteration_added

        Z_obs, Y_obs, Z_new, Y_new, smiles_new_gen, smiles_new_val = run_bo(
            Z_obs, Y_obs, bounds,
            n_iter=args.bo_points_per_iter,
            latent_size=args.latent_size,
            model=model, tk=my_tokenizer, max_len=max_len, device=device,
            main_iteration=iteration,
            total_iterations=args.bo_idl_iterations, 
            mu_target_np=mu_target_for_bo,
            std_target_np=std_target_for_bo,
            acquisition_samples=args.bo_acquisition_samples,
            noise_std_explore=exploration_noise_std
        )
        all_smiles_generated.extend(smiles_new_gen)
        all_smiles_valid.extend(smiles_new_val)

        num_new_points = Z_new.shape[0]
        if num_new_points > 0:
            new_iterations = np.full(num_new_points, current_iter_label, dtype=int)
            iteration_added = np.concatenate([iteration_added_before_bo, new_iterations])

        print(f"BO finished. Found {Z_new.shape[0]} new valid points. Total observed points: {Z_obs.shape[0]}")

        # Update bounds based on new observations
        lb = Z_obs.min(axis=0) - 1.0
        ub = Z_obs.max(axis=0) + 1.0
        bounds = (lb, ub)

        # 2. Identify Elite Samples (Pareto Front)
        pareto_idx = get_pareto_front(Y_obs)
        Z_elite = Z_obs[pareto_idx]
        Y_elite = Y_obs[pareto_idx]

        # Regenerate SMILES for elite Z to ensure consistency
        _, _, smiles_elite, _ = compute_targets(Z_elite, model, my_tokenizer, max_len, device)
        valid_elite_indices = [i for i, smi in enumerate(smiles_elite) if Chem.MolFromSmiles(smi) is not None]
        Z_elite = Z_elite[valid_elite_indices]
        Y_elite = Y_elite[valid_elite_indices]
        smiles_elite = [smiles_elite[i] for i in valid_elite_indices]

        print(f"Identified {len(smiles_elite)} elite samples on the Pareto front.")
        if len(smiles_elite) > 0:
            print(f"Elite samples properties range:")
            print(f"  -SAS (Higher is better): min={np.min(Y_elite[:, 0]):.2f}, max={np.max(Y_elite[:, 0]):.2f}")
            print(f"   QED (Higher is better): min={np.min(Y_elite[:, 1]):.2f}, max={np.max(Y_elite[:, 1]):.2f}")
            pareto_fronts_Y.append(Y_elite)
            # Calculate Hypervolume (using simple range product as approximation)
            if Y_elite.shape[0] > 0:
                 ref_point = np.array([-10.0, 0.0]) # Fixed reference point
                 hv_approx = np.prod(np.max(Y_elite, axis=0) - ref_point)
            else:
                 hv_approx = 0
            hypervolumes.append(hv_approx)
            print(f"Approximate Hypervolume (ref=[-10, 0]): {hv_approx:.4f}")
        else:
            print("Warning: No valid elite samples found in this iteration.")
            pareto_fronts_Y.append(np.empty((0,2))) # Add empty array
            hypervolumes.append(hypervolumes[-1] if hypervolumes else 0) # Keep previous HV


        # 3. Iterative Distribution Learning (IDL) Step
        if Z_elite.shape[0] > 0:
             mu_target_for_bo, std_target_for_bo = update_decoder_idl(
                 model, Z_elite, smiles_elite, my_tokenizer, max_len, device,
                 idl_epochs=args.idl_epochs, idl_lr=args.idl_lr, lambda_kl=args.idl_kl_weight
             )
        else:
             print("  Skipping IDL step as Z_elite is empty.")
             mu_target_for_bo, std_target_for_bo = None, None

        # Optional: Save model checkpoint after each IDL update
        # torch.save(model.state_dict(), f'model/{args.model}_model_{args.cell_name}_bo_idl_iter_{iteration+1}.pth')

    # --- Save Results ---
    print("\nOptimization finished. Saving results...")
    results_dir = f"results/bo_idl_{args.cell_name}_{args.latent_size}d"
    os.makedirs(results_dir, exist_ok=True)

    if mu_target_for_bo is not None:
        np.save(os.path.join(results_dir, 'final_target_mu.npy'), mu_target_for_bo)
        print(f"Final target mu saved to {os.path.join(results_dir, 'final_target_mu.npy')}")
    else:
        print("Final target mu was not generated (likely skipped IDL).")

    if std_target_for_bo is not None:
        np.save(os.path.join(results_dir, 'final_target_std.npy'), std_target_for_bo)
        print(f"Final target std saved to {os.path.join(results_dir, 'final_target_std.npy')}")
    else:
        print("Final target std was not generated (likely skipped IDL).")

    # Save all observed points and valid SMILES
    np.save(os.path.join(results_dir, 'observed_Z.npy'), Z_obs)
    np.save(os.path.join(results_dir, 'observed_Y.npy'), Y_obs)
    np.save(os.path.join(results_dir, 'observed_iteration.npy'), iteration_added)
    with open(os.path.join(results_dir, 'valid_smiles.txt'), 'w') as f:
        for smi in all_smiles_valid:
            f.write(f"{smi}\n")
    with open(os.path.join(results_dir, 'all_generated_smiles.txt'), 'w') as f:
        for smi in all_smiles_generated:
            f.write(f"{smi}\n")

    # Save final Pareto front
    final_pareto_idx = get_pareto_front(Y_obs)
    final_Z_pareto = Z_obs[final_pareto_idx]
    final_Y_pareto = Y_obs[final_pareto_idx]
    _, _, final_smiles_pareto, _ = compute_targets(final_Z_pareto, model, my_tokenizer, max_len, device)

    np.save(os.path.join(results_dir, 'final_pareto_Z.npy'), final_Z_pareto)
    np.save(os.path.join(results_dir, 'final_pareto_Y.npy'), final_Y_pareto)
    with open(os.path.join(results_dir, 'final_pareto_smiles.txt'), 'w') as f:
        for smi, y in zip(final_smiles_pareto, final_Y_pareto):
             mol = Chem.MolFromSmiles(smi)
             if mol:
                 f.write(f"{smi}\t{-y[0]:.4f}\t{y[1]:.4f}\n")

    # Save the final optimized model
    final_model_path = os.path.join(results_dir, f'{args.model}_model_{args.cell_name}_final_optimized.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final optimized model saved to {final_model_path}")

    # --- Plotting ---
    # Hypervolume plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, args.bo_idl_iterations + 1), hypervolumes, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Approximate Hypervolume')
    plt.title(f'Hypervolume Improvement ({args.cell_name})')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'hypervolume_improvement.png'))
    # plt.show()
    plt.close()

    # Pareto front plot
    plt.figure(figsize=(8, 8))
    # Plot all observed points
    # plt.scatter(Y_obs[:, 0], Y_obs[:, 1], c='gray', alpha=0.3, label='Observed Points')
    if Y_obs.shape[0] > 0 and iteration_added.shape[0] == Y_obs.shape[0]:
        # Use a colormap (e.g., viridis, plasma, coolwarm)
        cmap = plt.get_cmap('viridis', args.bo_idl_iterations + 1) # +1 for iteration 0
        scatter_bg = plt.scatter(
            Y_obs[:, 0],
            Y_obs[:, 1],
            c=iteration_added, # Color based on the iteration number
            cmap=cmap,         # Apply the colormap
            alpha=0.4,         # Make points slightly transparent
            s=20,              # Adjust point size if needed
            label='Observed Points (by Iteration)', # General label
            vmin=0,            # Ensure colorbar starts at iteration 0
            vmax=args.bo_idl_iterations # Ensure colorbar ends at the last iteration
        )
        # Add a colorbar
        cbar = plt.colorbar(scatter_bg, label='Iteration Added')
        # Optional: Set specific ticks on the colorbar if needed
        # tick_locs = np.linspace(0, args.bo_idl_iterations, num=min(11, args.bo_idl_iterations + 1)) # Example: max 11 ticks
        # cbar.set_ticks(tick_locs)
        # cbar.set_ticklabels([f'{int(t)}' for t in tick_locs])
    else:
        print("Warning: Cannot plot observed points by iteration due to data mismatch or empty data.")
        # Fallback to gray scatter plot if iteration data is missing/mismatched
        plt.scatter(Y_obs[:, 0], Y_obs[:, 1], c='gray', alpha=0.3, s=20, label='Observed Points')
    # Plot initial Pareto front
    # init_pareto_idx = get_pareto_front(Y_init[np.all(Y_init != [-10.0, 0.0], axis=1)]) # Use only valid init points
    # init_Y_pareto = Y_init[np.all(Y_init != [-10.0, 0.0], axis=1)][init_pareto_idx]
    # if init_Y_pareto.shape[0] > 0:
    #     plt.scatter(init_Y_pareto[:, 0], init_Y_pareto[:, 1], marker='s', s=80, edgecolor='k', facecolor='orange', label=f'Initial Pareto ({init_Y_pareto.shape[0]})')

    # initial_indices = np.where(iteration_added == 0)[0]
    # if len(initial_indices) > 0:
    #     Y_obs_initial = Y_obs[initial_indices]
    #     if Y_obs_initial.shape[0] > 0:
    #         init_pareto_idx_in_initial = get_pareto_front(Y_obs_initial)
    #         init_Y_pareto = Y_obs_initial[init_pareto_idx_in_initial]
    #         if init_Y_pareto.shape[0] > 0:
    #             plt.scatter(init_Y_pareto[:, 0], init_Y_pareto[:, 1], marker='s', s=100, # Increased size
    #                         edgecolor='black', linewidth=1, facecolor='orange', label=f'Initial Pareto ({init_Y_pareto.shape[0]})', zorder=3) # Ensure drawn on top

    # Plot final Pareto front
    # if final_Y_pareto.shape[0] > 0:
    #     plt.scatter(final_Y_pareto[:, 0], final_Y_pareto[:, 1], marker='*', s=150, edgecolor='k', facecolor='red', label=f'Final Pareto ({final_Y_pareto.shape[0]})')
    #     # Sort for line plotting
    #     final_Y_pareto_sorted = final_Y_pareto[np.argsort(final_Y_pareto[:, 0])]
    #     plt.plot(final_Y_pareto_sorted[:, 0], final_Y_pareto_sorted[:, 1], 'r--', alpha=0.7)


    # plt.xlabel('-SAS (Higher is Better)')
    # plt.ylabel('QED (Higher is Better)')
    # plt.title(f'Pareto Front Evolution ({args.cell_name})')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(os.path.join(results_dir, 'pareto_front_final.png'))
    # # plt.show()
    # plt.close()

    # print(f"Results saved in {results_dir}")

    # if Y_obs.shape[0] > 0: # Ensure Y_obs is not empty before finding final pareto
    #     final_pareto_idx = get_pareto_front(Y_obs)
    #     final_Y_pareto = Y_obs[final_pareto_idx]
    #     if final_Y_pareto.shape[0] > 0:
    #         plt.scatter(final_Y_pareto[:, 0], final_Y_pareto[:, 1], marker='*', s=250, # Increased size
    #                     edgecolor='black', linewidth=1, facecolor='red', label=f'Final Pareto ({final_Y_pareto.shape[0]})', zorder=4) # Ensure drawn on top
    #         # Sort for line plotting
    #         if final_Y_pareto.shape[0] > 1:
    #              final_Y_pareto_sorted = final_Y_pareto[np.argsort(final_Y_pareto[:, 0])]
    #              plt.plot(final_Y_pareto_sorted[:, 0], final_Y_pareto_sorted[:, 1], 'r--', alpha=0.7, linewidth=1.5, zorder=2)


    plt.xlabel('-SAS (Higher is Better)')
    plt.ylabel('QED (Higher is Better)')
    plt.title(f'Pareto Front Evolution ({args.cell_name} - Color by Iteration)')
    plt.legend(loc='best') # Adjust legend location if needed
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout() # Adjust layout
    plt.savefig(os.path.join(results_dir, 'pareto_front_final_colored.png'))
    # plt.show()
    plt.close()

    print(f"Results saved in {results_dir}")
