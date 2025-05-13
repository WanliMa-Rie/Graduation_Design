import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import rdkit
from rdkit import Chem
from rdkit.Chem import QED, RDConfig, AllChem
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer #type: ignore
SAS_AVAILABLE = True
from typing import List
from typing import Tuple
from torch.utils.data import DataLoader
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls import ExactMarginalLogLikelihood

# 假设 conditionVAE 模块已正确导入
from conditionVAE import *
from utils import *

# ====== Helper Functions ======

def generate_candidates_near(Z_obs, n_samples, latent_size, noise_std, lb, ub):
    """Generate candidate points near observed points with noise."""
    n_obs = Z_obs.shape[0]
    indices = np.random.randint(0, n_obs, size=n_samples)
    Z_candidates = Z_obs[indices] + np.random.normal(0, noise_std, size=(n_samples, latent_size))
    Z_candidates = np.clip(Z_candidates, lb, ub)
    return Z_candidates

def generate_candidates_target(mu, std, n_samples, latent_size):
    """Generate candidate points from target Gaussian distribution."""
    if mu is None or std is None:
        return np.random.normal(0, 1, (n_samples, latent_size))
    return np.random.normal(mu, std, (n_samples, latent_size))

def compute_targets(z, model, init_smiles, device):
    model.eval()
    targets = []
    all_smiles = []
    valid_smiles_list = []
    valid_z = []
    init_data = SubgraphDataset(init_smiles, args.atom_vocab, args.batch_size, args.num_decode)
    loader = DataLoader(init_data, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0]) 
    for init_mol in loader:
        init_mols = init_mol  # ['c1ccc[c:1]c1']
    batch_size = z.shape[0]
    z_tensor = torch.tensor(z, dtype=torch.float32, device=device)

    with torch.no_grad():
        for i in range(batch_size):
            z_batch = z_tensor[i:i+1].repeat(args.num_decode, 1)
            generated_smiles = model.decoder.decode(z_batch, init_mols, max_decode_step=args.max_decode_step)

            for smi in generated_smiles:
                all_smiles.append(smi)
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol and Chem.SanitizeMol(mol, catchErrors=True) == 0:
                        qed = QED.qed(mol)
                        sas = sascorer.calculateScore(mol)
                        targets.append([-sas, qed])
                        valid_smiles_list.append(smi)
                        valid_z.append(z[i])
                    else:
                        targets.append([-10.0, 0.0])
                except:
                    targets.append([-10.0, 0.0])

    return np.array(targets), all_smiles, valid_smiles_list, np.array(valid_z)

def compute_metrics(generated_smiles: List[str], train_molecules: List[str]) -> Tuple[float, float, float]:
    """
    Computes validity, diversity, and novelty of generated molecules.
    """
    valid_molecules = []
    valid_smiles = []

    # Validity
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

    # Diversity
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

    # Novelty
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

def get_pareto_front(Y):
    """Identifies the Pareto front from a set of observations Y."""
    is_pareto = np.ones(Y.shape[0], dtype=bool)
    for i, y in enumerate(Y):
        if is_pareto[i]:
            is_pareto[i] = not np.any(np.all(Y[is_pareto] >= y, axis=1) & np.any(Y[is_pareto] > y, axis=1))
    return np.where(is_pareto)[0]

# ====== Bayesian Optimization ======

def run_bo(
    Z_obs, Y_obs, bounds, n_iter, model, init_smiles, device):
    """Runs Bayesian Optimization to select new latent points."""
    Z_obs_tensor = torch.tensor(Z_obs, dtype=torch.float64, device=device)
    Y_obs_tensor = torch.tensor(Y_obs, dtype=torch.float64, device=device)
    bounds_tensor = torch.tensor(bounds, dtype=torch.float64, device=device).T

    # ------ Train GP models ------ 
    gp_models = []
    for i in range(Y_obs.shape[1]):
        X = Z_obs_tensor
        Y = Y_obs_tensor[:, i].unsqueeze(-1)
        model_gp = SingleTaskGP(X, Y, input_transform=Normalize(d=X.shape[-1]), outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model_gp.likelihood, model_gp)
        fit_gpytorch_mll(mll)
        gp_models.append(model_gp)
    model_list = ModelListGP(*gp_models)

    # ------ Generate candidate points ------
    Z_candidates = np.random.normal(0, 1, (args.bo_acquisition_samples, args.latent_size))
    Z_candidates_tensor = torch.tensor(Z_candidates, dtype=torch.float64, device=device)

    # ------ Compute EHVI acquisition function ------ 
    pareto_mask = is_non_dominated(Y_obs_tensor)
    current_pareto_Y = Y_obs_tensor[pareto_mask].to(torch.float64)
    ref_point = torch.tensor([-10.0, 0.0], dtype=torch.float64, device=device)
    partitioning = NondominatedPartitioning(ref_point, Y=current_pareto_Y)
    acquisition_function = qExpectedHypervolumeImprovement(
        model=model_list, ref_point=ref_point, partitioning=partitioning
    )

    # ------ Evaluate EHVI for candidates ------
    with torch.no_grad():
        acq_values = acquisition_function(Z_candidates_tensor.unsqueeze(1))

    # ------ Select top candidates ------
    num_to_select = min(n_iter, len(acq_values))
    _, best_indices = torch.topk(acq_values, k=num_to_select, largest=True)
    best_indices = best_indices.cpu().numpy()
    print(f"Selecting {len(best_indices)} points based on qEHVI scores")

    # Evaluate selected candidates
    Z_new_list = []
    Y_new_list = []
    smiles_new_all_this_run = []
    smiles_new_valid_this_run = []

    for idx in tqdm(best_indices, desc='BO Steps (Evaluating Selected)'):
        z_next = Z_candidates[idx:idx+1]
        y_next, smiles_next_all, smiles_next_valid, z_next_valid = compute_targets(
            z_next, model, init_smiles, device
        )
        if len(smiles_next_valid) > 0:
            Z_new_list.extend(z_next_valid)
            Y_new_list.extend(y_next)
            smiles_new_all_this_run.extend(smiles_next_all)
            smiles_new_valid_this_run.extend(smiles_next_valid)
    
    Z_new_array = np.array(Z_new_list)
    print(Z_new_array.shape)
    Y_new_array = np.array(Y_new_list)

    # Update observed data
    if Z_new_array.shape[0] > 0:
        updated_Z_obs = np.vstack([Z_obs, Z_new_array])
        updated_Y_obs = np.vstack([Y_obs, Y_new_array])
    else:
        updated_Y_obs = Y_obs
        updated_Z_obs = Z_obs
        print("No valid points added in this BO run")

    return updated_Z_obs, updated_Y_obs, Z_new_array, Y_new_array, smiles_new_all_this_run, smiles_new_valid_this_run

# ====== Iterative Distribution Learning (IDL) ======

def update_decoder_idl(model, smiles_elite):
    """
    Updates the AtomVGNN decoder using IDL based on elite samples.
    Args:
        model (AtomVGNN): The conditional VAE model.
        Z_elite (np.array): Latent vectors of elite samples.
        smiles_elite (list): SMILES strings of elite samples.
        rationale_smiles (str): The rationale SMILES.
        device (torch.device): Computation device.
    Returns:
        params of the fitted target Gaussian distribution.
    """
    
    optimizer = optim.Adam(model.parameters(), lr=args.idl_lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
    
    print(smiles_elite)
    smiles_elite = unique_rationales(smiles_elite)
    print(smiles_elite)
    smiles_elite_dataset = SubgraphDataset(smiles_elite, args.atom_vocab, args.decode_batch_size, args.num_decode)
    
    # IDL Training Loop
    for epoch in range(args.idl_epochs):
        optimizer.zero_grad()
        mol_elite = decode_smiles(model, smiles_elite_dataset)
        mol_elite = list(set(mol_elite))
        random.shuffle(mol_elite)

        dataset = MoleculeDataset(mol_elite, args.atom_vocab, args.batch_size)
        # dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=lambda x:x[0])

        model.train()
        meters = np.zeros(5)
        for total_step, batch in enumerate(dataset):
            if batch is None: continue

            model.zero_grad()
            loss, kl_div, wacc, tacc, sacc = model(*batch, beta=args.beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            meters = meters + np.array([kl_div, loss.item(), wacc * 100, tacc * 100, sacc * 100])

        if (epoch + 1) % 10 == 0:
            print(f"Beta: {args.beta:.3f}, KL: {meters[0]:.2f}, loss: {meters[1]:.3f}, Word: {meters[2]:.3f}, Topo: {meters[3]:.3f}, Assm: {meters[4]:.3f}")
            sys.stdout.flush()

        scheduler.step()

        

    model.eval()

# ====== Main Optimization Loop ======

def optimize_molecules(args):
    """Main function to run the MOBO-IDL optimization with AtomVGNN."""

    print("Starting Molecular Optimization...")
    print(f"Arguments: {args}")

    # ------ Initialization ------
    device = torch.device(args.device)
    init_smiles = [line.strip() for line in open(args.init_smiles)]
    print(init_smiles)

    # ------ Load AtomVGNN model ------
    print(f"Loading pre-trained AtomVGNN model from: {args.model}")
    model = AtomVGNN(args).to(device)
    model_ckpt = torch.load(args.model, map_location=device, weights_only=True)
    if isinstance(model_ckpt, tuple):
        print('Loading model with rationale distribution', file=sys.stderr)
        model.load_state_dict(model_ckpt[1])
    else:
        print('Loading pre-trained model', file=sys.stderr)
        model.load_state_dict(model_ckpt)
    model.eval()

    # ------ Get initial latent vectors -------
    print(f"Generating {args.bo_initial_points} initial latent points...")
    Z_init = np.random.normal(0, 1, (args.bo_initial_points, args.latent_size))

    # ------ Compute initial objective values ------
    print("Evaluating initial points...")
    Y_init, smiles_init_all, smiles_init_valid, Z_init_valid = compute_targets(
        Z_init, model, init_smiles, device)
    print(f"Initial evaluation complete. Found {len(smiles_init_valid)} valid molecules.")

    # Compute initial metrics
    # print("\nComputing initial metrics...")
    # validity, diversity, novelty = compute_metrics(smiles_init_all, train_molecules)

    Z_obs = Z_init_valid
    Y_obs = Y_init[np.all(Y_init != [-10.0, 0.0], axis=1)]
    num_initial_valid = Y_obs.shape[0]
    iteration_added = np.zeros(num_initial_valid, dtype=int)

    print(f"Starting BO with {Z_obs.shape[0]} valid points.")
    print(f"Initial Z_obs shape: {Z_obs.shape}, Initial Y_obs shape: {Y_obs.shape}")

    # Define latent space bounds
    lb = Z_obs.min(axis=0) - 1.0
    ub = Z_obs.max(axis=0) + 1.0
    bounds = (lb, ub)

    # Main BO-IDL Loop
    all_smiles_generated = list(smiles_init_all)
    all_smiles_valid = list(smiles_init_valid)
    hypervolumes = []
    pareto_fronts_Y = []
    mu_target_for_bo = None
    std_target_for_bo = None
    exploration_noise_std = 0.5
    # metrics_history = [(validity, diversity, novelty)]

    for iteration in range(args.iterations):
        print(f"\n{'='*20} Iteration {iteration + 1}/{args.iterations} {'='*20}")

        # Bayesian Optimization Step
        iteration_added_before_bo = iteration_added.copy()

        Z_obs, Y_obs, Z_new, Y_new, smiles_new_gen, smiles_new_val = run_bo(
            Z_obs, Y_obs, bounds, args.bo_points_per_iter, model,
            init_smiles, device)
        all_smiles_generated.extend(smiles_new_gen)
        all_smiles_valid.extend(smiles_new_val)

        num_new_points = Z_new.shape[0]
        if num_new_points > 0:
            new_iterations = np.full(num_new_points, iteration + 1, dtype=int)
            iteration_added = np.concatenate([iteration_added_before_bo, new_iterations])

        print(f"BO finished. Found {Z_new.shape[0]} new valid points. Total observed points: {Z_obs.shape[0]}")
        
        # Compute metrics for new molecules
        # print("\nComputing metrics for new molecules...")
        # validity, diversity, novelty = compute_metrics(smiles_new_gen, train_molecules)
        # metrics_history.append((validity, diversity, novelty))

        # Update bounds
        lb = Z_obs.min(axis=0) - 1.0
        ub = Z_obs.max(axis=0) + 1.0
        bounds = (lb, ub)

        # Identify Elite Samples (Pareto Front)
        Y_obs_tensor = torch.tensor(Y_obs, dtype=torch.float64, device=device)
        pareto_mask = is_non_dominated(Y_obs_tensor)
        Z_elite = Z_obs[pareto_mask.cpu().numpy()]
        Y_elite = Y_obs_tensor[pareto_mask].cpu().numpy()
        smiles_elite = [all_smiles_valid[i] for i, is_pareto in enumerate(pareto_mask) if is_pareto]

        print(f"Identified {len(smiles_elite)} elite samples on the Pareto front.")
        if len(smiles_elite) > 0:
            print(f"Elite samples properties range:")
            print(f"  -SAS: min={np.min(Y_elite[:, 0]):.2f}, max={np.max(Y_elite[:, 0]):.2f}")
            print(f"   QED: min={np.min(Y_elite[:, 1]):.2f}, max={np.max(Y_elite[:, 1]):.2f}")
            hv = Hypervolume(ref_point=torch.tensor([-10.0, 0.0], dtype=torch.float64, device=device))
            hypervolume = hv.compute(torch.tensor(Y_elite, dtype=torch.float64, device=device))
            hypervolumes.append(hypervolume)
            print(f"Hypervolume (ref=[-10, 0]): {hypervolume:.4f}")
            pareto_fronts_Y.append(Y_elite)
        else:
            print("Warning: No valid elite samples found in this iteration.")
            hypervolumes.append(hypervolumes[-1] if hypervolumes else 0)
            pareto_fronts_Y.append(np.empty((0, 2)))

        # Iterative Distribution Learning (IDL) Step
        # print(smiles_elite)
        if Z_elite.shape[0] > 0:
            update_decoder_idl(
                model, smiles_elite)
        else:
            print("Skipping IDL step as Z_elite is empty.")
            mu_target_for_bo, std_target_for_bo = None, None

    # Final Pareto Front and Hypervolume
    Y_obs_tensor = torch.tensor(Y_obs, dtype=torch.float64, device=device)
    final_pareto_mask = is_non_dominated(Y_obs_tensor)
    final_pareto_Y = Y_obs_tensor[final_pareto_mask].cpu().numpy()
    final_pareto_Z = Z_obs[final_pareto_mask.cpu().numpy()]
    final_pareto_smiles = [all_smiles_valid[i] for i, is_pareto in enumerate(final_pareto_mask.cpu().numpy()) if is_pareto]

    ref_point = torch.tensor([-10.0, 0.0], dtype=torch.float64, device=device)
    hv = Hypervolume(ref_point=ref_point)
    final_hypervolume = hv.compute(torch.tensor(final_pareto_Y, dtype=torch.float64, device=device))
    print(f"\nFinal Pareto front size: {len(final_pareto_Y)} points")
    print(f"Final Hypervolume: {final_hypervolume:.4f}")

    # Final metrics
    # print("\nComputing final metrics...")
    # final_validity, final_diversity, final_novelty = compute_metrics(all_smiles_generated, train_molecules)
    # metrics_history.append((final_validity, final_diversity, final_novelty))

    # Save Results
    print("\nOptimization finished. Saving results...")
    results_dir = f"results/bo_idl_{args.cell_name}_{args.latent_size}d"
    os.makedirs(results_dir, exist_ok=True)

    if mu_target_for_bo is not None:
        np.save(os.path.join(results_dir, 'final_target_mu.npy'), mu_target_for_bo)
    if std_target_for_bo is not None:
        np.save(os.path.join(results_dir, 'final_target_std.npy'), std_target_for_bo)

    np.save(os.path.join(results_dir, 'observed_Z.npy'), Z_obs)
    np.save(os.path.join(results_dir, 'observed_Y.npy'), Y_obs)
    np.save(os.path.join(results_dir, 'observed_iteration.npy'), iteration_added)
    with open(os.path.join(results_dir, 'valid_smiles.txt'), 'w') as f:
        for smi in all_smiles_valid:
            f.write(f"{smi}\n")
    with open(os.path.join(results_dir, 'all_generated_smiles.txt'), 'w') as f:
        for smi in all_smiles_generated:
            f.write(f"{smi}\n")

    np.save(os.path.join(results_dir, 'final_pareto_Z.npy'), final_pareto_Z)
    np.save(os.path.join(results_dir, 'final_pareto_Y.npy'), final_pareto_Y)
    with open(os.path.join(results_dir, 'final_pareto_smiles.txt'), 'w') as f:
        for smi, y in zip(final_pareto_smiles, final_pareto_Y):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                f.write(f"{smi}\t{-y[0]:.4f}\t{y[1]:.4f}\n")

    final_model_path = os.path.join(results_dir, f'atom_vgnn_model_{args.cell_name}_final_optimized.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final optimized model saved to {final_model_path}")

    # Save metrics history
    # np.save(os.path.join(results_dir, 'metrics_history.npy'), np.array(metrics_history))
    # with open(os.path.join(results_dir, 'metrics_history.txt'), 'w') as f:
    #     f.write("Iteration,Validity,Diversity,Novelty\n")
    #     f.write(f"0,{metrics_history[0][0]:.4f},{metrics_history[0][1]:.4f},{metrics_history[0][2]:.4f}\n")
    #     for i, (v, d, n) in enumerate(metrics_history[1:-1], 1):
    #         f.write(f"{i},{v:.4f},{d:.4f},{n:.4f}\n")
    #     f.write(f"Final,{metrics_history[-1][0]:.4f},{metrics_history[-1][1]:.4f},{metrics_history[-1][2]:.4f}\n")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, args.bo_idl_iterations + 1), hypervolumes, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Approximate Hypervolume')
    plt.title(f'Hypervolume Improvement ({args.cell_name})')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'hypervolume_improvement.png'))
    plt.close()

    plt.figure(figsize=(8, 8))
    if Y_obs.shape[0] > 0 and iteration_added.shape[0] == Y_obs.shape[0]:
        cmap = plt.get_cmap('viridis', args.bo_idl_iterations + 1)
        scatter_bg = plt.scatter(
            Y_obs[:, 0], Y_obs[:, 1], c=iteration_added, cmap=cmap, alpha=0.4, s=20,
            label='Observed Points (by Iteration)', vmin=0, vmax=args.bo_idl_iterations
        )
        plt.colorbar(scatter_bg, label='Iteration Added')
    else:
        plt.scatter(Y_obs[:, 0], Y_obs[:, 1], c='gray', alpha=0.3, s=20, label='Observed Points')

    if final_pareto_Y.shape[0] > 0:
        plt.scatter(final_pareto_Y[:, 0], final_pareto_Y[:, 1], marker='*', s=150, edgecolor='k', facecolor='red', label=f'Final Pareto ({final_pareto_Y.shape[0]})')
        if final_pareto_Y.shape[0] > 1:
            final_pareto_sorted = final_pareto_Y[np.argsort(final_pareto_Y[:, 0])]
            plt.plot(final_pareto_sorted[:, 0], final_pareto_sorted[:, 1], 'r--', alpha=0.7)

    plt.xlabel('-SAS (Higher is Better)')
    plt.ylabel('QED (Higher is Better)')
    plt.title(f'Pareto Front Evolution ({args.cell_name} - Color by Iteration)')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'pareto_front_final_colored.png'))
    plt.close()

    # # Plot metrics history
    # plt.figure(figsize=(10, 6))
    # iterations = list(range(len(metrics_history)-1)) + ['Final']
    # validity = [m[0] for m in metrics_history]
    # diversity = [m[1] for m in metrics_history]
    # novelty = [m[2] for m in metrics_history]
    # plt.plot(iterations, validity, marker='o', label='Validity')
    # plt.plot(iterations, diversity, marker='s', label='Diversity')
    # plt.plot(iterations, novelty, marker='^', label='Novelty')
    # plt.xlabel('Iteration')
    # plt.ylabel('Metric Value')
    # plt.title(f'Metrics Evolution ({args.cell_name})')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(os.path.join(results_dir, 'metrics_evolution.png'))
    # plt.close()

    # print(f"Results saved in {results_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_smiles', default='c1ccc[c:1]c1', help='Rationale SMILES')
    parser.add_argument('--train_molecules', default='data/ZINC/zinc.txt', help='Path to training molecules')
    parser.add_argument('--model', required=True, help='Path to pretrained AtomVGNN model')
    parser.add_argument('--num_decode', type=int, default=10, help='Number of molecules to generate per latent vector')
    parser.add_argument('--bo_initial_points', type=int, default=100, help='Number of initial latent points')
    parser.add_argument('--bo_points_per_iter', type=int, default=10, help='Number of points to evaluate per BO iteration')
    parser.add_argument('--bo_acquisition_samples', type=int, default=1000, help='Number of acquisition samples')
    parser.add_argument('--beta', type=float, default=0.3)
    parser.add_argument('--iterations', type=int, default=10, help='Number of S-IDL iterations')
    parser.add_argument('--idl_epochs', type=int, default=10, help='Number of IDL epochs')
    parser.add_argument('--idl_lr', type=float, default=1e-4, help='IDL learning rate')
    parser.add_argument('--anneal_rate', type=float, default=1.0)
    parser.add_argument('--idl_kl_weight', type=float, default=0.01, help='KL divergence weight for IDL')
    parser.add_argument('--latent_size', type=int, default=64, help='Latent space dimension')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--cell_name', default='default', help='Cell line name for results')
    parser.add_argument('--atom_vocab', default=common_atom_vocab, help='Atom vocabulary')
    parser.add_argument('--rnn_type', default='LSTM', help='RNN type for MPNN')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size for MPNN')
    parser.add_argument('--embed_size', type=int, default=256, help='Embedding size for atoms/bonds')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for decoding')
    parser.add_argument('--decode_batch_size', type=int, default=20)
    parser.add_argument('--depth', type=int, default=10, help='MPNN depth')
    parser.add_argument('--diter', type=int, default=3, help='Number of decoder iterations')
    parser.add_argument('--max_decode_step', type=int, default=80, help='Maximum decoding steps')
    args = parser.parse_args()
    optimize_molecules(args)