from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*") # Disable RDKit logging
import matplotlib.pyplot as plt
# KDE is no longer needed for the evaluate function itself
# from sklearn.neighbors import KernelDensity # Keep if latent_visulization needs it
import torch
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import QED, DataStructs # Added DataStructs for Tanimoto
from model import VAE_Autoencoder
from rdkit.Contrib.SA_Score import sascorer # type: ignore
from dataset import * # Assuming tokenizer is defined here
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import pandas as pd # Added for loading training data
# ... (imports and helper functions remain the same) ...
# --- Helper Functions (Keep these as they are) ---
# indices_to_smiles
# check_validity
# check_uniqueness
# check_novelty
# check_diversity
# (These functions remain unchanged from the previous version)
def indices_to_smiles(indices, int_to_char):
    smiles_list = []
    for seq in indices:
        end_idx = len(seq)
        eos_token_code = -1
        pad_token_code = -1
        sos_token_code = -1

        # Get codes for special tokens, handle if not present
        # Use .get() on the reversed dictionary to avoid KeyError if special tokens aren't in vocab
        char_to_code = {v: k for k, v in int_to_char.items()}
        eos_token_code = char_to_code.get('$', -1)
        pad_token_code = char_to_code.get(' ', -1)
        sos_token_code = char_to_code.get('^', -1)


        try:
            # Find first occurrence of EOS or PAD
            indices_list = list(seq)
            # Find index only if the token exists in the sequence
            eos_idx = indices_list.index(eos_token_code) if eos_token_code != -1 and eos_token_code in indices_list else len(seq)
            pad_idx = indices_list.index(pad_token_code) if pad_token_code != -1 and pad_token_code in indices_list else len(seq)
            end_idx = min(eos_idx, pad_idx)
        except ValueError:
             # Should not happen with the checks above, but as fallback:
             end_idx = len(seq)

        # Extract sequence up to end_idx, skipping SOS if present
        smiles_chars = []
        start_idx = 0
        # Check if the first token is SOS and SOS exists
        if len(seq) > 0 and seq[0] == sos_token_code and sos_token_code != -1:
            start_idx = 1 # Skip SOS token

        for i in range(start_idx, end_idx):
             char_code = seq[i]
             # Ensure the code is actually in the map before trying to get the char
             if char_code in int_to_char:
                 smiles_chars.append(int_to_char[char_code])
             # else: # Optional: handle unknown codes if necessary
                 # print(f"Warning: Unknown character code {char_code} encountered.")

        smiles = "".join(smiles_chars)
        smiles_list.append(smiles)
    return smiles_list

# Validity
def check_validity(smiles_list):
    """Calculates validity and returns valid SMILES list."""
    valid_smiles = []
    if not smiles_list:
        return [], 0.0
    count = 0
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                 # Optional basic sanity check (e.g., number of atoms)
                 # Check prevents RDKit from producing molecules with no atoms from empty strings etc.
                 if mol.GetNumAtoms() > 0:
                    valid_smiles.append(smi)
                    count += 1
                 # else:
                 #     print(f"Debug: Valid syntax but 0 atoms for SMILES: '{smi}'")
            # else:
            #      print(f"Debug: Invalid SMILES syntax: '{smi}'")
        except Exception as e: # Catch potential RDKit errors
            # print(f"Debug: RDKit error for SMILES '{smi}': {e}")
            pass

    validity = len(valid_smiles) / len(smiles_list) if smiles_list else 0.0
    # print(f"Debug: Validity check - {count} valid molecules with atoms found out of {len(smiles_list)} raw strings.")
    return valid_smiles, validity

# Uniqueness
def check_uniqueness(valid_smiles):
    """Calculates uniqueness based on canonical SMILES."""
    if not valid_smiles:
        return set(), 0.0
    canonical_smiles = []
    processed_count = 0
    unique_count = 0
    for smi in valid_smiles:
        processed_count += 1
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol and mol.GetNumAtoms() > 0: # Ensure mol is valid before canonicalizing
                canon_smi = Chem.MolToSmiles(mol, canonical=True)
                canonical_smiles.append(canon_smi)
            # else:
                 # This case should ideally not happen if input is from check_validity
                 # print(f"Debug: Mol object issue during canonicalization for: {smi}")

        except Exception as e:
            # print(f"Debug: Canonicalization error for SMILES '{smi}': {e}")
            pass # Ignore SMILES that fail canonicalization

    unique_smiles_set = set(canonical_smiles)
    unique_count = len(unique_smiles_set)
    uniqueness = unique_count / len(valid_smiles) if valid_smiles else 0.0
    # print(f"Debug: Uniqueness check - Found {unique_count} unique canonical SMILES from {len(valid_smiles)} valid input SMILES.")
    # Return the set of unique *canonical* SMILES
    return unique_smiles_set, uniqueness

# Novelty
def check_novelty(unique_generated_smiles_set, reference_smiles_set):
    """Calculates novelty against a reference set (both sets should contain canonical SMILES)."""
    if not unique_generated_smiles_set:
        # print("Debug: Novelty check - Input set of unique generated SMILES is empty.")
        return set(), 0.0
    if not reference_smiles_set:
        # print("Debug: Novelty check - Reference set is empty, assuming all generated are novel.")
        # Decide behavior: return 1.0 or NaN? Let's return 1.0, as technically none are in the (empty) reference.
        # However, typically novelty implies comparison against a non-empty known set.
        # Returning NaN might be safer if a reference set was expected but missing. Let's stick to calculation.
         novel_smiles_set = unique_generated_smiles_set
         novelty = 1.0
         return novel_smiles_set, novelty


    # Ensure reference is a set
    if not isinstance(reference_smiles_set, set):
        reference_smiles_set = set(reference_smiles_set)

    novel_smiles_set = unique_generated_smiles_set - reference_smiles_set
    novelty = len(novel_smiles_set) / len(unique_generated_smiles_set) if unique_generated_smiles_set else 0.0
    # print(f"Debug: Novelty check - {len(novel_smiles_set)} novel SMILES found out of {len(unique_generated_smiles_set)} unique generated.")
    return novel_smiles_set, novelty

# Diversity
def check_diversity(valid_smiles):
    """Calculates internal diversity using Tanimoto similarity of Morgan fingerprints."""
    if len(valid_smiles) < 2:
        # print("Debug: Diversity check - Input list has < 2 valid SMILES.")
        return 0.0 # Cannot compute diversity for 0 or 1 molecule

    mols = []
    valid_smiles_for_diversity = [] # Keep track of SMILES corresponding to successful mol objects
    mol_creation_success = 0
    for smi in valid_smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            # Ensure mol is valid and has atoms for meaningful fingerprinting
            if mol and mol.GetNumAtoms() > 0:
                mols.append(mol)
                valid_smiles_for_diversity.append(smi)
                mol_creation_success += 1
            # else:
                 # print(f"Debug: Diversity check - Skipping SMILES '{smi}' (invalid or 0 atoms).")

        except Exception as e:
             # print(f"Debug: Diversity check - RDKit error creating molecule from '{smi}': {e}")
             pass


    if len(mols) < 2: # Recheck after ensuring valid mols with atoms
         # print(f"Debug: Diversity check - Could only create {len(mols)} valid RDKit molecules with atoms from {len(valid_smiles)} valid SMILES strings. Cannot calculate diversity.")
         return 0.0

    # Generate Morgan fingerprints
    generator = GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    fp_generation_success = 0
    for i, mol in enumerate(mols):
        try:
            fp = generator.GetFingerprint(mol)
            fps.append(fp)
            fp_generation_success += 1
        except Exception as e:
            # print(f"Debug: Diversity check - Fingerprint generation failed for SMILES: {valid_smiles_for_diversity[i]}. Error: {e}")
            # We need to handle the case where fps list might become shorter than mols list if we skip.
            # Easier to just proceed and check for zero fps later. Let's keep fps aligned with mols for now.
            # A placeholder or skipping might be needed depending on how critical alignment is.
            # For now, let's assume GetFingerprint is robust or errors are rare after mol validation.
            # If errors occur frequently, a more robust skipping mechanism is needed.
             pass # Or append None and handle later? For simplicity, assume success or error is rare.

    if len(fps) < 2:
        # print(f"Debug: Diversity check - Generated {len(fps)} fingerprints, need at least 2.")
        return 0.0


    # Calculate pairwise similarities
    similarities = []
    zero_fp_pairs_skipped = 0
    valid_pairs_calculated = 0
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            # Check for None if fingerprint generation could fail and we appended None
            if fps[i] is None or fps[j] is None:
                 zero_fp_pairs_skipped += 1
                 continue

            # Check for zero fingerprints (can happen for very small/weird molecules)
            # Make sure GetNumOnBits() exists for the fp object type
            fp_i_bits = fps[i].GetNumOnBits() if hasattr(fps[i], 'GetNumOnBits') else -1
            fp_j_bits = fps[j].GetNumOnBits() if hasattr(fps[j], 'GetNumOnBits') else -1

            if fp_i_bits == 0 or fp_j_bits == 0:
                # print(f"Warning: Skipping similarity check due to zero fingerprint for SMILES pair involving: {valid_smiles_for_diversity[i]} or {valid_smiles_for_diversity[j]}")
                zero_fp_pairs_skipped += 1
                continue # Skip pair if either FP is zero

            # Use DataStructs.TanimotoSimilarity
            try:
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                similarities.append(sim)
                valid_pairs_calculated += 1
            except Exception as e:
                 # print(f"Debug: Tanimoto calculation error between {i} and {j}: {e}")
                 zero_fp_pairs_skipped +=1 # Count as skipped


    if not similarities: # Could happen if all pairs had zero fps or other errors
        # print(f"Debug: Diversity check - No valid fingerprint pairs found for similarity calculation ({valid_pairs_calculated} successful pairs, {zero_fp_pairs_skipped} pairs skipped).")
        return 0.0

    avg_similarity = np.mean(similarities)
    diversity = 1.0 - avg_similarity
    # print(f"Debug: Diversity check - Calculated average similarity {avg_similarity:.4f} from {len(similarities)} pairs. Diversity: {diversity:.4f}")
    return diversity


def evaluate(args, my_tokenizer):
    """
    Evaluates the VAE model by sampling from the standard Gaussian prior N(0, I)
    and calculating Validity, Uniqueness, Novelty, Diversity, QED, and SAS.
    Loads reference SMILES from a plain text file (one SMILES per line).
    """
    print("-" * 50)
    print(f"Starting evaluation for model: {args.model}...")
    print(f"Sampling Method: Standard Multivariate Normal N(0, I)")

    # --- 1. Load Model ---
    # (Model loading code remains the same as previous version)
    print("Loading model...")
    vocab_size = my_tokenizer.vocab_size
    if not hasattr(my_tokenizer, 'char_to_int') or not hasattr(my_tokenizer, 'int_to_char'):
         print("Error: Tokenizer object must have 'char_to_int' and 'int_to_char' mappings.")
         return None # Return None or raise error
    if not my_tokenizer.int_to_char:
        print("Error: Tokenizer 'int_to_char' mapping is empty.")
        return None

    model = VAE_Autoencoder(vocab_size, args.embedding_dim, args.hidden_size, args.latent_size, args.max_len, model_type=args.model)
    model.vocab = my_tokenizer.char_to_int
    model.int_to_char = my_tokenizer.int_to_char
    model.to(args.device)

    try:
        model_path = f'model/{args.model}_model_zinc.pth'
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        print(f"Loaded model weights from: {model_path}")
    except FileNotFoundError:
        print(f"Error: Model weights file not found at {model_path}")
        print("Evaluation cannot proceed without the model.")
        return None
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Check if model definition (embedding_dim, hidden_size, latent_size etc.) matches the saved weights.")
        return None

    model.eval()

    # --- 2. Sample from Standard Normal Distribution ---
    # (Sampling code remains the same)
    print(f"Sampling {args.num_samples} points from N(0, I) latent space (size: {args.latent_size})...")
    z_samples = torch.randn(args.num_samples, args.latent_size, dtype=torch.float32, device=args.device)


    # --- 3. Decode Samples ---
    # (Decoding code remains the same)
    print("Decoding sampled latent vectors...")
    with torch.no_grad():
        sampled_logits = model.decode(z_samples, args.max_len)
        sampled_indices = torch.argmax(sampled_logits, dim=-1).cpu().numpy()


    # --- 4. Convert Indices to SMILES ---
    # (Conversion code remains the same)
    print("Converting indices to SMILES strings...")
    sampled_smiles_raw = indices_to_smiles(sampled_indices, my_tokenizer.int_to_char)
    print(f"Generated {len(sampled_smiles_raw)} raw SMILES strings.")


    # --- 5. Load Reference Training Data for Novelty (from .txt file) --- <<< MODIFIED PART
    print(f"Loading reference (training) data for novelty calculation from: {args.train_data_path}")
    reference_smiles_set = set()
    if hasattr(args, 'train_data_path') and args.train_data_path:
        try:
            processed_lines = 0
            canonicalized_count = 0
            print(f"Reading and canonicalizing SMILES from {args.train_data_path}...")
            with open(args.train_data_path, 'r') as f:
                for line in f:
                    processed_lines += 1
                    smi = line.strip() # Remove leading/trailing whitespace (like newline)
                    if not smi: # Skip empty lines
                        continue
                    try:
                        mol = Chem.MolFromSmiles(smi)
                        if mol is not None:
                            canon_smi = Chem.MolToSmiles(mol, canonical=True)
                            reference_smiles_set.add(canon_smi)
                            canonicalized_count += 1
                        # else: # Optional: Log invalid SMILES in reference data
                            # print(f"Warning: Invalid SMILES in reference file: '{smi}'")
                    except Exception as e:
                        # print(f"Warning: RDKit error processing reference SMILES: '{smi}'. Error: {e}")
                        pass # Ignore invalid reference SMILES

            print(f"Processed {processed_lines} lines from reference file. Loaded and canonicalized {len(reference_smiles_set)} unique valid training SMILES.")
        except FileNotFoundError:
            print(f"Warning: Training data file not found at {args.train_data_path}. Cannot calculate novelty.")
        except Exception as e:
            print(f"Warning: Error reading or processing training data file {args.train_data_path}: {e}. Cannot calculate novelty.")
    else:
        print("Warning: `args.train_data_path` not specified. Cannot calculate novelty.")


    # --- 6. Calculate Metrics ---
    # (Metric calculation logic remains the same as the previous N(0,I) sampling version)
    print("-" * 50)
    print("Calculating evaluation metrics...")

    # 6.1 Validity
    valid_smiles, validity = check_validity(sampled_smiles_raw)
    print(f"Validity: {validity:.4f} ({len(valid_smiles)} / {len(sampled_smiles_raw)})")

    if not valid_smiles:
        print("No valid SMILES generated. Cannot calculate further metrics.")
        print("-" * 50)
        # Return empty/default results
        return {
            'validity': 0.0, 'uniqueness': 0.0, 'novelty': float('nan'),
            'diversity': 0.0, 'avg_qed': 0.0, 'avg_sas': 0.0,
            'num_generated': len(sampled_smiles_raw), 'num_valid': 0,
            'num_unique_canonical': 0, 'num_novel': 0
        }

    # 6.2 Uniqueness
    unique_canonical_smiles_set, uniqueness = check_uniqueness(valid_smiles)
    print(f"Uniqueness (among valid): {uniqueness:.4f} ({len(unique_canonical_smiles_set)} unique canonical / {len(valid_smiles)} valid)")

    # 6.3 Novelty
    novelty = float('nan')
    num_novel = 0
    if not unique_canonical_smiles_set:
         print("Novelty: Not calculated (no unique generated SMILES).")
    elif not reference_smiles_set:
        print("Novelty: Not calculated (reference data unavailable or empty).")
    else:
        novel_smiles_set, novelty = check_novelty(unique_canonical_smiles_set, reference_smiles_set)
        num_novel = len(novel_smiles_set)
        print(f"Novelty (vs training): {novelty:.4f} ({num_novel} novel / {len(unique_canonical_smiles_set)} unique)")

    # 6.4 Diversity
    diversity = check_diversity(valid_smiles)
    print(f"Diversity (among valid): {diversity:.4f}")

    # 6.5 QED and SAS
    qed_values = []
    sas_values = []
    print("Calculating QED and SAS for valid SMILES...")
    qed_success_count = 0
    sas_success_count = 0
    for smiles in valid_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            try:
                qed_val = QED.qed(mol)
                qed_values.append(qed_val)
                qed_success_count += 1
            except Exception: pass
            try:
                sas_val = sascorer.calculateScore(mol)
                sas_values.append(sas_val)
                sas_success_count += 1
            except Exception: pass

    avg_qed = np.mean(qed_values) if qed_values else 0.0
    avg_sas = np.mean(sas_values) if sas_values else 0.0
    print(f"Average QED (calculated on {qed_success_count}/{len(valid_smiles)} valid SMILES): {avg_qed:.4f}")
    print(f"Average SAS (calculated on {sas_success_count}/{len(valid_smiles)} valid SMILES): {avg_sas:.4f}")

    print("-" * 50)
    print("Evaluation finished.")

    results = {
        'validity': validity,
        'uniqueness': uniqueness,
        'novelty': novelty,
        'diversity': diversity,
        'avg_qed': avg_qed,
        'avg_sas': avg_sas,
        'num_generated': len(sampled_smiles_raw),
        'num_valid': len(valid_smiles),
        'num_unique_canonical': len(unique_canonical_smiles_set),
        'num_novel': num_novel
    }
    return results
# ... (rest of the file, including latent_visulization, latent_sample if present) ...


# --- Example Usage (You'll need to set up your args and tokenizer) ---
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate VAE model by sampling from N(0,I)")
    # Add arguments required by your model and evaluate function
    parser.add_argument('--model', type=str, default='VAE', help='Model type (e.g., VAE, LSTM_VAE)')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of character embeddings')
    parser.add_argument('--hidden_size', type=int, default=256, help='Size of RNN hidden states')
    parser.add_argument('--latent_size', type=int, default=64, help='Dimension of the latent space')
    parser.add_argument('--max_len', type=int, default=100, help='Maximum SMILES sequence length')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run evaluation on (cuda/cpu)')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate for evaluation')
    # ---> MODIFIED Default Path for Training Data <---
    parser.add_argument('--train_data_path', type=str, default='data/zinc.txt', help='Path to training SMILES text file (one SMILES per line) for novelty check')
    # Add any other necessary args for model loading or paths
    # e.g., parser.add_argument('--model_load_path', type=str, default='model/VAE_model.pth', help='Explicit path to load model weights')

    args = parser.parse_args()

    print("--- Evaluation Setup ---")
    print(f"Model Type: {args.model}")
    print(f"Latent Size: {args.latent_size}")
    print(f"Device: {args.device}")
    print(f"Number of Samples: {args.num_samples}")
    print(f"Training Data for Novelty: {args.train_data_path if args.train_data_path else 'Not provided'}")
    print("-" * 25)


    print("Initializing tokenizer...")
    try:
        my_tokenizer = tokenizer() # Or however you initialize your specific tokenizer

        # *** Crucial Check: Ensure vocab is populated ***
        if my_tokenizer.vocab_size == 0 or not my_tokenizer.char_to_int or not my_tokenizer.int_to_char:
            # Attempt to load or build vocab if possible
            # MODIFY this part if your tokenizer needs building from the .txt file
            if args.train_data_path:
                print(f"Attempting to build vocab from {args.train_data_path}...")
                try:
                    # Read SMILES from the text file for vocab building
                    with open(args.train_data_path, 'r') as f:
                        vocab_data = [line.strip() for line in f if line.strip()] # Read non-empty lines

                    # Assuming your tokenizer has a build_vocab method that takes a list of strings
                    if hasattr(my_tokenizer, 'build_vocab'):
                        my_tokenizer.build_vocab(vocab_data)
                        print("Vocab built successfully from list.")
                    else:
                         raise NotImplementedError("Tokenizer does not have a 'build_vocab' method.")

                    # Re-check after building
                    if my_tokenizer.vocab_size == 0 or not my_tokenizer.char_to_int or not my_tokenizer.int_to_char:
                         raise ValueError("Vocab building did not populate the tokenizer mappings.")

                except FileNotFoundError:
                     raise ValueError(f"Tokenizer vocab is empty, and train data file {args.train_data_path} not found for building.")
                except Exception as e:
                     raise ValueError(f"Tokenizer vocab is empty, and failed to build from {args.train_data_path}. Error: {e}")
            else:
                 raise ValueError("Tokenizer vocab is empty. Provide a way to load/build vocab or ensure it's initialized correctly.")

        print(f"Tokenizer vocabulary size: {my_tokenizer.vocab_size}")
        # Final check on mappings
        if not hasattr(my_tokenizer, 'int_to_char') or not my_tokenizer.int_to_char:
             raise ValueError("Tokenizer must have a populated 'int_to_char' map.")
        if not hasattr(my_tokenizer, 'char_to_int') or not my_tokenizer.char_to_int:
             raise ValueError("Tokenizer must have a populated 'char_to_int' map.")


    except NameError:
        print("Error: The 'tokenizer' class is not defined. Make sure it's imported from dataset.py or defined.")
        exit()
    except Exception as e:
        print(f"Error initializing or validating tokenizer: {e}")
        exit()


    # Call the evaluate function
    results = evaluate(args, my_tokenizer)

    # Print results summary
    if results:
        print("\n--- Final Results Summary ---")
        for key, value in results.items():
            if isinstance(value, float):
                # Handle NaN for printing
                value_str = f"{value:.4f}" if not np.isnan(value) else "NaN"
                print(f"{key.replace('_', ' ').capitalize()}: {value_str}")
            else:
                print(f"{key.replace('_', ' ').capitalize()}: {value}")
        print("--------------------------")