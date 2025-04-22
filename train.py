from tqdm import tqdm
from dataset import *
from model import VAE_Autoencoder
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


def get_data(args):
    gene_dataloader = Gene_Dataloader(args.batch_size, args.path, args.cell_name, args.train_rate)
    train_loader, test_loader = gene_dataloader.get_dataloader()
    return train_loader, test_loader

def kl_divergence_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def train(args):
    train_losses = []
    train_latent_vectors = []
    my_tokenizer = tokenizer()
    vocab_size = my_tokenizer.vocab_size
    train_loader, test_loader = get_data(args)
    print(len(train_loader.dataset))
    model = VAE_Autoencoder(vocab_size, args.embedding_dim, args.hidden_size, args.latent_size,args.max_len, model_type=args.model).to(args.device)
    model.vocab = my_tokenizer.char_to_int

    criterion = nn.CrossEntropyLoss(ignore_index= my_tokenizer.char_to_int[' '], reduction='sum') 
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0


        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}"):
            batch = batch.to(args.device)  # (batch_size, seq_len)  # 注意这里的 seq_len 是可变的
            optimizer.zero_grad()

            if args.model == 'VAE':
                recon_batch, mu, logvar = model(batch)  # recon_batch: (batch_size, max_len, vocab_size)
            elif args.model == 'AE':
                recon_batch, latent_batch = model(batch)

            seq_len = batch.size(1)

            # Reconstruction Loss
            # recon_loss = criterion(recon_batch[:, :seq_len - 1, :].reshape(-1, vocab_size),
            #                 batch[:, 1:].reshape(-1))
            input_size = recon_batch[:, :seq_len - 1, :].reshape(-1, vocab_size).size(0)
            target_size = batch[:, 1:].reshape(-1).size(0)
            
            if input_size != target_size:
                print(f"Warning: Input size ({input_size}) doesn't match target size ({target_size})")
                min_size = min(input_size, target_size)
                if input_size > target_size:
                    input_reshape = recon_batch[:, :seq_len - 1, :].reshape(-1, vocab_size)[:target_size]
                    target_reshape = batch[:, 1:].reshape(-1)
                else:
                    input_reshape = recon_batch[:, :seq_len - 1, :].reshape(-1, vocab_size)
                    target_reshape = batch[:, 1:].reshape(-1)[:input_size]
                
                recon_loss = criterion(input_reshape, target_reshape)
            else:
                recon_loss = criterion(recon_batch[:, :seq_len - 1, :].reshape(-1, vocab_size),
                                batch[:, 1:].reshape(-1))


            if args.model == 'VAE':
                kl_loss = kl_divergence_loss(mu, logvar)
                loss = recon_loss + kl_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            if args.model == 'VAE':
                total_kl_loss += kl_loss.item()



        avg_train_loss = total_loss / len(train_loader.dataset) # 平均loss
        avg_recon_loss = total_recon_loss / len(train_loader.dataset)
        if args.model == 'VAE':
            avg_kl_loss = total_kl_loss / len(train_loader.dataset)

        train_losses.append(avg_train_loss)
        if args.model == 'VAE':
            print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {avg_train_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}")


    # train_latent_vectors = np.concatenate(train_latent_vectors, axis=0)
    model.eval()
    train_latent_vectors = []
    mu_list = []
    var_list = []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc='Computing latent vectors'):
            if args.model == 'VAE':
                _, mu, logvar = model.encode(batch)  # 只需 mu 和 logvar
                mu_list.append(mu)
                var = torch.exp(0.5 * logvar)
                var_list.append(var)
                z = model.reparameterize(mu, logvar).detach().cpu().numpy()
            train_latent_vectors.append(z)
    
    mu_list = np.concatenate(mu_list, axis=0)
    var_list = np.concatenate(var_list, axis=0)
    train_latent_vectors = np.concatenate(train_latent_vectors, axis=0)
    print(f"Final mu shape: {mu_list.shape}")
    print(f"Final var shape: {var_list.shape}")
    print(f"Final train_latent_vectors shape: {train_latent_vectors.shape}")



    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curve ({args.model})') # Add model type to title
    plt.legend()
    plt.savefig(f'results/training_loss_curve_{args.model}.png')
    plt.show()

    torch.save(model.state_dict(), f'model/{args.model}_model_{args.cell_name}.pth')
    print(f'Model saved to model/{args.model}_model_{args.cell_name}.pth')

    np.save(f"results/{args.model}_mu_{args.latent_size}_{args.cell_name}", mu_list)
    np.save(f"results/{args.model}_var_{args.latent_size}_{args.cell_name}", var_list)
    np.save(f'results/{args.model}_latent_vectors_{args.latent_size}_{args.cell_name}', train_latent_vectors)

    print(f"Save mu, var and latent vector in results")
