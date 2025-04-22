import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import tokenizer


class VAE_Autoencoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, latent_size, max_len, model_type='VAE'):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.max_len = max_len
        self.model_type = model_type # 'VAE' or 'AE'
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)
        self.vocab = None # Initialize vocab attribute
        # Encoder (GRU)
        self.encoder = nn.GRU(embed_size, hidden_size, batch_first=True)
        ## VAE: mean and log variance
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)
        ## AE: directly to latent space
        self.encoding_to = nn.Linear(hidden_size, latent_size)

        # Decoder (GRU)
        self.decoder = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.decoding_to = nn.Linear(hidden_size, vocab_size)

        # From latent to hidden
        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)

    def reparameterize(self, mu, logvar): # Only for VAE
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, inputs):
        embed = self.embedding(inputs) # [batch_size, embed_size]
        output, hd = self.encoder(embed) # [1, batch_size, hidden_size]

        if self.model_type == 'VAE':
            mu = self.fc_mu(hd[-1])  # [batch_size, hidden_size]
            logvar = self.fc_logvar(hd[-1]) # [batch_size, hidden_size]
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z
        else:
            raise ValueError("model_type must be 'VAE'")


    def decode(self, z, max_len):
            if self.vocab is None:
                raise ValueError("Model vocabulary not set. Assign char_to_int dictionary to model.vocab")
            # 将潜在向量映射到解码器的初始隐藏状态
            hidden = self.latent_to_hidden(z)  # (batch_size, latent_size) -> (batch_size, hidden_size)
            hidden = torch.tanh(hidden) 
            hidden = hidden.unsqueeze(0)  # (batch_size, hidden_size) -> (1, batch_size, hidden_size)

            # The first input is <sos>.
            # batch_size = z.size(0)
            # decoder_input = torch.full((batch_size, 1), self.vocab['^'], dtype=torch.long, device=z.device)
            # decoder_input_embedded = self.embedding(decoder_input) # (batch_size, 1) -> (batch_size, 1, embed_size)
            # # store every step
            # outputs = []
            batch_size = z.size(0)
            start_token_idx = self.vocab['^'] # Get start token index from vocab
            decoder_input = torch.full((batch_size, 1), start_token_idx, dtype=torch.long, device=z.device)
            decoder_input_embedded = self.embedding(decoder_input) # (batch_size, 1) -> (batch_size, 1, embed_size)
            # store every step
            outputs = []

            # Recurrent decoding
            for i in range(max_len):
                # decoder_input_embedded: [batch_size, 1, embed_size]   hidden: [1, batch_size, hidden_szie]
                output, hidden = self.decoder(decoder_input_embedded, hidden) #output: (batch_size, 1, hidden_size), hidden: (1, batch_size, hidden_size)
                output = self.decoding_to(output.squeeze(1))  # (batch_size, 1, hidden_size) -> (batch_size, hidden_size) -> (batch_size, vocab_size)
                outputs.append(output) #output: (batch_size, vocab_size)

                # Max prob token
                top1 = output.argmax(1) #shape: (batch_size)

                # top1 token as the input of next time step
                decoder_input = top1.unsqueeze(1) # 下一个decoder input: (batch_size, 1)
                decoder_input_embedded = self.embedding(decoder_input) # (batch_size, 1) -> (batch_size, 1, embed_size)

            outputs = torch.stack(outputs, dim=1)  # (batch_size, max_len, vocab_size)

            return outputs

    def forward(self, x):
        if self.model_type == 'VAE':
            mu, logvar, z = self.encode(x)
            x_recon = self.decode(z, self.max_len)
            return x_recon, mu, logvar
        else:
            raise ValueError("model_type must be 'VAE'")