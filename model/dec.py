import torch
import torch.nn as nn

class DEC(nn.Module):
    def __init__(self, input_dim, latent_dim, n_clusters):
        super(DEC, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 50),
            nn.ReLU(),
            nn.Linear(50, input_dim)  # Ensure this matches the input dimension
        )
        self.cluster_centers = nn.Parameter(torch.zeros(n_clusters, latent_dim))

    def forward(self, x):
        #print("Input shape:", x.shape)
        z = self.encoder(x)
        #print("Encoded shape (latent):", z.shape)
        output = self.decoder(z)
        #print("Decoded shape (output):", output.shape)
        return output, z

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_centers) ** 2, dim=2))
        q = q ** 2
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q