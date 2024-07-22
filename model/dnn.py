from torch import nn

class DNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 30)
        self.fc3 = nn.Linear(30, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class DenoisingAutoencoder(nn.Module):
    def __init__(self, channels=1, latent_dim=128):
        super(DenoisingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, latent_dim, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
