import torch
import torch.nn as nn

class MNISTGenerator(nn.Module):
    def __init__(self, latent_dim):
        super(MNISTGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(-1, 1, 28, 28)

class MNISTDiscriminator(nn.Module):
    def __init__(self):
        super(MNISTDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)

class MNISTGAN(nn.Module):
    def __init__(self, latent_dim):
        super(MNISTGAN, self).__init__()
        self.latent_dim = latent_dim
        self.generator = MNISTGenerator(latent_dim)
        self.discriminator = MNISTDiscriminator()

    def generate(self, z):
        return self.generator(z)

    def discriminate(self, img):
        return self.discriminator(img)

    def forward(self, z, real_imgs):
        fake_imgs = self.generate(z)
        d_real = self.discriminate(real_imgs)
        d_fake = self.discriminate(fake_imgs)
        return fake_imgs, d_real, d_fake