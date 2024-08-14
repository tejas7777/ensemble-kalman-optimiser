import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from optimiser.enkf_gan import EnKFGAN
import os
from datetime import datetime
from model.gan import MNISTGAN
from data.dataloader.mnist_loader import MNISTDataLoader
from typing import Optional

class GANTrainer:
    def __init__(self, gan_model, lr:float =0.01, sigma:float =0.001, k:int = 10, gamma: float=1e-1, max_iterations: Optional[int]=1, loss_type: Optional[str]='mse', latent_dim=100):
        self.gan_model = gan_model
        self.latent_dim = latent_dim
        self.generator_optim = EnKFGAN(gan_model.generator, 0.0001, sigma, k, gamma, max_iterations, debug_mode=False)
        self.discriminator_optim = EnKFGAN(gan_model.discriminator, 0.05, sigma, 5, gamma, max_iterations, debug_mode=False)
        self.criterion = nn.BCELoss()

    def load_data(self, dataset_loader):
        self.train_loader, self.val_loader, self.test_loader = dataset_loader.get_loaders()

    def train(self, num_epochs=100, save_interval=10):
        g_losses, d_losses = None, None
        for epoch in range(num_epochs):
            g_losses, d_losses = self.train_one_epoch()
            
            print(f'Epoch {epoch+1}/{num_epochs}, '
                  f'G Loss: {sum(g_losses)/len(g_losses):.4f}, '
                  f'D Loss: {sum(d_losses)/len(d_losses):.4f}')

            if (epoch + 1) % save_interval == 0:
                #self.save_model(f'gan_model_epoch_{epoch+1}.pth')
                self.generate_and_save_images(epoch + 1)
        self.plot_losses(g_losses, d_losses)

    def train_one_epoch(self):
        self.gan_model.train()
        g_losses, d_losses = [], []

        for real_imgs, _ in self.train_loader:
            batch_size = real_imgs.size(0)
            
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # Train Discriminator
            z = torch.randn(batch_size, self.latent_dim)
            fake_imgs = self.gan_model.generate(z)
            
            # Real images
            real_output = self.gan_model.discriminate(real_imgs)
            real_loss = self.criterion(real_output, real_labels)
            
            # Fake images
            fake_output = self.gan_model.discriminate(fake_imgs.detach())
            fake_loss = self.criterion(fake_output, fake_labels)
            
            d_loss = real_loss + fake_loss

            # Optimize discriminator with EnKF
            #self.discriminator_optim.step(train=torch.cat([real_imgs, fake_imgs.detach()]), obs=torch.cat([real_labels, fake_labels]))
            self.discriminator_optim.step(train=real_imgs, obs=real_labels)
            self.discriminator_optim.step(train=fake_imgs.detach(), obs=fake_labels)

            # Train Generator
            gen_labels = torch.ones(batch_size, 1)
            z = torch.randn(batch_size, self.latent_dim)
            fake_imgs = self.gan_model.generate(z)
            g_output = self.gan_model.discriminate(fake_imgs)
            g_loss = self.criterion(g_output, gen_labels)
            
            # Optimize generator with EnKF
            self.generator_optim.step(train=z, obs=gen_labels)

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        return g_losses, d_losses


    def generate_and_save_images(self, epoch):
        self.gan_model.eval()
        with torch.no_grad():
            z = torch.randn(25, self.latent_dim)
            generated_imgs = self.gan_model.generate(z).cpu()

        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            ax.imshow(generated_imgs[i, 0, :, :], cmap='gray')
            ax.axis('off')

        plt.savefig(f'generated_images_epoch_{epoch}.png')
        plt.close()

    def save_model(self, filename=None):
        if filename is None:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'gan_model_enkf_{current_time}.pth'
        save_path = os.path.join('./saved_models', filename)
        torch.save(self.gan_model.state_dict(), save_path)
        print(f'GAN model saved to {save_path}')

    def plot_losses(self, g_losses, d_losses):
        """
        Plot the Generator and Discriminator losses over epochs.
        
        :param g_losses: List of average generator losses per epoch
        :param d_losses: List of average discriminator losses per epoch
        """
        epochs = range(1, len(g_losses) + 1)

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, g_losses, 'b', label='Generator Loss')
        plt.plot(epochs, d_losses, 'r', label='Discriminator Loss')
        plt.title('Generator and Discriminator Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Save the plot
        plt.savefig('gan_losses.png')
        plt.close()

        print("Loss plot saved as gan_losses.png")

if __name__ == '__main__':
    latent_dim = 50
    gan_model = MNISTGAN(latent_dim)
    dataset_loader = MNISTDataLoader(batch_size=32)

    trainer = GANTrainer(gan_model, latent_dim=latent_dim)
    trainer.load_data(dataset_loader)
    trainer.train(num_epochs=50, save_interval=10)