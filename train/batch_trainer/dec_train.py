import torch
import torch.nn as nn
from optimiser.enkf_clustering import EnKF
from datetime import datetime
import os
from typing import Optional
from data.dataloader.iris_dataloader import IrisDataLoader
import matplotlib.pyplot as plt
from model.dec import DEC
from sklearn.cluster import KMeans

class DECTrainer:
    def __init__(self, model, lr: float = 0.1, sigma: float = 0.1, k: int = 100, gamma: float = 1e-1, max_iterations: Optional[int] = 1, alpha: float = 1.0):
        self.model = model
        self.alpha = alpha
        self.loss_function = nn.MSELoss()
        self.optimiser = EnKF(model, lr, sigma, k, gamma, max_iterations=max_iterations, debug_mode=False, loss_type='mse')

    def load_data(self, dataset_loader):
        self.train_loader, self.val_loader, self.test_loader = dataset_loader.get_loaders()

    def pretrain_autoencoder(self, num_epochs=50):
        print("Pretraining Autoencoder...")
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in self.train_loader:
                train, _ = batch
                reconstructed, _ = self.model(train)
                loss = self.loss_function(reconstructed, train)  # Ensure dimensions match
                self.optimiser.step(train=train, obs=train)
                total_loss += loss.item()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(self.train_loader)}')

    def initialize_cluster_centers(self):
        print("Initializing cluster centers using K-means...")
        features = []
        with torch.no_grad():
            for data, _ in self.train_loader:
                z, _ = self.model(data)
                features.append(z)
        features = torch.cat(features, dim=0)
        kmeans = KMeans(n_clusters=self.model.cluster_centers.shape[0], n_init=20)
        y_pred = kmeans.fit_predict(features.cpu().numpy())
        self.model.cluster_centers.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(features.device)

    def train(self, num_epochs=100, is_plot_graph=1):
        train_losses = []
        val_losses = []

        print("TRAINING STARTED ...")
        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.evaluate(self.val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')

        if is_plot_graph:
            self.plot_train_graph(train_losses, val_losses)

        self.train_loss = train_losses
        self.val_loss = val_losses

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            train, _ = batch
            self.optimiser.step(train=train, obs=train)
            total_loss += self.evaluate_single_batch(train)
        return total_loss / len(self.train_loader)

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, _ in data_loader:
                total_loss += self.evaluate_single_batch(inputs)
        return total_loss / len(data_loader)

    def evaluate_single_batch(self, inputs):
        _, outputs = self.model(inputs)
        loss = self.loss_function(outputs, inputs)
        return loss.item()

    def evaluate_test(self):
        test_loss = self.evaluate(self.test_loader)
        print(f'Test Loss: {test_loss}')

    def plot_train_graph(self, train_losses, val_losses):
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.show()

    def save_model(self, filename=None):
        if filename is None:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'model_enkf_{current_time}.pth'
        save_path = os.path.join('./saved_models', filename)
        torch.save(self.model, save_path)
        print(f'Complete model saved to {save_path}')


if __name__ == '__main__':
    dataset_loader = IrisDataLoader(num_features=2, batch_size=32)  # Adjust num_features as needed
    input_size = dataset_loader.get_feature_dim()
    output_size = 3  # Number of classes in the Iris dataset
    model = DEC(input_dim=input_size, latent_dim=200, n_clusters=output_size)  # Adjust latent_dim as needed
    model_train = DECTrainer(model=model)
    model_train.load_data(dataset_loader)
    model_train.pretrain_autoencoder(num_epochs=50)  # Pretrain the autoencoder
    model_train.initialize_cluster_centers()  # Initialize cluster centers
    model_train.train(num_epochs=100, is_plot_graph=1)
    model_train.evaluate_test()
    model_train.save_model()