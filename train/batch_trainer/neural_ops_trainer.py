import torch
import torch.nn as nn
#import matplotlib.pyplot as plt
from optimiser.enkf import EnKF
from datetime import datetime
import os
from typing import Optional
from data.dataloader.pde_loader import PDEDataLoader
from model.fno import FNO
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, model, lr: float = 0.1, sigma: float = 0.01, k: int = 50, gamma: float = 0.1, max_iterations: Optional[int] = 1, loss_type: Optional[str] = 'mse', debug_mode = True):
        self.model = model
        self.loss_function_mapper = {
            'mse': nn.MSELoss(),
            'cross_entropy': nn.CrossEntropyLoss()
        }
        self.loss_function = self.loss_function_mapper[loss_type]
        self.optimiser = EnKF(model, lr, sigma, k, gamma, max_iterations=max_iterations, debug_mode=debug_mode, loss_type=loss_type)

    def load_data(self, dataset_loader):
        self.train_loader, self.val_loader, self.test_loader = dataset_loader.get_loaders()

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
            train, obs = batch
            self.optimiser.step(train=train, obs=obs)
            total_loss += self.evaluate_single_batch(train, obs)
        return total_loss / len(self.train_loader)

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in data_loader:
                total_loss += self.evaluate_single_batch(inputs, targets)
        return total_loss / len(data_loader)

    def evaluate_single_batch(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.loss_function(outputs, targets)
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
            filename = f'model_fno_{current_time}.pth'
        save_path = os.path.join('./saved_models', filename)
        torch.save(self.model.state_dict(), save_path)
        print(f'Model saved to {save_path}')


if __name__ == '__main__':

    file_path = './dataset/pde_data_subset.pt'  # Update the path as needed
    subset_size = 1000  # Example subset size
    batch_size = 1000  # Example batch size

    pde_data_loader = PDEDataLoader(file_path, subset_size=subset_size, batch_size=batch_size)
    train_loader, val_loader, test_loader = pde_data_loader.get_loaders()

    in_channels = 4260  # Input shape based on your data
    out_channels = 14  # Output shape based on your data
    modes = 50
    width = 40

    model = FNO(in_channels, out_channels, modes, width)
    model_trainer = ModelTrainer(model=model, lr=0.01, max_iterations=1, loss_type='mse', debug_mode=False)
    model_trainer.load_data(pde_data_loader)
    model_trainer.train(num_epochs=200, is_plot_graph=1)
    model_trainer.evaluate_test()
    model_trainer.save_model()