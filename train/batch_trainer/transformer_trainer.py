import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from optimiser.enkf import EnKF
from data.dataloader.IMDb_data_loader import IMDBDataLoader
import os
from datetime import datetime
from model.transformer import Transformer

class TransformerTrainer:
    def __init__(self, model, lr=0.1, sigma=0.01, k=50, gamma=1e-1, max_iterations=1, loss_type='cross_entropy'):
        self.model = model
        self.loss_function_mapper = {
            'mse': nn.MSELoss(),
            'cross_entropy': nn.CrossEntropyLoss()
        }
        self.loss_function = self.loss_function_mapper[loss_type]
        self.optimiser = EnKF(model, lr, sigma, k, gamma, max_iterations=max_iterations, debug_mode=False, loss_type=loss_type)

    def load_data(self, dataset_loader):
        self.train_loader, self.val_loader, self.test_loader = dataset_loader.get_loaders()

    def train(self, num_epochs=20, is_plot_graph=1):
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
            inputs, targets = batch
            self.optimiser.step(train=inputs, obs=targets)
            total_loss += self.evaluate_single_batch(inputs, targets)
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
        # Plot training and validation loss
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
    dataset_loader = IMDBDataLoader(batch_size=100, val_size=0.2, max_vocab_size=5)
    vocab_size = len(dataset_loader.vocab)
    embed_dim = 4
    num_heads = 2
    num_layers = 1
    num_classes = 2

    model = Transformer(vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, num_classes=num_classes)
    
    model_train = TransformerTrainer(model=model, lr=0.1, sigma=0.01, k=10, gamma=1e-1, max_iterations=1, loss_type='cross_entropy')
    model_train.load_data(dataset_loader)
    model_train.train(num_epochs=10, is_plot_graph=1)
    model_train.evaluate_test()
    model_train.save_model()
