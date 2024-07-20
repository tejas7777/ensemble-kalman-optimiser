import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from data.dataloader.regression_loader import OscillatoryDataLoader
import os
from datetime import datetime
from model.dnn import DNN

class BatchTrainer:
    def __init__(self, model, lr=0.001, batch_size=32):
        self.model = model
        self.loss_function = nn.MSELoss()
        self.optimiser = Adam(model.parameters(), lr=lr)
        self.batch_size = batch_size

    def load_data(self, dataset_loader):
        self.train_loader, self.val_loader, self.test_loader = dataset_loader.get_loaders()

    def train(self, num_epochs=1000, is_plot_graph=1):
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
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, targets)
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(data_loader)
    
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
            filename = f'model_adam_{current_time}.pth'
        save_path = os.path.join('./saved_models', filename)
        torch.save(self.model.state_dict(), save_path)
        print(f'Complete model saved to {save_path}')

if __name__ == '__main__':
    dataset_loader = OscillatoryDataLoader(test_size=0.3, val_size=0.3, batch_size=32)
    model_train = BatchTrainer(model=DNN(input_size=dataset_loader.train_dataset.X.shape[1], output_size=dataset_loader.train_dataset.y.shape[1]), batch_size=32)
    model_train.load_data(dataset_loader)
    model_train.train(num_epochs=1000, is_plot_graph=1)
    model_train.evaluate_test()
    model_train.save_model()