import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from optimiser.enkf_adaptive import EnKFAdaptive
from model.dnn import DNN, DenoisingAutoencoder
from data.dataloader.regression_loader import OscillatoryDataLoader
import pandas as pd
import os
from datetime import datetime
from typing import Optional

class BatchTrainer:
    def __init__(self, model, lr:float =0.5, sigma:float =0.01, k:int =100, gamma: float=1e-1, max_iterations: Optional[int]=1, loss_type: Optional[str]='mse', online_learning:Optional[bool] = False):
        self.model = model
        self.loss_function_mapper ={
            'mse': nn.MSELoss(),
            'cross_entropy': nn.CrossEntropyLoss()
        }
        self.loss_function = self.loss_function_mapper[loss_type]
        self.optimiser = EnKFAdaptive(model, lr, sigma, k, gamma, max_iterations=max_iterations, debug_mode=False,loss_type=loss_type)

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
            self.optimiser.step(train=train, obs=obs)  # Batch training: update per batch
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
        #Plot training and validation loss
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.show()
        pass

    def save_model(self, filename=None):
        if filename is None:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'model_enkf_{current_time}.pth'
        save_path = os.path.join('./saved_models', filename)
        torch.save(self.model, save_path)
        print(f'Complete model saved to {save_path}')

if __name__ == '__main__':
    #dataset_loader = OscillatoryDataLoader(test_size=0.3, val_size=0.3, batch_size=100)
    dataset_loader = OscillatoryDataLoader(batch_size=64, val_size=0.2)
    input_size = dataset_loader.train_dataset.X.shape[1]
    output_size = dataset_loader.train_dataset.y.shape[1]
    model = DNN(input_size=input_size, output_size=output_size)
    model_train = BatchTrainer(model=model)
    model_train.load_data(dataset_loader)
    model_train.train(num_epochs=1000, is_plot_graph=0)
    model_train.evaluate_test()
    model_train.save_model()