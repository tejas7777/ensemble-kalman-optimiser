import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from optimiser.enkf_cnn import EnKFCNN
from model.dnn import DenoisingAutoencoder
from data.dataloader.noisy_mnist import NoisyMNISTDataLoader
import pandas as pd
import os
from datetime import datetime
from typing import Optional
from visualisation.plotter import Plotter
from itertools import product

class BatchTrainer:
    def __init__(self, model, lr:float =0.1, sigma:float =0.01, k:int =50, gamma: float=1e-1, max_iterations: Optional[int]=1, loss_type: Optional[str]='mse', online_learning:Optional[bool] = False):
        self.model = model
        self.loss_function_mapper ={
            'mse': nn.MSELoss(),
            'cross_entropy': nn.CrossEntropyLoss()
        }
        self.loss_function = self.loss_function_mapper[loss_type]
        self.optimiser = EnKFCNN(model, lr, sigma, k, gamma, max_iterations=max_iterations, debug_mode=False,loss_type=loss_type)
        self.plotter = Plotter()

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
            self.plotter.plot_train_graph(
                (train_losses, "Train Loss"), 
                (val_losses, "Validation Loss"),
                log_scale=True, 
                xlabel='Epochs', 
                ylabel='Loss', 
                title='Training and Validation Loss',
                legend=True, 
                save_path=None
            )

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

    def evaluate_accuracy(self):
        self.model.eval()
        
        train_correct = 0
        train_total = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            # Evaluate on the training set
            for inputs, targets in self.train_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
                
            # Evaluate on the validation (test) set
            for inputs, targets in self.test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        
        return train_accuracy, val_accuracy

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

    def grid_search(self, param_grid, num_epochs=100):
        best_params = None
        best_val_loss = float('inf')
        
        param_combinations = list(product(*param_grid.values()))
        
        for param_combination in param_combinations:
            # Unpack the parameter combination
            param_dict = dict(zip(param_grid.keys(), param_combination))
            
            # Initialize the model and optimizer with the current parameters
            self.model.initialize_weights()  # Reinitialize model weights
            self.optimiser = EnKF(
                self.model, 
                lr=param_dict.get('lr', 0.1), 
                sigma=param_dict.get('sigma', 0.01), 
                k=param_dict.get('k', 100), 
                gamma=param_dict.get('gamma', 1e-2), 
                max_iterations=param_dict.get('max_iterations', 1), 
                debug_mode=False, 
                loss_type=param_dict.get('loss_type', 'mse')
            )
            
            # Train the model
            self.train(num_epochs=num_epochs, is_plot_graph=0)
            
            # Evaluate validation loss
            val_loss = self.evaluate(self.val_loader)
            
            print(f"Params: {param_dict}, Validation Loss: {val_loss}")
            
            # Update the best parameters if current val_loss is lower
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = param_dict
        
        print(f"Best Parameters: {best_params}, Best Validation Loss: {best_val_loss}")
        return best_params, best_val_loss
    


if __name__ == '__main__':
    #Initialise Dataloader
    dataset_loader = NoisyMNISTDataLoader(
        subset_size=500, 
        set_standardize=False, 
        test_size=0.2, 
        val_size=0.2, 
        batch_size=1000
    )

    model = DenoisingAutoencoder()
    model_train = BatchTrainer(model=model)
    model_train.load_data(dataset_loader)
    param_grid = {
    'lr': [0.1, 0.5 ],
    'sigma': [0.0001, 0.001, 0.01],
    'k': [100, 150],
    'gamma': [1e-1, 1e-2, 1e-3],
    # 'max_iterations': [1],
    # 'loss_type': ['mse']
    }
    #model_train.grid_search(param_grid, num_epochs=200)
    model_train.train(num_epochs=200, is_plot_graph=1)
    model_train.evaluate_test()
    model_train.save_model()