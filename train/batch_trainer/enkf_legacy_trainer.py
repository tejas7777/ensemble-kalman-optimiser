import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
import os
from typing import Optional
from visualisation.plotter import Plotter
from optimiser.enkf_iterative import EnKFOptimizerIterative
from optimiser.enkf_legacy import EnKFOriginal
from data.dataloader.regression_loader import OscillatoryDataLoader
from model.dnn import DNN

class ModelTrainer:
    def __init__(self, model, sigma: float = 0.001, J: int = 50, gamma: float = 1e-1, max_iterations: Optional[int] = 1, loss_type: Optional[str] = 'mse'):
        self.model = model
        self.loss_function = nn.MSELoss()
        #self.optimiser = EnKFOptimizerIterative(model, sigma=sigma, J=J, gamma=gamma, max_iterations=max_iterations, debug_mode=False)
        self.optimiser = EnKFOriginal(model, k=J)
        self.plotter = Plotter()

    def load_data(self, dataset_loader):
        (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test) = dataset_loader.get_data_as_tensors()

    def train(self, num_epochs=100, is_plot_graph=1):
        train_losses = []
        val_losses = []

        print("TRAINING STARTED ...")
        dataset = list(zip(self.X_train, self.y_train))
        num_output = self.y_train.shape[1]

        for epoch in range(num_epochs):
            self.optimiser.step(dataset=dataset, num_output=num_output)

            train_loss = self.evaluate(self.X_train, self.y_train)
            val_loss = self.evaluate(self.X_val, self.y_val)
            
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

    def evaluate(self, X, y):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            output = self.model(X)
            total_loss = self.loss_function(output, y).item()
        return total_loss
    
    def evaluate_test(self):
        test_loss = self.evaluate(self.X_test, self.y_test)
        print(f'Test Loss: {test_loss}')

    def save_model(self, filename=None):
        if filename is None:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'model_enkf_{current_time}.pth'
        save_path = os.path.join('./saved_models', filename)
        torch.save(self.model, save_path)
        print(f'Complete model saved to {save_path}')

if __name__ == '__main__':
    dataset_loader = OscillatoryDataLoader(batch_size=1, val_size=0.2)
    input_size = dataset_loader.train_dataset.X.shape[1]
    output_size = dataset_loader.train_dataset.y.shape[1]
    model = DNN(input_size=input_size, output_size=output_size)
    model_train = ModelTrainer(model=model)
    model_train.load_data(dataset_loader)
    model_train.train(num_epochs=100, is_plot_graph=1)
    model_train.evaluate_test()
    model_train.save_model()
