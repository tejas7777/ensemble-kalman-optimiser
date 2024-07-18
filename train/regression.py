import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from optimiser.enkf import EnKF
from model.dnn import DNN
from dataloader.regression_loader import OscillatoryDataLoader
import pandas as pd
import os
from datetime import datetime

class ModelTrainer:
    def __init__(self, model, lr=0.9, sigma=0.01, k=10, gamma=1e-1, max_iterations=1):
        self.model = model
        self.loss_function = nn.MSELoss()
        self.optimiser = EnKF(model, lr, sigma, k, gamma, max_iterations=1, debug_mode=False)

    def load_data(self, dataset_loader):
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = dataset_loader.get_data()

    def train(self, num_epochs=1000, is_plot_graph=1):
        train_losses = []
        val_losses = []

        obs = self.y_train
        train = self.X_train

        print("TRAINING STARTED ...")
        for epoch in range(num_epochs):
            self.optimiser.step(train=train, obs=obs)

            train_loss = self.evaluate(self.X_train, self.y_train)
            val_loss = self.evaluate(self.X_val, self.y_val)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')

        if is_plot_graph:
            self.plot_train_graph(train_losses, val_losses)

        self.train_loss = train_losses
        self.val_loss = val_losses

    def evaluate(self, inputs, targets):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, targets)
        return loss.item()
    
    def evaluate_test(self):
        test_loss = self.evaluate(self.X_test, self.y_test)
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
    dataset_loader = OscillatoryDataLoader(test_size=0.3, val_size=0.3)
    model_train = ModelTrainer(model=DNN(input_size=dataset_loader.X.shape[1], output_size=dataset_loader.y.shape[1]))
    model_train.load_data(dataset_loader)
    model_train.train(is_plot_graph=1)
    model_train.evaluate_test()
    model_train.save_model()
