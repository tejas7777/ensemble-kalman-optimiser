import torch
import torch.nn as nn
import matplotlib.pyplot as plt
#from optimiser.enkf import EnKF
from optimiser.enkf_bnn import EnKFOptimiser
from model.bnn import BayesianNet, bnn_loss_function
from model.dnn import DNNClassification
from data.dataloader.iris_dataloader import IrisDataLoader
import os
from datetime import datetime
from typing import Optional

class BNNTrainer:
    def __init__(self, model,  lr:float =0.1, sigma:float =0.001, k:int =100, gamma: float=1e-1, max_iterations: Optional[int]=1, loss_type: Optional[str]='mse'):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = EnKFOptimiser(model, lr, sigma, k, gamma, max_iterations=max_iterations, debug_mode=False,loss_type=loss_type)
        self.debug_mode = False

    def load_data(self, dataset_loader):
        self.train_loader, self.val_loader, self.test_loader = dataset_loader.get_loaders()

    def train(self, num_epochs=100, is_plot_graph=1):
        train_losses = []
        val_losses = []
        val_accuracies = []

        print("TRAINING STARTED ...")
        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch()
            val_loss, val_accuracy = self.evaluate(self.val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        if is_plot_graph:
            self.plot_train_graph(train_losses, val_losses, val_accuracies)

        self.train_loss = train_losses
        self.val_loss = val_losses

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        for inputs, targets in self.train_loader:
            self.optimizer.update(inputs, targets)
            total_loss += self.evaluate_single_batch(inputs, targets)
        return total_loss / len(self.train_loader)

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        accuracy = 100. * correct / total
        return total_loss / len(data_loader), accuracy

    def evaluate_single_batch(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        return loss.item()

    def evaluate_test(self):
        test_loss, test_accuracy = self.evaluate(self.test_loader)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    def plot_train_graph(self, train_losses, val_losses, val_accuracies):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Validation Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def save_model(self, filename=None):
        if filename is None:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'bnn_model_enkf_{current_time}.pth'
        save_path = os.path.join('./saved_models', filename)
        torch.save(self.model.state_dict(), save_path)
        print(f'BNN model saved to {save_path}')

# Usage example:
if __name__ == '__main__':
    dataset_loader = IrisDataLoader(batch_size=20, val_size=0.2)
    input_size = len(dataset_loader.get_selected_features())
    hidden_size = 10
    output_size = 3  # Iris dataset has 3 classes
    model = BayesianNet(input_size, hidden_size, output_size)
    
    trainer = BNNTrainer(model=model)
    trainer.load_data(dataset_loader)
    trainer.train(num_epochs=100, is_plot_graph=1)
    trainer.evaluate_test()
    trainer.save_model()