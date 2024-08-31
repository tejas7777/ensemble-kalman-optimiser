import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from optimiser.enkf import EnKF
from model.dnn import DNNClassification as DNN
from data.dataloader.iris_dataloader import IrisDataLoader
from data.dataloader.classification_loader import ClassificationDataLoader
import os
from datetime import datetime
from typing import Optional
from visualisation.plotter import Plotter
from itertools import product
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

class BatchTrainer:
    def __init__(self, model, lr: float = 0.2, sigma: float = 0.01, k: int = 50, gamma: float = 1e-1, max_iterations: Optional[int] = 1, loss_type: Optional[str] = 'cross_entropy', online_learning: Optional[bool] = False):
        self.model = model
        self.loss_function_mapper = {
            'mse': nn.MSELoss(),
            'cross_entropy': nn.CrossEntropyLoss()
        }
        self.loss_function = self.loss_function_mapper[loss_type]
        self.optimiser = EnKF(model, lr, sigma, k, gamma, max_iterations=max_iterations, debug_mode=False, loss_type=loss_type)
        self.plotter = Plotter()

    def load_data(self, dataset_loader):
        self.train_loader, self.val_loader, self.test_loader = dataset_loader.get_loaders()

    def train(self, num_epochs=100, is_plot_graph=1):
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_losses = []
        self.val_losses = []

        print("TRAINING STARTED ...")
        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.evaluate(self.val_loader)

            train_accuracy = self.evaluate_accuracy(self.train_loader)
            val_accuracy = self.evaluate_accuracy(self.val_loader)
            
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f'Epoch {epoch+1}/{num_epochs}, Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if is_plot_graph:
            self.plot_train_graph(
                (self.train_accuracies, "Train Accuracy"), 
                (self.val_accuracies, "Validation Accuracy"),
                (self.train_losses, "Train Loss"),
                (self.val_losses, "Validation Loss")
            )

        self.train_accuracy = self.train_accuracies
        self.val_accuracy = self.val_accuracies
        self.train_loss = self.train_losses
        self.val_loss = self.val_losses

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            inputs, targets = batch
            self.optimiser.step(train=inputs, obs=targets)
            total_loss += self.evaluate_single_batch(inputs, targets)
        return total_loss / len(self.train_loader)

    def evaluate_single_batch(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.loss_function(outputs, targets)
        return loss.item()

    def evaluate_accuracy(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        accuracy = 100 * correct / total
        return accuracy

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in data_loader:
                total_loss += self.evaluate_single_batch(inputs, targets)
        return total_loss / len(data_loader)

    
    def evaluate_test(self):
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        cm = confusion_matrix(all_targets, all_predictions)

        # Use the class indices as the target names if `classes` attribute is not available
        if hasattr(self.test_loader.dataset, 'classes'):
            target_names = self.test_loader.dataset.classes
        else:
            target_names = [f'Class {i}' for i in range(len(np.unique(all_targets)))]

        report = classification_report(all_targets, all_predictions, target_names=target_names)
        
        test_accuracy = 100 * sum(np.array(all_predictions) == np.array(all_targets)) / len(all_targets)
        
        print(f'Test Accuracy: {test_accuracy:.2f}%')
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", report)
        return cm, report, test_accuracy

    def save_model(self, filename=None):
        if filename is None:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'model_enkf_{current_time}.pth'
        save_path = os.path.join('./saved_models', filename)
        torch.save(self.model.state_dict(), save_path)
        print(f'Model saved to {save_path}')

    def grid_search(self, param_grid, num_epochs=100):
        best_params = None
        best_val_accuracy = 0
        
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
                loss_type=param_dict.get('loss_type', 'cross_entropy')
            )
            
            # Train the model
            self.train(num_epochs=num_epochs, is_plot_graph=0)
            
            # Evaluate validation accuracy
            val_accuracy = self.evaluate_accuracy(self.val_loader)
            
            print(f"Params: {param_dict}, Validation Accuracy: {val_accuracy:.2f}%")
            
            # Update the best parameters if current val_accuracy is higher
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_params = param_dict
        
        print(f"Best Parameters: {best_params}, Best Validation Accuracy: {best_val_accuracy:.2f}%")
        return best_params, best_val_accuracy

    def plot_train_graph(self, train_accuracy_data, val_accuracy_data, train_loss_data, val_loss_data):

        train_accuracies = list(map(float, train_accuracy_data[0])) 
        val_accuracies = list(map(float, val_accuracy_data[0]))   
        train_losses = list(map(float, train_loss_data[0])) 
        val_losses = list(map(float, val_loss_data[0]))

        # Pass the formatted data to the plotting function
        self.plotter.plot_train_accuracy_loss_graph(
            train_accuracies, 
            val_accuracies, 
            train_losses,
            val_losses,
            xlabel='Epochs', 
            accuracy_label='Accuracy (%)', 
            loss_label='Loss', 
        )

if __name__ == '__main__':
    #dataset_loader = IrisDataLoader(num_features=2, set_standardize=False, batch_size=100000)
    file_path = './dataset/multi_class_classification_data.csv'  # Update with your file path
    dataset_loader = ClassificationDataLoader(file_path=file_path, set_standardize=True, test_size=0.2, val_size=0.1, batch_size=100000000)
    input_size, output_size = dataset_loader.get_feature_dim()
    #output_size = 3  # For Iris dataset (3 classes)
    model = DNN(input_size=input_size, output_size=output_size)
    model_train = BatchTrainer(model=model, loss_type='cross_entropy')
    model_train.load_data(dataset_loader)
    model_train.train(num_epochs=100, is_plot_graph=1)
    model_train.evaluate_test()
    model_train.save_model()
