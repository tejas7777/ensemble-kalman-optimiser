import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from data.dataloader.iris_dataloader import IrisDataLoader  # Example for classification
from model.dnn import DNNClassification as DNN  # Assuming DNNClassification is suitable for classification tasks
import os
from datetime import datetime
from typing import Optional
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

class BatchTrainer:
    def __init__(self, model, lr=0.001, batch_size=32):
        self.model = model
        self.loss_function = nn.CrossEntropyLoss()  # For classification
        adam_params = {
            'lr': 0.01, 
            'betas': (0.9, 0.999),
            'weight_decay': 0.001,
            'eps': 1e-05
        }
        self.optimiser = Adam(model.parameters(), **adam_params)
        self.batch_size = batch_size

    def load_data(self, dataset_loader):
        self.train_loader, self.val_loader, self.test_loader = dataset_loader.get_loaders()

    def train(self, num_epochs=1000, is_plot_graph=1):
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
            self.plot_train_graph(self.train_accuracies, self.val_accuracies, self.train_losses, self.val_losses)

        self.train_accuracy = self.train_accuracies
        self.val_accuracy = self.val_accuracies
        self.train_loss = self.train_losses
        self.val_loss = self.val_losses

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
    
    def plot_train_graph(self, train_accuracies, val_accuracies, train_losses, val_losses):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot accuracies on the primary y-axis
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy (%)', color='g')
        ax1.plot(range(len(train_accuracies)), train_accuracies, 'g-', label='Train Accuracy', linewidth=2)
        ax1.plot(range(len(val_accuracies)), val_accuracies, 'g--', label='Validation Accuracy', linewidth=2)
        ax1.tick_params(axis='y', labelcolor='g')

        # Plot losses on the secondary y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Loss', color='b')
        ax2.plot(range(len(train_losses)), train_losses, 'b-', label='Train Loss', linewidth=2)
        ax2.plot(range(len(val_losses)), val_losses, 'b--', label='Validation Loss', linewidth=2)
        ax2.tick_params(axis='y', labelcolor='b')

        # Combine legends from both axes and place it outside the plot
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='center left', bbox_to_anchor=(1, 0.5))

        # Add title
        plt.title('Training and Validation Accuracy & Loss')

        # Adjust layout to make space for the legend
        fig.tight_layout(rect=[0, 0, 0.85, 1])

        plt.show()

    def save_model(self, filename=None):
        if filename is None:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'model_adam_{current_time}.pth'
        save_path = os.path.join('./saved_models', filename)
        torch.save(self.model.state_dict(), save_path)
        print(f'Complete model saved to {save_path}')

    def grid_search(self, param_grid, num_epochs=1000):
        best_params = None
        best_val_loss = float('inf')
        
        for lr in param_grid['lr']:
            for betas in param_grid['betas']:
                for weight_decay in param_grid['weight_decay']:
                    for eps in param_grid['eps']:
                        print(f'Training with lr={lr}, betas={betas}, weight_decay={weight_decay}, eps={eps}')
                        
                        # Update optimizer with current set of parameters
                        adam_params = {
                            'lr': lr, 
                            'betas': betas,
                            'weight_decay': weight_decay,
                            'eps': eps
                        }
                        self.optimiser = Adam(self.model.parameters(), **adam_params)
                        
                        # Train the model
                        self.train(num_epochs=num_epochs, is_plot_graph=0)
                        
                        # Evaluate on validation set
                        val_loss = self.evaluate(self.val_loader)
                        print(f'Validation Loss: {val_loss}')
                        
                        # Check if the current configuration is the best
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_params = adam_params
        
        print(f'Best Params: {best_params}, Best Validation Loss: {best_val_loss}')
        return best_params

if __name__ == '__main__':
    dataset_loader = IrisDataLoader(num_features=4, set_standardize=True, batch_size=32)
    input_size = dataset_loader.get_feature_dim()
    output_size = 3  # For Iris dataset (3 classes)
    model = DNN(input_size=input_size, output_size=output_size)
    model_train = BatchTrainer(model=model, batch_size=32)
    model_train.load_data(dataset_loader)
    # model_train.grid_search(
    #     param_grid={
    #         'lr': [0.001, 0.01, 0.1],
    #         'betas': [(0.9, 0.999), (0.9, 0.95)],
    #         'weight_decay': [0.0001, 0.001],
    #         'eps': [1e-08, 1e-05]
    #     },
    #     num_epochs=400,
    # )
    model_train.train(num_epochs=400, is_plot_graph=0)
    model_train.evaluate_test()
    model_train.save_model()
