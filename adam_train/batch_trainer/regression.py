import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from data.dataloader.regression_loader import OscillatoryDataLoader
from data.dataloader.saddle_point_dataset_loader import SaddlePointDataLoader
from data.dataloader.ackley_function_dataset_loader import AckleyFunctionDataLoader
from data.dataloader.greiwank_function_dataset_loader import GriewankFunctionDataLoader
from data.dataloader.michalewicz_function_dataset_loader import MichalewiczFunctionDataLoader
import os
from datetime import datetime
from model.dnn import DNN

class BatchTrainer:
    def __init__(self, model, lr=0.001, batch_size=32):
        self.model = model
        self.loss_function = nn.MSELoss()
        # adam_params = { //Best for Oscillatory
        #     'lr': 0.01, 
        #     'betas': (0.9, 0.999),
        #     'weight_decay': 0.0001,
        #     'eps': 1e-05
        # }
        # adam_params = {'lr': 0.01, 'betas': (0.9, 0.95), 'weight_decay': 0.001, 'eps': 1e-05} 
        #adam_params = {'lr': 0.1, 'betas': (0.9, 0.95), 'weight_decay': 0.001, 'eps': 1e-08}
        #adam_params = {'lr': 0.1, 'betas': (0.9, 0.95), 'weight_decay': 0.001, 'eps': 1e-08}
        adam_params = {'lr': 0.01, 'betas': (0.9, 0.999), 'weight_decay': 0.001, 'eps': 1e-05}
        self.optimiser = Adam(model.parameters(), **adam_params)
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

    def save_model(self, filename=None):
        if filename is None:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'model_adam_{current_time}.pth'
        save_path = os.path.join('./saved_models', filename)
        torch.save(self.model.state_dict(), save_path)
        print(f'Complete model saved to {save_path}')

if __name__ == '__main__':
    #dataset_loader = OscillatoryDataLoader(test_size=0.3, val_size=0.3, batch_size=32)
    #dataset_loader = SaddlePointDataLoader(num_samples=10000, noise_level=0.1, val_size=0.2, test_size=0.2, batch_size=10000)
    #dataset_loader = AckleyFunctionDataLoader(num_samples=10000, dimension=100, noise_level=0.9, batch_size=10000000)
    #dataset_loader = GriewankFunctionDataLoader(num_samples=10000, dimension=10, noise_level=0.9, batch_size=10000000)
    dataset_loader = MichalewiczFunctionDataLoader(num_samples=10000, dimension=10, m=10, noise_level=0.1, batch_size=10000)
    model_train = BatchTrainer(model=DNN(input_size=dataset_loader.train_dataset.X.shape[1], output_size=dataset_loader.train_dataset.y.shape[1]), batch_size=32)
    model_train.load_data(dataset_loader)
    model_train.grid_search(
    param_grid={
        'lr': [0.001, 0.01, 0.1],
        'betas': [(0.9, 0.999), (0.9, 0.95)],
        'weight_decay': [0.0001, 0.001],
        'eps': [1e-08, 1e-05]
    },
    num_epochs=400,
    )
    # model_train.train(num_epochs=400, is_plot_graph=1)
    # model_train.evaluate_test()
    # model_train.save_model()