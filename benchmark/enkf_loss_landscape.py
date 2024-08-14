import torch
import numpy as np
import matplotlib.pyplot as plt
from model.dnn import DNN
from data.dataloader.regression_loader import OscillatoryDataLoader
from optimiser.enkf import EnKF

class LossLandscapeVisualizer:
    def __init__(self, model, optimiser, loss_function, X_train, y_train):
        self.model = model
        self.optimiser = optimiser
        self.loss_function = loss_function
        self.X_train = X_train
        self.y_train = y_train

    def plot_loss_landscape(self, n_points=100, epsilon=0.5):
        theta = self.optimiser.theta
        direction1 = torch.randn_like(theta)
        direction2 = torch.randn_like(theta)

        direction1 = direction1 / torch.norm(direction1)
        direction2 = direction2 / torch.norm(direction2)

        loss_values = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(n_points):
                perturbed_theta = (theta + (i - n_points // 2) * epsilon * direction1 
                                   + (j - n_points // 2) * epsilon * direction2)

                with torch.no_grad():
                    perturbed_params = self.optimiser._EnKF__unflatten_parameters(perturbed_theta)
                    for original_param, new_param in zip(self.model.parameters(), perturbed_params):
                        original_param.data.copy_(new_param.data)
                    output = self.model(self.X_train)
                    loss = self.loss_function(output, self.y_train)
                    loss_values[i, j] = loss.item()

        X = np.linspace(-n_points // 2, n_points // 2, n_points) * epsilon
        Y = np.linspace(-n_points // 2, n_points // 2, n_points) * epsilon

        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, loss_values, levels=50, cmap='viridis')
        plt.colorbar()
        plt.xlabel('Direction 1')
        plt.ylabel('Direction 2')
        plt.title('2D Loss Landscape')
        plt.show()

if __name__ == "__main__":
    dataset_loader = OscillatoryDataLoader(batch_size=100, val_size=0.2)
    input_size = dataset_loader.train_dataset.X.shape[1]
    output_size = dataset_loader.train_dataset.y.shape[1]

    model = DNN(input_size=input_size, output_size=output_size)
    optimiser = EnKF(model, lr=0.5, sigma=0.001, k=100, gamma=1e-1)
    loss_function = torch.nn.MSELoss()

    X_train, y_train = next(iter(dataset_loader.train_loader))

    visualizer = LossLandscapeVisualizer(model, optimiser, loss_function, X_train, y_train)
    visualizer.plot_loss_landscape(n_points=100, epsilon=0.5)
