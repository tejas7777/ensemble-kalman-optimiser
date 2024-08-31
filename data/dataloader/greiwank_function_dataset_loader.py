import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class GriewankFunctionDataset(Dataset):
    def __init__(self, num_samples=1000, dimension=2, noise_level=0.0):
        self.num_samples = num_samples
        self.dimension = dimension
        self.noise_level = noise_level
        self.X, self.y = self.generate_data()

    def griewank_function(self, X):
        part1 = torch.sum(X ** 2, dim=1) / 4000
        part2 = torch.prod(torch.cos(X / torch.sqrt(torch.arange(1, X.shape[1] + 1, dtype=torch.float))), dim=1)
        result = part1 - part2 + 1
        return result

    def generate_data(self):
        X = torch.rand(self.num_samples, self.dimension) * 1200 - 600  # Typically bounded between [-600, 600]
        y = self.griewank_function(X)

        if self.noise_level > 0.0:
            noise = self.noise_level * torch.randn_like(y)
            y += noise

        return X, y.view(-1, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GriewankFunctionDataLoader:
    def __init__(self, num_samples=1000, dimension=2, noise_level=0.0, batch_size=32, val_size=0.2, test_size=0.2):
        self.num_samples = num_samples
        self.dimension = dimension
        self.noise_level = noise_level
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.load_and_process_data()

    def load_and_process_data(self):
        self.dataset = GriewankFunctionDataset(num_samples=self.num_samples, dimension=self.dimension, noise_level=self.noise_level)
        num_val = int(self.val_size * len(self.dataset))
        num_test = int(self.test_size * len(self.dataset))
        num_train = len(self.dataset) - num_val - num_test

        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [num_train, num_val, num_test])

        self.train_dataset.X = self.dataset.X[self.train_dataset.indices]
        self.train_dataset.y = self.dataset.y[self.train_dataset.indices]

        self.val_dataset.X = self.dataset.X[self.val_dataset.indices]
        self.val_dataset.y = self.dataset.y[self.val_dataset.indices]

        self.test_dataset.X = self.dataset.X[self.test_dataset.indices]
        self.test_dataset.y = self.dataset.y[self.test_dataset.indices]

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader


if __name__ == '__main__':
    dataset_loader = GriewankFunctionDataLoader(num_samples=10000, dimension=2, noise_level=0.1, batch_size=32)
    train_loader, val_loader, test_loader = dataset_loader.get_loaders()
    
    input_size = dataset_loader.train_dataset.X.shape[1]
    print(f"Input size: {input_size}")
