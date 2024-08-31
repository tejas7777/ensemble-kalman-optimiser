import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class SaddlePointDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SaddlePointDataLoader:
    def __init__(self, num_samples=10000, noise_level=0.1, val_size=0.2, test_size=0.2, batch_size=32):
        self.num_samples = num_samples
        self.noise_level = noise_level
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.generate_data()
        self.split_data()
        
    def rosenbrock_function(self, x, y):
        """Modified Rosenbrock function to introduce a high saddle point with one path leading to global minimum."""
        return (1 - x)**2 + 100 * (y - x**2)**2

    def generate_data(self):
        """Generate data with a saddle point characteristic."""
        x = np.random.uniform(-2, 2, self.num_samples)
        y = np.random.uniform(-1, 3, self.num_samples)
        z = self.rosenbrock_function(x, y) + self.noise_level * np.random.randn(self.num_samples)

        # Reshape and combine into input features
        self.X = np.vstack((x, y)).T
        self.y = z.reshape(-1, 1)

    def split_data(self):
        """Split data into training, validation, and test sets."""
        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, test_size=self.val_size + self.test_size, random_state=42)
        val_size_adjusted = self.val_size / (self.val_size + self.test_size)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42)

        self.train_dataset = SaddlePointDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        self.val_dataset = SaddlePointDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        self.test_dataset = SaddlePointDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

if __name__ == '__main__':
    data_loader = SaddlePointDataLoader(num_samples=10000, noise_level=0.1, val_size=0.2, test_size=0.2, batch_size=32)
    train_loader, val_loader, test_loader = data_loader.get_loaders()

    # Example usage: iterate through train_loader
    for batch in train_loader:
        X_batch, y_batch = batch
        print(X_batch, y_batch)
        break
