import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np

class CaliforniaHousingDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CaliforniaHousingDataLoader:
    def __init__(self, test_size=0.2, val_size=0.2, batch_size=32):
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.load_and_process_data()

    def load_and_process_data(self):
        data = fetch_california_housing()
        X = data['data']
        y = data['target'].reshape(-1, 1)  # Reshape y to be a 2D array

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=self.test_size + self.val_size, random_state=42)
        val_size_adjusted = self.val_size / (self.test_size + self.val_size)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42)

        self.train_dataset = CaliforniaHousingDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        self.val_dataset = CaliforniaHousingDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        self.test_dataset = CaliforniaHousingDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

if __name__ == "__main__":
    data_loader = CaliforniaHousingDataLoader(batch_size=64, val_size=0.2)
    train_loader, val_loader, test_loader = data_loader.get_loaders()

    for X, y in train_loader:
        print(X.shape, y.shape)
        break