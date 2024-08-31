import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

class MNISTDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NoisyMNISTDataLoader:
    def __init__(self, subset_size=500, set_standardize=False, test_size=0.2, val_size=0.2, batch_size=32):
        self.subset_size = subset_size
        self.set_standardize = set_standardize
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.load_and_process_data()

    def load_and_process_data(self):
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        X_train, y_train = mnist_train.data[:self.subset_size].unsqueeze(1).float() / 255.0, mnist_train.targets[:self.subset_size]
        X_test, y_test = mnist_test.data[:self.subset_size].unsqueeze(1).float() / 255.0, mnist_test.targets[:self.subset_size]

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.val_size, random_state=42)

        noise_factor = 0.5
        X_train_noisy = self.add_noise(X_train, noise_factor)
        X_val_noisy = self.add_noise(X_val, noise_factor)
        X_test_noisy = self.add_noise(X_test, noise_factor)

        self.train_dataset = MNISTDataset(X_train_noisy, y_train)
        self.val_dataset = MNISTDataset(X_val_noisy, y_val)
        self.test_dataset = MNISTDataset(X_test_noisy, y_test)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def add_noise(self, data, noise_factor):
        noisy_data = data + noise_factor * torch.randn(*data.shape)
        return torch.clip(noisy_data, 0., 1.)

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader