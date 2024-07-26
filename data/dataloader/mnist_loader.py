import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class MNISTDataLoader:
    def __init__(self, batch_size=64, val_split=0.1, test_split=0.1):
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.load_and_process_data()

    def load_and_process_data(self):
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])

        # Download and load the training data
        full_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        
        # Split into train, validation, and test sets
        total_size = len(full_dataset)
        test_size = int(self.test_split * total_size)
        val_size = int(self.val_split * total_size)
        train_size = total_size - test_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

    def get_data_dims(self):
        # MNIST images are 28x28 and grayscale (1 channel)
        return 1, 28, 28

    def get_num_classes(self):
        # MNIST has 10 classes (digits 0-9)
        return 10