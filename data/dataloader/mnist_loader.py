# import torch
# from torch.utils.data import DataLoader, random_split
# from torchvision import datasets, transforms

class MNISTDataLoader:
    pass

# class MNISTDataLoader:
#     def __init__(self, batch_size=32, val_size=0.2):
#         self.batch_size = batch_size
#         self.val_size = val_size
#         self.load_and_process_data()

#     def load_and_process_data(self):
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])

#         # Load the MNIST dataset
#         full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#         test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

#         # Split the training dataset into training and validation sets
#         val_size = int(self.val_size * len(full_train_dataset))
#         train_size = len(full_train_dataset) - val_size
#         train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

#         self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
#         self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
#         self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

#     def get_loaders(self):
#         return self.train_loader, self.val_loader, self.test_loader