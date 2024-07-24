import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class PDEData(Dataset):
    def __init__(self, input_tensor, output_tensor):
        self.inputs = input_tensor
        self.outputs = output_tensor

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

class PDEDataLoader:
    def __init__(self, file_path, subset_size=None, batch_size=32, shuffle=True):
        self.file_path = file_path
        self.subset_size = subset_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.load_and_process_data()

    def load_and_process_data(self):
        data = torch.load(self.file_path)
        inputs = data['input']
        outputs = data['output']

        print(f"Input shape: {data['input'].shape}")
        print(f"Output shape: {data['output'].shape}")
        
        if self.subset_size is not None:
            inputs = inputs[:self.subset_size]
            outputs = outputs[:self.subset_size]

        self.train_dataset = PDEData(inputs, outputs)
        self.val_dataset = PDEData(inputs, outputs)  # Adjust as necessary for actual validation data
        self.test_dataset = PDEData(inputs, outputs)  # Adjust as necessary for actual test data
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader
