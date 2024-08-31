import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

class ClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ClassificationDataLoader:
    def __init__(self, file_path='./dataset/multi_class_classification_data.csv', set_standardize=False, test_size=0.2, val_size=0.2, batch_size=32):
        self.file_path = file_path
        self.set_standardize = set_standardize
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.load_and_process_data()

    def load_and_process_data(self):
        data = pd.read_csv(self.file_path)
        input_cols = [col for col in data.columns if col.startswith('Theta')]
        target_col = 'Label'
        self.X = data[input_cols].values
        self.y = data[target_col].values
        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, test_size=self.test_size + self.val_size, random_state=42)
        val_size_adjusted = self.val_size / (self.test_size + self.val_size)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42)

        if self.set_standardize:
            X_train, X_val, X_test = self.standardize_data(X_train, X_val, X_test)

        self.train_dataset = ClassificationDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        self.val_dataset = ClassificationDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
        self.test_dataset = ClassificationDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def standardize_data(self, X_train, X_val, X_test):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        return X_train, X_val, X_test

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_feature_dim(self):
        input_dim = self.X.shape[1]  # Number of features
        output_dim = len(torch.unique(torch.tensor(self.y)))  # Number of unique classes
        return input_dim, output_dim


# Example usage:
if __name__ == '__main__':
    file_path = './dataset/multi_class_classification_data.csv'  # Update with your file path
    classification_data_loader = ClassificationDataLoader(file_path=file_path, set_standardize=True, test_size=0.2, val_size=0.1, batch_size=32)
    train_loader, val_loader, test_loader = classification_data_loader.get_loaders()

    for batch in train_loader:
        inputs, targets = batch
        print(inputs.shape, targets.shape)
        break
