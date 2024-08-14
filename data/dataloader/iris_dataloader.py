import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import numpy as np

class IrisDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class IrisDataLoader:
    def __init__(self, num_features=2, set_standardize=False, test_size=0.2, val_size=0.2, batch_size=32):
        self.num_features = num_features
        self.set_standardize = set_standardize
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.load_and_process_data()

    def load_and_process_data(self):
        iris = load_iris()
        data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        data['target'] = iris.target
        
        X = data[iris.feature_names].values
        y = data['target'].values

        # Feature selection
        selector = SelectKBest(f_classif, k=self.num_features)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        self.selected_features = np.array(iris.feature_names)[selector.get_support()]
        print(f"Selected features: {self.selected_features}")

        X_train, X_temp, y_train, y_temp = train_test_split(X_selected, y, test_size=self.test_size + self.val_size, random_state=42)
        val_size_adjusted = self.val_size / (self.test_size + self.val_size)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42)

        if self.set_standardize:
            X_train, X_val, X_test = self.standardize_data(X_train, X_val, X_test)

        self.train_dataset = IrisDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        self.val_dataset = IrisDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
        self.test_dataset = IrisDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

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
        return self.num_features

    def get_selected_features(self):
        return self.selected_features