import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

class OscillatoryDataLoader:
    def __init__(self, file_path = './dataset/oscillatory_data_small.csv', set_standardize=False, test_size=0.2, val_size=0.2):
        self.file_path = file_path
        self.set_standardize = set_standardize
        self.test_size = test_size
        self.val_size = val_size
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.load_and_process_data()

    def load_and_process_data(self):
        data = pd.read_csv(self.file_path)
        input_cols = [col for col in data.columns if 'Theta' in col]
        target_cols = [col for col in data.columns if 'F_Theta' in col]
        self.X = data[input_cols].values
        self.y = data[target_cols].values
        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, test_size=self.test_size + self.val_size, random_state=42)
        val_size_adjusted = self.val_size / (self.test_size + self.val_size)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42)

        if self.set_standardize:
            X_train, X_val, X_test = self.standardize_data(X_train, X_val, X_test)

        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.X_val = torch.tensor(X_val, dtype=torch.float32)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        self.y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        self.y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    def standardize_data(self, X_train, X_val, X_test):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        return X_train, X_val, X_test

    def get_data(self):
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
