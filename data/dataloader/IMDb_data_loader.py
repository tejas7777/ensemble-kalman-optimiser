import os
import tarfile
import urllib.request
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import word_tokenize
import nltk
from collections import Counter

# nltk.download('punkt')

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = word_tokenize(self.texts[idx])
        indices = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

class IMDBDataLoader:
    def __init__(self, batch_size=32, val_size=0.2, max_vocab_size=5000):
        self.batch_size = batch_size
        self.val_size = val_size
        self.vocab = {"<unk>": 0}
        self.max_vocab_size = max_vocab_size
        self.load_and_process_data()

    def download_and_extract(self, url, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)
        filepath = os.path.join(dest, "aclImdb_v1.tar.gz")
        if not os.path.exists(filepath):
            urllib.request.urlretrieve(url, filepath)
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(path=dest)

    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            tokens = word_tokenize(text)
            counter.update(tokens)
        most_common = counter.most_common(self.max_vocab_size)
        for token, _ in most_common:
            self.vocab[token] = len(self.vocab)

    def load_and_process_data(self):
        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        data_dir = "./data"
        self.download_and_extract(url, data_dir)
        
        texts, labels = [], []
        for label in ["pos", "neg"]:
            dir_path = os.path.join(data_dir, "aclImdb", "train", label)
            for fname in os.listdir(dir_path):
                if fname.endswith(".txt"):
                    with open(os.path.join(dir_path, fname), "r") as f:
                        texts.append(f.read())
                    labels.append(1 if label == "pos" else 0)

        self.build_vocab(texts)
        
        texts_train, texts_val, labels_train, labels_val = train_test_split(texts, labels, test_size=self.val_size, random_state=42)

        self.train_dataset = IMDBDataset(texts_train, labels_train, self.vocab)
        self.val_dataset = IMDBDataset(texts_val, labels_val, self.vocab)

        test_texts, test_labels = [], []
        for label in ["pos", "neg"]:
            dir_path = os.path.join(data_dir, "aclImdb", "test", label)
            for fname in os.listdir(dir_path):
                if fname.endswith(".txt"):
                    with open(os.path.join(dir_path, fname), "r") as f:
                        test_texts.append(f.read())
                    test_labels.append(1 if label == "pos" else 0)
        
        self.test_dataset = IMDBDataset(test_texts, test_labels, self.vocab)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_batch)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_batch)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_batch)

    def collate_batch(self, batch):
        texts, labels = zip(*batch)
        texts_padded = pad_sequence(texts, padding_value=self.vocab["<unk>"])
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return texts_padded.t(), labels_tensor

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

