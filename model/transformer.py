import torch
import torch.nn as nn
import torch.nn.functional as functional


class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, num_classes, dropout=0.1):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, embed_dim]
        x = self.encoder(x)  # [seq_len, batch_size, embed_dim]
        x = x[0]  # Use the first token's embedding for classification (assuming [CLS] token) [batch_size, embed_dim]
        x = self.dropout(x)
        x = self.fc(x)  # [batch_size, num_classes]
        return x



