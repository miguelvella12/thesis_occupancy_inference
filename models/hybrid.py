import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Adds positional context to the LSTM outputs
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # Create a matrix of positional encodings
        pe = torch.zeros(max_len, d_model)

        # Create a tensor of positions from 0 to max_len
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).float()

        # Calculate the div_term for sine and cosine functions at even/odd dimensions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd dimensions
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension for broadcasting during training
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)

        # Register the positional encodings as a buffer so that it is not a model parameter
        self.register_buffer('pe', pe)


    def forward(self, x):
        """
        Forward pass through the positional encoding layer.
        :param x: Input tensor of shape (batch_size, seq_len, d_model).
        :return: Output tensor of the same shape with positional encodings added.
        """
        # Add positional encodings to the input embeddings
        x = x + self.pe[:, :x.size(1), :] # Add positional encoding to input, Shape: (batch_size, seq_len, d_model)
        return x

class TransformerEncoderBlock(nn.Module):
    """
    A single Transformer encoder block consisting of multi-head self-attention and feed-forward network.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model) # Normalization after attention
        self.norm2 = nn.LayerNorm(d_model) # Normalization after feed-forward network
        self.dropout = nn.Dropout(dropout) # Dropout layer for regularization

    def forward(self, x):
        # Multi-head self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        # Feed-forward network
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class HybridLSTMTransformerModel(nn.Module):
    """
    Hybrid model combining pre-trained LSTM and Transformer for occupancy prediction using sensor data.
    """
    def __init__(self, lstm_model, transformer_model, dropout=0.2):
        super().__init__()

        self.lstm_model = lstm_model
        self.transformer_model = transformer_model

        # Freeze parameters of the pre-trained models
        for param in self.lstm_model.parameters():
            param.requires_grad = False
        for param in self.transformer_model.parameters():
            param.requires_grad = False

        # Get feature sizes
        lstm_feat_dim = 64
        transformer_feat_dim = 64

        # Fusion dimension
        fused_dim = lstm_feat_dim + transformer_feat_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, 1)
        )

    def forward(self, x):
        """
        Forward pass through the hybrid model.
        :param x: Input tensor of shape (batch_size, seq_len, input_dim).
        :return: Output tensor of shape (batch_size,), representing the predicted occupancy probability.
        """

        feat_lstm = self.lstm_model.extract_features(x) # LSTM output shape: (batch_size, lstm_hidden_dim)
        feat_transformer = self.transformer_model.extract_features(x) # Transformer output shape: (batch_size, transformer_hidden_dim)

        # Fusion of features
        fused = torch.cat([feat_lstm, feat_transformer], dim=1) # Shape: (batch_size, lstm_hidden_dim + transformer_hidden_dim)
        return self.classifier(fused).squeeze(-1) # Shape: (batch_size, 1)