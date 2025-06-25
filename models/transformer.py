import torch
import torch.nn as nn
import math

class _PositionalEncoding(nn.Module):
    """
    Add positional encoding to the input embeddings so that the model can learn the time-ordering of the sequence.
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

    # Add forward method to add positional encodings to the input
    def forward(self, x):
        """
        Add positional encodings to the input tensor.
        :param x: Input tensor of shape (batch_size, seq_len, d_model).
        :return: Tensor with positional encodings added.
        """

        # Add positional encodings to the input tensor
        x = x + self.pe[:, :x.size(1)]
        return x



class TransformerModel(nn.Module):
    """
    Transformer model for time-series classification.
    The model consists of an embedding layer, a transformer encoder, and a fully connected layer.
    Takes in a sequence of environmental sensor readings and predicts occupancy.
    """
    def __init__(
            self,
            input_dim: int, # Number of features in the input data (e.g., number of sensors)
            d_model: int, # Dimension of input and attention layers (embedding size)
            nhead: int, # Number of attention heads
            num_encoder_layers: int, # Number of transformer encoder layers
            dim_feedforward:int = 128, # Size of intermediate feedforward layer in encoder (default: 128)
            dropout: float = 0.1, # Dropout rate (default: 0.1)
    ):
        super().__init__()

        # Project raw input features to d_model dimensions
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding to add time information to the input and help the model learn the order of the sequence
        self.positional_encoding = _PositionalEncoding(d_model)

        # Transformer encoder layer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Set batch_first=True to match input shape (batch_size, seq_len, features)
        )

        # Stack multiple transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        # Fully connected output layer to reduce the output to a single value (occupancy prediction)
        self.fc = nn.Linear(d_model, 1)

        # Sigmoid activation function to output a probability between 0 and 1
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        """
        Forward pass through the transformer model.
        :param x: Input tensor of shape (batch_size, seq_len, input_dim).
        :return: Output tensor of shape (batch_size, 1).
        """

        # Project input features to d_model dimensions [batch_size, seq_len, d_model]
        x = self.input_proj(x)

        # Add time information to the input using positional encoding
        x = self.positional_encoding(x)

        # Pass the input through the transformer encoder [batch_size, seq_len, d_model]
        encoded = self.transformer_encoder(x)

        # Mean pooling over the sequence dimension to get a single representation for the entire sequence
        pooled = encoded.mean(dim=1) # Shape: (batch_size, d_model)

        # Pass the pooled output through the fully connected layer
        x = self.fc(pooled) # Shape: (batch_size, 1)

        # Apply sigmoid activation to get the final output
        # return self.sigmoid(x) # Output shape: (batch_size, 1)
        return x

    def extract_features(self, x):
        """
        Extract features from the transformer model.
        :param x: Input tensor of shape (batch_size, seq_len, input_dim).
        :return: Output tensor of shape (batch_size, d_model).
        """
        # Project input features to d_model dimensions
        x = self.input_proj(x)

        # Add time information to the input using positional encoding
        x = self.positional_encoding(x)

        # Pass the input through the transformer encoder
        encoded = self.transformer_encoder(x)

        # Mean pooling over the sequence dimension to get a single representation for the entire sequence
        return encoded.mean(dim=1)
