import torch.nn as nn

class LSTMModel(nn.Module):
    """
    Long Short-Term Memory (LSTM) model for occupancy inference.
    Learns to predict occupancy based on a sequence of environmental sensor readings.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1):
        """
        Initialize the LSTM model
        :param input_dim: Number of features in the input data (e.g., number of sensors).
        :param hidden_dim: Number of features in the hidden state of the LSTM.
        :param num_layers: Number of stacked LSTM layers.
        :param dropout: Dropout rate for regularization (default: 0.1).
        """

        super(LSTMModel, self).__init__()

        # LSTM layer with optional dropout
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True, # Set batch_first=True to match input shape (batch_size, seq_len, features)
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected output layer to reduce the output to a single value (occupancy prediction)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the LSTM model.

        Parameters:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, 1).
        """
        output, (hn, cn) = self.lstm(x)  # hn shape: [num_layers, batch_size, hidden_dim]
        last_hidden = hn[-1]  # Take the last layer's hidden state
        x = self.fc(last_hidden) # Fully connected layer, convert to scalar output
        # x = self.sigmoid(x) # Sigmoid activation to output a probability between 0 and 1
        return x

    def extract_features(self, x):
        """
        Extract features from the LSTM model.
        :param x: Input tensor of shape (batch_size, seq_len, input_dim).
        :return: Output tensor of shape (batch_size, hidden_dim).
        """
        _, (hn, _) = self.lstm(x)
        return hn[-1]