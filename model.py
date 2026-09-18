import torch
import torch.nn as nn

class OrbitalLSTM(nn.Module):
    """
    LSTM-based regression model for predicting future satellite/debris coordinates.
    Takes historical sequences of orbital parameters and outputs coordinates for the next time step(s).
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int, 
        output_size: int, 
        dropout: float = 0.2
    ):
        """
        Initializes the OrbitalLSTM.

        Args:
            input_size (int): Number of features in the input sequence (e.g., 3 for just x,y,z).
            hidden_size (int): Number of features in the LSTM hidden state.
            num_layers (int): Number of stacked LSTM layers.
            output_size (int): Number of target features to predict (e.g., 3 for future x,y,z).
            dropout (float): Dropout probability between LSTM layers (only applies if num_layers > 1).
        """
        super(OrbitalLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Batch_first=True expects input tensors of shape (batch, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Deep fully connected layers mapping the LSTM's final hidden state to the target output
        self.fc_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: Predicted coordinates of shape (batch_size, output_size).
        """
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        # h_n, c_n represent the hidden and cell states
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # We only care about the output from the final time step of the sequence
        final_timestep_out = lstm_out[:, -1, :]
        
        # Pass through fully connected network
        predictions = self.fc_network(final_timestep_out)
        
        return predictions
