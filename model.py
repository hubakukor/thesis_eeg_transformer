import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
import torch.optim as optim
#import data_processing


# Define the model architecture
class EEGTransformerModel(nn.Module):
    def __init__(self, input_channels=63, seq_len=1501, d_model=128, nhead=4, num_encoder_layers=2, num_classes=2):
        """
        Args:
            input_channels (int): Number of input channels (EEG channels).
            seq_len (int): Length of the time series (sequence length).
            d_model (int): Dimensionality of the transformer input and output.
            nhead (int): Number of attention heads in the transformer.
            num_encoder_layers (int): Number of encoder layers in the transformer.
            num_classes (int): Number of output classes for classification.
        """
        super(EEGTransformerModel, self).__init__()

        # CNN for local feature extraction
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Linear projection to match Transformer input size
        self.proj = nn.Linear(128, d_model)

        # TransformerEncoder block for temporal dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.1, activation="relu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Final classifier
        self.fc = nn.Linear(d_model, num_classes) #lowered from seq_length * d_model to d_model

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_channels, seq_len].
        Returns:
            torch.Tensor: Output logits of shape [batch_size, num_classes].
        """
        # CNN for feature extraction
        x = self.conv(x)  # [batch_size, 128, seq_len]

        # Permute to match transformer input (seq_len, batch_size, d_model)
        x = x.permute(2, 0, 1)  # [seq_len, batch_size, 128]

        # Linear projection to d_model
        x = self.proj(x)  # [seq_len, batch_size, d_model]

        # TransformerEncoder
        x = self.transformer(x)  # [seq_len, batch_size, d_model]

        # Flatten the sequence and classify
        # x = x.permute(1, 0, 2).contiguous()  # [batch_size, seq_len, d_model]
        # x = x.mean(dim=1)  # Flatten to [batch_size, seq_len * d_model]
        # x = self.fc(x)  # [batch_size, num_classes]

        # Permute: [batch, seq_len, d_model]
        x = x.permute(1, 0, 2)
        # Mean pooling over time (seq_len)
        x = x.mean(dim=1)  # → [batch, d_model]
        # Final classification
        x = self.fc(x)  # → [batch, num_classes]

        return x
