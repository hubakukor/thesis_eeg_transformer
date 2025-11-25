import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
import torch.optim as optim
#import data_processing
from positional_embedding import SinusoidalPositionalEmbedding, LearnablePositionalEmbedding


# Define the model architecture
class EEGTransformerModel(nn.Module):
    def __init__(self, input_channels=63, seq_len=1501, d_model=128, nhead=4, num_encoder_layers=2, num_classes=2, embedding_type='sinusoidal', conv_block_type='multi'):
        """
        Class for the EEG Transformer model.

        Args:
            input_channels (int): Number of input channels (EEG channels).
            seq_len (int): Length of the time series (sequence length).
            d_model (int): Dimensionality of the transformer input and output.
            nhead (int): Number of attention heads in the transformer.
            num_encoder_layers (int): Number of encoder layers in the transformer.
            num_classes (int): Number of output classes for classification.
            embedding_type (str): Type of embedding to use. Options are 'sinusoidal', 'learnable', and 'none'.
            conv_block_type (str): Changes the temporal convolution part. (multi: combines multiple kernel sizes, single: all kernels are the same size [25])

        returns:
            model: EEGTransformerModel instance.
        """
        super(EEGTransformerModel, self).__init__()

        # CNN for local feature extraction
        if conv_block_type == 'multi':
            self.conv = MultiKernelTemporalSpatial(
                input_channels=input_channels,
                # kernel_sizes=(17, 21, 25, 43, 51, 63),  # 6-kernel set for cybathlon data
                #kernel_sizes=(17, 25, 51, 101),  # 4-kernel set for cybathlon data
                # kernel_sizes = (11, 15, 21, 25, 31, 39), #6-kernel set for bci comp data
                kernel_sizes = (5, 7, 9, 13, 17, 21), #6-kernel set for physionet data
                total_time_channels=96,  # 16 ch per branch
                out_channels_after_spatial=128
            )
        elif conv_block_type == 'single':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 25), padding=(0, 12)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=(input_channels, 1), padding=(0, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )

        # Linear projection to match Transformer input size
        self.proj = nn.Linear(128, d_model)

        # Add positional embedding
        # Choose embedding type
        if embedding_type == 'none':
            self.positional_embedding = lambda x: x  # no embedding
        elif embedding_type == 'sinusoidal':
            self.positional_embedding = SinusoidalPositionalEmbedding(seq_len, d_model)
        elif embedding_type == 'learnable':
            self.positional_embedding = LearnablePositionalEmbedding(seq_len, d_model)

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
        x = x.unsqueeze(1)  # → [batch, 1, 63, 1501]
        x = self.conv(x)  # → [batch, 128, 1, 1501]

        x = x.squeeze(2)  # remove channel dim → [batch, 128, 1501]

        # Permute for transformer: [seq_len, batch, feature_dim]
        x = x.permute(2, 0, 1)  # → [1501, batch, 128]

        # Linear projection to d_model
        x = self.proj(x)  # [seq_len, batch_size, d_model]

        # Add positional embedding
        x = self.positional_embedding(x)

        # TransformerEncoder
        x = self.transformer(x)  # [seq_len, batch_size, d_model]


        # Permute: [batch, seq_len, d_model]
        x = x.permute(1, 0, 2)
        # Mean pooling over time (seq_len)
        x = x.mean(dim=1)  # → [batch, d_model]
        # Final classification
        x = self.fc(x)  # → [batch, num_classes]

        return x



class ShallowConvNet(nn.Module):
    def __init__(self, input_channels=63, input_time_length=1501, num_classes=2):
        """
        ShallowConvNet based on Dose et al. (2018) and Schirrmeister et al. (2017).

        Args:
            input_channels (int): Number of input channels (EEG channels).
            input_time_length (int): Length of the time series (sequence length).
            num_classes (int): Number of output classes for classification.
        """
        super(ShallowConvNet, self).__init__()

        # 1D convolution along the time axis
        self.conv_time = nn.Conv2d(
            in_channels=1,
            out_channels=40,
            kernel_size=(1, 25),
            stride=1,
            padding=(0, 12),
            bias=False
        )

        # 1D convolution along the spatial axis
        self.conv_spat = nn.Conv2d(
            in_channels=40,
            out_channels=40,
            kernel_size=(input_channels, 1),
            stride=1,
            bias=False
        )

        self.batch_norm = nn.BatchNorm2d(40) # Batch normalization
        self.pooling = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15)) # Average Pooling
        self.classifier = nn.Linear(self._calculate_flatten_size(input_time_length), num_classes) # Final classifier

    # Calculate the size of the flattened tensor fot the fc layer
    def _calculate_flatten_size(self, input_time_length):
        with torch.no_grad():
            x = torch.zeros(1, 1, 63, input_time_length)
            x = self.conv_time(x)
            x = self.conv_spat(x)
            x = self.batch_norm(x)
            x = x ** 2
            x = self.pooling(x)
            x = torch.log(torch.clamp(x, min=1e-6))
            x = x.view(1, -1)
            return x.shape[1]

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, channels, time]
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.batch_norm(x)
        x = x ** 2
        x = self.pooling(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class MultiscaleConvolution(nn.Module):
    def __init__(self, input_channels=22, input_time_length=751, num_classes=4,
                 kernel_sizes=(11, 15, 21, 25, 31, 39), total_time_channels=48):
        """
        Temporal convolution with different kernel sizes.


        (An improved multi‑scale convolution and transformer network
        for EEG‑based motor imagery decoding, Zhu et al.)
        """

        super().__init__()

        assert total_time_channels % len(kernel_sizes) == 0, "total_time_channels must be divisible by number of kernel sizes"
        branch_out = total_time_channels // len(kernel_sizes)

        # Parallel temporal convs (1 x k) with 'same' padding on time axis
        self.temporal_branches = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=branch_out,
                kernel_size=(1, k),
                stride=1,
                padding=(0, k // 2),
                bias=False
            ) for k in kernel_sizes
        ])


        self.conv_spat = nn.Conv2d(
            in_channels=total_time_channels,
            out_channels=total_time_channels,
            kernel_size=(input_channels, 1),
            stride=1,
            bias=False
        )

        self.batch_norm = nn.BatchNorm2d(total_time_channels)
        self.pooling = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.classifier = nn.Linear(self._calculate_flatten_size(input_channels, input_time_length),
                                    num_classes)

    def _calculate_flatten_size(self, input_channels, input_time_length):
        with torch.no_grad():
            x = torch.zeros(1, input_channels, input_time_length)
            x = x.unsqueeze(1)  # [1, 1, C, T]
            # parallel temporal convs + concat
            feats = [branch(x) for branch in self.temporal_branches]
            x = torch.cat(feats, dim=1)  # [1, total_time_channels, C, T]
            # spatial + rest identical to ShallowConvNet
            x = self.conv_spat(x)
            x = self.batch_norm(x)
            x = x ** 2
            x = self.pooling(x)
            x = torch.log(torch.clamp(x, min=1e-6))
            x = x.view(1, -1)
            return x.shape[1]

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, channels, time]
        feats = [branch(x) for branch in self.temporal_branches]
        x = torch.cat(feats, dim=1)  # [B, total_time_channels, C, T]
        x = self.conv_spat(x)
        x = self.batch_norm(x)
        x = x ** 2
        x = self.pooling(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class MultiKernelTemporalSpatial(nn.Module):
    """
    Convolutional block for the main model.
    Temporal covolution has multiple kernel with different sizes. Spatial convolution is unchanged.
    """
    def __init__(self, input_channels=63, kernel_sizes=(17, 21, 25, 43, 51, 63),
                 total_time_channels=96, out_channels_after_spatial=128):
        super().__init__()
        assert total_time_channels % len(kernel_sizes) == 0, \
            "total_time_channels must be divisible by number of kernel sizes"
        branch_out = total_time_channels // len(kernel_sizes)
        # parallel temporal convs with same-length time via padding
        self.temporal_branches = nn.ModuleList([
            nn.Conv2d(1, branch_out, kernel_size=(1, k), stride=1, padding=(0, k//2), bias=False)
            for k in kernel_sizes
        ])
        self.bn_time = nn.BatchNorm2d(total_time_channels)
        self.act_time = nn.ReLU(inplace=True)

        # spatial conv collapses electrodes C -> 1 and expands channels to 128 (to match your model)
        self.conv_spat = nn.Conv2d(
            in_channels=total_time_channels,
            out_channels=out_channels_after_spatial,
            kernel_size=(input_channels, 1),
            stride=1,
            bias=False
        )
        self.bn_spat = nn.BatchNorm2d(out_channels_after_spatial)
        self.act_spat = nn.ReLU(inplace=True)

    def forward(self, x):  # x: [B, 1, C, T]
        feats = [b(x) for b in self.temporal_branches]      # each: [B, branch_out, C, T]
        x = torch.cat(feats, dim=1)                         # [B, total_time_channels, C, T]
        x = self.act_time(self.bn_time(x))
        x = self.conv_spat(x)                               # [B, 128, 1, T]
        x = self.act_spat(self.bn_spat(x))
        return x


