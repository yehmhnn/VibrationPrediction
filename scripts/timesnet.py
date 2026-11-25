"""Compact univariate TimesNet forecasting baseline.

The block structure follows the official TimesNet forecasting implementation:
batch-level FFT period discovery, 1D-to-2D reshaping, Inception-style 2D
convolutions, and sample-dependent spectral aggregation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def fft_for_period(x, top_k=3):
    """Return shared dominant periods and per-sample spectral weights.

    Args:
        x: Tensor of shape ``[batch, time, channels]``.
    """
    spectrum = torch.fft.rfft(x, dim=1)
    amplitudes = spectrum.abs()
    global_amplitude = amplitudes.mean(dim=0).mean(dim=-1)
    global_amplitude = global_amplitude.clone()
    global_amplitude[0] = 0.0
    available = max(1, global_amplitude.numel() - 1)
    k = min(top_k, available)
    frequency_indices = torch.topk(global_amplitude, k=k).indices
    periods = torch.div(x.shape[1], frequency_indices, rounding_mode="floor")
    periods = periods.clamp_min(1)
    sample_weights = amplitudes.mean(dim=-1)[:, frequency_indices]
    return periods, sample_weights


class InceptionBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=4):
        super().__init__()
        self.kernels = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=2 * index + 1,
                    padding=index,
                )
                for index in range(num_kernels)
            ]
        )
        for layer in self.kernels:
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return torch.stack([kernel(x) for kernel in self.kernels], dim=-1).mean(dim=-1)


class TimesBlock(nn.Module):
    def __init__(
        self,
        sequence_length,
        hidden_channels,
        feedforward_channels,
        top_k=3,
        num_kernels=4,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.top_k = top_k
        self.conv = nn.Sequential(
            InceptionBlock2D(hidden_channels, feedforward_channels, num_kernels),
            nn.GELU(),
            InceptionBlock2D(feedforward_channels, hidden_channels, num_kernels),
        )

    def forward(self, x):
        batch_size, time_length, channels = x.shape
        if time_length != self.sequence_length:
            raise ValueError(
                f"Expected length {self.sequence_length}, got {time_length}"
            )
        periods, period_weights = fft_for_period(x, self.top_k)
        outputs = []
        for period_tensor in periods:
            period = int(period_tensor.item())
            padded_length = math.ceil(time_length / period) * period
            if padded_length > time_length:
                padding = x.new_zeros(batch_size, padded_length - time_length, channels)
                padded = torch.cat([x, padding], dim=1)
            else:
                padded = x
            two_dimensional = (
                padded.reshape(batch_size, padded_length // period, period, channels)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            transformed = self.conv(two_dimensional)
            transformed = transformed.permute(0, 2, 3, 1).reshape(
                batch_size, padded_length, channels
            )[:, :time_length]
            outputs.append(transformed)

        stacked = torch.stack(outputs, dim=-1)
        weights = torch.softmax(period_weights, dim=1)
        weights = weights[:, None, None, :].expand_as(stacked)
        return x + torch.sum(stacked * weights, dim=-1)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, channels, maximum_length):
        super().__init__()
        position = torch.arange(maximum_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, channels, 2, dtype=torch.float32)
            * (-math.log(10000.0) / channels)
        )
        encoding = torch.zeros(maximum_length, channels)
        encoding[:, 0::2] = torch.sin(position * divisor)
        if channels > 1:
            encoding[:, 1::2] = torch.cos(
                position * divisor[: encoding[:, 1::2].shape[1]]
            )
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, x):
        return x + self.encoding[: x.shape[1]].unsqueeze(0)


class TimesNet(nn.Module):
    def __init__(
        self,
        input_length=192,
        prediction_length=96,
        channels=1,
        hidden_channels=32,
        feedforward_channels=64,
        num_blocks=2,
        top_k=3,
        num_kernels=4,
        dropout=0.1,
        condition_channels=0,
    ):
        super().__init__()
        if channels != 1:
            raise ValueError(
                "This controlled baseline currently supports one target channel"
            )
        self.input_length = input_length
        self.prediction_length = prediction_length
        self.total_length = input_length + prediction_length
        self.condition_channels = int(condition_channels)
        self.value_embedding = nn.Conv1d(
            channels, hidden_channels, kernel_size=3, padding=1
        )
        self.position_encoding = SinusoidalPositionEncoding(
            hidden_channels, self.total_length
        )
        self.dropout = nn.Dropout(dropout)
        self.condition_embedding = (
            nn.Sequential(
                nn.Linear(self.condition_channels, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            if self.condition_channels > 0
            else None
        )
        self.temporal_projection = nn.Linear(input_length, self.total_length)
        self.blocks = nn.ModuleList(
            [
                TimesBlock(
                    self.total_length,
                    hidden_channels,
                    feedforward_channels,
                    top_k,
                    num_kernels,
                )
                for _ in range(num_blocks)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_channels) for _ in range(num_blocks)]
        )
        self.output_projection = nn.Linear(hidden_channels, channels)

    def forward(self, x, condition=None):
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        if x.ndim != 3 or x.shape[1] != self.input_length or x.shape[2] != 1:
            raise ValueError(
                f"Expected [batch, {self.input_length}, 1], got {tuple(x.shape)}"
            )

        mean = x.mean(dim=1, keepdim=True).detach()
        centered = x - mean
        standard_deviation = torch.sqrt(
            centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        normalized = centered / standard_deviation

        embedded = self.value_embedding(normalized.transpose(1, 2)).transpose(1, 2)
        embedded = self.temporal_projection(embedded.transpose(1, 2)).transpose(1, 2)
        embedded = self.dropout(self.position_encoding(embedded))
        if self.condition_embedding is not None:
            if (
                condition is None
                or condition.ndim != 2
                or condition.shape[1] != self.condition_channels
            ):
                raise ValueError(
                    f"Expected condition [batch, {self.condition_channels}], "
                    f"got {None if condition is None else tuple(condition.shape)}"
                )
            embedded = embedded + self.condition_embedding(condition)[:, None, :]
        elif condition is not None:
            raise ValueError("This TimesNet instance was created without conditioning")
        for block, norm in zip(self.blocks, self.norms):
            embedded = norm(block(embedded))
        output = self.output_projection(embedded)
        output = output * standard_deviation + mean
        return output[:, -self.prediction_length :, 0]
