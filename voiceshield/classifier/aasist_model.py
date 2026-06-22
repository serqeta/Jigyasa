"""
AASIST-L architecture.
Adapted from clovaai/aasist (MIT License) for local weight loading.
Config: first_conv=128, filts=[24,[1,32],[32,32],[32,32],[32,32]],
        gat_dims=[64,32], pool_ratios=[0.5,0.5,0.5,0.5]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Sinc filter bank (SincNet-style)
# ---------------------------------------------------------------------------


class SincConv(nn.Module):
    """Learnable sinc filter bank for raw waveform."""

    window_: torch.Tensor
    n_: torch.Tensor
    low_hz_: nn.Parameter
    band_hz_: nn.Parameter

    @staticmethod
    def to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    @staticmethod
    def to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        sample_rate: int = 16000,
        min_low_hz: float = 50.0,
        min_band_hz: float = 50.0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = 30.0
        high_hz = sample_rate / 2.0 - (min_low_hz + min_band_hz)

        mel = torch.linspace(self.to_mel(low_hz), self.to_mel(high_hz), out_channels + 1)
        hz = 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        self.low_hz_ = nn.Parameter(hz[:-1].unsqueeze(1))
        self.band_hz_ = nn.Parameter((hz[1:] - hz[:-1]).unsqueeze(1))

        n = (self.kernel_size - 1) / 2.0
        self.n_ = torch.arange(-n, 0).float()

        window = torch.hamming_window(self.kernel_size)
        self.register_buffer("window_", window)
        self.register_buffer("n_", self.n_)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (batch, 1, samples)
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2.0,
        )
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_.unsqueeze(0))
        f_times_t_high = torch.matmul(high, self.n_.unsqueeze(0))

        band_pass_left = (
            (torch.sin(f_times_t_high * 2 * math.pi) - torch.sin(f_times_t_low * 2 * math.pi))
            / (self.n_ / 2.0)
            * self.window_[: self.kernel_size // 2]
        )
        band_pass_center = 2 * band.unsqueeze(1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        filters = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        filters = filters / (2.0 * band.unsqueeze(1))
        filters = filters.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveform, filters, stride=1, padding=self.kernel_size // 2, bias=None)


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------


class Residual_block(nn.Module):
    def __init__(self, nb_filts: list, first: bool = False):
        super().__init__()
        self.first = first

        if not first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])

        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        self.conv1 = nn.Conv2d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(1, 1),
            stride=1,
        )
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(
            in_channels=nb_filts[1],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(0, 1),
            stride=1,
        )

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(
                in_channels=nb_filts[0],
                out_channels=nb_filts[1],
                kernel_size=(1, 1),
                padding=0,
                stride=1,
            )
        else:
            self.downsample = False

        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out = out + identity[:, :, : out.shape[2], : out.shape[3]]
        out = self.mp(out)
        return out


# ---------------------------------------------------------------------------
# Graph Attention Layer
# ---------------------------------------------------------------------------


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, temperature: float):
        super().__init__()
        self.temperature = temperature
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * out_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, N, in_dim)
        h = self.fc(x)  # (B, N, out_dim)
        B, N, D = h.shape

        a_input = torch.cat(
            [h.unsqueeze(2).expand(-1, -1, N, -1), h.unsqueeze(1).expand(-1, N, -1, -1)], dim=-1
        )
        e = self.attn(a_input).squeeze(-1)  # (B, N, N)
        attn = F.softmax(e / self.temperature, dim=-1)
        return torch.bmm(attn, h)  # (B, N, out_dim)


# ---------------------------------------------------------------------------
# Heterogeneous Temporal–Spectral Graph Attention Layer
# ---------------------------------------------------------------------------


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, temperature: float):
        super().__init__()
        self.temperature = temperature
        self.fc_s = nn.Linear(in_dim, out_dim, bias=False)  # spectral
        self.fc_t = nn.Linear(in_dim, out_dim, bias=False)  # temporal
        self.attn_st = nn.Linear(2 * out_dim, 1, bias=False)
        self.attn_ts = nn.Linear(2 * out_dim, 1, bias=False)

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x_s / x_t: (B, Ns/Nt, in_dim)
        h_s = self.fc_s(x_s)  # (B, Ns, out_dim)
        h_t = self.fc_t(x_t)  # (B, Nt, out_dim)
        Ns = h_s.size(1)
        Nt = h_t.size(1)

        # spectral → temporal cross attention
        a_st = torch.cat(
            [h_s.unsqueeze(2).expand(-1, -1, Nt, -1), h_t.unsqueeze(1).expand(-1, Ns, -1, -1)],
            dim=-1,
        )
        e_st = self.attn_st(a_st).squeeze(-1)  # (B, Ns, Nt)
        alpha_st = F.softmax(e_st / self.temperature, dim=-1)
        out_s = torch.bmm(alpha_st, h_t)  # (B, Ns, out_dim)

        # temporal → spectral cross attention
        a_ts = torch.cat(
            [h_t.unsqueeze(2).expand(-1, -1, Ns, -1), h_s.unsqueeze(1).expand(-1, Nt, -1, -1)],
            dim=-1,
        )
        e_ts = self.attn_ts(a_ts).squeeze(-1)  # (B, Nt, Ns)
        alpha_ts = F.softmax(e_ts / self.temperature, dim=-1)
        out_t = torch.bmm(alpha_ts, h_s)  # (B, Nt, out_dim)

        return out_s, out_t


# ---------------------------------------------------------------------------
# Graph Pool
# ---------------------------------------------------------------------------


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        scores = self.proj(x).squeeze(-1)  # (B, N)
        N = x.size(1)
        k = max(1, int(self.k * N))
        _, idx = scores.topk(k, dim=-1)
        idx = idx.sort(dim=-1)[0]
        return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


# ---------------------------------------------------------------------------
# AASIST-L
# ---------------------------------------------------------------------------


class AASIST(nn.Module):
    """
    AASIST-L: Anti-Spoofing using Integrated Spectro-Temporal Graph Attention.
    Lite configuration matching clovaai/aasist AASIST-L pretrained weights.
    """

    def __init__(
        self,
        nb_samp: int = 64600,
        first_conv: int = 128,
        filts: list | None = None,
        gat_dims: list | None = None,
        pool_ratios: list | None = None,
        temperatures: list | None = None,
    ):
        super().__init__()

        if filts is None:
            filts = [24, [1, 32], [32, 32], [32, 32], [32, 32]]
        if gat_dims is None:
            gat_dims = [64, 32]
        if pool_ratios is None:
            pool_ratios = [0.5, 0.5, 0.5, 0.5]
        if temperatures is None:
            temperatures = [2.0, 2.0, 100.0, 100.0]

        self.sinc_conv = SincConv(
            out_channels=first_conv,
            kernel_size=1024,
            sample_rate=16000,
        )

        in_ch = first_conv
        self.res_blocks = nn.ModuleList()
        for i, filt in enumerate(filts[1:]):
            self.res_blocks.append(
                Residual_block(
                    nb_filts=[in_ch if i == 0 else filts[i][1], filt[0] if i == 0 else filts[i][1]],
                    first=(i == 0),
                )
            )

        # Rebuild properly using the filts spec
        self.res_blocks = nn.ModuleList()
        # filts[0] is the number of sinc filter channels (first_conv already = 128)
        # filts[1..] are [in, out] pairs for residual blocks
        self.res_blocks.append(Residual_block(nb_filts=[first_conv, filts[1][1]], first=True))
        for i in range(2, len(filts)):
            self.res_blocks.append(Residual_block(nb_filts=filts[i], first=False))

        self.bn_after_sinc = nn.BatchNorm2d(first_conv)
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)

        last_ch = filts[-1][1]

        # Spectral and temporal graph attention
        self.gat_s = GraphAttentionLayer(last_ch, gat_dims[0], temperatures[0])
        self.gat_t = GraphAttentionLayer(last_ch, gat_dims[0], temperatures[1])
        self.pool_s = GraphPool(pool_ratios[0], gat_dims[0])
        self.pool_t = GraphPool(pool_ratios[1], gat_dims[0])

        # Heterogeneous cross-graph attention
        self.hgat = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperatures[2])
        self.pool_hs = GraphPool(pool_ratios[2], gat_dims[1])
        self.pool_ht = GraphPool(pool_ratios[3], gat_dims[1])

        # Classifier: max + avg readout → fc
        self.fc = nn.Linear(gat_dims[1] * 4, 2)
        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, nb_samp)
        x = x.unsqueeze(1)  # (B, 1, T)
        x = torch.abs(self.sinc_conv(x))  # (B, first_conv, T)
        x = x.unsqueeze(2)  # (B, first_conv, 1, T)
        x = self.bn_after_sinc(x)
        x = self.lrelu(x)

        for block in self.res_blocks:
            x = block(x)

        # Reshape to (B, T', F', C') for graph processing
        # x: (B, C, H, W) after res blocks
        B, C, H, W = x.shape

        # Spectral nodes: average over time dim
        x_s = x.mean(dim=3)  # (B, C, H)
        x_s = x_s.permute(0, 2, 1)  # (B, H, C) = (B, N_freq, C)

        # Temporal nodes: average over freq dim
        x_t = x.mean(dim=2)  # (B, C, W)
        x_t = x_t.permute(0, 2, 1)  # (B, W, C) = (B, N_time, C)

        # Spectral GAT
        x_s = self.gat_s(x_s)
        x_s = self.pool_s(x_s)

        # Temporal GAT
        x_t = self.gat_t(x_t)
        x_t = self.pool_t(x_t)

        # Heterogeneous cross-graph attention
        x_s, x_t = self.hgat(x_s, x_t)
        x_s = self.pool_hs(x_s)
        x_t = self.pool_ht(x_t)

        # Readout: max + avg for each branch
        x_s_max = x_s.max(dim=1)[0]
        x_s_avg = x_s.mean(dim=1)
        x_t_max = x_t.max(dim=1)[0]
        x_t_avg = x_t.mean(dim=1)

        out = torch.cat([x_s_max, x_s_avg, x_t_max, x_t_avg], dim=-1)
        out = self.drop(out)
        return self.fc(out)  # (B, 2)
