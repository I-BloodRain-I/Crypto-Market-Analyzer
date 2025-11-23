import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn


@dataclass
class MultiHorizonTransformerConfig:
    """
    Configuration for MultiHorizonTransformer model.
    
    This dataclass contains all hyperparameters for the multi-horizon forecasting model,
    organized into logical groups: architecture, sequence lengths, feature dimensions,
    output settings, positional encodings, temporal convolutions, and variable selection.
    
    Architecture Parameters:
        d_model: Model embedding dimension
        n_heads: Number of attention heads in transformer layers
        d_ff: Feedforward network hidden dimension
        n_enc_layers: Number of transformer encoder layers
        n_dec_layers: Number of transformer decoder layers
        dropout: Dropout probability applied throughout the model
        layer_norm_eps: Epsilon for layer normalization numerical stability
    
    Sequence Length Parameters:
        context_len: Length of historical context window (L)
        max_horizon: Maximum prediction horizon (H_max)
        target_horizons: Specific horizons to extract predictions for (e.g., 1, 3, 5, 10, 15 steps ahead)
    
    Feature Dimension Parameters:
        past_feat_dim: Number of dynamic historical features (F_p)
        future_feat_dim: Number of known future features/covariates (F_f)
        static_feat_dim: Number of static features (F_s); set to 0 if none
    
    Output Parameters:
        out_dim: Output dimension (1 for point forecast, n_quantiles for probabilistic)
        use_quantiles: Whether to use quantile regression for probabilistic forecasting
        quantiles: Quantile levels to predict (e.g., 0.1, 0.5, 0.9 for 10th, 50th, 90th percentiles)
        per_horizon_heads: Whether to use separate output heads for each target horizon
    
    Positional Encoding Parameters:
        use_sinusoidal_pe: Use sinusoidal positional encodings (True) or learned embeddings (False)
    
    Temporal Convolution Parameters:
        use_tcn_branch: Enable multi-resolution TCN branch for pattern extraction
        tcn_channels: Number of channels per TCN block (before GLU activation)
        tcn_kernel_sizes: Tuple of kernel sizes for multi-scale convolutions
        tcn_dilations: Tuple of dilation rates for expanding receptive fields
    
    Variable Selection Parameters:
        use_variable_selection: Enable VSN for dynamic feature importance learning
        vsn_hidden: Hidden dimension for VSN GRN modules
        grn_hidden: Hidden dimension for general GRN modules
    
    Initialization Parameters:
        init_gain: Gain factor for Xavier initialization
    """
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    n_enc_layers: int = 3
    n_dec_layers: int = 2
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    context_len: int = 120
    max_horizon: int = 15
    target_horizons: Tuple[int, ...] = (1, 3, 5, 10, 15)

    past_feat_dim: int = 16
    future_feat_dim: int = 8
    static_feat_dim: int = 0

    out_dim: int = 1
    use_quantiles: bool = False
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)
    per_horizon_heads: bool = False

    use_sinusoidal_pe: bool = True

    use_tcn_branch: bool = True
    tcn_channels: int = 128
    tcn_kernel_sizes: Tuple[int, ...] = (3, 5, 9)
    tcn_dilations: Tuple[int, ...] = (1, 2, 4)

    use_variable_selection: bool = True
    vsn_hidden: int = 128
    grn_hidden: int = 256

    init_gain: float = 1.0

    def save(self, path: str):
        """Saves the configuration to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def load(path: str) -> "MultiHorizonTransformerConfig":
        """Loads the configuration from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return MultiHorizonTransformerConfig(**data)


def causal_mask(sz: int, device: torch.device) -> torch.Tensor:
    """
    Creates a causal (upper triangular) attention mask for autoregressive decoding.
    
    Args:
        sz: Size of the square mask (sequence length)
        device: Device to create the tensor on
    
    Returns:
        Boolean mask of shape [sz, sz] where True indicates masked (blocked) positions
    """
    return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

class SinusoidalPE(nn.Module):
    """
    Sinusoidal Positional Encoding layer.
    
    Implements the classic sinusoidal positional encoding from "Attention is All You Need".
    Uses sine and cosine functions of different frequencies to encode absolute position information.
    
    The encoding is computed as:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        d_model: Dimensionality of the model embeddings
        max_len: Maximum sequence length supported
    
    Input Shape:
        x: [B, T, D] where B=batch, T=time steps, D=d_model
    
    Output Shape:
        [B, T, D] with positional encodings added to input
    """
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].to(dtype=x.dtype, device=x.device)


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) from Temporal Fusion Transformer.
    
    A flexible feature transformation module that applies non-linear processing with gating
    mechanism and residual connections. The gating allows the network to suppress or amplify
    certain features dynamically during training.
    
    Architecture:
        1. Two-layer MLP with ELU activation
        2. Gating mechanism (sigmoid) to control information flow
        3. Skip connection for residual learning
        4. Layer normalization for training stability
    
    Formula:
        h = Dropout(FC2(ELU(FC1(x))))
        g = Sigmoid(Gate(h))
        output = LayerNorm(g * h + (1 - g) * Skip(x))
    
    Args:
        in_dim: Input feature dimensionality
        hidden_dim: Hidden layer dimensionality
        out_dim: Output feature dimensionality
        dropout: Dropout probability
        eps: Epsilon for layer normalization
    
    Input Shape:
        x: [*, in_dim] (arbitrary leading dimensions)
    
    Output Shape:
        [*, out_dim]
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float, eps: float = 1e-5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(out_dim, out_dim)
        self.sigmoid = nn.Sigmoid()
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        g = self.sigmoid(self.gate(h))
        y = g * h + (1.0 - g) * self.skip(x)
        return self.norm(y)


class VariableSelectionNetwork(nn.Module):
    """
    Feature-wise Variable Selection Network from Temporal Fusion Transformer.
    
    Performs dynamic feature selection by learning importance weights for each input feature.
    The network processes each feature independently, computes importance scores, and produces
    a weighted mixture of feature embeddings.
    
    Architecture:
        1. Per-feature linear projections to d_model dimensions
        2. Per-feature importance scorers (using GRN) producing logits
        3. Softmax over features to normalize importance weights
        4. Weighted sum of feature embeddings
        5. Post-processing GRN for final transformation
    
    Supports both temporal (dynamic) and static features:
        - Dynamic features: x ∈ ℝ^{B×T×F} → output ∈ ℝ^{B×T×D}
        - Static features: x ∈ ℝ^{B×F} → output ∈ ℝ^{B×D}
    
    Args:
        num_features: Number of input features (F)
        d_model: Output embedding dimension (D)
        hidden: Hidden dimension for GRN modules
        dropout: Dropout probability
    
    Input Shape:
        x: [B, T, F] for dynamic features or [B, F] for static features
    
    Output Shape:
        [B, T, D] for dynamic or [B, D] for static
    """
    def __init__(self, num_features: int, d_model: int, hidden: int, dropout: float):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        self.feature_proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_features)])
        self.feature_scorers = nn.ModuleList([GatedResidualNetwork(d_model, hidden, 1, dropout) for _ in range(num_features)])
        self.post_grn = GatedResidualNetwork(d_model, hidden, d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            B, T, F = x.shape
            assert F == self.num_features, f"Expected {self.num_features} features, got {F}"
            embs = []
            scores = []
            for f in range(F):
                xf = x[:, :, f:f+1]                      # [B, T, 1]
                ef = self.feature_proj[f](xf)            # [B, T, D]
                sf = self.feature_scorers[f](ef)         # [B, T, 1] — importance score logits
                embs.append(ef)
                scores.append(sf)
            E = torch.stack(embs, dim=2)
            S = torch.stack(scores, dim=2)

            W = torch.softmax(S, dim=2)

            mix = (E * W).sum(dim=2)

            out = self.post_grn(mix)
            return out

        elif x.dim() == 2:
            B, F = x.shape
            assert F == self.num_features
            embs = []
            scores = []
            for f in range(F):
                xf = x[:, f:f+1]                         # [B, 1]
                ef = self.feature_proj[f](xf)            # [B, D]
                sf = self.feature_scorers[f](ef)         # [B, 1]
                embs.append(ef)
                scores.append(sf)
            E = torch.stack(embs, dim=1)                 # [B, F, D]
            S = torch.stack(scores, dim=1)               # [B, F, 1]
            W = torch.softmax(S, dim=1)                  # [B, F, 1]
            mix = (E * W).sum(dim=1)                     # [B, D]
            out = self.post_grn(mix)                     # [B, D]
            return out

        else:
            raise ValueError("VariableSelectionNetworkFW expects [B,T,F] or [B,F].")


class TemporalConvBlock(nn.Module):
    """
    Single temporal convolution block with causal padding.
    
    Implements a dilated causal convolution followed by normalization, dropout, and GLU activation.
    The causal padding ensures that the convolution only sees past and present information,
    maintaining temporal ordering for time series prediction.
    
    Architecture:
        1. Dilated 1D convolution with causal padding
        2. Right-side padding removal to maintain causality
        3. Batch normalization
        4. Dropout for regularization
        5. Gated Linear Unit (GLU) activation
    
    Args:
        in_ch: Number of input channels
        out_ch: Number of output channels (must be even for GLU)
        kernel_size: Convolution kernel size
        dilation: Dilation rate for dilated convolution
        dropout: Dropout probability
    
    Input Shape:
        x: [B, D, T] where B=batch, D=channels, T=time steps
    
    Output Shape:
        [B, out_ch//2, T] (GLU halves the channel dimension)
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(out_ch)
        self.glu = nn.GLU(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = y[:, :, : x.size(-1)]
        y = self.dropout(self.norm(y))
        return self.glu(y)


class MultiResolutionTCN(nn.Module):
    """
    Multi-resolution Temporal Convolutional Network.
    
    Captures temporal patterns at multiple scales by combining multiple dilated causal convolutions
    with different kernel sizes and dilation rates. Each combination creates a different receptive
    field, allowing the network to learn both short-term and long-term dependencies.
    
    Architecture:
        1. Multiple parallel TemporalConvBlocks with varying kernels and dilations
        2. Concatenation of all outputs along channel dimension
        3. Linear fusion layer to project back to d_model dimensions
    
    The receptive field for each block is: RF = (kernel_size - 1) * dilation + 1
    
    Args:
        d_model: Model dimensionality for input and output
        tcn_channels: Output channels per TCN block (before GLU)
        ks: Tuple of kernel sizes for convolutions
        ds: Tuple of dilation rates
        dropout: Dropout probability
    
    Input Shape:
        x: [B, T, D] where B=batch, T=time steps, D=d_model
    
    Output Shape:
        [B, T, D]
    """
    def __init__(self, d_model: int, tcn_channels: int, ks: Tuple[int, ...], ds: Tuple[int, ...], dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList([
            TemporalConvBlock(d_model, tcn_channels * 2, k, d, dropout)
            for k in ks for d in ds
        ])
        self.fuse = nn.Linear(len(ks) * len(ds) * tcn_channels, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ch = x.transpose(1, 2)
        feats = [blk(x_ch) for blk in self.blocks]
        cat = torch.cat(feats, dim=1)
        cat = cat.transpose(1, 2)
        return self.fuse(cat)


# ---------------------------
# Core Model
# ---------------------------

class MultiHorizonTransformer(nn.Module):
    """
    Production-grade Multi-Horizon Transformer for time series forecasting.
    
    A sophisticated encoder-decoder architecture combining Transformer attention mechanisms with
    Variable Selection Networks (VSN), Gated Residual Networks (GRN), and optional multi-resolution
    Temporal Convolutional Networks (TCN) for robust multi-step-ahead predictions.
    
    Key Features:
        - Variable Selection: Learns to focus on important features dynamically
        - Multi-Resolution TCN: Captures patterns at different temporal scales
        - Horizon Embeddings: Network learns horizon-specific characteristics
        - Static Context: Supports time-invariant features (e.g., asset IDs)
        - Quantile Regression: Probabilistic forecasting support
        - Per-Horizon Heads: Optional specialized output layers for each horizon
    
    Architecture Flow:
        1. Feature Processing:
           - Past features → VSN/Linear → [B, L, D]
           - Future features → VSN/Linear → [B, H, D]
           - Static features → VSN → [B, D] (optional)
        
        2. Encoder Path:
           - Add positional encodings
           - Optional TCN branch for multi-scale patterns
           - Fuse TCN output with main path via GRN
           - Add static context (broadcast over time)
           - Transformer encoder layers
        
        3. Decoder Path:
           - Add horizon embeddings to future features
           - Add positional encodings
           - Add static context
           - Transformer decoder with causal masking
        
        4. Output:
           - Full sequence predictions: [B, H, out_dim]
           - Selected horizon predictions: [B, K, out_dim]
           - Optional per-horizon refinement heads
    
    Args:
        cfg: Configuration dataclass containing all hyperparameters
    
    Input Shapes:
        past_feats: [B, L, F_p] - Historical features
        future_feats: [B, H, F_f] - Known future covariates (e.g., time features)
        static_feats: [B, F_s] - Time-invariant features (optional)
    
    Output:
        Tuple containing:
            - 'y_hat': [B, K, out_dim] - Predictions for target horizons
            - 'y_hat_full': [B, H, out_dim] - Full sequence (if return_full=True)
    
    Example:
        >>> cfg = MultiHorizonTransformerConfig(
        ...     d_model=256, n_heads=8, context_len=120, max_horizon=15,
        ...     past_feat_dim=16, future_feat_dim=8
        ... )
        >>> model = MultiHorizonTransformer(cfg)
        >>> past = torch.randn(32, 120, 16)  # [B, L, F_p]
        >>> future = torch.randn(32, 15, 8)  # [B, H, F_f]
        >>> output = model(past, future)
        >>> print(output['y_hat'].shape)  # [32, 5, 1] for 5 target horizons
    """
    def __init__(self, cfg: MultiHorizonTransformerConfig):
        super().__init__()
        self.cfg = cfg

        assert cfg.d_model % cfg.n_heads == 0, \
            f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})"
        assert (not cfg.use_quantiles) or (cfg.out_dim == len(cfg.quantiles)), \
            "out_dim must equal len(quantiles) when use_quantiles=True"
        assert cfg.past_feat_dim  > 0 and cfg.future_feat_dim > 0, "Feature dims must be > 0"

        # Static pipeline (optional)
        if cfg.static_feat_dim > 0:
            self.static_vsn = VariableSelectionNetwork(cfg.static_feat_dim, cfg.d_model, cfg.vsn_hidden, cfg.dropout)
        else:
            self.static_vsn = None

        # Dynamic pipelines
        if cfg.use_variable_selection:
            self.past_vsn = VariableSelectionNetwork(cfg.past_feat_dim, cfg.d_model, cfg.vsn_hidden, cfg.dropout)
            self.future_vsn = VariableSelectionNetwork(cfg.future_feat_dim, cfg.d_model, cfg.vsn_hidden, cfg.dropout)
        else:
            self.past_proj = nn.Linear(cfg.past_feat_dim, cfg.d_model)
            self.future_proj = nn.Linear(cfg.future_feat_dim, cfg.d_model)

        # Positional encodings
        if cfg.use_sinusoidal_pe:
            self.pe_past = SinusoidalPE(cfg.d_model, max_len=cfg.context_len + 32)
            self.pe_fut = SinusoidalPE(cfg.d_model, max_len=cfg.max_horizon + 32)
        else:
            self.pe_past = nn.Embedding(cfg.context_len, cfg.d_model)
            self.pe_fut = nn.Embedding(cfg.max_horizon, cfg.d_model)

        # Horizon embeddings
        self.horizon_emb = nn.Embedding(cfg.max_horizon, cfg.d_model)

        # Optional multi-resolution temporal conv branch fused into encoder inputs
        if cfg.use_tcn_branch:
            self.tcn = MultiResolutionTCN(cfg.d_model, cfg.tcn_channels, cfg.tcn_kernel_sizes, cfg.tcn_dilations, cfg.dropout)
            self.fuse_gate = GatedResidualNetwork(cfg.d_model * 2, cfg.grn_hidden, cfg.d_model, cfg.dropout)
        else:
            self.tcn = None

        # Transformer encoder/decoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            layer_norm_eps=cfg.layer_norm_eps
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_enc_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            layer_norm_eps=cfg.layer_norm_eps
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.n_dec_layers)

        # Output heads
        if cfg.per_horizon_heads:
            self.per_head = nn.ModuleDict({
                str(h): nn.Sequential(
                    nn.Linear(cfg.d_model, cfg.d_model),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.d_model, cfg.out_dim)
                ) for h in cfg.target_horizons
            })
            self.full_head = nn.Linear(cfg.d_model, cfg.out_dim)
        else:
            self.full_head = nn.Linear(cfg.d_model, cfg.out_dim)

        self.reset_parameters(gain=cfg.init_gain)

    def reset_parameters(self, gain: float = 1.0):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _add_pe(self, x: torch.Tensor, which: str) -> torch.Tensor:
        if isinstance(getattr(self, f"pe_{which}"), SinusoidalPE):
            return getattr(self, f"pe_{which}")(x)
        else:
            B, T, D = x.shape
            idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            return x + getattr(self, f"pe_{which}")(idx)

    def _encode_past(self, past_feats: torch.Tensor, static_ctx: Optional[torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        if cfg.use_variable_selection:
            x_p = self.past_vsn(past_feats)  # [B, L, D]
        else:
            x_p = self.past_proj(past_feats)

        x_p = self._add_pe(x_p, "past")

        # Fuse TCN branch if enabled
        if self.tcn is not None:
            tcn_out = self.tcn(x_p)                 # [B, L, D]
            fused = torch.cat([x_p, tcn_out], dim=-1)
            x_p = self.fuse_gate(fused)             # [B, L, D]

        # Optionally inject static context (additive)
        if static_ctx is not None:
            x_p = x_p + static_ctx.unsqueeze(1)     # broadcast to time

        memory = self.encoder(x_p)                  # [B, L, D]
        return memory

    def _build_decoder_inputs(self, future_feats: torch.Tensor, static_ctx: Optional[torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        if cfg.use_variable_selection:
            d_f = self.future_vsn(future_feats)     # [B, H, D]
        else:
            d_f = self.future_proj(future_feats)

        # Horizon embeddings
        h_ids = torch.arange(cfg.max_horizon, device=d_f.device).unsqueeze(0).expand(d_f.size(0), -1)  # [B, H]
        d_f = d_f + self.horizon_emb(h_ids)

        # Positional encodings for future
        d_f = self._add_pe(d_f, "fut")

        if static_ctx is not None:
            d_f = d_f + static_ctx.unsqueeze(1)
        return d_f

    def forward(
        self,
        past_feats: torch.Tensor,     # [B, L, F_p]
        future_feats: torch.Tensor,   # [B, H, F_f] (truly future-known!)
        static_feats: Optional[torch.Tensor] = None,  # [B, F_s] or None
        return_full: bool = False
    ) -> Dict[str, torch.Tensor]:

        cfg = self.cfg
        B, L, Fp = past_feats.shape
        B2, H, Ff = future_feats.shape
        assert B == B2 and L == cfg.context_len and H == cfg.max_horizon, f"Invalid shapes for past/future. Expected B={B}, L={cfg.context_len}, H={cfg.max_horizon}."
        if static_feats is not None:
            assert static_feats.shape[0] == B and static_feats.shape[1] == cfg.static_feat_dim, f"Invalid static shape. Expected B={B}, F_s={cfg.static_feat_dim}."

        static_ctx = None
        if self.static_vsn is not None and static_feats is not None:
            static_ctx = self.static_vsn(static_feats)  # [B, D]

        memory = self._encode_past(past_feats, static_ctx)                  # [B, L, D]
        tgt = self._build_decoder_inputs(future_feats, static_ctx)          # [B, H, D]

        tgt_mask = causal_mask(H, tgt.device)                               # [H, H] bool

        dec_out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=None
        )                                                                   # [B, H, D]

        y_full = self.full_head(dec_out)                                    # [B, H, out_dim]

        # Select requested horizons
        idx = torch.tensor([h - 1 for h in cfg.target_horizons], device=y_full.device)
        y_sel = y_full.index_select(dim=1, index=idx)                       # [B, K, out_dim]

        if cfg.per_horizon_heads:
            # refine selected steps with per-horizon heads
            dec_sel = dec_out.index_select(dim=1, index=idx)                # [B, K, D]
            outs = []
            for j, h in enumerate(cfg.target_horizons):
                outs.append(self.per_head[str(h)](dec_sel[:, j, :]))        # [B, out_dim]
            y_sel = torch.stack(outs, dim=1)                                # [B, K, out_dim]

        y_sel = y_sel.permute(0, 2, 1)  # [B, out_dim, K]
        if return_full:
            return y_sel, y_full
        else:
            return y_sel