import torch
import math
from torch import nn
from attention_layer import Attention
import sidechainnet as scn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Transformer-style).

    Adds a deterministic position signal so self-attention can distinguish residue order.
    """

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.d_model = int(d_model)

        # Create positional encodings once (max_len, d_model)
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims

        # Register as buffer (not a parameter); persistent=False to avoid breaking older checkpoints
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        returns: (B, L, D) with positional encoding added
        """
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0).to(dtype=x.dtype, device=x.device)


class ProteinNet(nn.Module):
    def __init__(self,
                 d_hidden=768,
                 dim=384,
                 d_in=49,
                 d_embedding=48,
                 heads=12,
                 dim_head=64,
                 attn_dropout=0.0,
                 integer_sequence=False,
                 pos_enc_max_len: int = 4096):
        super().__init__()
        self.integer_sequence = integer_sequence
        self.d_embedding = d_embedding

        # Embedding or input projection
        if integer_sequence:
            # amino acids -> indices, padding_idx=20 (as in original)
            self.embedding = nn.Embedding(21, d_embedding, padding_idx=20)
        else:
            self.input_proj = nn.Linear(d_in, d_embedding)

        # Positional encoding (added here)
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_embedding, max_len=pos_enc_max_len)

        # Original layers
        self.hidden2out = nn.Sequential(
            nn.Linear(d_embedding, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 24),   # 12 angles * 2 (sin, cos)
        )

        self.out2attn = nn.Linear(24, dim)
        self.activation_0 = nn.GELU()
        self.norm_0 = nn.LayerNorm(dim)

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)

        self.activation_1 = nn.GELU()
        self.norm_1 = nn.LayerNorm(dim)

        self.final = nn.Sequential(
            nn.Linear(dim, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 24),
        )

        self.output_activation = nn.Tanh()

    def forward(self, x, mask=None):
        """
        x:
          - if integer_sequence: (B, L) int indices
          - else: (B, L, d_in) float features (PSSM)
        mask: (B, L) boolean/0-1 mask (batch.msks)
        """

        # Embed/proj
        if self.integer_sequence:
            # (B, L) -> (B, L, d_embedding)
            output = self.embedding(x)
        else:
            # (B, L, d_in) -> (B, L, d_embedding)
            output = self.input_proj(x)

        # Add positional encoding
        output = self.pos_enc(output)

        # Ensure padding positions stay zeroed (optional but recommended)
        if mask is not None:
            # mask: (B, L) -> (B, L, 1)
            output = output * mask.unsqueeze(-1).to(output.dtype)

        # --- Original forward pipeline ---
        output = self.hidden2out(output)
        output = self.out2attn(output)
        output = self.activation_0(output)
        output = self.norm_0(output)

        output = self.attn(output, mask=mask)

        output = self.activation_1(output)
        output = self.norm_1(output)
        output = self.final(output)

        # Bound output in [-1, 1]
        output = self.output_activation(output)

        # Reshape to (B, L, 12, 2)
        output = output.view(output.shape[0], output.shape[1], 12, 2)

        return output