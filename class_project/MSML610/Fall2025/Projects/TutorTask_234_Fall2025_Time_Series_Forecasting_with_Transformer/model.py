import jax.numpy as jnp
import flax.linen as nn

class PositionalEncoding(nn.Module):
    d_model: int

    @nn.compact
    def __call__(self, x):
        pos = self.param(
            "pos",
            nn.initializers.normal(stddev=0.01),
            (5000, self.d_model)
        )
        return x + pos[:x.shape[1]]


class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x, train=True):

        attn = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=0.0,   # NO DROPOUT
        )
        x2 = attn(x, deterministic=True)
        x = nn.LayerNorm()(x + x2)

        x2 = nn.Dense(self.mlp_dim)(x)
        x2 = nn.gelu(x2)
        x2 = nn.Dense(self.d_model)(x2)

        x = nn.LayerNorm()(x + x2)
        return x


class TimeSeriesTransformer(nn.Module):
    seq_len: int = 60
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    mlp_dim: int = 256
    out_len: int = 5
    num_features: int = 6   # ← MATCH THE CHECKPOINT (6 FEATURES)

    @nn.compact
    def __call__(self, x, train=True):

        # x shape must be (batch, seq_len, 6)
        x = nn.Dense(self.d_model)(x)

        x = PositionalEncoding(self.d_model)(x)

        for _ in range(self.num_layers):
            x = TransformerBlock(
                self.d_model,
                self.num_heads,
                self.mlp_dim
            )(x, train=train)

        last = x[:, -1]
        return nn.Dense(self.out_len)(last)
