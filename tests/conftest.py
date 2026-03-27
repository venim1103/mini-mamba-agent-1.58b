# tests/conftest.py
"""Shared pytest fixtures for the mini-mamba-agent-1.58b test suite."""

import pytest
import torch

# ---------------------------------------------------------------------------
# Device fixture — GPU when available, CPU otherwise
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Small model configs — keep dims tiny so every test completes in seconds
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_mamba_cfg():
    """A minimal BitMambaBlock configuration."""
    return dict(dim=128, d_state=16, d_conv=4, expand=2, headdim=32, ngroups=1, chunk_size=64)


@pytest.fixture(scope="session")
def tiny_llm_cfg():
    """A minimal BitMambaLLM configuration — pure Mamba (no attn)."""
    return dict(
        vocab_size=256, dim=128, n_layers=2, d_state=16,
        expand=2, headdim=32, ngroups=1, chunk_size=64,
        use_attn=False,
    )


@pytest.fixture(scope="session")
def tiny_hybrid_cfg():
    """A minimal BitMambaLLM with one attention layer (attn_pct=0.25).

    dim=256 → n_heads = 256//64 = 4 which satisfies AttentionBlock's
    default n_kv_heads=4 (GQA requires n_kv_heads ≤ n_heads).
    """
    return dict(
        vocab_size=256, dim=256, n_layers=4, d_state=16,
        expand=2, headdim=64, ngroups=1, chunk_size=64,
        use_attn=True, attn_pct=0.25,
    )
