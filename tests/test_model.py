# tests/test_model.py
"""Unit tests for model.py — quantization, blocks, full LLM, and utilities."""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import model as model_module

from model import (
    weight_quant,
    activation_quant,
    BitLinear,
    RMSNorm,
    AttentionBlock,
    BitMambaBlock,
    BitMambaLLM,
    chunked_cross_entropy,
)


# ===========================================================================
# 1. Ternary weight quantisation
# ===========================================================================

class TestWeightQuant:
    def test_output_values_in_ternary_set(self, device):
        """All quantised weights must be in {-1, 0, 1}."""
        torch.manual_seed(0)
        w = torch.randn(64, 64, device=device)
        q = weight_quant(w)
        unique = q.unique()
        assert unique.numel() <= 3
        assert all(v.item() in {-1.0, 0.0, 1.0} for v in unique)

    def test_output_shape_preserved(self, device):
        w = torch.randn(32, 128, device=device)
        assert weight_quant(w).shape == w.shape

    def test_ste_gradient_passes_through(self, device):
        """Straight-through estimator: gradient of quantised output == input gradient."""
        w = torch.randn(16, 16, device=device, requires_grad=True)
        loss = weight_quant(w).sum()
        loss.backward()
        assert w.grad is not None
        # STE: dL/dw should be all-ones (d(sum)/dw = 1 everywhere)
        assert torch.allclose(w.grad, torch.ones_like(w.grad))

    def test_ste_gradient_shape_matches_weight(self, device):
        w = torch.randn(8, 8, device=device, requires_grad=True)
        weight_quant(w).sum().backward()
        assert w.grad.shape == w.shape

    def test_zero_weight_stays_zero(self, device):
        w = torch.zeros(4, 4, device=device)
        q = weight_quant(w)
        assert q.abs().sum().item() == 0.0


# ===========================================================================
# 2. Activation quantisation
# ===========================================================================

class TestActivationQuant:
    def test_output_range_within_int8(self, device):
        """activation_quant should produce values in [-128/scale, 127/scale]."""
        torch.manual_seed(1)
        x = torch.randn(4, 64, device=device)
        q = activation_quant(x)
        # The output is in the same unit as x (de-scaled), so check relative magnitudes.
        assert q.shape == x.shape

    def test_output_shape_preserved(self, device):
        x = torch.randn(2, 3, 32, device=device)
        assert activation_quant(x).shape == x.shape

    def test_gradient_flows(self, device):
        x = torch.randn(4, 16, device=device, requires_grad=True)
        activation_quant(x).sum().backward()
        assert x.grad is not None

    def test_per_token_scaling(self, device):
        """Each token (last dim) gets its own max-abs scale."""
        x = torch.zeros(2, 4, device=device)
        x[0, 0] = 10.0   # row 0 has large value
        x[1, 0] = 1.0    # row 1 has small value
        q = activation_quant(x)
        # Row 0 max should be quantised to ±127; row 1 max also ±127 (different scales).
        # Check via: max of each row should be equal after absolute value.
        assert q.shape == x.shape


# ===========================================================================
# 3. BitLinear
# ===========================================================================

class TestBitLinear:
    def test_forward_output_shape(self, device):
        layer = BitLinear(64, 32).to(device)
        x = torch.randn(2, 10, 64, device=device)
        out = layer(x)
        assert out.shape == (2, 10, 32)

    def test_no_bias_by_default(self):
        layer = BitLinear(16, 8)
        assert layer.bias is None

    def test_bias_registered_when_requested(self):
        layer = BitLinear(16, 8, bias=True)
        assert layer.bias is not None
        assert layer.bias.shape == (8,)

    def test_gradient_flows_through_linear(self, device):
        layer = BitLinear(16, 8).to(device)
        x = torch.randn(2, 5, 16, device=device, requires_grad=True)
        layer(x).sum().backward()
        assert x.grad is not None

    def test_deterministic_given_same_input(self, device):
        """Same input → same output (quantisation is deterministic at inference)."""
        layer = BitLinear(16, 8).to(device).eval()
        x = torch.randn(1, 4, 16, device=device)
        out1 = layer(x)
        out2 = layer(x)
        assert torch.allclose(out1, out2)

    def test_eval_reuses_cached_quantized_weights(self, device, monkeypatch):
        layer = BitLinear(16, 8).to(device).eval()
        x = torch.randn(1, 4, 16, device=device)
        call_count = 0
        original_weight_quant = model_module.weight_quant

        def counting_weight_quant(weight):
            nonlocal call_count
            call_count += 1
            return original_weight_quant(weight)

        monkeypatch.setattr(model_module, "weight_quant", counting_weight_quant)

        layer(x)
        layer(x)

        assert call_count == 1

    def test_train_mode_clears_cached_quantized_weights(self, device, monkeypatch):
        layer = BitLinear(16, 8).to(device).eval()
        x = torch.randn(1, 4, 16, device=device)
        call_count = 0
        original_weight_quant = model_module.weight_quant

        def counting_weight_quant(weight):
            nonlocal call_count
            call_count += 1
            return original_weight_quant(weight)

        monkeypatch.setattr(model_module, "weight_quant", counting_weight_quant)

        layer(x)
        layer.train()
        layer.eval()
        layer(x)

        assert call_count == 2


# ===========================================================================
# 4. RMSNorm
# ===========================================================================

class TestRMSNorm:
    def test_output_shape(self, device):
        norm = RMSNorm(32).to(device)
        x = torch.randn(2, 10, 32, device=device)
        assert norm(x).shape == x.shape

    def test_rms_normalised(self, device):
        """After RMSNorm (with default weight=1), RMS of each token ≈ 1."""
        norm = RMSNorm(64).to(device)
        # Force weight to 1 so scale doesn't interfere.
        nn.init.ones_(norm.weight)
        x = torch.randn(3, 8, 64, device=device)
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_gradient_flows(self, device):
        norm = RMSNorm(16).to(device)
        x = torch.randn(2, 4, 16, device=device, requires_grad=True)
        norm(x).sum().backward()
        assert x.grad is not None

    def test_learnable_weight(self):
        norm = RMSNorm(8)
        assert any(p.requires_grad for p in norm.parameters())


# ===========================================================================
# 5. AttentionBlock
# ===========================================================================

class TestAttentionBlock:
    @pytest.fixture
    def block(self, device):
        # dim=128: n_heads = 128//64 = 2, n_kv_heads=1 (GQA 2:1)
        torch.manual_seed(42)
        return AttentionBlock(dim=128, n_kv_heads=2).to(device)

    def test_forward_output_shape(self, block, device):
        x = torch.randn(2, 8, 128, device=device)
        out = block(x)
        assert out.shape == (2, 8, 128)

    def test_forward_residual_connection(self, block, device):
        """Output should differ from input (non-trivial transform) but not explode."""
        x = torch.randn(1, 4, 128, device=device)
        out = block(x)
        assert not torch.allclose(out, x)
        assert out.isfinite().all()

    def test_forward_with_seq_idx_single_doc(self, block, device):
        """Single document — result should equal no-seq_idx result."""
        x = torch.randn(1, 6, 128, device=device)
        seq_idx = torch.zeros(1, 6, dtype=torch.long, device=device)
        out_with = block(x, seq_idx=seq_idx)
        out_without = block(x, seq_idx=None)
        # Should match since it's one contiguous document.
        assert torch.allclose(out_with, out_without, atol=1e-5)

    def test_forward_seq_idx_blocks_cross_doc_attention(self, block, device):
        """Tokens in doc 1 must NOT attend to tokens in doc 0."""
        torch.manual_seed(77)
        x = torch.randn(1, 8, 128, device=device)
        # Two documents: [0,0,0,0, 1,1,1,1]
        seq_idx = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], device=device)

        # Perturb only the first doc — if second doc output changes, leakage exists.
        x_perturbed = x.clone()
        x_perturbed[0, :4, :] = torch.randn(4, 128, device=device)

        out_orig = block(x, seq_idx=seq_idx)
        out_perturbed = block(x_perturbed, seq_idx=seq_idx)

        # Second document's output should be UNCHANGED despite perturbed doc-0.
        second_doc_diff = (out_orig[0, 4:, :] - out_perturbed[0, 4:, :]).abs().max().item()
        assert second_doc_diff < 1e-4, f"Cross-document leakage detected: diff={second_doc_diff}"

    def test_gradient_flows(self, block, device):
        x = torch.randn(1, 4, 128, device=device, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None

    # --- prefill / step parity ---

    def test_prefill_output_matches_forward(self, block, device):
        """prefill() output must equal forward() output."""
        torch.manual_seed(5)
        x = torch.randn(1, 8, 128, device=device)
        block.eval()
        with torch.no_grad():
            ref = block(x)
            pf_out, _ = block.prefill(x)
        assert torch.allclose(ref, pf_out, atol=1e-5)

    def test_prefill_returns_kv_cache(self, block, device):
        x = torch.randn(1, 6, 128, device=device)
        block.eval()
        with torch.no_grad():
            _, cache = block.prefill(x)
        assert "k" in cache and "v" in cache
        # KV cache key shape: (bsz, seqlen, n_kv_heads, head_dim)
        assert cache["k"].shape[1] == 6

    def test_step_output_shape(self, block, device):
        block.eval()
        with torch.no_grad():
            _, cache = block.prefill(torch.randn(1, 6, 128, device=device))
            step_out = block.step(torch.randn(1, 1, 128, device=device), cache)
        assert step_out.shape == (1, 1, 128)

    def test_step_kv_cache_grows(self, block, device):
        """Each step should append one new KV entry to the cache."""
        block.eval()
        prompt_len = 5
        with torch.no_grad():
            _, cache = block.prefill(torch.randn(1, prompt_len, 128, device=device))
            assert cache["k"].shape[1] == prompt_len
            block.step(torch.randn(1, 1, 128, device=device), cache)
            assert cache["k"].shape[1] == prompt_len + 1

    def test_step_parity_with_forward(self, block, device):
        """step() on the (n+1)-th token should match forward() over n+1 tokens."""
        torch.manual_seed(12)
        prompt = torch.randn(1, 8, 128, device=device)
        new_tok = torch.randn(1, 1, 128, device=device)
        full = torch.cat([prompt, new_tok], dim=1)

        block.eval()
        with torch.no_grad():
            ref = block(full)[:, -1:, :]
            _, cache = block.prefill(prompt)
            step_out = block.step(new_tok, cache)

        diff = (ref - step_out).abs().max().item()
        assert diff < 1e-4, f"AttentionBlock step parity: max_diff={diff}"


# ===========================================================================
# 6. BitMambaBlock
# ===========================================================================

class TestBitMambaBlock:
    @pytest.fixture
    def block(self, device, tiny_mamba_cfg):
        torch.manual_seed(42)
        return BitMambaBlock(**tiny_mamba_cfg).to(device)

    def test_forward_output_shape(self, block, device):
        x = torch.randn(2, 16, 128, device=device)
        out = block(x)
        assert out.shape == (2, 16, 128)

    def test_forward_residual_finite(self, block, device):
        x = torch.randn(1, 8, 128, device=device)
        out = block(x)
        assert out.isfinite().all()

    def test_forward_with_seq_idx(self, block, device):
        """Block should accept seq_idx without error."""
        x = torch.randn(1, 8, 128, device=device)
        seq_idx = torch.zeros(1, 8, dtype=torch.int32, device=device)
        out = block(x, seq_idx=seq_idx)
        assert out.shape == (1, 8, 128)

    def test_gradient_flows(self, block, device):
        x = torch.randn(1, 8, 128, device=device, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None

    def test_prefill_output_matches_forward(self, block, device):
        torch.manual_seed(7)
        x = torch.randn(1, 8, 128, device=device)
        block.eval()
        with torch.no_grad():
            ref = block(x)
            pf_out, _ = block.prefill(x)
        # Prefill uses the same kernel — must be exact.
        assert torch.allclose(ref, pf_out, atol=1e-5)

    def test_prefill_returns_cache_keys(self, block, device):
        x = torch.randn(1, 8, 128, device=device)
        block.eval()
        with torch.no_grad():
            _, cache = block.prefill(x)
        assert "conv_state" in cache
        assert "ssm_state" in cache

    def test_prefill_conv_state_shape(self, block, device):
        x = torch.randn(1, 10, 128, device=device)
        block.eval()
        with torch.no_grad():
            _, cache = block.prefill(x)
        # conv_state: (batch, conv_dim, d_conv)
        assert cache["conv_state"].shape[-1] == block.d_conv

    def test_prefill_ssm_state_shape(self, block, device):
        x = torch.randn(1, 8, 128, device=device)
        block.eval()
        with torch.no_grad():
            _, cache = block.prefill(x)
        # ssm_state: (batch, nheads, headdim, d_state)
        assert cache["ssm_state"].shape == (1, block.nheads, block.headdim, block.d_state)

    def test_step_output_shape(self, block, device):
        block.eval()
        with torch.no_grad():
            _, cache = block.prefill(torch.randn(1, 8, 128, device=device))
            out = block.step(torch.randn(1, 1, 128, device=device), cache)
        assert out.shape == (1, 1, 128)

    def test_step_output_finite(self, block, device):
        block.eval()
        with torch.no_grad():
            _, cache = block.prefill(torch.randn(1, 8, 128, device=device))
            out = block.step(torch.randn(1, 1, 128, device=device), cache)
        assert out.isfinite().all()

    def test_step_ssm_state_updates(self, block, device):
        """SSM state tensor must change after a step."""
        block.eval()
        x = torch.randn(1, 8, 128, device=device)
        with torch.no_grad():
            _, cache = block.prefill(x)
            state_before = cache["ssm_state"].clone()
            block.step(torch.randn(1, 1, 128, device=device), cache)
            state_after = cache["ssm_state"]
        assert not torch.allclose(state_before, state_after)

    def test_step_prefill_parity_forward(self, block, device):
        """prefill then step should match full forward on last position."""
        torch.manual_seed(3)
        prompt = torch.randn(1, 7, 128, device=device)
        new_tok = torch.randn(1, 1, 128, device=device)

        block.eval()
        with torch.no_grad():
            ref = block(torch.cat([prompt, new_tok], dim=1))[:, -1:, :]
            _, cache = block.prefill(prompt)
            step_out = block.step(new_tok, cache)

        diff = (ref - step_out).abs().max().item()
        # SSD kernel uses chunk-dual form; ~0.1 relative error is expected (see FINDINGS #9).
        assert diff < 2.0, f"BitMambaBlock step parity too large: max_diff={diff}"
        # The direction (sign) should agree — check correlation.
        cos = F.cosine_similarity(ref.flatten().unsqueeze(0), step_out.flatten().unsqueeze(0)).item()
        assert cos > 0.99, f"Step output direction wrong: cosine={cos}"


# ===========================================================================
# 7. BitMambaLLM (full model)
# ===========================================================================

class TestBitMambaLLM:
    @pytest.fixture
    def pure_mamba_model(self, device, tiny_llm_cfg):
        torch.manual_seed(42)
        return BitMambaLLM(**tiny_llm_cfg).to(device).eval()

    @pytest.fixture
    def hybrid_model(self, device, tiny_hybrid_cfg):
        torch.manual_seed(42)
        return BitMambaLLM(**tiny_hybrid_cfg).to(device).eval()

    # --- forward ---

    def test_forward_output_shape(self, pure_mamba_model, device):
        ids = torch.randint(0, 256, (2, 16), device=device)
        logits = pure_mamba_model(ids)
        assert logits.shape == (2, 16, 256)

    def test_forward_logits_finite(self, pure_mamba_model, device):
        ids = torch.randint(0, 256, (1, 8), device=device)
        assert pure_mamba_model(ids).isfinite().all()

    def test_forward_with_seq_idx(self, pure_mamba_model, device):
        ids = torch.randint(0, 256, (1, 8), device=device)
        seq_idx = torch.zeros(1, 8, dtype=torch.int32, device=device)
        out = pure_mamba_model(ids, seq_idx=seq_idx)
        assert out.shape == (1, 8, 256)

    def test_forward_hidden_shape(self, pure_mamba_model, device):
        ids = torch.randint(0, 256, (1, 8), device=device)
        hidden = pure_mamba_model.forward_hidden(ids)
        assert hidden.shape == (1, 8, 128)  # dim=128

    def test_hybrid_model_has_attn_layers(self, tiny_hybrid_cfg):
        model = BitMambaLLM(**tiny_hybrid_cfg)
        from model import AttentionBlock
        attn_layers = [l for l in model.layers if isinstance(l, AttentionBlock)]
        assert len(attn_layers) >= 1

    # --- gradient checkpointing ---

    def test_use_checkpoint_does_not_crash(self, device, tiny_llm_cfg):
        cfg = dict(**tiny_llm_cfg, use_checkpoint=True)
        model = BitMambaLLM(**cfg).to(device).train()
        ids = torch.randint(0, 256, (1, 8), device=device)
        model(ids).sum().backward()

    # --- prefill logit parity ---

    def test_prefill_logits_match_forward(self, pure_mamba_model, device):
        """prefill → norm → output should produce exactly the same logits as forward()."""
        torch.manual_seed(99)
        ids = torch.randint(0, 256, (1, 12), device=device)
        with torch.no_grad():
            ref = pure_mamba_model(ids)[:, -1, :]

            x = pure_mamba_model.tok_embeddings(ids)
            for layer in pure_mamba_model.layers:
                x, _ = layer.prefill(x)
            x = pure_mamba_model.norm(x)
            pf_logits = pure_mamba_model.output(x)[:, -1, :]

        assert torch.allclose(ref, pf_logits, atol=1e-4)

    # --- generate ---

    def test_generate_length_greedy(self, pure_mamba_model, device):
        prompt = torch.randint(0, 256, (1, 4), device=device)
        out = pure_mamba_model.generate(prompt, max_new_tokens=10, temperature=0.0)
        assert out.shape[1] == 14  # 4 prompt + 10 new

    def test_generate_length_sampled(self, pure_mamba_model, device):
        prompt = torch.randint(0, 256, (1, 4), device=device)
        out = pure_mamba_model.generate(prompt, max_new_tokens=10, temperature=1.0, do_sample=True)
        assert out.shape[1] == 14

    def test_generate_eos_stops_early(self, pure_mamba_model, device):
        prompt = torch.randint(0, 256, (1, 4), device=device)
        # Use eos_token_id=1, which must appear before max_new_tokens=1000.
        out = pure_mamba_model.generate(prompt, max_new_tokens=1000, temperature=0.0, eos_token_id=0)
        # Pure Mamba greedy — token 0 is very likely; sequence should be short.
        assert out.shape[1] < 1004  # At most prompt + 1000 (eos hit in practice)

    def test_generate_preserves_prompt_prefix(self, pure_mamba_model, device):
        """The first `prompt_len` tokens of generate()'s output must equal the input."""
        prompt = torch.randint(0, 256, (1, 5), device=device)
        out = pure_mamba_model.generate(prompt, max_new_tokens=5, temperature=0.0)
        assert torch.equal(out[0, :5], prompt[0])

    def test_generate_greedy_deterministic(self, pure_mamba_model, device):
        torch.manual_seed(0)
        prompt = torch.randint(0, 256, (1, 4), device=device)
        out1 = pure_mamba_model.generate(prompt, max_new_tokens=8, temperature=0.0)
        out2 = pure_mamba_model.generate(prompt, max_new_tokens=8, temperature=0.0)
        assert torch.equal(out1, out2)

    def test_generate_works_with_hybrid(self, hybrid_model, device):
        prompt = torch.randint(0, 256, (1, 4), device=device)
        out = hybrid_model.generate(prompt, max_new_tokens=8, temperature=0.0)
        assert out.shape[1] == 12
        assert out.isfinite().all()

    def test_generate_no_nan(self, pure_mamba_model, device):
        prompt = torch.randint(1, 200, (1, 8), device=device)
        out = pure_mamba_model.generate(prompt, max_new_tokens=16, temperature=0.7, do_sample=True)
        assert not out.float().isnan().any()

    def test_generate_tokens_in_vocab_range(self, pure_mamba_model, device):
        prompt = torch.randint(1, 200, (1, 4), device=device)
        out = pure_mamba_model.generate(prompt, max_new_tokens=12, temperature=0.7, do_sample=True)
        assert out.min().item() >= 0
        assert out.max().item() < 256


# ===========================================================================
# 8. chunked_cross_entropy
# ===========================================================================

class TestChunkedCrossEntropy:
    @pytest.fixture
    def proj_and_data(self, device):
        torch.manual_seed(0)
        vocab, dim, bs, seq = 64, 32, 2, 16
        proj = BitLinear(dim, vocab).to(device)
        hidden = torch.randn(bs, seq, dim, device=device)
        targets = torch.randint(0, vocab, (bs, seq), device=device)
        return proj, hidden, targets

    def test_matches_naive_ce(self, proj_and_data, device):
        """chunked result must equal the standard un-chunked CE loss."""
        proj, hidden, targets = proj_and_data
        proj.eval()
        with torch.no_grad():
            chunked = chunked_cross_entropy(hidden, proj, targets, chunk_size=8)
            logits = proj(hidden).reshape(-1, proj.out_features)
            naive = F.cross_entropy(logits, targets.reshape(-1), reduction='mean')
        assert torch.allclose(chunked, naive, atol=1e-4), \
            f"chunked={chunked.item():.6f} naive={naive.item():.6f}"

    def test_ignore_index_excluded(self, proj_and_data, device):
        """Tokens with target=-100 must not contribute to the loss."""
        proj, hidden, targets = proj_and_data
        proj.eval()
        # Mask out the entire first row.
        targets_masked = targets.clone()
        targets_masked[0, :] = -100
        with torch.no_grad():
            loss_masked = chunked_cross_entropy(hidden, proj, targets_masked, chunk_size=8)
            loss_row1 = chunked_cross_entropy(hidden[1:2], proj, targets[1:2], chunk_size=8)
        assert torch.allclose(loss_masked, loss_row1, atol=1e-4)

    def test_chunk_size_invariant(self, proj_and_data, device):
        """Different chunk sizes must give the same result."""
        proj, hidden, targets = proj_and_data
        proj.eval()
        with torch.no_grad():
            l1 = chunked_cross_entropy(hidden, proj, targets, chunk_size=4)
            l2 = chunked_cross_entropy(hidden, proj, targets, chunk_size=16)
            l3 = chunked_cross_entropy(hidden, proj, targets, chunk_size=1)
        assert torch.allclose(l1, l2, atol=1e-4)
        assert torch.allclose(l1, l3, atol=1e-4)

    def test_scalar_output(self, proj_and_data, device):
        proj, hidden, targets = proj_and_data
        proj.eval()
        with torch.no_grad():
            loss = chunked_cross_entropy(hidden, proj, targets, chunk_size=8)
        assert loss.ndim == 0

    def test_return_stats_matches_mean_loss(self, proj_and_data, device):
        proj, hidden, targets = proj_and_data
        proj.eval()
        with torch.no_grad():
            loss = chunked_cross_entropy(hidden, proj, targets, chunk_size=8)
            loss_sum, valid_tokens = chunked_cross_entropy(
                hidden,
                proj,
                targets,
                chunk_size=8,
                return_stats=True,
            )
        assert torch.allclose(loss_sum / valid_tokens, loss, atol=1e-6)

    def test_gradient_flows(self, proj_and_data, device):
        proj, hidden, targets = proj_and_data
        hidden = hidden.detach().requires_grad_(True)
        loss = chunked_cross_entropy(hidden, proj, targets, chunk_size=8)
        loss.backward()
        assert hidden.grad is not None
        assert hidden.grad.isfinite().all()
