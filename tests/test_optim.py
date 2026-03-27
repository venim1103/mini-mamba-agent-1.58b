# tests/test_optim.py
"""Unit tests for optim.py — Muon optimiser, parameter routing, FG-WSD scheduler."""

import pytest
import torch
import torch.nn as nn

from optim import Muon, setup_mamba_optimizers, FGWSD_Scheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_linear_param(shape=(16, 16), requires_grad=True):
    p = nn.Parameter(torch.randn(*shape))
    p.requires_grad_(requires_grad)
    return p


# ===========================================================================
# 1. Muon
# ===========================================================================

class TestMuon:
    def test_parameter_updates_after_step(self):
        """A gradient step must change the parameter values."""
        p = _tiny_linear_param()
        optim = Muon([p], lr=0.1)
        before = p.data.clone()

        p.grad = torch.ones_like(p)
        optim.step()

        assert not torch.allclose(p.data, before), "Parameter was not updated"

    def test_gradient_zeroed_after_step(self):
        """Muon sets p.grad = None after consuming it (G5 — free BF16 gradient memory)."""
        p = _tiny_linear_param()
        optim = Muon([p], lr=0.01)
        p.grad = torch.ones_like(p)
        optim.step()
        assert p.grad is None

    def test_no_update_with_no_grad(self):
        p = _tiny_linear_param()
        optim = Muon([p], lr=0.1)
        before = p.data.clone()
        optim.step()  # no grad set
        assert torch.allclose(p.data, before)

    def test_momentum_buffer_initialised_on_first_step(self):
        p = _tiny_linear_param()
        optim = Muon([p], lr=0.01)
        p.grad = torch.ones_like(p)
        optim.step()
        assert "momentum_buffer" in optim.state[p]

    def test_momentum_buffer_shape_matches_param(self):
        p = _tiny_linear_param((8, 4))
        optim = Muon([p], lr=0.01)
        p.grad = torch.ones_like(p)
        optim.step()
        assert optim.state[p]["momentum_buffer"].shape == p.shape

    def test_lr_zero_gives_no_update(self):
        p = _tiny_linear_param()
        optim = Muon([p], lr=0.0)
        before = p.data.clone()
        p.grad = torch.ones_like(p)
        optim.step()
        assert torch.allclose(p.data, before)

    def test_update_direction_opposes_gradient(self):
        """For a positive gradient and lr > 0, the update should be negative (descent)."""
        torch.manual_seed(0)
        p = nn.Parameter(torch.zeros(8, 8))
        optim = Muon([p], lr=1.0, ns_steps=1)
        p.grad = torch.ones_like(p)
        optim.step()
        # Mean should have decreased (moved in negative direction).
        assert p.data.mean().item() < 0, "Update should oppose the gradient direction"

    def test_multiple_param_groups(self):
        p1 = _tiny_linear_param()
        p2 = _tiny_linear_param()
        optim = Muon([p1, p2], lr=0.01)
        p1.grad = torch.ones_like(p1)
        p2.grad = torch.ones_like(p2)
        before1 = p1.data.clone()
        optim.step()
        assert not torch.allclose(p1.data, before1)


# ===========================================================================
# 2. setup_mamba_optimizers — parameter routing
# ===========================================================================

class TestSetupMambaOptimizers:
    @pytest.fixture
    def tiny_model(self):
        """Minimal model that mimics the parameter landscape of BitMambaLLM."""
        from model import BitMambaLLM
        torch.manual_seed(0)
        return BitMambaLLM(
            vocab_size=64, dim=128, n_layers=2, d_state=16,
            expand=2, headdim=32, ngroups=1, chunk_size=64,
        )

    @pytest.fixture
    def config(self):
        return {"peak_lr": 1e-3, "end_lr": 1e-5}

    def test_returns_three_optimizers(self, tiny_model, config):
        opts = setup_mamba_optimizers(tiny_model, config, use_8bit=False)
        assert len(opts) == 3

    def test_mamba_sensitive_params_isolated(self, tiny_model, config):
        """A_log, D, dt_bias must end up in the mamba_core optimiser only."""
        muon_opt, adam_opt, mamba_opt = setup_mamba_optimizers(tiny_model, config, use_8bit=False)
        mamba_param_ids = {id(p) for g in mamba_opt.param_groups for p in g["params"]}
        muon_param_ids = {id(p) for g in muon_opt.param_groups for p in g["params"]}
        adam_param_ids = {id(p) for g in adam_opt.param_groups for p in g["params"]}

        for name, p in tiny_model.named_parameters():
            if any(k in name for k in ("A_log", ".D", "dt_bias")):
                assert id(p) in mamba_param_ids, \
                    f"Expected {name} in mamba_core, but not found"
                assert id(p) not in muon_param_ids, \
                    f"Mamba param {name} leaked into Muon"
                assert id(p) not in adam_param_ids, \
                    f"Mamba param {name} leaked into AdamW"

    def test_2d_weights_go_to_muon(self, tiny_model, config):
        """2D weight matrices (BitLinear) should be in the Muon group."""
        muon_opt, _, _ = setup_mamba_optimizers(tiny_model, config, use_8bit=False)
        muon_param_ids = {id(p) for g in muon_opt.param_groups for p in g["params"]}

        for name, p in tiny_model.named_parameters():
            # Skip mamba sensitive params and embedding / norm
            if any(k in name for k in ("A_log", ".D", "dt_bias")):
                continue
            if "norm" in name or "tok_embeddings" in name:
                continue
            if p.ndim == 2 and "weight" in name:
                assert id(p) in muon_param_ids, \
                    f"2D weight {name} not in Muon group"

    def test_no_parameter_is_missing(self, tiny_model, config):
        """Every trainable parameter must appear in exactly one optimizer."""
        muon_opt, adam_opt, mamba_opt = setup_mamba_optimizers(tiny_model, config, use_8bit=False)
        all_ids = (
            {id(p) for g in muon_opt.param_groups for p in g["params"]} |
            {id(p) for g in adam_opt.param_groups for p in g["params"]} |
            {id(p) for g in mamba_opt.param_groups for p in g["params"]}
        )
        for name, p in tiny_model.named_parameters():
            if p.requires_grad:
                assert id(p) in all_ids, f"Parameter {name} is not in any optimizer"

    def test_no_parameter_is_duplicated(self, tiny_model, config):
        """No parameter should appear in more than one optimizer."""
        muon_opt, adam_opt, mamba_opt = setup_mamba_optimizers(tiny_model, config, use_8bit=False)
        sets = [
            {id(p) for g in opt.param_groups for p in g["params"]}
            for opt in [muon_opt, adam_opt, mamba_opt]
        ]
        # Pairwise intersection must be empty.
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                overlap = sets[i] & sets[j]
                assert len(overlap) == 0, \
                    f"Optimizer {i} and {j} share {len(overlap)} parameters"

    def test_mamba_core_lr_is_lower(self, tiny_model, config):
        muon_opt, adam_opt, mamba_opt = setup_mamba_optimizers(tiny_model, config, use_8bit=False)
        mamba_lr = mamba_opt.param_groups[0]["lr"]
        main_lr = adam_opt.param_groups[0]["lr"]
        assert mamba_lr == pytest.approx(main_lr * 0.1, rel=1e-5)


# ===========================================================================
# 3. FGWSD_Scheduler
# ===========================================================================

class TestFGWSDScheduler:
    """
    Phase schedule for the real training (4 phases, total_steps=1000):
      Phase 1 (warmup):  0% → 10%  → steps 0-99    → linear LR ramp
      Phase 2 (plateau): 10% → 50% → steps 100-499 → constant LR
      Phase 3 (plateau): 50% → 70% → steps 500-699 → constant LR
      Phase 4 (cosine):  70% → 100%→ steps 700-999 → cosine decay
    """

    TOTAL = 1000
    PHASES = [
        {"pct": 0.10, "ctx": 2048},
        {"pct": 0.40, "ctx": 4096},
        {"pct": 0.20, "ctx": 8192},
        {"pct": 0.30, "ctx": 8192},
    ]
    CFG = {"peak_lr": 1e-3, "end_lr": 1e-5, "phases": PHASES}

    @pytest.fixture
    def sched(self):
        # Use dummy optimisers (one param group each is enough).
        p = nn.Parameter(torch.zeros(2))
        muon = Muon([p], lr=1e-3)
        adam = torch.optim.AdamW([p], lr=1e-3)
        mamba = torch.optim.AdamW([p], lr=1e-4)
        return FGWSD_Scheduler(muon, adam, mamba, self.TOTAL, self.CFG)

    def test_warmup_lr_increases(self, sched):
        lr0, _, _ = sched.get_lr_and_ctx(0)
        lr5, _, _ = sched.get_lr_and_ctx(50)
        assert lr5 > lr0

    def test_warmup_ends_at_peak(self, sched):
        lr, _, _ = sched.get_lr_and_ctx(99)
        assert lr == pytest.approx(self.CFG["peak_lr"], rel=0.05)

    def test_plateau_lr_constant(self, sched):
        lrs = [sched.get_lr_and_ctx(s)[0] for s in [100, 200, 300, 499]]
        assert max(lrs) - min(lrs) < 1e-9

    def test_cosine_decay_lr_decreasing(self, sched):
        lr_start, _, _ = sched.get_lr_and_ctx(700)
        lr_end, _, _ = sched.get_lr_and_ctx(999)
        assert lr_end < lr_start

    def test_cosine_decay_ends_near_end_lr(self, sched):
        lr, _, _ = sched.get_lr_and_ctx(999)
        assert lr == pytest.approx(self.CFG["end_lr"], rel=0.10)

    def test_correct_ctx_returned_per_phase(self, sched):
        _, ctx0, _ = sched.get_lr_and_ctx(0)    # Phase 1
        _, ctx1, _ = sched.get_lr_and_ctx(100)   # Phase 2
        _, ctx2, _ = sched.get_lr_and_ctx(500)   # Phase 3
        assert ctx0 == 2048
        assert ctx1 == 4096
        assert ctx2 == 8192

    def test_phase_names_returned(self, sched):
        _, _, p1 = sched.get_lr_and_ctx(0)
        _, _, p2 = sched.get_lr_and_ctx(100)
        _, _, p_done = sched.get_lr_and_ctx(1000)
        assert "Phase_1" in p1
        assert "Phase_2" in p2
        assert "Complete" in p_done

    def test_step_updates_all_optimizers(self, sched):
        """Calling step() at warmup-start sets all optimizer lrs to end_lr (< peak_lr)."""
        # step=0 → Phase 1 start → lr = end_lr (1e-5), far below initial 1e-3.
        sched.step(0)
        new_lrs = [opt.param_groups[0]["lr"] for opt in sched.opts]
        # Main optimisers should now be at end_lr.
        assert new_lrs[0] == pytest.approx(self.CFG["end_lr"], rel=1e-3)
        assert new_lrs[1] == pytest.approx(self.CFG["end_lr"], rel=1e-3)

    def test_mamba_lr_always_10x_lower(self, sched):
        for step in [0, 100, 500, 999]:
            sched.step(step)
            main_lr = sched.opts[0].param_groups[0]["lr"]
            mamba_lr = sched.opts[2].param_groups[0]["lr"]
            assert mamba_lr == pytest.approx(main_lr * 0.1, rel=1e-5), \
                f"At step {step}: mamba_lr={mamba_lr}, main_lr={main_lr}"

    def test_lr_never_below_end_lr(self, sched):
        for step in range(self.TOTAL):
            lr, _, _ = sched.get_lr_and_ctx(step)
            assert lr >= self.CFG["end_lr"] - 1e-10, \
                f"LR below end_lr at step {step}: {lr}"

    def test_lr_never_above_peak_lr(self, sched):
        for step in range(self.TOTAL):
            lr, _, _ = sched.get_lr_and_ctx(step)
            assert lr <= self.CFG["peak_lr"] + 1e-10, \
                f"LR above peak_lr at step {step}: {lr}"
