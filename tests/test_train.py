import pytest
import torch
import sys
from unittest.mock import MagicMock, patch

# Mock ALL heavy dependencies BEFORE any imports
_mocked_modules = {
    'torch': MagicMock(),
    'torch.nn': MagicMock(),
    'torch.nn.functional': MagicMock(),
    'torch.cuda': MagicMock(),
    'torch.optim': MagicMock(),
    'torch.optim.lr_scheduler': MagicMock(),
    'torch.amp': MagicMock(),
    'torch.amp.GradScaler': MagicMock(),
    'torch.utils': MagicMock(),
    'torch.utils.data': MagicMock(),
    'torch.utils.data.distributed': MagicMock(),
    'model': MagicMock(),
    'model.BitMambaLLM': MagicMock(),
    'model.chunked_cross_entropy': MagicMock(),
    'model.maybe_autocast': MagicMock(),
    'model.RMSNorm': MagicMock(),
    'model.BitLinear': MagicMock(),
    'model.weight_quant': MagicMock(),
    'model.activation_quant': MagicMock(),
    'optim': MagicMock(),
    'optim.Muon': MagicMock(),
    'optim.setup_mamba_optimizers': MagicMock(),
    'optim.FGWSD_Scheduler': MagicMock(),
    'dist_utils': MagicMock(),
    'dist_utils.setup_distributed': MagicMock(),
    'dist_utils.cleanup_distributed': MagicMock(),
    'dist_utils.is_main_process': MagicMock(),
    'dist_utils.wrap_model_ddp': MagicMock(),
    'dist_utils.unwrap_model': MagicMock(),
    'dist_utils.barrier': MagicMock(),
    'dist_utils.get_world_size': MagicMock(),
    'wandb': MagicMock(),
    'transformers': MagicMock(),
    'transformers.AutoTokenizer': MagicMock(),
    'datasets': MagicMock(),
    'datasets.load_dataset': MagicMock(),
}

for _name, _obj in _mocked_modules.items():
    if _name not in sys.modules:
        sys.modules[_name] = _obj


class TestCreateSeqIdxBatch:
    """Test train.py's create_seq_idx_batch helper function."""

    def test_single_segment_all_zeros(self):
        from train import create_seq_idx_batch
        cu_seqlens = torch.tensor([[0, 8, -1, -1]], dtype=torch.int32)
        n_segs = torch.tensor([2])
        seq_idx = create_seq_idx_batch(cu_seqlens, n_segs, 8)
        assert seq_idx.shape == (1, 8)
        assert seq_idx[0].unique().item() == 0

    def test_two_segments(self):
        from train import create_seq_idx_batch
        cu_seqlens = torch.tensor([[0, 3, 8, -1, -1]], dtype=torch.int32)
        n_segs = torch.tensor([3])
        seq_idx = create_seq_idx_batch(cu_seqlens, n_segs, 8)
        assert seq_idx.shape == (1, 8)
        assert (seq_idx[0, :3] == 0).all()
        assert (seq_idx[0, 3:] == 1).all()

    def test_three_segments(self):
        from train import create_seq_idx_batch
        cu_seqlens = torch.tensor([[0, 2, 5, 8]], dtype=torch.int32)
        n_segs = torch.tensor([4])
        seq_idx = create_seq_idx_batch(cu_seqlens, n_segs, 8)
        expected = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2]], dtype=torch.int32, device=seq_idx.device)
        assert torch.equal(seq_idx, expected)

    def test_batch_size_two(self):
        from train import create_seq_idx_batch
        cu_seqlens = torch.tensor([[0, 4, 8, -1], [0, 2, 8, -1]], dtype=torch.int32)
        n_segs = torch.tensor([2, 3])
        seq_idx = create_seq_idx_batch(cu_seqlens, n_segs, 8)
        assert seq_idx.shape == (2, 8)
        assert (seq_idx[0, :4] == 0).all()
        assert (seq_idx[0, 4:] == 1).all()
        assert (seq_idx[1, :2] == 0).all()
        assert (seq_idx[1, 2:] == 1).all()

    def test_truncation_handling(self):
        from train import create_seq_idx_batch
        cu_seqlens = torch.tensor([[0, 3, 8, -1, -1]], dtype=torch.int32)
        n_segs = torch.tensor([3])
        seq_idx = create_seq_idx_batch(cu_seqlens, n_segs, 6)
        assert seq_idx.shape == (1, 6)
        assert (seq_idx[0, :3] == 0).all()
        assert (seq_idx[0, 3:] == 1).all()


class TestTrainConfigs:
    """Test train.py configuration constants."""

    def test_train_configs_has_required_phases(self):
        from train import TRAIN_CONFIGS
        required_phases = ["Phase_1", "Phase_2", "Phase_3", "Phase_4"]
        for phase in required_phases:
            assert phase in TRAIN_CONFIGS

    def test_each_phase_has_weighted_datasets(self):
        from train import TRAIN_CONFIGS
        for phase_name, datasets in TRAIN_CONFIGS.items():
            assert len(datasets) > 0
            weights = [d["weight"] for d in datasets]
            assert sum(weights) == pytest.approx(1.0, abs=0.01)

    def test_curriculum_config_valid(self):
        from train import CURRICULUM_CONFIG
        assert "peak_lr" in CURRICULUM_CONFIG
        assert "end_lr" in CURRICULUM_CONFIG
        assert "phases" in CURRICULUM_CONFIG
        assert len(CURRICULUM_CONFIG["phases"]) == 4

    def test_phase_percentages_sum_to_one(self):
        from train import CURRICULUM_CONFIG
        total_pct = sum(p["pct"] for p in CURRICULUM_CONFIG["phases"])
        assert total_pct == pytest.approx(1.0, abs=0.01)

    def test_early_phases_use_half_context(self):
        from train import CURRICULUM_CONFIG, CONTEXT_LENGTH, HALF_CONTEXT_LENGTH
        phases = CURRICULUM_CONFIG["phases"]
        assert phases[0]["ctx"] == HALF_CONTEXT_LENGTH
        assert phases[1]["ctx"] == HALF_CONTEXT_LENGTH
        assert phases[2]["ctx"] == HALF_CONTEXT_LENGTH
        assert phases[3]["ctx"] == CONTEXT_LENGTH

    def test_model_configs_valid_dimensions(self):
        from train import MODEL_CONFIG
        assert MODEL_CONFIG["vocab_size"] > 0
        assert MODEL_CONFIG["dim"] > 0
        assert MODEL_CONFIG["n_layers"] > 0

    def test_batch_size_and_accum_steps(self):
        from train import BATCH_SIZE, GRAD_ACCUM_STEPS, SAVE_EVERY
        assert BATCH_SIZE > 0
        assert GRAD_ACCUM_STEPS > 0
        assert SAVE_EVERY > 0

    def test_checkpoint_dir_defined(self):
        from train import CHECKPOINT_DIR
        assert "checkpoints" in CHECKPOINT_DIR


class TestSafeDivisorMath:
    """Verify that SAFE_DIVISOR divide-before-backward/multiply-back-into-grad_scale
    produces mathematically identical gradients to a direct normalised backward pass."""

    def test_safe_divisor_gradient_math(self):
        import torch
        import torch.nn as nn

        torch.manual_seed(0)
        model = nn.Linear(8, 8, bias=False)
        x = torch.randn(2, 8)
        targets = torch.randint(0, 8, (2,))
        SAFE_DIVISOR = 16384.0 * 2

        model.zero_grad()
        loss_sum = nn.functional.cross_entropy(model(x), targets, reduction='sum')
        safe_loss = loss_sum / SAFE_DIVISOR
        safe_loss.backward()
        grad_scale = (1 * SAFE_DIVISOR) / targets.numel()
        model.weight.grad.mul_(grad_scale)
        grad_scaled = model.weight.grad.clone()

        model.zero_grad()
        loss_sum2 = nn.functional.cross_entropy(model(x), targets, reduction='sum')
        loss_sum2.backward()
        model.weight.grad.mul_(1.0 / targets.numel())
        grad_direct = model.weight.grad.clone()

        assert torch.allclose(grad_scaled, grad_direct, atol=1e-5), \
            "SAFE_DIVISOR scaling produced different gradients than direct scaling"


class TestRunTrainingStepsIntegration:
    """Tiny integration test: runs two full pretraining steps on CPU."""

    @patch('train.wandb')
    @patch('train.barrier')
    @patch('train.is_main_process', return_value=True)
    @patch('train.create_dataloaders')
    def test_run_training_steps_completes_and_saves(
        self, mock_create_dl, mock_is_main, mock_barrier, mock_wandb,
        tmp_path, monkeypatch
    ):
        import train

        TINY_BATCH = 2
        SEQ_LEN = 16
        TOTAL = 2

        monkeypatch.setattr(train, 'DEVICE', 'cpu')
        monkeypatch.setattr(train, 'CHECKPOINT_DIR', str(tmp_path))
        monkeypatch.setattr(train, 'BATCH_SIZE', TINY_BATCH)
        monkeypatch.setattr(train, 'GRAD_ACCUM_STEPS', 1)
        monkeypatch.setattr(train, 'SAVE_EVERY', 1)
        monkeypatch.setattr(train, 'SAFE_DIVISOR', float(TINY_BATCH * train.CONTEXT_LENGTH))

        class DummyOptimizer:
            def __init__(self, lr=1e-4):
                self.param_groups = [{"lr": lr}]

            def zero_grad(self):
                return None

            def step(self, closure=None):
                if closure is not None:
                    closure()
                return None

            def state_dict(self):
                return {"state": {}, "param_groups": self.param_groups}

        class DummyScheduler:
            def get_lr_and_ctx(self, _step):
                return 1e-4, SEQ_LEN, "Phase_1"

            def step(self, _step):
                return 1e-4, SEQ_LEN, "Phase_1"

        p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, requires_grad=True))
        model = MagicMock()
        model.parameters.return_value = [p]
        model.state_dict.return_value = {"dummy": torch.tensor([1.0])}
        model.return_value = (
            torch.tensor(1.0, dtype=torch.float32, requires_grad=True),
            torch.tensor(float(TINY_BATCH * SEQ_LEN), dtype=torch.float32),
        )

        muon_opt = DummyOptimizer()
        adam_opt = DummyOptimizer()
        mamba_core_opt = DummyOptimizer()
        scheduler = DummyScheduler()

        scaler = torch.amp.GradScaler(enabled=False)

        def make_batch():
            x = torch.randint(0, 64, (TINY_BATCH, SEQ_LEN))
            y = torch.randint(0, 64, (TINY_BATCH, SEQ_LEN))
            cu = torch.tensor([[0, SEQ_LEN, -1], [0, SEQ_LEN, -1]], dtype=torch.int32)
            n_segs = torch.tensor([2, 2], dtype=torch.int32)
            return x, y, cu, n_segs

        def infinite_iter():
            while True:
                yield make_batch()

        mock_loader = MagicMock()
        mock_loader.__iter__ = MagicMock(side_effect=infinite_iter)
        mock_create_dl.return_value = (mock_loader, MagicMock())

        # wandb.run must be None so wandb.run.id doesn't return an unpicklable MagicMock
        mock_wandb.run = None

        train.run_training_steps(
            model=model,
            raw_model=model,
            optimizers=(muon_opt, adam_opt, mamba_core_opt),
            scheduler=scheduler,
            train_loader=mock_loader,
            scaler=scaler,
            total_steps=TOTAL,
            checkpoint_dir=str(tmp_path),
            device='cpu',
            world_size=1,
        )

        ckpt_files = list(tmp_path.glob("*.pt"))
        assert len(ckpt_files) == 1, f"Expected 1 checkpoint, found: {ckpt_files}"

        # weights_only=False required: optimizer states contain complex Python objects
        ckpt = torch.load(ckpt_files[0], map_location='cpu', weights_only=False)
        assert 'model_state_dict' in ckpt
        assert 'step' in ckpt
        assert 'muon_opt_state' in ckpt
        assert 'adam_opt_state' in ckpt
        assert 'mamba_core_opt_state' in ckpt
        assert 'scaler_state' in ckpt
        assert 'total_tokens' in ckpt
        assert ckpt.get('wandb_run_id') is None  # wandb.run was None, so this should be None