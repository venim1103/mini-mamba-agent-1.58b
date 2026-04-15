import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "kaggle_link_datasets.sh"


def _touch_data_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("dummy")


def test_links_data_in_kaggle_mode_with_custom_root(tmp_path):
    kaggle_root = tmp_path / "kaggle_input"
    src = kaggle_root / "datasets/venim1103/mini-mamba-1b58-pretrain-smalls/code/tiny-codes/train.parquet"
    _touch_data_file(src)

    env = os.environ.copy()
    env["KAGGLE_INPUT_ROOT"] = str(kaggle_root)
    env["KAGGLE_KERNEL_RUN_TYPE"] = "Interactive"

    result = subprocess.run(
        ["bash", str(SCRIPT), "--env", "kaggle"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    linked = tmp_path / "local_data/train/code/tiny-codes/train.parquet"
    assert linked.exists()
    assert linked.is_symlink()


def test_links_data_in_colab_mode_with_versioned_layout(tmp_path):
    colab_root = tmp_path / "colab_cache"
    src = (
        colab_root
        / "datasets/venim1103/mini-mamba-1b58-pretrain-smalls/versions/7/code/tiny-codes/data.jsonl"
    )
    _touch_data_file(src)

    env = os.environ.copy()
    env["COLAB_KAGGLEHUB_ROOT"] = str(colab_root)

    result = subprocess.run(
        ["bash", str(SCRIPT), "--env", "colab", "--colab-version", "7"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    linked = tmp_path / "local_data/train/code/tiny-codes/data.jsonl"
    assert linked.exists()
    assert linked.is_symlink()
