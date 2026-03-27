import os

from rl_train import collect_data_files


def test_collect_data_files_follows_symlinked_directories(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    parquet_file = source_dir / "sample.parquet"
    parquet_file.write_text("dummy")

    rl_root = tmp_path / "rl"
    rl_root.mkdir()
    symlink_dir = rl_root / "reasoning"
    symlink_dir.symlink_to(source_dir, target_is_directory=True)

    files = collect_data_files(str(rl_root))

    assert len(files) == 1
    assert os.path.basename(files[0]) == "sample.parquet"