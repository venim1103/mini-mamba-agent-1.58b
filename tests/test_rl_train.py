import os
import pytest

from rl_train import (
    collect_data_files,
    compute_format_reward,
    compute_accuracy_reward,
    compute_conciseness_penalty,
    compute_rewards,
)


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


class TestComputeFormatReward:
    def test_complete_with_think_and_answer(self):
        completion = "<think>Let me think about this.</think>\nThe answer is 42."
        assert compute_format_reward(completion) == 1.0

    def test_no_tags(self):
        completion = "The answer is 42."
        assert compute_format_reward(completion) == 0.0

    def test_tags_present_but_empty_answer(self):
        completion = "<think>Let me think about this.</think>"
        assert compute_format_reward(completion) == 0.5

    def test_tags_present_but_no_closing_tag(self):
        completion = "<think>Just thinking"
        assert compute_format_reward(completion) == 0.0


class TestComputeAccuracyReward:
    def test_ground_truth_in_final_answer(self):
        completion = "<think>thinking.</think>\nThe answer is 42"
        assert compute_accuracy_reward(completion, "42") == 2.0

    def test_case_insensitive_match(self):
        completion = "<think>thinking.</think>\nThe answer is FORTY TWO"
        assert compute_accuracy_reward(completion, "forty two") == 2.0

    def test_no_closing_tag(self):
        completion = "<think>thinking"
        assert compute_accuracy_reward(completion, "42") == 0.0

    def test_no_match(self):
        completion = "<think>thinking.</think>\nThe answer is 99"
        assert compute_accuracy_reward(completion, "42") == 0.0


class TestComputeConcisenessPenalty:
    def test_verbose_thinking_penalized(self):
        completion = "<think>" + "x" * 1000 + "</think>\nhi"
        assert compute_conciseness_penalty(completion) == -0.5

    def test_concise_thinking_no_penalty(self):
        completion = "<think>short.</think>\nhi"
        assert compute_conciseness_penalty(completion) == 0.0

    def test_no_tags_no_penalty(self):
        completion = "just a short answer"
        assert compute_conciseness_penalty(completion) == 0.0


class TestComputeRewards:
    def test_combines_all_rewards(self):
        completions = ["<think>thinking.</think>\n42 is the answer"]
        ground_truth = "42"
        rewards = compute_rewards(completions, ground_truth)
        assert rewards.shape == (1,)
        assert rewards[0].item() > 0

    def test_empty_completions(self):
        rewards = compute_rewards([], "answer")
        assert len(rewards) == 0