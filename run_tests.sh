#!/bin/sh
# Skip tests that fail due to datasets/huggingface_hub import issues in CI
coverage erase
coverage run --branch --source=. -m pytest tests --ignore=tests/test_data.py --ignore=tests/test_rl_train.py --ignore=tests/test_sft_data.py
coverage html -i