"""End-to-end run of every script on tiny random models and fixture data (CPU, ~1-2 min)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    env = {**os.environ, "TOKENIZERS_PARALLELISM": "false"}
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True, env=env)


@pytest.mark.slow
def test_full_pipeline():
    c = ["--config", "configs/smoke.yaml"]
    run("tests/make_tiny_models.py", ".smoke/tiny")
    run("scripts/01_prepare_data.py", *c, "--no-download")
    run("scripts/02_train_recogniser.py", *c)
    run("scripts/03_cache_emotions.py", *c)
    run("scripts/04_train_generator.py", *c, "--stage", "base")
    for v in ("b0", "b1", "etg_nodelta", "etg_nogate", "etg"):
        run("scripts/04_train_generator.py", *c, "--variant", v)
        run("scripts/05_generate.py", *c, "--variant", v, "--swap")
    run("scripts/06_evaluate.py", *c)
    run("scripts/07_human_eval.py", *c, "make")
    assert (ROOT / ".smoke/outputs/results/summary.md").exists()
