"""Reproductibilité entre processus : le résultat ne doit pas dépendre de PYTHONHASHSEED."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Seed d'évaluation sensible à l'ordre de hachage avant la correction de
# rl_arena/environment.py (collided_static itéré sans tri).
SCRIPT = (
    "from training.evaluate import run_episode\n"
    "print(run_episode('scripted_nomap', 'default', 10000033).survival)\n"
)


def _run(hash_seed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    out = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, env=env, capture_output=True, text=True,
                         check=True, timeout=300)
    return out.stdout.strip()


class CrossProcessReproducibilityTests(unittest.TestCase):
    def test_episode_independent_of_hash_seed(self) -> None:
        results = {h: _run(h) for h in ("0", "2", "4")}
        self.assertEqual(len(set(results.values())), 1, results)


if __name__ == "__main__":
    unittest.main()
