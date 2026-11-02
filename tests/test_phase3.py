"""Tests phase 3 : Q-table, agent appris, mise à jour SMDP."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from random import Random

from agent.options import OPTION_NAMES
from agent.qpolicy import QAgent, QTable, Transition
from tests.test_agent import _random_observation
from training.train import GAMMA, Learner, play_episode


class QTableTests(unittest.TestCase):
    def test_save_load_roundtrip_and_greedy_masking(self) -> None:
        t = QTable()
        q, n = t.ensure("abc")
        q[0], q[3], n[0], n[3] = 1.0, 5.0, 2, 1
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "q.json"
            t.save(path)
            t2 = QTable.load(path)
        self.assertEqual(t2.q["abc"], t.q["abc"])
        self.assertEqual(t2.greedy("abc", [0, 3]), 3)
        self.assertEqual(t2.greedy("abc", [0, 1]), 0)      # 3 masquée
        self.assertIsNone(t2.greedy("abc", [1, 2]))       # jamais essayées
        self.assertIsNone(t2.greedy("inconnu", [0]))


class LearnerTests(unittest.TestCase):
    def test_death_has_no_future_but_truncation_bootstraps(self) -> None:
        lr = Learner(alpha_min=1.0)  # alpha = 1 : Q = cible
        q, n = lr.table.ensure("next")
        q[0], n[0] = 10.0, 1
        lr.update(Transition("s", 1, 2.0, 3, None, (), True))
        self.assertAlmostEqual(lr.table.q["s"][1], 2.0)
        lr.update(Transition("s", 2, 2.0, 3, "next", (0,), False))
        self.assertAlmostEqual(lr.table.q["s"][2], 2.0 + GAMMA ** 3 * 10.0)


class AgentTests(unittest.TestCase):
    def test_qagent_on_degraded_observations(self) -> None:
        rng = Random(2)
        sink: list = []
        agent = QAgent(QTable(), epsilon=0.5, seed=1, transition_sink=sink.append)
        agent.reset()
        for t in range(500):
            agent.act(_random_observation(rng, t))
            agent.record(0.1)
        agent.end_episode(None, terminal=True)
        self.assertTrue(sink)
        for tr in sink:
            self.assertTrue(0 <= tr.option < len(OPTION_NAMES))
            self.assertGreater(tr.duration, 0)
        self.assertTrue(sink[-1].terminal)

    def test_training_episode_produces_transitions(self) -> None:
        out = play_episode(QTable(), 1.0, 0, 123, 0.3)
        self.assertGreater(out.survival, 0)
        self.assertTrue(out.transitions)


if __name__ == "__main__":
    unittest.main()
