"""Tests de l'agent scripté (phase 1) : contrat public, robustesse, exécution."""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from random import Random

from rl_arena.arena_api import (
    Action,
    InventoryItem,
    ObjectKind,
    Observation,
    SelfState,
    SoundDetection,
    SoundKind,
    TouchContact,
    VisualDetection,
)
from rl_arena.environment import TrainingEnvironment

from agent.scripted import ScriptedAgent
from student_agent import MyAgent

ROOT = Path(__file__).resolve().parents[1]
AGENT_FILES = [ROOT / "student_agent.py", *sorted((ROOT / "agent").glob("*.py"))]


class ContractImportTests(unittest.TestCase):
    def test_agent_imports_only_public_contract(self) -> None:
        stdlib = set(sys.stdlib_module_names)
        for path in AGENT_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.level > 0:
                        continue
                    mod = node.module or ""
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".")[0]
                        self.assertTrue(root in stdlib or root == "agent", f"{path.name}: import {alias.name}")
                    continue
                else:
                    continue
                root = mod.split(".")[0]
                allowed = mod == "rl_arena.arena_api" or root == "agent" or root in stdlib
                self.assertTrue(allowed, f"{path.name}: from {mod} import ...")


def _random_observation(rng: Random, tick: int) -> Observation:
    kinds = list(ObjectKind)
    vision = tuple(
        VisualDetection(
            ref=i + 1,
            kind=rng.choice(kinds),
            distance=rng.uniform(0.0, 12.0),
            bearing=rng.uniform(-1.05, 1.05),
            angular_size=rng.uniform(0.0, 3.1),
            properties=rng.choice([{}, {"open": rng.random() < 0.5}, {"label": "x"}, {"armed": True}]),
        )
        for i in range(rng.randint(0, 12))
    )
    inventory = tuple(
        InventoryItem(slot, rng.choice(kinds), rng.uniform(0.1, 3.0),
                      rng.choice([{}, {"weapon_type": "potato_launcher"}, {"hydration": "?"}, {"heal": None}]))
        for slot in range(rng.randint(0, 5))
    )
    s = SelfState(
        health=rng.random(), energy=rng.random(), hydration=rng.random(), satiety=rng.random(),
        body_mass=70.0, carried_mass=2.0, speed=0.0, orientation=rng.uniform(-3.14, 3.14),
        injuries=(), inventory=inventory,
        equipped_slot=rng.choice([None, *[i.slot for i in inventory]]),
    )
    hearing = tuple(
        SoundDetection(rng.choice(list(SoundKind)), rng.random(), rng.uniform(-3.14, 3.14), 0.3,
                       rng.choice([None, rng.uniform(0, 20)]), 0.1)
        for _ in range(rng.randint(0, 3))
    )
    touch = tuple(TouchContact(100 + i, rng.choice(kinds), rng.uniform(-3.14, 3.14)) for i in range(rng.randint(0, 2)))
    return Observation(tick, s, vision, hearing, touch, ())


class RobustnessTests(unittest.TestCase):
    def test_scripted_agent_survives_degraded_observations(self) -> None:
        rng = Random(0)
        agent = ScriptedAgent()
        agent.reset()
        for t in range(2000):
            action = agent.act(_random_observation(rng, t))  # ne doit pas lever
            self.assertIsInstance(action, Action)


class PureQAgentTests(unittest.TestCase):
    def test_pure_agent_never_calls_scripted_policy(self) -> None:
        from unittest import mock

        from agent import qpolicy
        from agent.qpolicy import QAgent, QTable

        table = QTable()
        table.meta["pure"] = True
        q, n = table.ensure("2" * 14)
        q[0], n[0] = 1.0, 3
        rng = Random(1)
        with mock.patch.object(qpolicy, "scripted_choice", side_effect=AssertionError("scripted appelé")):
            for eps in (0.0, 0.3):
                agent = QAgent(table, epsilon=eps, seed=2, explore_scripted=1.0, prior_bonus=5.0)
                self.assertTrue(agent.pure)
                agent.reset()
                for t in range(300):
                    self.assertIsInstance(agent.act(_random_observation(rng, t)), Action)
                self.assertEqual(agent.prior_bonus, 0.0)

    def test_nearest_greedy_uses_closest_known_state(self) -> None:
        from agent.qpolicy import QTable

        table = QTable(n_options=3)
        q, n = table.ensure("000")
        q[:], n[:] = [0.0, 5.0, 1.0], [1, 1, 1]
        q, n = table.ensure("222")
        q[:], n[:] = [9.0, 0.0, 0.0], [1, 1, 1]
        self.assertEqual(table.nearest_greedy("001", [0, 1, 2]), 1)
        self.assertEqual(table.nearest_greedy("122", [0, 1, 2]), 0)
        self.assertEqual(table.nearest_greedy("001", [0, 2]), 2)
        self.assertIsNone(QTable().nearest_greedy("001", [0]))


class EpisodeTests(unittest.TestCase):
    def test_my_agent_runs_full_default_episode(self) -> None:
        env = TrainingEnvironment()
        env.register_agent("student", MyAgent())
        env.add_default_bots()
        env.reset(seed=3)
        while True:
            res = env.step({})
            if res.truncated or res.terminated["student"]:
                break
        self.assertGreater(res.info["student"].survival_ticks, 0)


if __name__ == "__main__":
    unittest.main()
