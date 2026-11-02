"""Tests phase 2 : représentation d'état, masque d'options, environnements durcis."""

from __future__ import annotations

import unittest
from random import Random

from rl_arena.arena_api import ObjectKind

from agent.features import State, encode_state, valid_options
from agent.perception import Memory
from agent.options import make_options
from tests.test_agent import _random_observation
from training.envs import SCENARIOS, STAGES, STUDENT_ID, build_env, sample_curriculum_spec


class FeatureTests(unittest.TestCase):
    def test_encode_state_ranges_and_key(self) -> None:
        rng = Random(1)
        mem = Memory()
        opts = make_options(rng)
        for t in range(1000):
            obs = _random_observation(rng, t)
            mem.update(obs)
            st = encode_state(obs, mem)
            self.assertIsInstance(st, State)
            for v, hi in zip(st, (2, 2, 2, 1, 1, 3, 2, 1, 1, 1, 1, 1)):
                self.assertTrue(0 <= int(v) <= hi)
            self.assertEqual(len(st.key()), len(State._fields))
            self.assertEqual(State(*map(int, st.key())), State(*map(int, st)))
            valid = valid_options(obs, mem, opts)
            self.assertIn("EXPLORE", valid)
            if not any(i.kind in (ObjectKind.FOOD, ObjectKind.DRINK, ObjectKind.MEDICINE)
                       for i in obs.self_state.inventory):
                self.assertNotIn("CONSUME", valid)


class EnvironmentTests(unittest.TestCase):
    def test_all_scenarios_and_stages_build_and_step(self) -> None:
        rng = Random(0)
        specs = [sc.spec(5) for sc in SCENARIOS.values()]
        specs += [st.sample_spec(rng, 7) for st in STAGES]
        specs.append(sample_curriculum_spec(3, rng, 9))
        for spec in specs:
            env = build_env(spec)  # agent externe
            obs = env.reset(seed=11)
            self.assertIn(STUDENT_ID, obs)
            for _ in range(20):
                res = env.step({})
                if res.truncated:
                    break

    def test_loadout_equips_opponents(self) -> None:
        env = build_env(SCENARIOS["hostile"].spec(0))
        env.reset(seed=0)
        hunters = env.debug_state()["observations"]
        launcher = hunters["hunter0"].self_state
        self.assertIsNotNone(launcher.equipped_slot)
        self.assertTrue(any(i.kind is ObjectKind.AMMUNITION for i in launcher.inventory))

    def test_scenarios_are_deterministic(self) -> None:
        def run():
            env = build_env(SCENARIOS["hostile"].spec(3))
            env.reset(seed=3)
            for _ in range(60):
                res = env.step({})
            return res.observations[STUDENT_ID].self_state.health if STUDENT_ID in res.observations else -1

        self.assertEqual(run(), run())


if __name__ == "__main__":
    unittest.main()
