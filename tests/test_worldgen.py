"""Cartes procédurales (training/worldgen.py) et extensions du moteur d'entraînement."""

from __future__ import annotations

import unittest
from random import Random

from rl_arena.arena_api import Action, Movement, ObjectKind, Shoot
from rl_arena.geometry import Rect
from rl_arena.world import AreaEffect, InventoryEntry

from training.envs import MAP_SCENARIOS, SCENARIOS, build_env
from training.opponents import ArenaEnvironment
from training.worldgen import CATALOG, PROFILES, generate


def _entry(env, name: str) -> InventoryEntry:
    tpl = CATALOG[name]
    return InventoryEntry(env._alloc_uid(), tpl.kind, tpl.mass, dict(tpl.props))


def _duel(distance: float = 1.2) -> ArenaEnvironment:
    """Deux agents externes face à face, sans carte autour."""
    from rl_arena.generation import GeneratedWorld

    env = ArenaEnvironment(world_generator=lambda cfg, rng: GeneratedWorld())
    env.register_agent("a")
    env.register_agent("b")
    env.reset(seed=1)
    a, b = env._runtimes["a"].state, env._runtimes["b"].state
    a.x, a.y, a.orientation = 10.0, 10.0, 0.0
    b.x, b.y = 10.0 + distance, 10.0
    env._last_observations = env._observe_all()
    return env


class WorldgenTests(unittest.TestCase):
    def test_every_profile_generates_and_hosts_eight_agents(self):
        for name, prof in PROFILES.items():
            for seed in range(5):
                rng = Random(seed)
                cfg = prof.config(rng)
                world = generate(prof, cfg, rng)
                self.assertGreater(sum(o.portable for o in world.objects), 0, name)
                env = ArenaEnvironment(cfg, world_generator=prof.generator())
                for i in range(8):
                    env.register_agent(f"a{i}")
                env.reset(seed=seed)  # lève si moins de 8 points d'apparition libres

    def test_only_kit_items_are_generated(self):
        kit = {"sandwich", "fruit", "bottle", "water_bottle", "bandage", "medicine", "rolling_pin",
               "potato_launcher", "potato", "reinforced_spring", "wood", "metal", "cloth", "plastic"}
        self.assertLessEqual(set(CATALOG), kit)
        for name, prof in PROFILES.items():
            rng = Random(0)
            world = generate(prof, prof.config(rng), rng)
            labels = {o.properties.get("label") for o in world.objects if o.portable}
            self.assertLessEqual(labels, kit, name)

    def test_generation_is_deterministic(self):
        prof = PROFILES["town"]
        w1 = generate(prof, prof.config(Random(3)), Random(4))
        w2 = generate(prof, prof.config(Random(3)), Random(4))
        self.assertEqual([(o.kind, o.x, o.y) for o in w1.objects], [(o.kind, o.x, o.y) for o in w2.objects])

    def test_map_scenarios_run(self):
        from agent.scripted import ScriptedAgent

        for name in MAP_SCENARIOS:
            env = build_env(SCENARIOS[name].spec(7), student=ScriptedAgent())
            env.reset(seed=7)
            for _ in range(30):
                env.step({})


class EngineExtensionTests(unittest.TestCase):
    def test_kit_weapons_still_work(self):
        env = _duel(4.0)
        st = env._runtimes["a"].state
        st.inventory[0] = _entry(env, "potato_launcher")
        st.inventory[1] = _entry(env, "potato")
        st.equipped_slot = 0
        env.step({"a": Action(interaction=Shoot(0.0))})
        self.assertNotIn(1, st.inventory)

    def test_hazard_damages_and_ice_slows(self):
        env = _duel(5.0)
        env.world.areas.append(AreaEffect(999, ObjectKind.HAZARD, Rect(10.0, 10.0, 3.0, 3.0), {"damage_per_tick": 0.02}))
        env.step({})
        self.assertLess(env._runtimes["a"].state.health, 0.985)  # 0.02 de dégâts, 0.0015 de soin passif

        env = _duel(5.0)
        env.world.areas.append(AreaEffect(999, ObjectKind.ICE, Rect(10.0, 10.0, 6.0, 6.0), {"speed_factor": 0.5}))
        env.step({"a": Action(movement=Movement(1.0, 0.0, 3.2))})
        self.assertLessEqual(env._runtimes["a"].state.speed, 1.6 + 1e-6)



if __name__ == "__main__":
    unittest.main()
