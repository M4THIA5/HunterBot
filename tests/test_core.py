import unittest
from math import hypot, pi

from rl_arena.arena_api import Action, Movement, Pickup
from rl_arena.config import EnvironmentConfig
from rl_arena.environment import TrainingEnvironment
from rl_arena.geometry import torus_distance, torus_vector


class CoreEnvironmentTests(unittest.TestCase):
    def make_env(self, max_ticks=20):
        cfg = EnvironmentConfig(max_ticks=max_ticks)
        env = TrainingEnvironment(cfg)
        env.register_agent("a")
        env.register_agent("b")
        return env

    def test_reset_is_deterministic(self):
        env1 = self.make_env()
        obs1 = env1.reset(seed=123)
        state1 = env1.debug_state()

        env2 = self.make_env()
        obs2 = env2.reset(seed=123)
        state2 = env2.debug_state()

        self.assertEqual(obs1, obs2)
        for aid in ("a", "b"):
            a = state1["agents"][aid]
            b = state2["agents"][aid]
            self.assertAlmostEqual(a.x, b.x)
            self.assertAlmostEqual(a.y, b.y)
            self.assertAlmostEqual(a.orientation, b.orientation)

    def test_same_actions_same_trajectory(self):
        env1 = self.make_env(max_ticks=10)
        env2 = self.make_env(max_ticks=10)
        env1.reset(seed=9)
        env2.reset(seed=9)

        for _ in range(6):
            r1 = env1.step({"a": Action(), "b": Action()})
            r2 = env2.step({"a": Action(), "b": Action()})
            self.assertEqual(r1.observations, r2.observations)
            self.assertEqual(r1.rewards, r2.rewards)
            self.assertEqual(r1.terminated, r2.terminated)

    def test_truncation(self):
        env = self.make_env(max_ticks=3)
        env.reset(seed=1)
        result = None
        for _ in range(3):
            result = env.step({"a": Action(), "b": Action()})
        self.assertIsNotNone(result)
        self.assertTrue(result.truncated)
        self.assertEqual(env.tick, 3)
        self.assertFalse(any(result.terminated.values()))


    def test_pickup_uses_ephemeral_ref_and_rewards_once(self):
        env = TrainingEnvironment(EnvironmentConfig(max_ticks=10))
        env.register_agent("a")
        env.reset(seed=4)
        obj = next(o for o in env.world.objects if o.portable)
        st = env.debug_state()["agents"]["a"]
        st.x, st.y = obj.x, obj.y
        env._last_observations = env._observe_all()
        obs = env._last_observations["a"]
        det = next(d for d in obs.vision if d.ref in env._ref_maps["a"] and env._ref_maps["a"][d.ref] == obj.uid)
        result = env.step({"a": Action(interaction=Pickup(det.ref))})
        self.assertEqual(result.info["a"].pickups, 1)
        self.assertGreaterEqual(result.rewards["a"], env.config.reward.first_pickup)
        self.assertEqual(len(env.debug_state()["agents"]["a"].inventory), 1)

    def test_agent_collision_removes_inward_velocity(self):
        env = self.make_env(max_ticks=10)
        env.reset(seed=1)
        a = env.debug_state()["agents"]["a"]
        b = env.debug_state()["agents"]["b"]
        a.x, a.y, a.orientation = 14.2, 12.0, 0.0
        b.x, b.y, b.orientation = 15.8, 12.0, pi
        env._last_observations = env._observe_all()

        env.step({
            "a": Action(movement=Movement(1.0, 0.0, 3.0)),
            "b": Action(movement=Movement(1.0, 0.0, 3.0)),
        })

        a = env.debug_state()["agents"]["a"]
        b = env.debug_state()["agents"]["b"]
        distance = torus_distance(a.x, a.y, b.x, b.y, env.config.map_width, env.config.map_height)
        self.assertGreaterEqual(distance, a.radius + b.radius - 1e-3)

        dx, dy = torus_vector(a.x, a.y, b.x, b.y, env.config.map_width, env.config.map_height)
        norm = hypot(dx, dy)
        ux, uy = dx / norm, dy / norm
        closing_speed = (a.vx * ux + a.vy * uy) - (b.vx * ux + b.vy * uy)
        self.assertLessEqual(closing_speed, 1e-6)

    def test_default_bots_run(self):
        env = TrainingEnvironment(EnvironmentConfig(max_ticks=50))
        env.add_default_bots()
        obs = env.reset(seed=42)
        self.assertEqual(set(obs), {"bot_random", "bot_hunter"})
        for _ in range(5):
            result = env.step({})
        self.assertEqual(env.tick, 5)
        self.assertIn("bot_random", result.info)


if __name__ == "__main__":
    unittest.main()
