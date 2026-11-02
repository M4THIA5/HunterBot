"""Lance votre agent contre les deux bots de référence.

Usage :
    python run_student_agent.py          # mode headless
    python run_student_agent.py --gui    # visualisation Arcade
    python run_student_agent.py --seed 7
"""

from __future__ import annotations

import argparse

from rl_arena.environment import TrainingEnvironment
from student_agent import MyAgent


def build_env(gui: bool, seed: int) -> TrainingEnvironment:
    env = TrainingEnvironment(render=gui)
    env.register_agent("student", MyAgent())
    env.add_default_bots()
    env.reset(seed=seed)
    return env


def run_headless(env: TrainingEnvironment) -> None:
    result = None
    while True:
        result = env.step({})
        if result.truncated or result.terminated.get("student", False):
            break

    assert result is not None
    info = result.info["student"]
    print("Episode terminé")
    print(f"  survie : {info.survival_ticks} ticks ({info.survival_ticks * 0.5:.1f} s simulées)")
    print(f"  reward cumulé : {info.cumulative_reward:.2f}")
    print(f"  kills : {info.kills}")
    print(f"  pickups : {info.pickups}")
    print(f"  dégâts reçus : {info.damage_received:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="ouvrir l'interface Arcade")
    parser.add_argument("--seed", type=int, default=42, help="seed de l'épisode")
    args = parser.parse_args()

    env = build_env(args.gui, args.seed)
    if args.gui:
        from rl_arena.render_arcade import run_arcade
        run_arcade(env)
    else:
        run_headless(env)


if __name__ == "__main__":
    main()
