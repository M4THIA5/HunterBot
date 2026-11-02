from rl_arena.environment import TrainingEnvironment
from rl_arena.render_arcade import run_arcade


def main():
    env = TrainingEnvironment(render=True)
    env.add_default_bots()
    env.reset(seed=42)
    run_arcade(env)


if __name__ == "__main__":
    main()
