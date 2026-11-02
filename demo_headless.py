from rl_arena.environment import TrainingEnvironment


def main():
    env = TrainingEnvironment()
    env.add_default_bots()
    observations = env.reset(seed=42)
    print("Agents:", list(observations))

    for _ in range(120):
        result = env.step({})
        if result.truncated:
            break
        if all(result.terminated.values()):
            break

    print("tick:", env.tick)
    for agent_id, info in result.info.items():
        print(
            f"{agent_id:12s} alive={not result.terminated[agent_id]} "
            f"reward={info.cumulative_reward:7.2f} kills={info.kills} "
            f"pickups={info.pickups} damage={info.damage_received:.3f}"
        )


if __name__ == "__main__":
    main()
