"""Squelette minimal d'une boucle d'entraînement RL externe.

Ce fichier montre où se placent :
- la représentation/discrétisation de l'état ;
- le choix de l'action ;
- la mise à jour Q / modèle.

Aucune discrétisation n'est imposée : c'est volontairement votre travail.
"""

from __future__ import annotations

from rl_arena.arena_api import Action, Movement, Observation
from rl_arena.environment import TrainingEnvironment


def state_representation(obs: Observation):
    """À remplacer par VOTRE représentation de l'état."""
    # Exemple volontairement pauvre uniquement pour rendre le script exécutable.
    return (round(obs.self_state.energy, 1), bool(obs.vision))


def choose_action(state) -> Action:
    """À remplacer par epsilon-greedy / politique / modèle."""
    return Action(movement=Movement(forward=1.0, right=0.0, speed=1.2))


def update_model(state, action, reward, next_state, terminated):
    """À remplacer par votre mise à jour Q-learning / approximation."""
    pass


def main() -> None:
    env = TrainingEnvironment()
    env.register_agent("student")  # contrôle externe de cet agent
    env.add_default_bots()

    for episode in range(10):
        observations = env.reset(seed=1000 + episode)
        obs = observations["student"]
        total_reward = 0.0

        while True:
            state = state_representation(obs)
            action = choose_action(state)
            result = env.step({"student": action})
            reward = result.rewards["student"]
            total_reward += reward
            terminated = result.terminated["student"]

            if terminated or result.truncated:
                next_state = None
            else:
                obs_next = result.observations["student"]
                next_state = state_representation(obs_next)

            update_model(state, action, reward, next_state, terminated)

            if terminated or result.truncated:
                break
            obs = obs_next

        print(
            f"episode={episode:02d} "
            f"survival={result.info['student'].survival_ticks:4d} ticks "
            f"reward={total_reward:8.2f}"
        )


if __name__ == "__main__":
    main()
