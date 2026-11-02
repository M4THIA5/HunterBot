"""Votre agent.

Vous pouvez modifier librement ce fichier et ajouter vos propres modules.
Pour rester compatible avec l'arène finale, le code de décision doit dépendre
uniquement de rl_arena.arena_api et des données reçues dans Observation.
"""

from rl_arena.arena_api import Action, Movement, Observation, Rotation


class MyAgent:
    def reset(self) -> None:
        """Appelé au début de chaque épisode.

        Réinitialisez ici uniquement la mémoire temporaire de l'épisode.
        """
        pass

    def act(self, observation: Observation) -> Action:
        """Retourne l'action demandée pour les 0,5 s simulées suivantes."""
        # Politique volontairement triviale : avance lentement.
        # À vous de construire représentation d'état, mémoire, Q-table, etc.
        return Action(
            movement=Movement(forward=1.0, right=0.0, speed=1.0),
            rotation=Rotation(angular_speed=0.0),
        )

    def on_episode_end(self, result) -> None:
        """Appelé lorsque l'épisode de l'agent se termine."""
        pass
