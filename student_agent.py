"""Votre agent.

Vous pouvez modifier librement ce fichier et ajouter vos propres modules.
Pour rester compatible avec l'arène finale, le code de décision doit dépendre
uniquement de rl_arena.arena_api et des données reçues dans Observation.

Phase 1 : politique scriptée sur options (agent/scripted.py). Les phases
suivantes remplaceront le choix d'option par une Q-table apprise, la
politique scriptée restant le repli pour les états inconnus.
"""

from rl_arena.arena_api import Action, Observation

from agent.scripted import ScriptedAgent


class MyAgent:
    def __init__(self) -> None:
        self._policy = ScriptedAgent()

    def reset(self) -> None:
        """Appelé au début de chaque épisode : réinitialise la mémoire d'épisode."""
        self._policy.reset()

    def act(self, observation: Observation) -> Action:
        """Retourne l'action demandée pour les 0,5 s simulées suivantes."""
        try:
            return self._policy.act(observation)
        except Exception:
            # Une exception ne doit jamais coûter l'épisode dans l'arène finale.
            return Action(rest=True)

    def on_episode_end(self, result) -> None:
        """Appelé lorsque l'épisode de l'agent se termine."""
        self._policy.on_episode_end(result)
