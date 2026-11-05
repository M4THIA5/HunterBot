"""Votre agent.

Vous pouvez modifier librement ce fichier et ajouter vos propres modules.
Pour rester compatible avec l'arène finale, le code de décision doit dépendre
uniquement de rl_arena.arena_api et des données reçues dans Observation.

Politique livrée : Q-table seule (agent/qpolicy.py, mode `pure`), apprise par
Q-learning SMDP sur les options (agent/options.py). Aucune décision n'est
déléguée à la politique scriptée : les états jamais visités prennent la
décision de l'état connu le plus proche dans la table.
"""

from rl_arena.arena_api import Action, Observation

from agent.qpolicy import QAgent, QTable, load_default_agent


class MyAgent:
    def __init__(self) -> None:
        # Table absente/illisible : table vide (option EXPLORE partout), jamais le scripté.
        self._policy = load_default_agent(pure=True) or QAgent(QTable(), pure=True)

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
