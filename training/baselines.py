"""Agents de référence pour comparer nos politiques."""

from __future__ import annotations

from rl_arena.arena_api import Action, Movement, Observation, Rotation
from rl_arena.bots import HunterBot, RandomBot


class TrivialAgent:
    """Copie de l'agent fourni dans le kit : avance tout droit à 1 m/s."""

    def reset(self) -> None:
        pass

    def act(self, observation: Observation) -> Action:
        return Action(movement=Movement(1.0, 0.0, 1.0), rotation=Rotation(0.0))

    def on_episode_end(self, result) -> None:
        pass


def make_agent(name: str):
    if name == "trivial":
        return TrivialAgent()
    if name == "random_bot":
        return RandomBot(seed=7)
    if name == "hunter_bot":
        return HunterBot(seed=7)
    if name == "scripted":
        from agent.scripted import ScriptedAgent

        return ScriptedAgent()
    if name.startswith("scripted_map_"):
        # Ablation : scripted_map_items, scripted_map_doors, scripted_map_novelty...
        from agent.scripted import ScriptedAgent

        return ScriptedAgent(spatial_uses=frozenset(name.removeprefix("scripted_map_").split("+")))
    if name == "scripted_noblind":
        from agent.scripted import ScriptedAgent

        return ScriptedAgent(use_blind_hit=False)
    if name.startswith("scripted_commit_"):
        # scripted_commit_off / _always / _serious (+ suffixe _noblind)
        from agent.scripted import ScriptedAgent

        mode = name.removeprefix("scripted_commit_").removesuffix("_noblind")
        return ScriptedAgent(flee_commit=mode, use_blind_hit=not name.endswith("_noblind"))
    if name == "scripted_nomap":
        from agent.scripted import ScriptedAgent

        return ScriptedAgent(use_spatial=False)
    if name == "qlearned":
        # Q-table : variable d'environnement RL_QTABLE, sinon agent/q_table.json.
        import os
        from pathlib import Path

        from agent.qpolicy import DEFAULT_TABLE, QAgent, QTable

        return QAgent(QTable.load(Path(os.environ.get("RL_QTABLE", DEFAULT_TABLE))))
    if name.startswith("hybrid"):
        # hybrid, hybrid_<gate>, hybrid_<gate>_<hold> (Q-table : RL_QTABLE).
        import os
        from pathlib import Path

        from agent.hybrid import HybridAgent
        from agent.qpolicy import DEFAULT_TABLE, QTable

        parts = name.split("_")
        gate = parts[1] if len(parts) > 1 else "seen"
        hold = int(parts[2]) if len(parts) > 2 else 0
        return HybridAgent(QTable.load(Path(os.environ.get("RL_QTABLE", DEFAULT_TABLE))), gate, hold)
    if name == "student":
        from student_agent import MyAgent

        return MyAgent()
    raise ValueError(f"agent inconnu : {name}")
