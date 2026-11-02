"""Configurations d'environnement d'entraînement / d'évaluation.

Ce module modifie librement l'environnement de test (autorisé par le contrat).
Il ne doit JAMAIS être importé par le code de décision de l'agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from rl_arena.bots import HunterBot, RandomBot
from rl_arena.config import EnvironmentConfig, GenerationConfig, PhysiologyConfig

LONG_EPISODE = 3000

OpponentFactory = Callable[[int], list[tuple[str, object]]]


def _default_opponents(seed: int) -> list[tuple[str, object]]:
    return [("bot_random", RandomBot(seed=101 + seed)), ("bot_hunter", HunterBot(seed=202 + seed))]


def _hostile_opponents(seed: int) -> list[tuple[str, object]]:
    return [(f"hunter{i}", HunterBot(seed=1000 * i + seed)) for i in range(4)]


def _blind_opponents(seed: int) -> list[tuple[str, object]]:
    return [("bot_random", RandomBot(seed=101 + seed))] + [
        (f"hunter{i}", HunterBot(seed=1000 * i + seed)) for i in range(2)
    ]


@dataclass(slots=True)
class Scenario:
    name: str
    config_factory: Callable[[], EnvironmentConfig]
    opponents: OpponentFactory = field(default=_default_opponents)


def _default_cfg() -> EnvironmentConfig:
    return EnvironmentConfig(max_ticks=LONG_EPISODE)


def _hostile_cfg() -> EnvironmentConfig:
    return EnvironmentConfig(max_ticks=LONG_EPISODE)


def _scarce_cfg() -> EnvironmentConfig:
    return EnvironmentConfig(
        max_ticks=LONG_EPISODE,
        generation=GenerationConfig(item_count=(4, 8)),
        physiology=PhysiologyConfig(hydration_decay_per_tick=0.0009, satiety_decay_per_tick=0.0006),
    )


def _blind_cfg() -> EnvironmentConfig:
    return EnvironmentConfig(
        max_ticks=LONG_EPISODE,
        generation=GenerationConfig(fog_probability=1.0, low_light_probability=1.0),
    )


def _big_map_cfg() -> EnvironmentConfig:
    return EnvironmentConfig(max_ticks=LONG_EPISODE, map_width=50.0, map_height=50.0)


SCENARIOS: dict[str, Scenario] = {
    "default": Scenario("default", _default_cfg),
    "hostile": Scenario("hostile", _hostile_cfg, _hostile_opponents),
    "scarce": Scenario("scarce", _scarce_cfg),
    "blind": Scenario("blind", _blind_cfg, _blind_opponents),
    "big_map": Scenario("big_map", _big_map_cfg),
}

# Seeds d'évaluation : disjointes des seeds d'entraînement [0, 1_000_000).
EVAL_SEEDS = range(10_000_000, 10_000_200)
