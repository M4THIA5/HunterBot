"""Environnements d'entraînement / d'évaluation durcis et curriculum.

Ce module modifie librement l'environnement de test (autorisé par le contrat).
Il ne doit JAMAIS être importé par le code de décision de l'agent.

Deux familles :
- `SCENARIOS` : configurations FIXES (sauf la seed) pour l'évaluation (§8) ;
- `STAGES` : étapes du curriculum (§6.2), avec randomisation de domaine :
  chaque épisode tire sa configuration et ses adversaires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Callable

from rl_arena.bots import HunterBot, RandomBot
from rl_arena.config import EnvironmentConfig, GenerationConfig, PhysiologyConfig

from training.opponents import (
    AggressiveHunter,
    ArenaEnvironment,
    FrozenAgent,
    Loadout,
    launcher_loadout,
    melee_loadout,
)

LONG_EPISODE = 3000
STUDENT_ID = "student"

Opponents = list[tuple[str, object]]


def _scripted_copy() -> FrozenAgent:
    from agent.scripted import ScriptedAgent

    return FrozenAgent(ScriptedAgent)


@dataclass(slots=True)
class EpisodeSpec:
    """Tout ce qu'il faut pour construire un épisode."""

    config: EnvironmentConfig
    opponents: Opponents
    loadouts: dict[str, Loadout] = field(default_factory=dict)


def build_env(spec: EpisodeSpec, student=None, student_loadout: Loadout | None = None) -> ArenaEnvironment:
    """`student=None` : agent contrôlé de l'extérieur (boucle d'entraînement)."""
    loadouts = dict(spec.loadouts)
    if student_loadout:
        loadouts[STUDENT_ID] = student_loadout
    env = ArenaEnvironment(spec.config, loadouts=loadouts)
    env.register_agent(STUDENT_ID, student)
    for aid, bot in spec.opponents:
        env.register_agent(aid, bot)
    return env


# ----------------------------------------------------------------------
# Scénarios d'évaluation (§8)
# ----------------------------------------------------------------------
@dataclass(slots=True)
class Scenario:
    name: str
    make: Callable[[int], EpisodeSpec]  # seed -> spec (déterministe)
    description: str = ""

    def spec(self, seed: int) -> EpisodeSpec:
        return self.make(seed)


def _default(seed: int) -> EpisodeSpec:
    return EpisodeSpec(
        EnvironmentConfig(max_ticks=LONG_EPISODE),
        [("bot_random", RandomBot(seed=101 + seed)), ("bot_hunter", HunterBot(seed=202 + seed))],
    )


def _hostile(seed: int) -> EpisodeSpec:
    # 4 chasseurs armés d'office (2 à distance, 2 au contact) + 2 copies figées.
    opponents: Opponents = [(f"hunter{i}", AggressiveHunter(seed=1000 * i + seed)) for i in range(4)]
    opponents += [(f"copy{i}", _scripted_copy()) for i in range(2)]
    loadouts = {"hunter0": launcher_loadout(), "hunter1": launcher_loadout(), "hunter2": melee_loadout(),
                "hunter3": melee_loadout()}
    return EpisodeSpec(EnvironmentConfig(max_ticks=LONG_EPISODE), opponents, loadouts)


def _hostile_v1(seed: int) -> EpisodeSpec:
    # Ancienne version (phase 1) : 4 HunterBot non armés.
    return EpisodeSpec(EnvironmentConfig(max_ticks=LONG_EPISODE),
                       [(f"hunter{i}", HunterBot(seed=1000 * i + seed)) for i in range(4)])


def _scarce(seed: int) -> EpisodeSpec:
    spec = _default(seed)
    spec.config = EnvironmentConfig(
        max_ticks=LONG_EPISODE,
        generation=GenerationConfig(item_count=(4, 8)),
        physiology=PhysiologyConfig(hydration_decay_per_tick=0.0009, satiety_decay_per_tick=0.0006),
    )
    return spec


def _blind(seed: int) -> EpisodeSpec:
    opponents: Opponents = [("bot_random", RandomBot(seed=101 + seed))]
    opponents += [(f"hunter{i}", AggressiveHunter(seed=1000 * i + seed)) for i in range(2)]
    return EpisodeSpec(
        EnvironmentConfig(max_ticks=LONG_EPISODE,
                          generation=GenerationConfig(fog_probability=1.0, low_light_probability=1.0)),
        opponents,
        {"hunter0": launcher_loadout()},
    )


def _big_map(seed: int) -> EpisodeSpec:
    spec = _default(seed)
    spec.config = EnvironmentConfig(max_ticks=LONG_EPISODE, map_width=50.0, map_height=50.0)
    return spec


SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in (
        Scenario("default", _default, "kit fourni, max_ticks=3000"),
        Scenario("hostile", _hostile, "4 AggressiveHunter armés + 2 copies scriptées"),
        Scenario("hostile_v1", _hostile_v1, "4 HunterBot non armés (phase 1)"),
        Scenario("scarce", _scarce, "objets rares, physiologie x1.5"),
        Scenario("blind", _blind, "brouillard + faible lumière, 2 chasseurs dont 1 armé"),
        Scenario("big_map", _big_map, "carte 50x50"),
    )
}
MAIN_SCENARIOS = ("default", "hostile", "scarce", "blind", "big_map")

# Seeds d'évaluation : disjointes des seeds d'entraînement [0, 1_000_000).
EVAL_SEEDS = range(10_000_000, 10_000_200)
TRAIN_SEED_RANGE = (0, 1_000_000)


# ----------------------------------------------------------------------
# Curriculum avec randomisation de domaine (§6)
# ----------------------------------------------------------------------
@dataclass(slots=True)
class Stage:
    index: int
    name: str
    sample: Callable[[Random, int], EpisodeSpec]  # (rng, seed) -> spec
    goal: str = ""

    def sample_spec(self, rng: Random, seed: int) -> EpisodeSpec:
        return self.sample(rng, seed)


def _random_generation(rng: Random, items: tuple[int, int], fog: float, low_light: float) -> GenerationConfig:
    return GenerationConfig(
        obstacle_count=(rng.randint(3, 6), rng.randint(7, 12)),
        item_count=items,
        water_probability=rng.uniform(0.4, 0.9),
        fog_probability=fog,
        low_light_probability=low_light,
    )


def _random_physiology(rng: Random, speedup: float) -> PhysiologyConfig:
    return PhysiologyConfig(
        hydration_decay_per_tick=0.0006 * speedup * rng.uniform(0.85, 1.15),
        satiety_decay_per_tick=0.0004 * speedup * rng.uniform(0.85, 1.15),
    )


def _stage0(rng: Random, seed: int) -> EpisodeSpec:
    cfg = EnvironmentConfig(
        max_ticks=LONG_EPISODE,
        generation=_random_generation(rng, (20, 28), 0.0, 0.0),
        physiology=_random_physiology(rng, 1.5),
    )
    return EpisodeSpec(cfg, [("bot_random", RandomBot(seed=seed))])


def _stage1(rng: Random, seed: int) -> EpisodeSpec:
    cfg = EnvironmentConfig(
        max_ticks=LONG_EPISODE,
        generation=_random_generation(rng, (14, 22), 0.2, 0.2),
        physiology=_random_physiology(rng, 1.3),
    )
    opponents: Opponents = [("bot_random", RandomBot(seed=seed))]
    loadouts: dict[str, Loadout] = {}
    for i in range(rng.randint(1, 2)):
        aid = f"hunter{i}"
        opponents.append((aid, HunterBot(seed=seed + 17 * i) if rng.random() < 0.5 else AggressiveHunter(seed=seed + 17 * i)))
        if rng.random() < 0.5:
            loadouts[aid] = launcher_loadout(rng.randint(2, 6)) if rng.random() < 0.5 else melee_loadout()
    return EpisodeSpec(cfg, opponents, loadouts)


def _stage2(rng: Random, seed: int) -> EpisodeSpec:
    size = rng.choice([26.0, 30.0, 30.0, 36.0, 44.0])
    cfg = EnvironmentConfig(
        max_ticks=LONG_EPISODE,
        map_width=size,
        map_height=size,
        generation=_random_generation(rng, (4, 14), rng.uniform(0.3, 1.0), rng.uniform(0.3, 1.0)),
        physiology=_random_physiology(rng, rng.uniform(1.0, 1.6)),
    )
    opponents: Opponents = []
    loadouts: dict[str, Loadout] = {}
    for i in range(rng.randint(3, 4)):
        aid = f"hunter{i}"
        opponents.append((aid, AggressiveHunter(seed=seed + 17 * i)))
        r = rng.random()
        if r < 0.4:
            loadouts[aid] = launcher_loadout(rng.randint(2, 6))
        elif r < 0.7:
            loadouts[aid] = melee_loadout()
    return EpisodeSpec(cfg, opponents, loadouts)


# Pool de politiques figées pour le self-play : la phase 3 y ajoutera les
# meilleures Q-tables successives (fabriques sans argument).
SELF_PLAY_POOL: list[Callable[[], object]] = []
_TABLE_CACHE: dict[str, object] = {}


def set_self_play_tables(paths) -> None:
    """Remplit SELF_PLAY_POOL avec des QAgent gloutons sur ces Q-tables (cache par processus)."""
    from pathlib import Path

    from agent.qpolicy import QAgent, QTable

    SELF_PLAY_POOL.clear()
    for path in paths:
        if path not in _TABLE_CACHE:
            _TABLE_CACHE[path] = QTable.load(Path(path))
        table = _TABLE_CACHE[path]
        SELF_PLAY_POOL.append(lambda t=table: QAgent(t))


def _stage3(rng: Random, seed: int) -> EpisodeSpec:
    spec = _stage2(rng, seed)
    spec.opponents = spec.opponents[:2]
    spec.loadouts = {k: v for k, v in spec.loadouts.items() if k in dict(spec.opponents)}
    pool = SELF_PLAY_POOL or []
    for i in range(rng.randint(2, 3)):
        factory = rng.choice(pool) if pool and rng.random() < 0.7 else None
        spec.opponents.append((f"copy{i}", FrozenAgent(factory) if factory else _scripted_copy()))
    return spec


STAGES: list[Stage] = [
    Stage(0, "physiology", _stage0, "gérer énergie, boire, manger, se reposer"),
    Stage(1, "first_threats", _stage1, "fuir / se cacher, s'armer"),
    Stage(2, "degraded", _stage2, "décider sous perception dégradée, adversaires armés"),
    Stage(3, "self_play", _stage3, "robustesse face à des agents « intelligents »"),
]


def sample_curriculum_spec(stage: int, rng: Random, seed: int, replay: float = 0.2) -> EpisodeSpec:
    """Épisode de l'étape `stage`, ou (proba `replay`) d'une étape antérieure (anti-oubli)."""
    if stage > 0 and rng.random() < replay:
        stage = rng.randrange(stage)
    return STAGES[stage].sample_spec(rng, seed)
