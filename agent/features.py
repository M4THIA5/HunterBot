"""Représentation discrète de l'état (§3) et masquage des options (§7.1).

Tout est calculé depuis l'Observation courante et la Memory de l'agent :
grandeurs relatives, ratios, présence/absence. Aucune constante propre à
l'environnement de test (slots, charge, taille de carte, valeurs d'objets).
"""

from __future__ import annotations

from typing import NamedTuple

from rl_arena.arena_api import ObjectKind, Observation

from .navigation import LOOKAHEAD, blocking_obstacles, ray_clearance
from .perception import USEFUL_KINDS, Memory, is_armed

THREAT_NONE, THREAT_FAR, THREAT_MID, THREAT_CLOSE = 0, 1, 2, 3
SIDE_AHEAD, SIDE_LEFT, SIDE_RIGHT = 0, 1, 2


class State(NamedTuple):
    health: int          # 0: < 0.35, 1: < 0.7, 2: sinon
    energy: int          # 0: < 0.2,  1: < 0.5, 2: sinon
    need: int            # min(hydration, satiety) : 0: < 0.3, 1: < 0.6, 2: sinon
    has_need_item: bool  # boisson/nourriture couvrant le besoin le plus bas
    has_medicine: bool
    threat: int          # agent vu le plus proche : aucun / > 8 m / 3–8 m / < 3 m
    side: int            # côté de cette menace (devant si aucune)
    heard_threat: bool
    recent_damage: bool  # DAMAGE_RECEIVED dans les 4 derniers ticks
    armed: bool          # arme équipée (et munition si arme à distance)
    item_visible: bool   # objet utile visible
    obstacle_ahead: bool

    def key(self) -> str:
        """Clé compacte et stable pour la sérialisation JSON de la Q-table."""
        return "".join(str(int(v)) for v in self)


def bin3(x: float, a: float, b: float) -> int:
    return 0 if x < a else (1 if x < b else 2)


def encode_state(obs: Observation, mem: Memory) -> State:
    s = obs.self_state
    inv_kinds = {i.kind for i in s.inventory}
    need_kind = ObjectKind.DRINK if s.hydration <= s.satiety else ObjectKind.FOOD

    threat, side = THREAT_NONE, SIDE_AHEAD
    agents = [d for d in obs.vision if d.kind is ObjectKind.AGENT]
    if agents:
        t = min(agents, key=lambda d: d.distance)
        threat = THREAT_CLOSE if t.distance < 3.0 else (THREAT_MID if t.distance < 8.0 else THREAT_FAR)
        side = SIDE_AHEAD if abs(t.bearing) < 0.4 else (SIDE_LEFT if t.bearing < 0 else SIDE_RIGHT)
    elif any(c.kind is ObjectKind.AGENT for c in obs.touch):
        threat = THREAT_CLOSE
        b = next(c.bearing for c in obs.touch if c.kind is ObjectKind.AGENT)
        side = SIDE_AHEAD if abs(b) < 0.4 else (SIDE_LEFT if b < 0 else SIDE_RIGHT)

    obstacle_ahead = ray_clearance(0.0, blocking_obstacles(obs)) < min(1.5, LOOKAHEAD)

    return State(
        health=bin3(s.health, 0.35, 0.7),
        energy=bin3(s.energy, 0.2, 0.5),
        need=bin3(min(s.hydration, s.satiety), 0.3, 0.6),
        has_need_item=need_kind in inv_kinds,
        has_medicine=ObjectKind.MEDICINE in inv_kinds,
        threat=threat,
        side=side,
        heard_threat=mem.heard_threat,
        recent_damage=mem.ticks_since_damage <= 4,
        armed=is_armed(obs),
        item_visible=any(d.kind in USEFUL_KINDS for d in obs.vision),
        obstacle_ahead=obstacle_ahead,
    )


def valid_options(obs: Observation, mem: Memory, options: dict) -> list[str]:
    """Options exécutables maintenant (masque fondé uniquement sur l'observation).

    `EXPLORE`, `REST` et `HIDE` sont toujours valides, la liste n'est donc
    jamais vide.
    """
    return [name for name, opt in options.items() if opt.available(obs, mem)]
