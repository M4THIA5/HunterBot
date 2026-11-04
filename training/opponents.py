"""Adversaires d'entraînement plus dangereux que les bots du kit.

- `AggressiveHunter` : HunterBot qui tire de plus loin, poursuit au moindre son
  et cherche activement les armes.
- `FrozenAgent` : copie figée d'une de nos politiques (self-play).
- `ArenaEnvironment` : TrainingEnvironment qui peut équiper des agents dès le
  reset (« loadout »). Modifier l'environnement de test est autorisé ; cet
  état interne n'est jamais lu par la politique.
"""

from __future__ import annotations

from typing import Callable

from rl_arena.arena_api import (
    Action,
    Movement,
    ObjectKind,
    Observation,
    Rotation,
    Shoot,
    TICK_SECONDS,
)
from rl_arena.bots import HunterBot
from rl_arena.environment import TrainingEnvironment
from rl_arena.world import InventoryEntry


class AggressiveHunter(HunterBot):
    """HunterBot durci : portée de tir 11 m, poursuite sonore sensible."""

    def __init__(self, seed: int = 0, shoot_range: float = 11.0, hearing_threshold: float = 0.08):
        super().__init__(seed)
        self.shoot_range = shoot_range
        self.hearing_threshold = hearing_threshold

    def act(self, obs: Observation) -> Action:
        s = obs.self_state
        enemies = [d for d in obs.vision if d.kind is ObjectKind.AGENT]
        equipped = next((i for i in s.inventory if i.slot == s.equipped_slot), None)
        has_launcher = equipped is not None and equipped.properties.get("weapon_type") == "potato_launcher"
        has_ammo = any(i.kind is ObjectKind.AMMUNITION for i in s.inventory)

        if enemies and has_launcher and has_ammo and s.health >= 0.45:
            t = min(enemies, key=lambda d: d.distance)
            if t.distance < self.shoot_range:
                self.last_seen_bearing, self.memory_ticks = t.bearing, 10
                fwd = -0.5 if t.distance < 3.0 else (0.6 if t.distance > 7.0 else 0.0)
                return Action(
                    movement=Movement(fwd, 0.4 * self.melee_strafe_sign, 2.0),
                    rotation=Rotation(t.bearing / TICK_SECONDS),
                    interaction=Shoot(t.bearing) if abs(t.bearing) < 0.25 else None,
                )

        action = super().act(obs)
        if not enemies and action.interaction is None and obs.hearing:
            # Poursuite au son plus sensible que HunterBot.
            snd = max(obs.hearing, key=lambda x: x.intensity)
            if self.hearing_threshold < snd.intensity <= 0.2:
                return Action(movement=Movement(1.0, 0.0, 2.2), rotation=Rotation(snd.bearing / TICK_SECONDS))
        return action


class FrozenAgent:
    """Enveloppe d'une politique figée utilisée comme adversaire."""

    def __init__(self, factory: Callable[[], object]):
        self._inner = factory()

    def reset(self) -> None:
        self._inner.reset()

    def act(self, obs: Observation) -> Action:
        try:
            return self._inner.act(obs)
        except Exception:
            return Action()

    def on_episode_end(self, result) -> None:
        self._inner.on_episode_end(result)


# ----------------------------------------------------------------------
# Équipements de départ
# ----------------------------------------------------------------------
LAUNCHER = (ObjectKind.WEAPON, 2.2, {"label": "potato_launcher", "weapon_type": "potato_launcher", "damage": 0.18})
POTATO = (ObjectKind.AMMUNITION, 0.18, {"label": "potato", "ammo_type": "potato"})
ROLLING_PIN = (ObjectKind.WEAPON, 0.65, {"label": "rolling_pin", "weapon_type": "rolling_pin", "damage": 0.12})

Loadout = list[tuple[ObjectKind, float, dict]]


def launcher_loadout(potatoes: int = 6) -> Loadout:
    return [LAUNCHER] + [POTATO] * potatoes


def melee_loadout() -> Loadout:
    return [ROLLING_PIN]


class ArenaEnvironment(TrainingEnvironment):
    """TrainingEnvironment avec équipements de départ par agent."""

    def __init__(self, config=None, render: bool = False, loadouts: dict[str, Loadout] | None = None,
                 world_generator=None):
        super().__init__(config, render)
        self.loadouts = loadouts or {}
        if world_generator is not None:
            self.world_generator = world_generator

    def reset(self, seed: int | None = None):
        super().reset(seed)
        for aid, items in self.loadouts.items():
            rt = self._runtimes.get(aid)
            if rt is None or rt.state is None:
                continue
            st = rt.state
            for slot, (kind, mass, props) in enumerate(items):
                st.inventory[slot] = InventoryEntry(self._alloc_uid(), kind, mass, dict(props))
                if kind is ObjectKind.WEAPON and st.equipped_slot is None:
                    st.equipped_slot = slot
        if self.loadouts:
            self._last_observations = self._observe_all()
        return dict(self._last_observations)
