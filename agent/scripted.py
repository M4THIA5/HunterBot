"""Politique scriptée de référence sur les options (phase 1).

Elle sert (1) de baseline que l'apprentissage devra battre, (2) de repli
pour les états jamais vus par la Q-table.
"""

from __future__ import annotations

from math import cos, pi, sin
from random import Random

from rl_arena.arena_api import Action, Movement, ObjectKind, Observation

from .options import Option, make_options
from .perception import Memory, equipped_item, is_armed, weapon_mode, wrap


class OptionRunner:
    """Exécute une option choisie par une politique ; gère commutation et durée."""

    def __init__(self, seed: int = 0, use_spatial: bool = True, spatial_uses: frozenset | None = None,
                 **mem_options) -> None:
        self._seed = seed
        self.use_spatial = use_spatial
        self.spatial_uses = spatial_uses
        self.mem_options = mem_options  # champs de Memory à fixer (ablations)
        self.reset()

    def reset(self) -> None:
        self.rng = Random(self._seed)
        self.mem = Memory(use_spatial=self.use_spatial, **self.mem_options)
        if self.spatial_uses is not None:
            self.mem.spatial_uses = frozenset(self.spatial_uses)
        self.options = make_options(self.rng)
        self.current: Option | None = None
        self.history: dict[str, int] = {}

    def needs_decision(self, obs: Observation) -> bool:
        return self.current is None or self.current.done(obs, self.mem)

    def switch(self, name: str, obs: Observation) -> None:
        opt = self.options[name]
        opt.start(obs, self.mem)
        self.current = opt

    def _unwedge(self, obs: Observation, action: Action) -> Action:
        return _unwedge_action(obs, self.mem, action)

    def step(self, obs: Observation) -> Action:
        assert self.current is not None
        self.history[self.current.name] = self.history.get(self.current.name, 0) + 1
        action = self.current.step(obs, self.mem)
        action = self._unwedge(obs, action)
        self.mem.moving_cmd = action.movement is not None and not action.rest and action.movement.speed > 0.2
        if self.mem.spatial is not None:
            # Odométrie : le mouvement commandé sert à intégrer le tick suivant.
            self.mem.spatial.last_move = None if action.rest else action.movement
        return action


UNWEDGE_AFTER = 12     # ticks coincé avant de forcer le dégagement
UNWEDGE_STEP = 3       # ticks par direction essayée
UNWEDGE_TRIES = 8      # directions balayées (tour complet)


def _unwedge_action(obs: Observation, mem: Memory, action: Action) -> Action:
    """Coincé (mêmes contacts plusieurs ticks) : balaie des directions absolues à pleine vitesse."""
    if not mem.use_unwedge:
        return action
    if mem.unwedge_ticks == 0 and mem.wedged_ticks >= UNWEDGE_AFTER:
        mem.unwedge_ticks = UNWEDGE_STEP * UNWEDGE_TRIES
        mem.unwedge_heading = obs.self_state.orientation + pi  # d'abord : demi-tour
        mem.wedged_ticks = 0
    if mem.unwedge_ticks > 0:
        if mem.unwedge_ticks % UNWEDGE_STEP == 0:
            mem.unwedge_heading = wrap(mem.unwedge_heading + 2 * pi / UNWEDGE_TRIES * 3)  # pas de 135°
        b = wrap(mem.unwedge_heading - obs.self_state.orientation)
        return Action(movement=Movement(cos(b), sin(b), 2.0), interaction=action.interaction)
    return action


def scripted_choice(obs: Observation, mem: Memory, opts: dict[str, Option]) -> str:
    """Règles écrites à la main, par ordre de priorité."""
    s = obs.self_state
    avail = lambda n: opts[n].available(obs, mem)  # noqa: E731

    agents = [d for d in obs.vision if d.kind is ObjectKind.AGENT]
    closest = min((d.distance for d in agents), default=99.0)
    touched = any(c.kind is ObjectKind.AGENT for c in obs.touch)
    attacked = mem.under_attack(3) and (mem.threat_active(4) or touched)
    ranged = weapon_mode(equipped_item(obs)) == "ranged" and is_armed(obs)

    # 1. Danger immédiat.
    if attacked or touched or closest < 2.5:
        if mem.fight_limit:
            # Mêlée seulement avec l'avantage (bonne santé, adversaire non perçu armé)
            # ou acculé (impossible de fuir) ; le tir à distance reste permis.
            cornered = not avail("FLEE")
            melee_ok = (s.health >= 0.6 and not mem.threat_armed) or cornered
            if avail("FIGHT") and (ranged or melee_ok):
                return "FIGHT"
        elif avail("FIGHT") and (ranged or closest < 1.6 or s.health > 0.5
                                 or (mem.counter_attack and mem.tick - mem.blind_hit_tick <= 2
                                     and s.health >= mem.counter_min_health)):
            return "FIGHT"
        if avail("FLEE"):
            return "FLEE"
    # 2. Soin / boire / manger.
    if avail("CONSUME"):
        return "CONSUME"
    # 3. S'équiper.
    if avail("ARM"):
        return "ARM"
    # 4. Agent proche : tir à distance si possible, sinon fuite.
    if agents and closest < 6.0:
        if ranged and avail("FIGHT"):
            return "FIGHT"
        if avail("FLEE") and mem.threat_dangerous():
            return "FLEE"
    # 5. Récupérer l'énergie au calme.
    if s.energy < 0.25 and not mem.threat_active(6):
        return "REST"
    # 6. Ressource visible.
    if avail("SEEK_ITEM"):
        return "SEEK_ITEM"
    # 7. Menace entendue mais invisible et énergie moyenne : se faire discret.
    if mem.heard_gunshot and not agents:
        return "HIDE"
    return "EXPLORE"


CALM_OPTIONS = frozenset({"EXPLORE", "SEEK_ITEM", "REST", "HIDE"})


class ScriptedAgent:
    """Agent complet : runner d'options + politique scriptée."""

    def __init__(self, seed: int = 0, use_spatial: bool = True, spatial_uses: frozenset | None = None,
                 **mem_options) -> None:
        self.runner = OptionRunner(seed, use_spatial, spatial_uses, **mem_options)

    def reset(self) -> None:
        self.runner.reset()

    def act(self, obs: Observation) -> Action:
        r = self.runner
        r.mem.update(obs)
        choice = scripted_choice(obs, r.mem, r.options)
        # Engagement dans la fuite : la menace sort du champ de vision dès qu'on
        # lui tourne le dos ; sans cela on repasse en EXPLORE au tick suivant,
        # on se retourne vers elle et on oscille.
        if (r.mem.serious_threat() and r.current is not None and r.current.name == "FLEE"
                and not r.needs_decision(obs) and choice in CALM_OPTIONS):
            choice = "FLEE"
        # Les options longues ne sont interrompues que par un changement de
        # priorité ; sinon on les laisse se terminer.
        if r.current is None or r.current.name != choice or r.needs_decision(obs):
            r.switch(choice, obs)
        return r.step(obs)

    def on_episode_end(self, result) -> None:
        pass
