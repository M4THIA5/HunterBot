"""Politique apprise sur options : Q-table tabulaire (SMDP, §7.1).

`QAgent` choisit une option aux points de décision :
- fin naturelle de l'option en cours (`done`) ;
- interruption : menace qui se rapproche, dégâts reçus, objet utile qui apparaît.

Parmi les options valides (masque), il prend argmax Q sur les couples déjà
visités ; si l'état n'a jamais été visité, il se replie sur la politique
scriptée. En entraînement, `epsilon > 0` et un `transition_sink` reçoivent les
transitions (s, o, R, k, s', valides(s'), terminal) pour la mise à jour.

Dépend uniquement de rl_arena.arena_api (via les modules agent.*) et de la
bibliothèque standard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Callable

from rl_arena.arena_api import Action, Observation

from .features import State, encode_state, valid_options
from .options import OPTION_NAMES
from .scripted import OptionRunner, scripted_choice

OPTION_INDEX = {name: i for i, name in enumerate(OPTION_NAMES)}
DEFAULT_TABLE = Path(__file__).with_name("q_table.json")


class QTable:
    def __init__(self, n_options: int = len(OPTION_NAMES)) -> None:
        self.n_options = n_options
        self.q: dict[str, list[float]] = {}
        self.n: dict[str, list[int]] = {}

    def values(self, key: str) -> list[float] | None:
        return self.q.get(key)

    def visits(self, key: str) -> list[int] | None:
        return self.n.get(key)

    def ensure(self, key: str) -> tuple[list[float], list[int]]:
        if key not in self.q:
            self.q[key] = [0.0] * self.n_options
            self.n[key] = [0] * self.n_options
        return self.q[key], self.n[key]

    def best_value(self, key: str, valid: list[int]) -> float:
        """max sur les options valides déjà essayées (0 si aucune)."""
        q, n = self.q.get(key), self.n.get(key)
        if q is None:
            return 0.0
        tried = [q[i] for i in valid if n[i] > 0]
        return max(tried) if tried else 0.0

    def greedy(self, key: str, valid: list[int]) -> int | None:
        q, n = self.q.get(key), self.n.get(key)
        if q is None:
            return None
        tried = [i for i in valid if n[i] > 0]
        if not tried:
            return None
        return max(tried, key=lambda i: q[i])

    # Sérialisation ------------------------------------------------------
    def to_json(self, meta: dict | None = None) -> dict:
        return {"options": list(OPTION_NAMES), "fields": list(State._fields), "meta": meta or {},
                "q": self.q, "n": self.n}

    def save(self, path: Path, meta: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(meta), separators=(",", ":")), encoding="utf-8")

    @classmethod
    def from_json(cls, data: dict) -> "QTable":
        names = data.get("options", list(OPTION_NAMES))
        table = cls()
        # Réordonner si l'ordre des options a changé entre deux versions.
        remap = [names.index(o) if o in names else None for o in OPTION_NAMES]
        for key, vals in data.get("q", {}).items():
            counts = data.get("n", {}).get(key, [1] * len(vals))
            table.q[key] = [vals[j] if j is not None else 0.0 for j in remap]
            table.n[key] = [counts[j] if j is not None else 0 for j in remap]
        return table

    @classmethod
    def load(cls, path: Path) -> "QTable":
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))


@dataclass(slots=True)
class Transition:
    state: str
    option: int
    reward: float      # R = Σ γ^i r_i sur la durée de l'option
    duration: int      # k ticks
    next_state: str | None
    next_valid: tuple[int, ...]
    terminal: bool     # mort : pas de valeur future (troncature : bootstrap)


class QAgent:
    """Agent à options piloté par une Q-table (ou par le scripté en repli)."""

    def __init__(self, table: QTable | None = None, epsilon: float = 0.0, seed: int = 0,
                 gamma: float = 0.998, transition_sink: Callable[[Transition], None] | None = None,
                 explore_scripted: float = 0.0) -> None:
        self.table = table if table is not None else QTable()
        self.epsilon = epsilon
        self.gamma = gamma
        self.sink = transition_sink
        self.explore_scripted = explore_scripted
        self._seed = seed
        self.runner = OptionRunner(seed)
        self.rng = Random(seed + 7919)
        self.decisions = {"q": 0, "fallback": 0, "random": 0}
        self._reset_episode()

    # Cycle de vie -------------------------------------------------------
    def _reset_episode(self) -> None:
        self._pending: tuple[str, int] | None = None
        self._ret = 0.0
        self._k = 0
        self._start_state: State | None = None

    def reset(self) -> None:
        self.runner.reset()
        self._reset_episode()

    def on_episode_end(self, result) -> None:
        pass

    # Décision -----------------------------------------------------------
    def _interrupt(self, st: State) -> bool:
        s0 = self._start_state
        if s0 is None:
            return False
        return (st.threat > s0.threat or (st.recent_damage and not s0.recent_damage)
                or (st.item_visible and not s0.item_visible))

    def _choose(self, obs: Observation, st: State, valid: list[int]) -> int:
        key = st.key()
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            self.decisions["random"] += 1
            if self.explore_scripted and self.rng.random() < self.explore_scripted:
                return OPTION_INDEX[scripted_choice(obs, self.runner.mem, self.runner.options)]
            return self.rng.choice(valid)
        best = self.table.greedy(key, valid)
        if best is None:
            self.decisions["fallback"] += 1
            name = scripted_choice(obs, self.runner.mem, self.runner.options)
            return OPTION_INDEX[name] if OPTION_INDEX[name] in valid else valid[0]
        self.decisions["q"] += 1
        return best

    def act(self, obs: Observation) -> Action:
        r = self.runner
        r.mem.update(obs)
        st = encode_state(obs, r.mem)
        if r.current is None or r.current.done(obs, r.mem) or self._interrupt(st):
            valid = [OPTION_INDEX[n] for n in valid_options(obs, r.mem, r.options)]
            self._close(st.key(), tuple(valid), terminal=False)
            o = self._choose(obs, st, valid)
            r.switch(OPTION_NAMES[o], obs)
            self._pending = (st.key(), o)
            self._start_state = st
        return r.step(obs)

    # Apprentissage ------------------------------------------------------
    def record(self, reward: float) -> None:
        """Récompense du tick qui vient d'être joué (appelé après env.step)."""
        if self._pending is not None:
            self._ret += (self.gamma ** self._k) * reward
            self._k += 1

    def _close(self, next_key: str | None, next_valid: tuple[int, ...], terminal: bool) -> None:
        if self._pending is not None and self.sink is not None and self._k > 0:
            s, o = self._pending
            self.sink(Transition(s, o, self._ret, self._k, next_key, next_valid, terminal))
        self._pending = None
        self._ret, self._k = 0.0, 0

    def end_episode(self, last_obs: Observation | None, terminal: bool) -> None:
        """Clôt l'option en cours. `terminal=True` : mort ; sinon troncature (bootstrap)."""
        if terminal or last_obs is None:
            self._close(None, (), terminal=True)
        else:
            self.runner.mem.update(last_obs)
            st = encode_state(last_obs, self.runner.mem)
            valid = tuple(OPTION_INDEX[n] for n in valid_options(last_obs, self.runner.mem, self.runner.options))
            self._close(st.key(), valid, terminal=False)


def load_default_agent() -> QAgent | None:
    """QAgent glouton sur la Q-table livrée, ou None si elle est absente/illisible."""
    try:
        return QAgent(QTable.load(DEFAULT_TABLE))
    except (OSError, ValueError, KeyError, TypeError):
        return None
