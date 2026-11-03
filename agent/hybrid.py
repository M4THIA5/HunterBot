"""Agent hybride : scripté au calme, Q-table apprise en situation de danger.

Constat du run 6 : la Q-table survit mieux face aux adversaires armés
(fuite / cachette plus fréquentes) mais gère moins bien les ressources que
le scripté. L'hybride délègue donc :
- calme : politique scriptée réévaluée à chaque tick (comme ScriptedAgent) ;
- danger : choix d'option par la Q-table aux points de décision (comme QAgent).

Le danger est défini uniquement depuis l'état encodé (Observation + mémoire).
`hold` prolonge le mode danger quelques ticks après sa disparition, pour
éviter d'osciller entre les deux politiques au bord du champ de vision.

Dépend uniquement de rl_arena.arena_api (via les modules agent.*).
"""

from __future__ import annotations

from rl_arena.arena_api import Action, Observation

from .features import State, encode_state, valid_options
from .options import OPTION_NAMES
from .qpolicy import OPTION_INDEX, QAgent, QTable
from .scripted import scripted_choice

GATES = {
    # agent vu/touché ou dégâts récents
    "seen": lambda st: st.threat > 0 or st.recent_damage,
    # + menace entendue
    "heard": lambda st: st.threat > 0 or st.recent_damage or st.heard_threat,
    # agent proche (< 8 m) ou dégâts récents
    "near": lambda st: st.threat >= 2 or st.recent_damage,
}


class HybridAgent:
    def __init__(self, table: QTable, gate: str = "seen", hold: int = 0, seed: int = 0) -> None:
        self.q = QAgent(table, seed=seed)
        self.gate = GATES[gate]
        self.hold = hold
        self.stats = {"calm": 0, "danger": 0}
        self.reset()

    @property
    def runner(self):
        return self.q.runner

    def reset(self) -> None:
        self.q.reset()
        self._danger_left = 0
        self._mode = "calm"

    def on_episode_end(self, result) -> None:
        pass

    def _calm_step(self, obs: Observation) -> Action:
        r = self.runner
        choice = scripted_choice(obs, r.mem, r.options)
        if r.current is None or r.current.name != choice or r.needs_decision(obs):
            r.switch(choice, obs)
        return r.step(obs)

    def _danger_step(self, obs: Observation, st: State, entering: bool) -> Action:
        r, q = self.runner, self.q
        if entering or r.current is None or r.current.done(obs, r.mem) or q._interrupt(st):
            valid = [OPTION_INDEX[n] for n in valid_options(obs, r.mem, r.options)]
            o = q._choose(obs, st, valid)
            r.switch(OPTION_NAMES[o], obs)
            q._start_state = st
        return r.step(obs)

    def act(self, obs: Observation) -> Action:
        r = self.runner
        r.mem.update(obs)
        st = encode_state(obs, r.mem)
        if self.gate(st):
            self._danger_left = self.hold + 1
        mode = "danger" if self._danger_left > 0 else "calm"
        self._danger_left = max(0, self._danger_left - 1)
        entering = mode == "danger" and self._mode != "danger"
        self._mode = mode
        self.stats[mode] += 1
        if mode == "danger":
            return self._danger_step(obs, st, entering)
        return self._calm_step(obs)
