"""Statistiques de visite des états discrets (taille effective de la Q-table).

Usage :
    python -m training.state_stats --episodes 40

Fait tourner la politique scriptée sur les étapes du curriculum (seeds
d'entraînement) et compte les états `encode_state` rencontrés aux points de
décision (début de chaque option).
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from random import Random

from agent.features import State, encode_state
from agent.scripted import ScriptedAgent
from training.envs import STAGES, STUDENT_ID, build_env


class _Recorder(ScriptedAgent):
    def __init__(self) -> None:
        super().__init__()
        self.visits: Counter = Counter()

    def act(self, obs):
        before = self.runner.current
        action = super().act(obs)
        if self.runner.current is not before or self.runner.current.ticks == 1:
            self.visits[encode_state(obs, self.runner.mem).key()] += 1
        return action


def _run(stage: int, seed: int) -> Counter:
    rng = Random(seed)
    agent = _Recorder()
    env = build_env(STAGES[stage].sample_spec(rng, seed), student=agent)
    env.reset(seed=seed)
    while True:
        res = env.step({})
        if res.truncated or res.terminated[STUDENT_ID]:
            return agent.visits


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=40, help="épisodes par étape")
    args = p.parse_args()
    jobs = [(st.index, 1000 * st.index + i) for st in STAGES for i in range(args.episodes)]
    with ProcessPoolExecutor() as pool:
        counters = list(pool.map(_run, *zip(*jobs)))
    total: Counter = Counter()
    for (stage, _), c in zip(jobs, counters):
        total.update(c)
    for st in STAGES:
        c: Counter = Counter()
        for (stage, _), cc in zip(jobs, counters):
            if stage == st.index:
                c.update(cc)
        print(f"étape {st.index} ({st.name}) : {len(c)} états distincts, {sum(c.values())} décisions")
    n = len(total)
    rare = sum(1 for v in total.values() if v < 10)
    print(f"\ntotal : {n} états distincts / {sum(total.values())} décisions ; "
          f"{rare} états ({rare / max(1, n):.0%}) vus < 10 fois")
    print("états les plus fréquents :")
    for key, v in total.most_common(8):
        print(f"  {key}  {v:6d}  {State(*map(int, key))}")


if __name__ == "__main__":
    main()
