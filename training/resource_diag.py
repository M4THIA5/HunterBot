"""Diagnostic de la collecte de ressources (outil hors politique : lit l'état du moteur).

Usage : python -m training.resource_diag [--agent scripted] [--scenario default] [--episodes 50]

Par épisode : boissons/nourriture présentes au départ, ramassées par nous,
par les autres, jamais ramassées ; tick de notre premier ramassage de
boisson ; survie. Permet de distinguer « pas trouvé » de « pris par un autre ».
"""

from __future__ import annotations

import argparse
import statistics
from concurrent.futures import ProcessPoolExecutor

from rl_arena.arena_api import EventKind, ObjectKind

from training.baselines import make_agent
from training.envs import EVAL_SEEDS, SCENARIOS, STUDENT_ID, build_env

KINDS = (ObjectKind.DRINK, ObjectKind.FOOD)


def _items(env) -> dict[int, ObjectKind]:
    return {o.uid: o.kind for o in env.debug_state()["world"].objects if o.kind in KINDS}


def run(agent_name: str, scenario: str, seed: int) -> dict:
    env = build_env(SCENARIOS[scenario].spec(seed), student=make_agent(agent_name))
    env.reset(seed=seed)
    start = _items(env)
    present = dict(start)
    mine = {k: 0 for k in KINDS}
    others = {k: 0 for k in KINDS}
    first_drink = None
    seen_drink = None
    while True:
        res = env.step({})
        obs = res.observations.get(STUDENT_ID)
        now = _items(env)
        st = env.debug_state()["agents"][STUDENT_ID]
        inv_uids = {i.uid for i in st.inventory.values()}
        for uid, kind in present.items():
            if uid not in now:
                # Disparu de la carte : ramassé par nous s'il est dans notre sac, sinon par un autre.
                (mine if uid in inv_uids else others)[kind] += 1
        # Objets lâchés à la mort d'un agent : ils réapparaissent sur la carte.
        present = now
        if obs is not None:
            if seen_drink is None and any(d.kind is ObjectKind.DRINK for d in obs.vision):
                seen_drink = obs.tick
            if first_drink is None and any(i.kind is ObjectKind.DRINK for i in obs.self_state.inventory):
                first_drink = obs.tick
        if res.terminated[STUDENT_ID] or res.truncated:
            break
    left = {k: sum(1 for v in present.values() if v is k) for k in KINDS}
    total = {k: sum(1 for v in start.values() if v is k) for k in KINDS}
    return dict(seed=seed, survival=res.info[STUDENT_ID].survival_ticks, total=total, mine=mine, others=others,
                left=left, first_drink=first_drink, seen_drink=seen_drink)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="scripted")
    p.add_argument("--scenario", default="default")
    p.add_argument("--episodes", type=int, default=50)
    a = p.parse_args()
    seeds = list(EVAL_SEEDS)[: a.episodes]
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(run, [a.agent] * len(seeds), [a.scenario] * len(seeds), seeds))
    print(f"{a.agent} / {a.scenario} : survie moyenne {statistics.fmean(r['survival'] for r in rows):.0f}")
    for k in KINDS:
        f = lambda key: statistics.fmean(r[key][k] for r in rows)  # noqa: E731
        print(f"  {k.name:5s} au départ {f('total'):.2f}  nous {f('mine'):.2f}  autres {f('others'):.2f}  "
              f"restants à la fin {f('left'):.2f}")
    none = [r for r in rows if r["first_drink"] is None]
    print(f"  épisodes sans aucune boisson ramassée : {len(none)}/{len(rows)} "
          f"(dont boisson jamais vue : {sum(1 for r in none if r['seen_drink'] is None)})")
    fd = [r["first_drink"] for r in rows if r["first_drink"] is not None]
    sd = [r["seen_drink"] for r in rows if r["seen_drink"] is not None]
    if fd:
        print(f"  1re boisson vue (médiane) : tick {statistics.median(sd):.0f} ; 1re ramassée : tick {statistics.median(fd):.0f}")
    print("  10 pires :")
    for r in sorted(rows, key=lambda r: r["survival"])[:10]:
        print(f"    seed {r['seed']} survie {r['survival']}  boissons départ {r['total'][ObjectKind.DRINK]} "
              f"nous {r['mine'][ObjectKind.DRINK]} autres {r['others'][ObjectKind.DRINK]} "
              f"restantes {r['left'][ObjectKind.DRINK]}  vue {r['seen_drink']}")


if __name__ == "__main__":
    main()
