"""Diagnostic des morts (outil hors politique : lit l'état interne du moteur).

Usage : python -m training.hostile_diag [--agent scripted] [--scenario hostile] [--episodes 50]

Pour chaque épisode : tueur, tick du premier dégât, dégâts subis par option
active, dégâts subis avec l'attaquant visible ou non, état (armé, santé).
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from rl_arena.arena_api import EventKind, ObjectKind

from training.baselines import make_agent
from training.envs import EVAL_SEEDS, SCENARIOS, STUDENT_ID, build_env


def _runner(agent):
    return getattr(agent, "runner", None) or getattr(getattr(agent, "_policy", None), "runner", None)


def run(agent_name: str, scenario: str, seed: int) -> dict:
    agent = make_agent(agent_name)
    env = build_env(SCENARIOS[scenario].spec(seed), student=agent)
    obs = env.reset(seed=seed)[STUDENT_ID]
    dmg_by_opt: Counter = Counter()
    dmg_visible = dmg_hidden = 0.0
    dmg_by_src: Counter = Counter()
    first_dmg = None
    first_seen = None
    armed_at_first = None
    opt_prev = None
    while True:
        res = env.step({})
        dead = res.terminated[STUDENT_ID]
        if STUDENT_ID in res.observations:
            obs = res.observations[STUDENT_ID]
        r = _runner(agent)
        opt = r.current.name if r and r.current else "?"
        st = env.debug_state()["agents"][STUDENT_ID]
        if first_seen is None and any(d.kind is ObjectKind.AGENT for d in obs.vision):
            first_seen = obs.tick
        for ev in obs.events:
            if ev.kind is EventKind.DAMAGE_RECEIVED:
                v = float(ev.value or 0.0)
                dmg_by_opt[opt_prev or opt] += v
                dmg_by_src[str(st.last_damage_source)] += v
                if any(d.kind is ObjectKind.AGENT for d in obs.vision):
                    dmg_visible += v
                else:
                    dmg_hidden += v
                if first_dmg is None:
                    first_dmg = obs.tick
                    armed_at_first = obs.self_state.equipped_slot is not None
        opt_prev = opt
        if dead or res.truncated:
            break
    killer = st.last_damage_source if dead else None
    return dict(seed=seed, survival=obs.tick, dead=dead, killer=killer, first_dmg=first_dmg, first_seen=first_seen,
                armed=armed_at_first, dmg_by_opt=dict(dmg_by_opt), dmg_by_src=dict(dmg_by_src),
                visible=dmg_visible, hidden=dmg_hidden)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="scripted")
    p.add_argument("--scenario", default="hostile")
    p.add_argument("--episodes", type=int, default=50)
    a = p.parse_args()
    seeds = list(EVAL_SEEDS)[: a.episodes]
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(run, [a.agent] * len(seeds), [a.scenario] * len(seeds), seeds))
    surv = [r["survival"] for r in rows]
    print(f"{a.agent} / {a.scenario} : moyenne {statistics.fmean(surv):.0f}")
    killers = Counter(("".join(c for c in str(r["killer"]) if not c.isdigit()) + str(r["killer"])[-1:]
                       if r["killer"] else ("alive" if not r["dead"] else "physio")) for r in rows)
    print("tueurs :", dict(killers.most_common()))
    opt: Counter = Counter()
    src: Counter = Counter()
    for r in rows:
        opt.update(r["dmg_by_opt"])
        src.update(r["dmg_by_src"])
    tot = sum(opt.values()) or 1
    print("dégâts par option active :", {k: f"{v / tot:.0%}" for k, v in opt.most_common()})
    print("dégâts par source :", {k: f"{v / tot:.0%}" for k, v in src.most_common()})
    vis = sum(r["visible"] for r in rows)
    print(f"dégâts attaquant visible : {vis / tot:.0%}")
    fd = [r["first_dmg"] for r in rows if r["first_dmg"] is not None]
    fs = [r["first_seen"] for r in rows if r["first_seen"] is not None]
    print(f"premier dégât (médiane) : {statistics.median(fd):.0f} ; premier agent vu (médiane) : {statistics.median(fs):.0f}")
    print(f"armé au premier dégât : {sum(1 for r in rows if r['armed']) / max(1, len(fd)):.0%}")
    combat = [r for r in rows if r["killer"]]
    print("morts au combat les plus précoces :")
    for r in sorted(combat, key=lambda r: r["survival"])[:10]:
        print(f"  seed {r['seed']} survie {r['survival']} tueur {r['killer']} 1er dégât {r['first_dmg']} "
              f"1er vu {r['first_seen']} options {{{', '.join(f'{k}:{v:.2f}' for k, v in r['dmg_by_opt'].items())}}}")


if __name__ == "__main__":
    main()
