"""Mêlée générale : tous nos agents dans la même arène.

Usage :
    python -m training.arena_ffa                 # une partie (seed 10000000) détaillée
    python -m training.arena_ffa --games 100     # classement sur 100 parties

Carte et physiologie du kit, max_ticks=3000, 8 agents (limite du moteur).
La cause de mort lit l'état interne du moteur : diagnostic uniquement.
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from rl_arena.config import EnvironmentConfig

from training.envs import EVAL_SEEDS, LONG_EPISODE
from training.opponents import ArenaEnvironment


def _q(path: str):
    from agent.qpolicy import QAgent, QTable

    return lambda: QAgent(QTable.load(Path(path)))


def _hybrid(path: str):
    from agent.hybrid import HybridAgent
    from agent.qpolicy import QTable

    return lambda: HybridAgent(QTable.load(Path(path)), "heard", 10)


def _scripted(**kw):
    from agent.scripted import ScriptedAgent

    return lambda: ScriptedAgent(**kw)


# id -> (description, fabrique)
ROSTER = {
    "student": ("student_agent.MyAgent (scripté, fuite « serious »)", None),
    "s_always": ("scripté, fuite « always »", _scripted(flee_commit="always")),
    "s_nocommit": ("scripté, coup invisible seul", _scripted(flee_commit="off")),
    "s_v_map": ("scripté avant correctifs hostile (mémoire ressources)",
                _scripted(flee_commit="off", use_blind_hit=False)),
    "s_v_nomap": ("scripté sans mémoire spatiale (phase 1)",
                  _scripted(flee_commit="off", use_blind_hit=False, use_spatial=False)),
    "q_run6": ("Q-learning run 6", _q("results/phase4_run6/q_best.json")),
    "q_run5": ("Q-learning run 5", _q("results/phase4_run5/q_best.json")),
    "hybrid": ("hybride heard_10 (run 6)", _hybrid("results/phase4_run6/q_best.json")),
}


def _make(aid: str):
    if aid == "student":
        from student_agent import MyAgent

        return MyAgent()
    return ROSTER[aid][1]()


def play(seed: int, verbose: bool = False) -> dict:
    env = ArenaEnvironment(EnvironmentConfig(max_ticks=LONG_EPISODE))
    for aid in ROSTER:
        env.register_agent(aid, _make(aid))
    env.reset(seed=seed)
    death: dict[str, tuple[int, str | None]] = {}
    while True:
        res = env.step({})
        state = env.debug_state()["agents"]
        for aid, dead in res.terminated.items():
            if dead and aid not in death:
                killer = state[aid].last_damage_source
                death[aid] = (res.info[aid].survival_ticks, killer)
                if verbose:
                    cause = f"tué par {killer}" if killer else "physiologie"
                    print(f"  tick {res.info[aid].survival_ticks:4d}  {aid:11s} meurt ({cause})")
        if res.truncated or len(death) == len(ROSTER):
            break
    out = {}
    for aid in ROSTER:
        info = res.info[aid]
        out[aid] = dict(survival=info.survival_ticks, alive=aid not in death,
                        killer=death.get(aid, (0, None))[1], kills=info.kills, pickups=info.pickups)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--seed", type=int, default=EVAL_SEEDS[0])
    a = p.parse_args()

    if a.games == 1:
        print(f"Partie seed {a.seed}, {len(ROSTER)} agents, max {LONG_EPISODE} ticks")
        res = play(a.seed, verbose=True)
        ranking = sorted(res.items(), key=lambda kv: -kv[1]["survival"])
        print()
        for rank, (aid, r) in enumerate(ranking, 1):
            status = "survivant" if r["alive"] else ("tué par " + r["killer"] if r["killer"] else "physiologie")
            print(f"{rank}. {aid:11s} {r['survival']:5d} ticks  kills {r['kills']}  pickups {r['pickups']:2d}  "
                  f"{status:22s} {ROSTER[aid][0]}")
        return

    seeds = [a.seed + i for i in range(a.games)]
    with ProcessPoolExecutor() as pool:
        games = list(pool.map(play, seeds))
    wins: Counter = Counter()
    for g in games:
        best = max(v["survival"] for v in g.values())
        for aid, v in g.items():
            if v["survival"] == best:
                wins[aid] += 1  # ex aequo : tous comptés
    rows = []
    for aid in ROSTER:
        surv = [g[aid]["survival"] for g in games]
        ranks = [1 + sum(1 for o in g.values() if o["survival"] > g[aid]["survival"]) for g in games]
        rows.append((aid, statistics.fmean(surv), statistics.median(surv), statistics.fmean(ranks),
                     sum(g[aid]["alive"] for g in games) / len(games),
                     sum(g[aid]["killer"] is not None for g in games) / len(games),
                     statistics.fmean(g[aid]["kills"] for g in games), wins[aid]))
    rows.sort(key=lambda r: -r[1])
    print(f"{a.games} parties, {len(ROSTER)} agents, max {LONG_EPISODE} ticks")
    print(f"{'agent':11s} {'moy':>6s} {'méd':>6s} {'rang':>5s} {'vivant':>7s} {'tué':>5s} {'kills':>6s} {'1er':>4s}")
    for aid, mean, med, rank, alive, killed, kills, w in rows:
        print(f"{aid:11s} {mean:6.0f} {med:6.0f} {rank:5.2f} {alive:7.0%} {killed:5.0%} {kills:6.2f} {w:4d}  {ROSTER[aid][0]}")


if __name__ == "__main__":
    main()
