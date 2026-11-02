"""Évaluation de la survie d'un agent sur des scénarios et des seeds fixes.

Usage :
    python -m training.evaluate --agents trivial scripted --scenarios hostile scarce --episodes 30

La cause de mort utilise l'état interne du moteur : c'est un outil de
diagnostic uniquement, jamais une entrée de la politique.
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from rl_arena.environment import TrainingEnvironment

from training.baselines import make_agent
from training.envs import EVAL_SEEDS, SCENARIOS


@dataclass(frozen=True, slots=True)
class EpisodeStats:
    agent: str
    scenario: str
    seed: int
    survival: int
    truncated: bool
    cause: str  # "alive", "combat", "physiology"
    kills: int
    pickups: int
    damage: float


def run_episode(agent_name: str, scenario_name: str, seed: int) -> EpisodeStats:
    scenario = SCENARIOS[scenario_name]
    env = TrainingEnvironment(scenario.config_factory())
    env.register_agent("student", make_agent(agent_name))
    for aid, bot in scenario.opponents(seed):
        env.register_agent(aid, bot)
    env.reset(seed=seed)

    last_source: str | None = None
    while True:
        res = env.step({})
        st = env.debug_state()["agents"]["student"]
        if st.alive:
            last_source = st.last_damage_source
        dead = res.terminated["student"]
        if dead or res.truncated:
            break

    info = res.info["student"]
    if not dead:
        cause = "alive"
    else:
        # last_damage_source vaut l'id de l'agent responsable, ou None pour
        # faim/soif/saignement.
        src = env.debug_state()["agents"]["student"].last_damage_source
        cause = "combat" if src is not None else "physiology"
    return EpisodeStats(
        agent_name, scenario_name, seed, info.survival_ticks, not dead, cause,
        info.kills, info.pickups, info.damage_received,
    )


def percentile(values: list[int], q: float) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(stats: list[EpisodeStats]) -> dict[str, float]:
    surv = [s.survival for s in stats]
    causes = Counter(s.cause for s in stats)
    n = len(stats)
    return {
        "mean": statistics.fmean(surv),
        "median": statistics.median(surv),
        "p10": percentile(surv, 0.10),
        "alive": causes["alive"] / n,
        "combat": causes["combat"] / n,
        "physio": causes["physiology"] / n,
        "kills": statistics.fmean(s.kills for s in stats),
        "pickups": statistics.fmean(s.pickups for s in stats),
    }


def evaluate(agents: list[str], scenarios: list[str], episodes: int, workers: int | None = None):
    seeds = list(EVAL_SEEDS)[:episodes]
    jobs = [(a, sc, seed) for a in agents for sc in scenarios for seed in seeds]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_episode, *zip(*jobs), chunksize=1))
    table: dict[tuple[str, str], dict[str, float]] = {}
    for a in agents:
        for sc in scenarios:
            table[(a, sc)] = summarize([r for r in results if r.agent == a and r.scenario == sc])
    return table, results


def print_table(table) -> None:
    header = f"{'scenario':<9} {'agent':<11} {'mean':>7} {'median':>7} {'p10':>7} {'alive':>6} {'combat':>7} {'physio':>7} {'kills':>6} {'pick':>5}"
    print(header)
    print("-" * len(header))
    for (a, sc), m in sorted(table.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print(
            f"{sc:<9} {a:<11} {m['mean']:7.0f} {m['median']:7.0f} {m['p10']:7.0f} "
            f"{m['alive']:6.0%} {m['combat']:7.0%} {m['physio']:7.0%} {m['kills']:6.2f} {m['pickups']:5.1f}"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", nargs="+", default=["trivial", "scripted"])
    p.add_argument("--scenarios", nargs="+", default=["default", "hostile", "scarce"])
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--worst", type=int, default=0, help="affiche les N pires épisodes par (agent, scénario)")
    args = p.parse_args()
    table, results = evaluate(args.agents, args.scenarios, args.episodes, args.workers)
    print_table(table)
    if args.worst:
        for a in args.agents:
            for sc in args.scenarios:
                rows = sorted((r for r in results if r.agent == a and r.scenario == sc), key=lambda r: r.survival)
                print(f"\n{a} / {sc} : pires épisodes")
                for r in rows[: args.worst]:
                    print(f"  seed {r.seed}  survie {r.survival:5d}  {r.cause:<10} dégâts {r.damage:.2f}  pickups {r.pickups}")


if __name__ == "__main__":
    main()
