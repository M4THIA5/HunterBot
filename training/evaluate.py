"""Protocole d'évaluation de la survie (§8).

Usage :
    python -m training.evaluate --agents trivial scripted --episodes 50
    python -m training.evaluate --agents scripted --scenarios hostile --worst 5
    python -m training.evaluate --agents trivial random_bot hunter_bot scripted \\
        --episodes 50 --out results/baselines.json --markdown results/baselines.md

Métriques par (agent, scénario) : survie moyenne / médiane / 10ᵉ centile, taux
de troncature, cause de mort, kills, pickups, fréquence d'usage des options.
Score combiné = moyenne des 10ᵉ centiles sur les scénarios (le classement
punit les morts précoces).

La cause de mort lit l'état interne du moteur : outil de diagnostic
uniquement, jamais une entrée de la politique.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from training.baselines import make_agent
from training.envs import EVAL_SEEDS, MAIN_SCENARIOS, SCENARIOS, STUDENT_ID, build_env


@dataclass(frozen=True, slots=True)
class EpisodeStats:
    agent: str
    scenario: str
    seed: int
    survival: int
    truncated: bool
    cause: str  # "alive", "combat", "physiology"
    killer: str | None
    kills: int
    pickups: int
    damage: float
    options: dict = field(default_factory=dict)


def _option_history(agent) -> dict:
    """Histogramme d'options si l'agent en expose un (ScriptedAgent, MyAgent)."""
    for path in (("runner",), ("_policy", "runner")):
        obj = agent
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "history"):
            return dict(obj.history)
    return {}


def run_episode(agent_name: str, scenario_name: str, seed: int) -> EpisodeStats:
    agent = make_agent(agent_name)
    env = build_env(SCENARIOS[scenario_name].spec(seed), student=agent)
    env.reset(seed=seed)
    while True:
        res = env.step({})
        dead = res.terminated[STUDENT_ID]
        if dead or res.truncated:
            break
    info = res.info[STUDENT_ID]
    killer = env.debug_state()["agents"][STUDENT_ID].last_damage_source if dead else None
    cause = "alive" if not dead else ("combat" if killer is not None else "physiology")
    return EpisodeStats(agent_name, scenario_name, seed, info.survival_ticks, not dead, cause, killer,
                        info.kills, info.pickups, info.damage_received, _option_history(agent))


def percentile(values: list[int], q: float) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(stats: list[EpisodeStats]) -> dict:
    surv = [s.survival for s in stats]
    causes = Counter(s.cause for s in stats)
    n = len(stats)
    opt_total: Counter = Counter()
    for s in stats:
        opt_total.update(s.options)
    total = sum(opt_total.values()) or 1
    return {
        "episodes": n,
        "mean": statistics.fmean(surv),
        "median": statistics.median(surv),
        "p10": percentile(surv, 0.10),
        "alive": causes["alive"] / n,
        "combat": causes["combat"] / n,
        "physio": causes["physiology"] / n,
        "kills": statistics.fmean(s.kills for s in stats),
        "pickups": statistics.fmean(s.pickups for s in stats),
        "damage": statistics.fmean(s.damage for s in stats),
        "options": {k: v / total for k, v in opt_total.most_common()},
    }


def combined_score(table: dict, agent: str, scenarios) -> float:
    return statistics.fmean(table[(agent, sc)]["p10"] for sc in scenarios)


def evaluate(agents: list[str], scenarios: list[str], episodes: int, workers: int | None = None):
    seeds = list(EVAL_SEEDS)[:episodes]
    jobs = [(a, sc, seed) for a in agents for sc in scenarios for seed in seeds]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_episode, *zip(*jobs), chunksize=1))
    table = {(a, sc): summarize([r for r in results if r.agent == a and r.scenario == sc])
             for a in agents for sc in scenarios}
    return table, results


def format_table(table, agents, scenarios) -> str:
    header = (f"{'scenario':<10} {'agent':<11} {'mean':>6} {'median':>6} {'p10':>6} {'alive':>6} "
              f"{'combat':>6} {'physio':>6} {'kills':>5} {'pick':>5} {'dmg':>5}")
    lines = [header, "-" * len(header)]
    for sc in scenarios:
        for a in agents:
            m = table[(a, sc)]
            lines.append(
                f"{sc:<10} {a:<11} {m['mean']:6.0f} {m['median']:6.0f} {m['p10']:6.0f} {m['alive']:6.0%} "
                f"{m['combat']:6.0%} {m['physio']:6.0%} {m['kills']:5.2f} {m['pickups']:5.1f} {m['damage']:5.2f}"
            )
    lines.append("")
    lines.append("score combiné (moyenne des p10) : " + ", ".join(
        f"{a}={combined_score(table, a, scenarios):.0f}" for a in agents))
    for a in agents:
        opts = {}
        for sc in scenarios:
            for k, v in table[(a, sc)]["options"].items():
                opts[k] = opts.get(k, 0.0) + v / len(scenarios)
        if opts:
            lines.append(f"options {a} : " + ", ".join(f"{k} {v:.0%}" for k, v in sorted(opts.items(), key=lambda kv: -kv[1])))
    return "\n".join(lines)


def markdown_table(table, agents, scenarios, episodes: int) -> str:
    out = [f"Seeds d'évaluation : {episodes} premières de `EVAL_SEEDS`, `max_ticks=3000`.", "",
           "| Scénario | Agent | Moyenne | Médiane | 10ᵉ centile | Survivants | Morts combat | Morts physio | Kills | Pickups |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for sc in scenarios:
        for a in agents:
            m = table[(a, sc)]
            out.append(f"| {sc} | {a} | {m['mean']:.0f} | {m['median']:.0f} | {m['p10']:.0f} | {m['alive']:.0%} | "
                       f"{m['combat']:.0%} | {m['physio']:.0%} | {m['kills']:.2f} | {m['pickups']:.1f} |")
    out += ["", "| Agent | Score combiné (moyenne des p10) |", "|---|---|"]
    out += [f"| {a} | {combined_score(table, a, scenarios):.0f} |" for a in agents]
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", nargs="+", default=["trivial", "scripted"])
    p.add_argument("--scenarios", nargs="+", default=list(MAIN_SCENARIOS))
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--worst", type=int, default=0, help="affiche les N pires épisodes par (agent, scénario)")
    p.add_argument("--out", type=Path, help="fichier JSON (résumé + épisodes)")
    p.add_argument("--markdown", type=Path, help="tableau Markdown")
    args = p.parse_args()

    table, results = evaluate(args.agents, args.scenarios, args.episodes, args.workers)
    print(format_table(table, args.agents, args.scenarios))

    if args.worst:
        for a in args.agents:
            for sc in args.scenarios:
                rows = sorted((r for r in results if r.agent == a and r.scenario == sc), key=lambda r: r.survival)
                print(f"\n{a} / {sc} : pires épisodes")
                for r in rows[: args.worst]:
                    print(f"  seed {r.seed}  survie {r.survival:5d}  {r.cause:<10} ({r.killer})  "
                          f"dégâts {r.damage:.2f}  pickups {r.pickups}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "summary": {f"{a}/{sc}": m for (a, sc), m in table.items()},
            "episodes": [asdict(r) for r in results],
        }, indent=1), encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_table(table, args.agents, args.scenarios, args.episodes), encoding="utf-8")


if __name__ == "__main__":
    main()
