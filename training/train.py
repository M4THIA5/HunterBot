"""Q-learning tabulaire sur options (SMDP) avec curriculum (§6, §7).

Usage :
    python -m training.train --stages 0 1 --episodes 640 960 --out results/phase3

Organisation (§7.4) : à chaque « round », des workers jouent des épisodes en
parallèle avec une copie figée de la Q-table et renvoient leurs transitions
(s, o, R, k, s', valides(s'), terminal) ; le processus principal applique les
mises à jour Q-learning (off-policy, donc tolérant au léger décalage).

Mise à jour SMDP :
    Q(s,o) <- Q(s,o) + α [ R + γ^k max_{o' valide} Q(s',o') - Q(s,o) ]
avec valeur future nulle à la mort et bootstrap en cas de troncature.

Récompense d'entraînement (§5.2, recalibrée) : RewardConfig(kill=0.5,
first_pickup=0.2, harmful_collision=0.05) + shaping par potentiel
Φ = 2 health + 4 min(hydration, satiety) + [boisson] + [nourriture].
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from random import Random

from rl_arena.arena_api import ObjectKind
from rl_arena.config import RewardConfig

from agent.options import OPTION_NAMES
from agent.qpolicy import QAgent, QTable, Transition
from training.envs import EVAL_SEEDS, STAGES, STUDENT_ID, build_env, sample_curriculum_spec, set_self_play_tables
from training.evaluate import percentile

GAMMA = 0.998
# Collision réduite : à -0.5 elle dominait le gain d'une ressource trouvée et
# l'agent apprenait à rester immobile (run 1, voir results/phase3_run1_myopic).
TRAIN_REWARD = dict(kill=0.5, first_pickup=0.2, harmful_collision=0.05)


def potential(obs) -> float:
    """Φ : santé, besoin vital le plus bas et réserves en inventaire.

    Les réserves rendent le ramassage immédiatement payant alors que son
    bénéfice réel (ne pas mourir de soif) arrive des centaines de ticks plus
    tard, au-delà de l'horizon effectif de γ.
    """
    s = obs.self_state
    kinds = {i.kind for i in s.inventory}
    stock = (ObjectKind.DRINK in kinds) + (ObjectKind.FOOD in kinds)
    return 2.0 * s.health + 4.0 * min(s.hydration, s.satiety) + 1.0 * stock


@dataclass(slots=True)
class EpisodeOutcome:
    survival: int
    died: bool
    killer: str | None
    transitions: list[Transition]
    decisions: dict


def play_episode(table: QTable, epsilon: float, stage: int, seed: int, explore_scripted: float,
                 learn: bool = True, replay: float = 0.2, spec=None, prior: tuple[float, float] = (0.0, 0.0),
                 selfplay: tuple[str, ...] = ()) -> EpisodeOutcome:
    set_self_play_tables(selfplay)
    rng = Random(seed)
    if spec is None:
        spec = sample_curriculum_spec(stage, rng, seed, replay=replay)
    spec.config.reward = RewardConfig(**TRAIN_REWARD)
    transitions: list[Transition] = []
    agent = QAgent(table, epsilon=epsilon, seed=seed, gamma=GAMMA,
                   transition_sink=transitions.append if learn else None, explore_scripted=explore_scripted,
                   prior_bonus=prior[0], prior_kappa=prior[1])
    env = build_env(spec)  # contrôle externe : on récupère les récompenses
    obs = env.reset(seed=seed)[STUDENT_ID]
    agent.reset()
    while True:
        action = agent.act(obs)
        res = env.step({STUDENT_ID: action})
        died = res.terminated[STUDENT_ID]
        r = res.rewards[STUDENT_ID]
        nxt = None if died else res.observations.get(STUDENT_ID)
        r += GAMMA * (potential(nxt) if nxt is not None else 0.0) - potential(obs)
        agent.record(r)
        if died or res.truncated:
            agent.end_episode(nxt, terminal=died)
            killer = env.debug_state()["agents"][STUDENT_ID].last_damage_source if died else None
            return EpisodeOutcome(res.info[STUDENT_ID].survival_ticks, died, killer, transitions, agent.decisions)
        obs = nxt


def _collect(args) -> EpisodeOutcome:
    return play_episode(*args)


def _eval_one(args) -> tuple[int, bool, str | None]:
    table, stage, seed, prior, selfplay = args
    set_self_play_tables(selfplay)
    rng = Random(seed)
    spec = STAGES[stage].sample_spec(rng, seed)  # pas de rejeu : étape pure
    out = play_episode(table, 0.0, stage, seed, 0.0, learn=False, spec=spec, prior=prior)
    return out.survival, out.died, out.killer


class Learner:
    def __init__(self, table: QTable | None = None, alpha_min: float = 0.02, alpha_power: float = 0.6):
        self.table = table or QTable()
        self.alpha_min = alpha_min
        self.alpha_power = alpha_power
        self.default_value = 0.0  # valeur d'un état jamais vu (moyenne des V connues)
        self.updates = 0

    def update(self, tr: Transition) -> None:
        q, n = self.table.ensure(tr.state)
        n[tr.option] += 1
        alpha = max(self.alpha_min, (1.0 + n[tr.option]) ** -self.alpha_power)
        if tr.terminal or tr.next_state is None:
            future = 0.0
        else:
            nq, nn = self.table.values(tr.next_state), self.table.visits(tr.next_state)
            tried = [nq[i] for i in tr.next_valid if nn[i] > 0] if nq is not None else []
            future = max(tried) if tried else self.default_value
        target = tr.reward + (GAMMA ** tr.duration) * future
        q[tr.option] += alpha * (target - q[tr.option])
        self.updates += 1

    def refresh_default_value(self) -> None:
        vals = [max(v for v, c in zip(q, n) if c > 0) for q, n in zip(self.table.q.values(), self.table.n.values())
                if any(n)]
        self.default_value = statistics.fmean(vals) if vals else 0.0


def epsilon_at(episode: int, total: int, start: float, end: float, frac: float) -> float:
    horizon = max(1, int(total * frac))
    return max(end, start - (start - end) * episode / horizon)


def evaluate_stage(pool, table: QTable, stage: int, episodes: int, prior=(0.0, 0.0), selfplay=()) -> dict:
    seeds = list(EVAL_SEEDS)[:episodes]
    rows = list(pool.map(_eval_one, [(table, stage, s, prior, tuple(selfplay)) for s in seeds], chunksize=1))
    surv = [r[0] for r in rows]
    return {
        "mean": statistics.fmean(surv), "median": statistics.median(surv), "p10": percentile(surv, 0.1),
        "combat": sum(1 for r in rows if r[1] and r[2] is not None) / len(rows),
        "physio": sum(1 for r in rows if r[1] and r[2] is None) / len(rows),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stages", type=int, nargs="+", default=[0, 1])
    p.add_argument("--episodes", type=int, nargs="+", default=[640, 960], help="budget max par étape")
    p.add_argument("--advance", type=float, nargs="+", default=[],
                   help="survie moyenne (éval) pour passer à l'étape suivante, par étape")
    p.add_argument("--round", type=int, default=32, help="épisodes par round")
    p.add_argument("--eval-every", type=int, default=160)
    p.add_argument("--eval-episodes", type=int, default=32)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-frac", type=float, default=0.6)
    p.add_argument("--explore-scripted", type=float, default=0.3,
                   help="part des actions exploratoires tirées de la politique scriptée")
    p.add_argument("--init", type=Path, help="Q-table de départ")
    p.add_argument("--prior-bonus", type=float, default=0.0,
                   help="bonus de Q ajouté à l'option scriptée au choix glouton (0 : désactivé)")
    p.add_argument("--prior-kappa", type=float, default=0.0,
                   help="décroissance du bonus : bonus*κ/(κ+N(s)) ; <= 0 : bonus constant")
    p.add_argument("--skip-reference", action="store_true", help="ne pas réévaluer le scripté")
    p.add_argument("--selfplay-size", type=int, default=3,
                   help="nombre de sauvegardes récentes de ce run utilisées comme adversaires (étape 3)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("results/phase3"))
    args = p.parse_args()
    prior = (args.prior_bonus, args.prior_kappa)
    prior_meta = {"prior": {"bonus": args.prior_bonus, "kappa": args.prior_kappa}}

    args.out.mkdir(parents=True, exist_ok=True)
    learner = Learner(QTable.load(args.init) if args.init else None)
    total = sum(args.episodes[: len(args.stages)])
    rng = Random(args.seed)
    log_path = args.out / "train_log.csv"
    eval_path = args.out / "eval_log.csv"
    log_f = log_path.open("w", newline="", encoding="utf-8")
    eval_f = eval_path.open("w", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    ev = csv.writer(eval_f)
    log.writerow(["episode", "stage", "epsilon", "train_mean", "train_p10", "deaths_combat", "states", "updates",
                  "fallback_share", "seconds"])
    ev.writerow(["episode", "stage", "mean", "median", "p10", "combat", "physio", "states", "snapshot"])

    best = {"score": -1.0, "path": None}
    snapshots: list[str] = []
    episode = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        # Référence : Q-table vide => repli scripté à 100 %, mêmes seeds d'éval.
        for stage in ([] if args.skip_reference else args.stages):
            m = evaluate_stage(pool, QTable(), stage, args.eval_episodes)
            ev.writerow([0, stage, f"{m['mean']:.1f}", f"{m['median']:.1f}", f"{m['p10']:.1f}",
                         f"{m['combat']:.3f}", f"{m['physio']:.3f}", 0, "scripted"])
            print(f"[référence scriptée, étape {stage}] moyenne {m['mean']:.0f}  médiane {m['median']:.0f}  "
                  f"p10 {m['p10']:.0f}  combat {m['combat']:.0%}", flush=True)
        eval_f.flush()
        for si, stage in enumerate(args.stages):
            budget = args.episodes[si]
            threshold = args.advance[si] if si < len(args.advance) else None
            done_in_stage = 0
            since_eval = 0
            print(f"=== étape {stage} ({STAGES[stage].name}) : budget {budget} épisodes")
            while done_in_stage < budget:
                eps = epsilon_at(episode, total, args.eps_start, args.eps_end, args.eps_frac)
                n = min(args.round, budget - done_in_stage)
                seeds = [rng.randrange(0, 1_000_000) for _ in range(n)]
                selfplay = tuple(snapshots[-args.selfplay_size:])
                jobs = [(learner.table, eps, stage, s, args.explore_scripted, True, 0.2, None, prior, selfplay)
                        for s in seeds]
                outcomes = list(pool.map(_collect, jobs, chunksize=1))
                for out in outcomes:
                    for tr in out.transitions:
                        learner.update(tr)
                learner.refresh_default_value()
                episode += n
                done_in_stage += n
                since_eval += n
                surv = [o.survival for o in outcomes]
                dec = {k: sum(o.decisions[k] for o in outcomes) for k in ("q", "fallback", "random")}
                fb = dec["fallback"] / max(1, dec["q"] + dec["fallback"])
                log.writerow([episode, stage, f"{eps:.3f}", f"{statistics.fmean(surv):.1f}",
                              f"{percentile(surv, 0.1):.1f}",
                              f"{sum(1 for o in outcomes if o.died and o.killer) / n:.3f}",
                              len(learner.table.q), learner.updates, f"{fb:.3f}", f"{time.time() - t0:.0f}"])
                log_f.flush()
                print(f"ep {episode:5d}  ε={eps:.2f}  survie={statistics.fmean(surv):6.0f}  "
                      f"états={len(learner.table.q):5d}  repli={fb:.0%}  {time.time() - t0:.0f}s", flush=True)

                last = done_in_stage >= budget
                if since_eval >= args.eval_every or last:
                    since_eval = 0
                    m = evaluate_stage(pool, learner.table, stage, args.eval_episodes, prior,
                                       snapshots[-args.selfplay_size:])
                    snap = args.out / f"q_ep{episode:05d}.json"
                    learner.table.save(snap, {"episode": episode, "stage": stage, "eval": m} | prior_meta)
                    snapshots.append(str(snap))
                    ev.writerow([episode, stage, f"{m['mean']:.1f}", f"{m['median']:.1f}", f"{m['p10']:.1f}",
                                 f"{m['combat']:.3f}", f"{m['physio']:.3f}", len(learner.table.q), snap.name])
                    eval_f.flush()
                    print(f"  [éval étape {stage}] moyenne {m['mean']:.0f}  médiane {m['median']:.0f}  "
                          f"p10 {m['p10']:.0f}  combat {m['combat']:.0%}", flush=True)
                    # Sélection : la dernière étape entraînée prime, puis le p10.
                    score = 10_000 * si + m["p10"] + 0.25 * m["mean"]
                    if score > best["score"]:
                        best = {"score": score, "path": snap}
                    if threshold is not None and m["mean"] >= threshold:
                        print(f"  critère atteint ({m['mean']:.0f} >= {threshold:.0f}) : étape suivante")
                        break

    learner.table.save(args.out / "q_final.json", {"episode": episode} | prior_meta)
    if best["path"] is not None:
        (args.out / "q_best.json").write_text(best["path"].read_text(encoding="utf-8"), encoding="utf-8")
    (args.out / "run.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()} |
                                                  {"best": str(best["path"]), "options": list(OPTION_NAMES)},
                                                  indent=1), encoding="utf-8")
    log_f.close()
    eval_f.close()
    print(f"terminé : {episode} épisodes, {len(learner.table.q)} états, meilleure version {best['path']}")


if __name__ == "__main__":
    main()
