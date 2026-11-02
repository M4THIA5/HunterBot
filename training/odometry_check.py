"""Diagnostic de l'odométrie de agent/spatial.py contre la vraie position du moteur.

Usage :
    python -m training.odometry_check --seeds 6 --ticks 600

Lit l'état interne du moteur : outil de calibration uniquement, la politique
n'y accède jamais.
"""

from __future__ import annotations

import argparse
import math
import statistics

from rl_arena.geometry import torus_delta

from agent import spatial
from agent.scripted import ScriptedAgent
from training.envs import EVAL_SEEDS, SCENARIOS, STUDENT_ID, build_env


def drift(seed: int, ticks: int, scenario: str = "default") -> list[float]:
    a = ScriptedAgent()
    spec = SCENARIOS[scenario].spec(seed)
    env = build_env(spec, student=a)
    env.reset(seed=seed)
    W, H = spec.config.map_width, spec.config.map_height
    st = env.debug_state()["agents"][STUDENT_ID]
    px, py, tx, ty = st.x, st.y, 0.0, 0.0
    prev = (0.0, 0.0)
    out = []
    for t in range(1, ticks + 1):
        res = env.step({})
        if res.terminated[STUDENT_ID]:
            break
        st = env.debug_state()["agents"][STUDENT_ID]
        tx += torus_delta(px, st.x, W)
        ty += torus_delta(py, st.y, H)
        px, py = st.x, st.y
        sp = a.runner.mem.spatial
        # L'estimation a un tick de retard ; y agent = -y moteur.
        if t % 100 == 0:
            out.append(math.hypot(sp.x - prev[0], sp.y + prev[1]))
        prev = (tx, ty)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--ticks", type=int, default=600)
    p.add_argument("--grid", action="store_true", help="balaye TOUCH_SCALE et COLLISION_SCALE")
    args = p.parse_args()
    seeds = list(EVAL_SEEDS)[: args.seeds]
    combos = [(t, c) for t in (0.0, 0.25, 0.5) for c in (0.25, 0.5, 0.75, 1.0)] if args.grid \
        else [(spatial.TOUCH_SCALE, spatial.COLLISION_SCALE)]
    for t, c in combos:
        spatial.TOUCH_SCALE, spatial.COLLISION_SCALE = t, c
        runs = [drift(s, args.ticks) for s in seeds]
        by_t = list(zip(*[r for r in runs if len(r) == len(runs[0])]))
        summary = "  ".join(f"t={100 * (i + 1)}: {statistics.median(v):5.1f} m" for i, v in enumerate(by_t))
        print(f"touch={t:.2f} coll={c:.2f}  dérive médiane  {summary}")


if __name__ == "__main__":
    main()
