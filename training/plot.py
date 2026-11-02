"""Courbes d'apprentissage en SVG (sans dépendance externe).

Usage :
    python -m training.plot results/phase3

Produit `learning_curves.svg` : survie en entraînement (moyenne glissante),
survie en évaluation gloutonne (moyenne et 10ᵉ centile) par étape, et
référence scriptée de chaque étape en pointillés.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

W, H = 900, 420
ML, MR, MT, MB = 70, 190, 40, 50
COLORS = {"train": "#9aa5b1", "mean": "#1f6feb", "p10": "#d1242f", "scripted": "#2da44e"}


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _smooth(ys: list[float], w: int = 5) -> list[float]:
    out = []
    for i in range(len(ys)):
        win = ys[max(0, i - w + 1): i + 1]
        out.append(sum(win) / len(win))
    return out


def render(run_dir: Path) -> Path:
    train = _read(run_dir / "train_log.csv")
    evals = _read(run_dir / "eval_log.csv")
    learned = [e for e in evals if e["snapshot"] != "scripted"]
    refs = [e for e in evals if e["snapshot"] == "scripted"]

    x_max = max([float(r["episode"]) for r in train] + [1.0])
    ys = [float(r["train_mean"]) for r in train] + [float(e["mean"]) for e in evals] + [float(e["p10"]) for e in evals]
    y_max = max(ys + [100.0]) * 1.08

    def px(x: float) -> float:
        return ML + (W - ML - MR) * x / x_max

    def py(y: float) -> float:
        return H - MB - (H - MT - MB) * y / y_max

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="12">',
             f'<rect width="{W}" height="{H}" fill="white"/>',
             f'<text x="{ML}" y="22" font-size="15" font-weight="bold">Q-learning SMDP sur options : survie (ticks)</text>']
    # Axes et grille.
    for i in range(6):
        y = y_max * i / 5
        parts.append(f'<line x1="{ML}" x2="{W - MR}" y1="{py(y):.1f}" y2="{py(y):.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{ML - 8}" y="{py(y) + 4:.1f}" text-anchor="end">{y:.0f}</text>')
    for i in range(6):
        x = x_max * i / 5
        parts.append(f'<text x="{px(x):.1f}" y="{H - MB + 18}" text-anchor="middle">{x:.0f}</text>')
    parts.append(f'<text x="{(ML + W - MR) / 2}" y="{H - 12}" text-anchor="middle">épisodes d\'entraînement</text>')
    parts.append(f'<line x1="{ML}" x2="{W - MR}" y1="{py(0)}" y2="{py(0)}" stroke="#333"/>')
    parts.append(f'<line x1="{ML}" x2="{ML}" y1="{MT}" y2="{py(0)}" stroke="#333"/>')

    # Frontières d'étapes.
    prev = None
    for r in train:
        if prev is not None and r["stage"] != prev["stage"]:
            x = px(float(prev["episode"]))
            parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{MT}" y2="{py(0)}" stroke="#bbb" stroke-dasharray="3,3"/>')
            parts.append(f'<text x="{x + 4:.1f}" y="{MT + 12}" fill="#666">étape {r["stage"]}</text>')
        prev = r
    if train:
        parts.append(f'<text x="{ML + 4}" y="{MT + 12}" fill="#666">étape {train[0]["stage"]}</text>')

    def polyline(points, color, width=2.0, dash=None):
        if len(points) < 2:
            return
        d = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in points)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="{width}"{extra}/>')

    tx = [float(r["episode"]) for r in train]
    polyline(list(zip(tx, _smooth([float(r["train_mean"]) for r in train]))), COLORS["train"], 1.5)
    for key in ("mean", "p10"):
        pts = [(float(e["episode"]), float(e[key])) for e in learned]
        polyline(pts, COLORS[key], 2.2)
        for x, y in pts:
            parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3" fill="{COLORS[key]}"/>')

    # Références scriptées : segment horizontal sur la plage de l'étape.
    for ref in refs:
        stage_rows = [r for r in train if r["stage"] == ref["stage"]]
        if not stage_rows:
            continue
        idx = train.index(stage_rows[0])
        x0 = float(train[idx - 1]["episode"]) if idx > 0 else 0.0
        x1 = float(stage_rows[-1]["episode"])
        for key, dash in (("mean", "6,4"), ("p10", "2,3")):
            polyline([(x0, float(ref[key])), (x1, float(ref[key]))], COLORS["scripted"], 1.6, dash)

    legend = [("entraînement (moy. glissante, ε-greedy)", COLORS["train"], None),
              ("éval gloutonne : moyenne", COLORS["mean"], None),
              ("éval gloutonne : 10ᵉ centile", COLORS["p10"], None),
              ("scripté : moyenne", COLORS["scripted"], "6,4"),
              ("scripté : 10ᵉ centile", COLORS["scripted"], "2,3")]
    for i, (label, color, dash) in enumerate(legend):
        y = MT + 30 + 22 * i
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{W - MR + 12}" x2="{W - MR + 36}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="2.2"{extra}/>')
        parts.append(f'<text x="{W - MR + 42}" y="{y + 4}" font-size="11">{label}</text>')
    parts.append("</svg>")
    out = run_dir / "learning_curves.svg"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(render(Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase3")))
