"""Mémoire spatiale par odométrie, construite uniquement depuis Observation.

L'agent n'a pas accès à sa position : il l'estime en intégrant, à chaque
tick, la vitesse perçue (`self_state.speed`), son orientation et la direction
locale du mouvement qu'il a lui-même commandé. L'estimation dérive (collisions,
bruit, carte torique de taille inconnue) ; les repères vieillissent donc et
sont oubliés.

Repère : (catégorie, position estimée, tick de dernière observation). Aucun
`ref` n'est stocké ; deux détections de même catégorie proches l'une de
l'autre sont fusionnées.

Repère frame : x = cos(h), y = sin(h) avec h angle absolu (sens horaire),
cohérent avec `orientation + bearing`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, hypot, pi, sin

from rl_arena.arena_api import EventKind, Movement, ObjectKind, Observation, TICK_SECONDS

from .perception import CONSUMABLE_KINDS, door_is_closed, wrap

CELL = 2.0               # taille d'une cellule de la grille de visite (m)
MERGE_RADIUS = 1.2       # deux repères de même catégorie plus proches sont fusionnés
FORGET_TICKS = 250       # un repère non revu depuis si longtemps est oublié (dérive de l'odométrie)
VERIFY_RADIUS = 2.0      # proche d'un repère attendu dans le champ de vision mais absent : oublié
TOUCH_SCALE = 0.5        # collision + contact persistant (calibré, voir training/odometry_check.py)
COLLISION_SCALE = 0.5    # collision ponctuelle : déplacement partiel
SOLID_TOUCH = frozenset({ObjectKind.WALL, ObjectKind.OBSTACLE, ObjectKind.MOVING_OBSTACLE, ObjectKind.DOOR,
                         ObjectKind.MACHINE, ObjectKind.CONTAINER, ObjectKind.AGENT})
MEMORY_KINDS = CONSUMABLE_KINDS | {ObjectKind.WEAPON, ObjectKind.AMMUNITION}


@dataclass(slots=True)
class Landmark:
    kind: ObjectKind
    x: float
    y: float
    seen_tick: int
    passed: bool = False     # porte : déjà franchie/atteinte
    closed: bool | None = None


@dataclass(slots=True)
class SpatialMap:
    x: float = 0.0
    y: float = 0.0
    items: list[Landmark] = field(default_factory=list)
    doors: list[Landmark] = field(default_factory=list)
    visits: dict = field(default_factory=dict)
    last_move: Movement | None = None

    # ------------------------------------------------------------------
    # Mise à jour
    # ------------------------------------------------------------------
    def integrate(self, obs: Observation) -> None:
        """Avance la position estimée du déplacement effectué au tick précédent."""
        s = obs.self_state
        mv = self.last_move
        if mv is not None and s.speed > 1e-3:
            local = atan2(mv.right, mv.forward) if (mv.forward or mv.right) else 0.0
            h = s.orientation + local
            step = s.speed * TICK_SECONDS
            vx, vy = step * cos(h), step * sin(h)
            # Collisions : le moteur annule tout ou partie du déplacement (axes
            # résolus séparément). Facteurs calibrés contre le moteur de test
            # (diagnostic hors politique) : voir training/odometry_check.py.
            collided = any(e.kind is EventKind.COLLISION and e.properties.get("with") != "agent"
                           for e in obs.events)
            if collided:
                touching = any(c.kind in SOLID_TOUCH for c in obs.touch)
                k = TOUCH_SCALE if touching else COLLISION_SCALE
                vx, vy = k * vx, k * vy
            self.x += vx
            self.y += vy
        cell = (round(self.x / CELL), round(self.y / CELL))
        self.visits[cell] = self.visits.get(cell, 0) + 1

    def observe(self, obs: Observation) -> None:
        s = obs.self_state
        tick = obs.tick
        seen_items: list[tuple[ObjectKind, float, float]] = []
        for d in obs.vision:
            if d.kind in MEMORY_KINDS or d.kind is ObjectKind.DOOR:
                h = s.orientation + d.bearing
                px, py = self.x + d.distance * cos(h), self.y + d.distance * sin(h)
                if d.kind is ObjectKind.DOOR:
                    lm = self._merge(self.doors, d.kind, px, py, tick)
                    closed = door_is_closed(d)
                    if closed is not None:
                        lm.closed = closed
                    if d.distance < 0.6:
                        lm.passed = True
                else:
                    self._merge(self.items, d.kind, px, py, tick)
                    seen_items.append((d.kind, px, py))

        # Oubli : repère proche, normalement dans le champ de vision, mais absent.
        kept = []
        for lm in self.items:
            dist = hypot(lm.x - self.x, lm.y - self.y)
            if tick - lm.seen_tick > FORGET_TICKS:
                continue
            if dist < VERIFY_RADIUS and tick != lm.seen_tick:
                b = wrap(atan2(lm.y - self.y, lm.x - self.x) - s.orientation)
                if abs(b) < pi / 3:
                    continue  # devrait être visible : ramassé par quelqu'un ou mal estimé
            kept.append(lm)
        self.items = kept
        self.doors = [d for d in self.doors if tick - d.seen_tick <= 2 * FORGET_TICKS]

        # Un objet ramassé par nous disparaît de la carte.
        if any(e.kind is EventKind.ITEM_PICKED_UP for e in obs.events):
            self.items = [lm for lm in self.items if hypot(lm.x - self.x, lm.y - self.y) > 1.5]

    @staticmethod
    def _merge(store: list[Landmark], kind: ObjectKind, x: float, y: float, tick: int) -> Landmark:
        for lm in store:
            if lm.kind is kind and hypot(lm.x - x, lm.y - y) < MERGE_RADIUS:
                lm.x, lm.y, lm.seen_tick = 0.5 * (lm.x + x), 0.5 * (lm.y + y), tick
                return lm
        lm = Landmark(kind, x, y, tick)
        store.append(lm)
        return lm

    # ------------------------------------------------------------------
    # Requêtes
    # ------------------------------------------------------------------
    def bearing_to(self, x: float, y: float, orientation: float) -> tuple[float, float]:
        return wrap(atan2(y - self.y, x - self.x) - orientation), hypot(x - self.x, y - self.y)

    def nearest_item(self, kinds) -> Landmark | None:
        cands = [lm for lm in self.items if lm.kind in kinds]
        return min(cands, key=lambda lm: hypot(lm.x - self.x, lm.y - self.y)) if cands else None

    def unvisited_door(self) -> Landmark | None:
        cands = [d for d in self.doors if not d.passed]
        return min(cands, key=lambda d: hypot(d.x - self.x, d.y - self.y)) if cands else None

    def novelty_heading(self, orientation: float, n: int = 12, horizon: int = 4) -> float:
        """Cap absolu menant vers les cellules les moins visitées."""
        best_h, best_score = orientation, -1e9
        for i in range(n):
            h = orientation + (2 * pi * i / n if i <= n // 2 else -2 * pi * (n - i) / n)
            score = 0.0
            for k in range(1, horizon + 1):
                cx, cy = self.x + k * CELL * cos(h), self.y + k * CELL * sin(h)
                score += 1.0 / (1.0 + self.visits.get((round(cx / CELL), round(cy / CELL)), 0))
            score -= 0.15 * abs(wrap(h - orientation))  # éviter de faire demi-tour sans raison
            if score > best_score:
                best_h, best_score = h, score
        return wrap(best_h)
