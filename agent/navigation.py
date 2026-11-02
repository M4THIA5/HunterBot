"""Pilotage bas niveau : choisir une direction libre proche d'une direction voulue.

Chaque détection bloquante est modélisée comme un segment perpendiculaire à sa
direction, situé à sa distance perçue, de demi-largeur déduite de
`angular_size`. Pour une direction candidate c, la distance libre est celle à
laquelle le rayon c rencontre ce segment (élargi du rayon du corps).
"""

from __future__ import annotations

from math import cos, pi, sin, tan

from rl_arena.arena_api import Movement, ObjectKind, Observation, Rotation, TICK_SECONDS

from .perception import Memory, is_blocking, wrap

BODY_MARGIN = 0.65          # demi-largeur du corps + marge (m)
LOOKAHEAD = 3.0             # au-delà, une direction est considérée libre (m)
HALF_FOV_KNOWN = pi / 3     # hors de ce cône, on n'a pas d'information visuelle
CANDIDATES = [i * pi / 16 for i in range(-16, 16)]


def _ray_clearance(c: float, obstacles) -> float:
    free = LOOKAHEAD
    for d, b, ext in obstacles:
        delta = wrap(c - b)
        if abs(delta) >= pi / 2:
            continue
        lateral = d * abs(tan(delta)) if abs(delta) < 1.5 else 1e9
        if lateral <= ext + BODY_MARGIN:
            dist = d / max(1e-3, cos(delta))
            if dist < free:
                free = dist
    return free


def blocking_obstacles(obs: Observation, ignore_ref: int | None = None, avoid_agents: bool = True):
    out = []
    for det in obs.vision:
        if det.ref == ignore_ref or det.distance > LOOKAHEAD + 1.0:
            continue
        if det.kind is ObjectKind.AGENT and not avoid_agents:
            continue
        if not is_blocking(det):
            continue
        ext = det.distance * tan(min(det.angular_size, 3.0) / 2) if det.distance > 0.05 else 1.0
        ext = min(ext, 6.0)
        if det.kind is ObjectKind.AGENT:
            ext = max(ext, 0.6)
        out.append((max(0.0, det.distance), det.bearing, ext))
    # Toucher : un contact solide bloque fortement la direction du contact.
    for c in obs.touch:
        if c.kind in (ObjectKind.WALL, ObjectKind.OBSTACLE, ObjectKind.MOVING_OBSTACLE, ObjectKind.DOOR,
                      ObjectKind.MACHINE, ObjectKind.CONTAINER, ObjectKind.HAZARD):
            if c.ref != ignore_ref:
                out.append((0.0, c.bearing, 0.8))
    return out


def choose_direction(obs: Observation, desired: float | None, mem: Memory, ignore_ref: int | None = None,
                     avoid_agents: bool = True) -> tuple[float, float]:
    """Retourne (bearing choisi, distance libre dans cette direction)."""
    obstacles = blocking_obstacles(obs, ignore_ref, avoid_agents)
    if desired is None:
        desired = 0.0
    best_c, best_score, best_free = 0.0, -1e9, 0.0
    for c in CANDIDATES:
        free = _ray_clearance(c, obstacles)
        if abs(c) > HALF_FOV_KNOWN:
            free = min(free, LOOKAHEAD * 0.7)  # inconnu : un peu pénalisé
        score = 2.0 * min(free, LOOKAHEAD) - 1.6 * abs(wrap(c - desired)) - 0.3 * abs(c)
        if free < 0.8:
            score -= 10.0
        if score > best_score:
            best_c, best_score, best_free = c, score, free
    return best_c, best_free


def drive(direction: float, speed: float, free: float | None = None, max_turn: float = pi) -> tuple[Movement, Rotation]:
    """Produit (Movement, Rotation) pour aller vers `direction` (bearing local).

    La rotation est appliquée avant le déplacement dans le tick : on tourne au
    maximum vers la direction, puis on se déplace selon l'écart restant.
    """
    turn = max(-max_turn * TICK_SECONDS, min(max_turn * TICK_SECONDS, direction))
    rest = wrap(direction - turn)
    fwd, right = cos(rest), sin(rest)
    if free is not None:
        speed = min(speed, max(0.4, free * 1.2))
    if abs(rest) > pi / 2:
        speed *= 0.5
    return Movement(fwd, right, speed), Rotation(turn / TICK_SECONDS)


def free_gap_bearing(obs: Observation, mem: Memory) -> float:
    """Centre du plus grand secteur angulaire libre de tout contact proche."""
    blocked = [c.bearing for c in obs.touch if c.kind not in (ObjectKind.WATER, ObjectKind.STEAM)]
    blocked += [d.bearing for d in obs.vision if d.distance < 0.9 and is_blocking(d)]
    tb = mem.threat_bearing(obs)
    if tb is not None and mem.threat_active(3) and mem.threat_distance < 1.5:
        blocked.append(tb)
    if not blocked:
        return pi
    return max(CANDIDATES, key=lambda c: min(abs(wrap(c - b)) for b in blocked))


def steer(obs: Observation, desired: float | None, speed: float, mem: Memory, ignore_ref: int | None = None,
          avoid_agents: bool = True) -> tuple[Movement, Rotation]:
    # Manœuvre de dégagement après collisions répétées (coin, agent qui pousse).
    if mem.escape_ticks > 0 and mem.escape_heading is not None:
        b = mem.heading_to_bearing(mem.escape_heading, obs.self_state.orientation)
        return drive(b, min(speed, 1.6))
    if mem.collisions_recent >= 3 and obs.tick - mem.last_collision_tick <= 1:
        gap = free_gap_bearing(obs, mem)
        mem.escape_heading = wrap(obs.self_state.orientation + gap)
        mem.escape_ticks = 4
        mem.collisions_recent = 0
        return drive(gap, 1.4)
    c, free = choose_direction(obs, desired, mem, ignore_ref, avoid_agents)
    return drive(c, speed, free)
