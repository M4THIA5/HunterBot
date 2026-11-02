from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def wrap_angle(angle: float) -> float:
    return (angle + pi) % (2 * pi) - pi


def wrap_pos(x: float, size: float) -> float:
    return x % size


def torus_delta(a: float, b: float, size: float) -> float:
    """Shortest signed delta from a to b on a torus axis."""
    d = (b - a + size / 2) % size - size / 2
    return d


def torus_vector(ax: float, ay: float, bx: float, by: float, width: float, height: float) -> tuple[float, float]:
    return torus_delta(ax, bx, width), torus_delta(ay, by, height)


def torus_distance(ax: float, ay: float, bx: float, by: float, width: float, height: float) -> float:
    dx, dy = torus_vector(ax, ay, bx, by, width, height)
    return hypot(dx, dy)


def heading_vector(theta: float) -> tuple[float, float]:
    # Positive angles are clockwise in a world where +y points upward.
    return cos(theta), -sin(theta)


def right_vector(theta: float) -> tuple[float, float]:
    return -sin(theta), -cos(theta)


def bearing_from_vector(dx: float, dy: float, orientation: float) -> float:
    world_angle_clockwise = atan2(-dy, dx)
    return wrap_angle(world_angle_clockwise - orientation)


def normalize(x: float, y: float) -> tuple[float, float]:
    n = hypot(x, y)
    if n <= 1e-12:
        return 0.0, 0.0
    return x / n, y / n


@dataclass(slots=True, frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def left(self) -> float:
        return self.x - self.w / 2

    @property
    def right(self) -> float:
        return self.x + self.w / 2

    @property
    def bottom(self) -> float:
        return self.y - self.h / 2

    @property
    def top(self) -> float:
        return self.y + self.h / 2


def point_in_rect(px: float, py: float, r: Rect) -> bool:
    return r.left <= px <= r.right and r.bottom <= py <= r.top


def circle_rect_overlap(cx: float, cy: float, radius: float, r: Rect) -> bool:
    qx = clamp(cx, r.left, r.right)
    qy = clamp(cy, r.bottom, r.top)
    return (cx - qx) ** 2 + (cy - qy) ** 2 < radius ** 2


def nearest_point_on_rect(px: float, py: float, r: Rect) -> tuple[float, float]:
    return clamp(px, r.left, r.right), clamp(py, r.bottom, r.top)


def segment_intersects_rect(x1: float, y1: float, x2: float, y2: float, r: Rect) -> bool:
    """Liang-Barsky line clipping."""
    dx = x2 - x1
    dy = y2 - y1
    p = (-dx, dx, -dy, dy)
    q = (x1 - r.left, r.right - x1, y1 - r.bottom, r.top - y1)
    u1, u2 = 0.0, 1.0
    for pi_, qi in zip(p, q):
        if abs(pi_) < 1e-12:
            if qi < 0:
                return False
            continue
        t = qi / pi_
        if pi_ < 0:
            if t > u2:
                return False
            u1 = max(u1, t)
        else:
            if t < u1:
                return False
            u2 = min(u2, t)
    return True



def nearest_toric_rect(px: float, py: float, r: Rect, width: float, height: float) -> Rect:
    """Return the periodic copy of *r* whose center is nearest to (px, py)."""
    dx = torus_delta(px, r.x, width)
    dy = torus_delta(py, r.y, height)
    return Rect(px + dx, py + dy, r.w, r.h)


def shifted_rects(r: Rect, width: float, height: float):
    for sx in (-width, 0.0, width):
        for sy in (-height, 0.0, height):
            yield Rect(r.x + sx, r.y + sy, r.w, r.h)
