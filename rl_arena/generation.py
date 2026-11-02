from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from .arena_api import ObjectKind
from .config import EnvironmentConfig
from .geometry import Rect, point_in_rect
from .world import AreaEffect, DoorEntity, RectEntity, WorldObject, Zone


@dataclass(frozen=True, slots=True)
class BuildingVisual:
    label: str
    rect: Rect
    theme: str


@dataclass(frozen=True, slots=True)
class DecorationVisual:
    kind: str
    x: float
    y: float
    w: float = 1.0
    h: float = 1.0
    rotation: float = 0.0


@dataclass(slots=True)
class GeneratedWorld:
    rects: list[RectEntity] = field(default_factory=list)
    doors: list[DoorEntity] = field(default_factory=list)
    objects: list[WorldObject] = field(default_factory=list)
    areas: list[AreaEffect] = field(default_factory=list)
    zones: list[Zone] = field(default_factory=list)
    buildings: list[BuildingVisual] = field(default_factory=list)
    decorations: list[DecorationVisual] = field(default_factory=list)
    visibility_factor: float = 1.0
    low_light_factor: float = 1.0


class _Ids:
    def __init__(self) -> None:
        self.value = 1

    def next(self) -> int:
        out = self.value
        self.value += 1
        return out


def _add_building(
    world: GeneratedWorld,
    ids: _Ids,
    *,
    cx: float,
    cy: float,
    w: float,
    h: float,
    door_side: str,
    label: str,
    theme: str,
) -> None:
    t = 0.35
    gap = 1.4
    world.buildings.append(BuildingVisual(label, Rect(cx, cy, w, h), theme))
    # Door is centered on the chosen side; wall is split around it.
    if door_side in {"top", "bottom"}:
        y = cy + (h / 2 if door_side == "top" else -h / 2)
        segment = (w - gap) / 2
        for sign in (-1, 1):
            x = cx + sign * (gap / 2 + segment / 2)
            world.rects.append(
                RectEntity(ids.next(), ObjectKind.WALL, Rect(x, y, segment, t), properties={"building": label})
            )
        world.rects.append(RectEntity(ids.next(), ObjectKind.WALL, Rect(cx - w / 2, cy, t, h), properties={"building": label}))
        world.rects.append(RectEntity(ids.next(), ObjectKind.WALL, Rect(cx + w / 2, cy, t, h), properties={"building": label}))
        opposite_y = cy - (h / 2 if door_side == "top" else -h / 2)
        world.rects.append(RectEntity(ids.next(), ObjectKind.WALL, Rect(cx, opposite_y, w, t), properties={"building": label}))
        world.doors.append(
            DoorEntity(ids.next(), ObjectKind.DOOR, Rect(cx, y, gap, t), properties={"building": label, "label": "door"}, open=False)
        )
    else:
        x = cx + (w / 2 if door_side == "right" else -w / 2)
        segment = (h - gap) / 2
        for sign in (-1, 1):
            y = cy + sign * (gap / 2 + segment / 2)
            world.rects.append(
                RectEntity(ids.next(), ObjectKind.WALL, Rect(x, y, t, segment), properties={"building": label})
            )
        world.rects.append(RectEntity(ids.next(), ObjectKind.WALL, Rect(cx, cy - h / 2, w, t), properties={"building": label}))
        world.rects.append(RectEntity(ids.next(), ObjectKind.WALL, Rect(cx, cy + h / 2, w, t), properties={"building": label}))
        opposite_x = cx - (w / 2 if door_side == "right" else -w / 2)
        world.rects.append(RectEntity(ids.next(), ObjectKind.WALL, Rect(opposite_x, cy, t, h), properties={"building": label}))
        world.doors.append(
            DoorEntity(ids.next(), ObjectKind.DOOR, Rect(x, cy, t, gap), properties={"building": label, "label": "door"}, open=False)
        )


def _spawn_item(ids: _Ids, kind: ObjectKind, x: float, y: float, mass: float, **properties) -> WorldObject:
    return WorldObject(
        uid=ids.next(),
        kind=kind,
        x=x,
        y=y,
        mass=mass,
        radius=0.20,
        portable=True,
        solid=False,
        properties=properties,
    )


def generate_city(config: EnvironmentConfig, rng: Random) -> GeneratedWorld:
    world = GeneratedWorld()
    ids = _Ids()
    W, H = config.map_width, config.map_height

    # Four semantically stable districts with light positional jitter.
    jitter = lambda s=0.7: rng.uniform(-s, s)
    world.zones = [
        Zone("Commerces", Rect(7.5, 22.5, 15.0, 15.0), "shops"),
        Zone("Parc", Rect(22.5, 22.5, 15.0, 15.0), "park"),
        Zone("Services", Rect(7.5, 7.5, 15.0, 15.0), "medical"),
        Zone("Industrie", Rect(22.5, 7.5, 15.0, 15.0), "industrial"),
    ]

    _add_building(world, ids, cx=5.0 + jitter(), cy=23.0 + jitter(), w=7.0, h=6.0, door_side="bottom", label="Epicerie", theme="shop")
    _add_building(world, ids, cx=5.5 + jitter(), cy=7.0 + jitter(), w=7.0, h=6.5, door_side="right", label="Infirmerie", theme="medical")
    _add_building(world, ids, cx=23.0 + jitter(), cy=6.5 + jitter(), w=9.0, h=7.0, door_side="top", label="Entrepot", theme="industrial")

    # A compact second shop makes the centre feel like a town instead of isolated buildings.
    _add_building(world, ids, cx=12.0 + jitter(0.4), cy=19.0 + jitter(0.4), w=5.0, h=4.5, door_side="right", label="Bazar", theme="shop")

    # Park water feature. It is traversable but slows agents.
    if rng.random() < config.generation.water_probability:
        world.areas.append(
            AreaEffect(ids.next(), ObjectKind.WATER, Rect(22.5, 23.0, 6.5, 4.0), {"speed_factor": 0.55, "energy_factor": 1.25})
        )

    # Steam in the industrial zone affects visibility but is not a wall.
    world.areas.append(
        AreaEffect(ids.next(), ObjectKind.STEAM, Rect(25.0, 9.5, 3.0, 2.4), {"visibility_factor": 0.55})
    )

    # Street furniture / industrial obstacles.
    obstacle_positions = [
        (14.8, 15.0, 1.4, 0.5, "bench"),
        (17.2, 14.0, 0.7, 2.2, "sign"),
        (20.0, 4.2, 1.8, 0.8, "pallet"),
        (27.0, 4.8, 1.3, 1.3, "crate"),
        (25.0, 14.0, 2.0, 0.7, "pipe"),
    ]
    for x, y, w, h, label in obstacle_positions:
        world.rects.append(
            RectEntity(ids.next(), ObjectKind.OBSTACLE, Rect(x, y, w, h), sound_absorption=0.15, properties={"label": label})
        )

    # Purely visual city dressing. These do not affect physics or observations.
    # They make the training world read like a small town rather than four
    # colored quadrants while keeping the simulation contract unchanged.
    world.decorations.extend([
        DecorationVisual("bush", 18.8, 21.7, 1.0, 0.6),
        DecorationVisual("bush", 25.2, 26.3, 1.1, 0.6),
        DecorationVisual("lamp", 14.0, 17.2, 0.3, 0.3),
        DecorationVisual("lamp", 16.0, 12.8, 0.3, 0.3),
        DecorationVisual("lamp", 14.0, 12.8, 0.3, 0.3),
        DecorationVisual("lamp", 16.0, 17.2, 0.3, 0.3),
        DecorationVisual("parking", 20.0, 2.3, 5.8, 1.8),
        DecorationVisual("parking", 9.5, 2.3, 4.8, 1.8),
        DecorationVisual("crosswalk", 15.0, 18.0, 3.6, 1.2, 0.0),
        DecorationVisual("crosswalk", 15.0, 12.0, 3.6, 1.2, 0.0),
        DecorationVisual("crosswalk", 12.0, 15.0, 1.2, 3.6, 0.0),
        DecorationVisual("crosswalk", 18.0, 15.0, 1.2, 3.6, 0.0),
    ])

    # Park trees are real solid/occluding objects, not decorative props.
    for tx, ty, radius in [
        (18.3, 25.2, 0.55),
        (20.2, 20.4, 0.50),
        (26.5, 20.3, 0.60),
        (27.6, 25.9, 0.48),
    ]:
        world.objects.append(
            WorldObject(
                uid=ids.next(),
                kind=ObjectKind.OBSTACLE,
                x=tx,
                y=ty,
                mass=250.0,
                radius=radius,
                portable=False,
                solid=True,
                properties={"label": "tree"},
            )
        )

    # One moving cart: it will be animated by the environment using these properties.
    world.objects.append(
        WorldObject(
            uid=ids.next(),
            kind=ObjectKind.MOVING_OBSTACLE,
            x=20.0,
            y=11.5,
            mass=80.0,
            radius=0.55,
            portable=False,
            solid=True,
            properties={"axis": "x", "origin_x": 20.0, "origin_y": 11.5, "amplitude": 4.0, "period_ticks": 24},
        )
    )

    # Seeded atmosphere. The agent never gets these values directly.
    if rng.random() < config.generation.fog_probability:
        world.visibility_factor = rng.uniform(0.62, 0.82)
    if rng.random() < config.generation.low_light_probability:
        world.low_light_factor = rng.uniform(0.72, 0.90)

    # Curated guaranteed items establish the semantics of each zone.
    curated = [
        _spawn_item(ids, ObjectKind.FOOD, 4.0, 23.0, 0.35, label="sandwich", nutrition=0.35),
        _spawn_item(ids, ObjectKind.DRINK, 6.0, 22.0, 0.55, label="bottle", hydration=0.40),
        _spawn_item(ids, ObjectKind.WEAPON, 10.8, 19.0, 0.65, label="rolling_pin", weapon_type="rolling_pin", damage=0.12),
        _spawn_item(ids, ObjectKind.MEDICINE, 5.5, 7.0, 0.25, label="bandage", medical_type="bandage", heal=0.04),
        _spawn_item(ids, ObjectKind.MEDICINE, 6.6, 7.8, 0.20, label="medicine", medical_type="medicine", heal=0.10),
        _spawn_item(ids, ObjectKind.WEAPON, 23.0, 6.5, 2.2, label="potato_launcher", weapon_type="potato_launcher", damage=0.18),
        _spawn_item(ids, ObjectKind.AMMUNITION, 24.0, 6.5, 0.18, label="potato", ammo_type="potato"),
        _spawn_item(ids, ObjectKind.AMMUNITION, 24.6, 6.8, 0.18, label="potato", ammo_type="potato"),
        _spawn_item(ids, ObjectKind.WEAPON_MODIFIER, 21.0, 7.0, 0.25, label="reinforced_spring", modifier="projectile_speed", factor=1.15),
        _spawn_item(ids, ObjectKind.RAW_MATERIAL, 22.0, 4.0, 1.2, label="metal", material="metal"),
        _spawn_item(ids, ObjectKind.RAW_MATERIAL, 27.0, 8.0, 0.5, label="cloth", material="cloth"),
    ]
    world.objects.extend(curated)

    # Additional random resources. Spawn areas are semantically biased.
    target_count = rng.randint(*config.generation.item_count)
    templates = [
        (ObjectKind.FOOD, 0.30, {"label": "fruit", "nutrition": 0.22}, world.zones[0].rect),
        (ObjectKind.DRINK, 0.50, {"label": "water_bottle", "hydration": 0.35}, world.zones[0].rect),
        (ObjectKind.MEDICINE, 0.20, {"label": "bandage", "medical_type": "bandage", "heal": 0.04}, world.zones[2].rect),
        (ObjectKind.RAW_MATERIAL, 0.80, {"label": "wood", "material": "wood"}, world.zones[1].rect),
        (ObjectKind.RAW_MATERIAL, 1.00, {"label": "plastic", "material": "plastic"}, world.zones[3].rect),
        (ObjectKind.AMMUNITION, 0.18, {"label": "potato", "ammo_type": "potato"}, world.zones[3].rect),
    ]

    def blocked(x: float, y: float) -> bool:
        for rect in world.rects:
            if point_in_rect(x, y, rect.rect):
                return True
        for door in world.doors:
            if point_in_rect(x, y, door.rect):
                return True
        return False

    attempts = 0
    while sum(1 for obj in world.objects if obj.portable) < target_count and attempts < 300:
        attempts += 1
        kind, mass, props, zone = rng.choice(templates)
        x = rng.uniform(zone.left + 0.7, zone.right - 0.7)
        y = rng.uniform(zone.bottom + 0.7, zone.top - 0.7)
        if not blocked(x, y):
            world.objects.append(_spawn_item(ids, kind, x, y, mass, **props))

    return world
