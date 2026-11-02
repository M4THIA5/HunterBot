from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .arena_api import InjuryKind, ObjectKind, SoundKind
from .geometry import Rect


@dataclass(slots=True)
class WorldObject:
    uid: int
    kind: ObjectKind
    x: float
    y: float
    mass: float = 0.0
    radius: float = 0.25
    portable: bool = False
    solid: bool = False
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RectEntity:
    uid: int
    kind: ObjectKind
    rect: Rect
    solid: bool = True
    occluding: bool = True
    sound_absorption: float = 0.35
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def x(self) -> float:
        return self.rect.x

    @property
    def y(self) -> float:
        return self.rect.y


@dataclass(slots=True)
class DoorEntity(RectEntity):
    open: bool = False

    @property
    def effective_solid(self) -> bool:
        return self.solid and not self.open

    @property
    def effective_occluding(self) -> bool:
        return self.occluding and not self.open


@dataclass(slots=True)
class AreaEffect:
    uid: int
    kind: ObjectKind
    rect: Rect
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def x(self) -> float:
        return self.rect.x

    @property
    def y(self) -> float:
        return self.rect.y


@dataclass(slots=True)
class InventoryEntry:
    uid: int
    kind: ObjectKind
    mass: float
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InjuryState:
    kind: InjuryKind
    severity: float
    bleeding: float


@dataclass(slots=True)
class AgentState:
    uid: int
    agent_id: str
    x: float
    y: float
    orientation: float
    body_mass: float
    radius: float
    health: float = 1.0
    energy: float = 1.0
    hydration: float = 1.0
    satiety: float = 1.0
    speed: float = 0.0
    alive: bool = True
    injuries: list[InjuryState] = field(default_factory=list)
    inventory: dict[int, InventoryEntry] = field(default_factory=dict)
    equipped_slot: int | None = None
    vx: float = 0.0
    vy: float = 0.0
    last_damage_source: str | None = None

    @property
    def carried_mass(self) -> float:
        return sum(item.mass for item in self.inventory.values())


@dataclass(slots=True)
class Projectile:
    uid: int
    owner_id: str
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    damage: float
    ttl: float
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SoundEventInternal:
    source_x: float
    source_y: float
    kind: SoundKind
    base_intensity: float
    source_agent_id: str | None = None


@dataclass(slots=True)
class Zone:
    name: str
    rect: Rect
    theme: str
