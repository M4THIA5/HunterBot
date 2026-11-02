from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Mapping, Protocol, TypeAlias

TICK_SECONDS = 0.5
PropertyValue: TypeAlias = int | float | bool | str | None
Properties: TypeAlias = Mapping[str, PropertyValue]
AgentId: TypeAlias = str


class ObjectKind(Enum):
    UNKNOWN = auto()
    AGENT = auto()
    WALL = auto()
    OBSTACLE = auto()
    MOVING_OBSTACLE = auto()
    DOOR = auto()
    CONTROL = auto()
    MACHINE = auto()
    RAW_MATERIAL = auto()
    WATER = auto()
    ICE = auto()
    STEAM = auto()
    WEAPON = auto()
    WEAPON_MODIFIER = auto()
    AMMUNITION = auto()
    PROJECTILE = auto()
    FOOD = auto()
    DRINK = auto()
    MEDICINE = auto()
    TOOL = auto()
    CONTAINER = auto()
    LIGHT_SOURCE = auto()
    HAZARD = auto()


class SoundKind(Enum):
    UNKNOWN = auto()
    MOVEMENT = auto()
    IMPACT = auto()
    GUNSHOT = auto()
    SPLASH = auto()
    MECHANISM = auto()
    OBJECT = auto()
    AGENT = auto()
    AMBIENT = auto()


class InjuryKind(Enum):
    BRUISE = auto()
    WOUND = auto()
    BURN = auto()


class EventKind(Enum):
    DAMAGE_RECEIVED = auto()
    COLLISION = auto()
    ITEM_PICKED_UP = auto()
    ITEM_DROPPED = auto()
    ITEM_EQUIPPED = auto()
    ITEM_USED = auto()
    ITEM_BROKEN = auto()
    SHOT_FIRED = auto()
    INTERACTION = auto()
    HEALED = auto()


@dataclass(frozen=True, slots=True)
class VisualDetection:
    ref: int
    kind: ObjectKind
    distance: float
    bearing: float
    angular_size: float
    properties: Properties = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SoundDetection:
    kind: SoundKind
    intensity: float
    bearing: float
    bearing_uncertainty: float
    distance_estimate: float | None
    muffling: float


@dataclass(frozen=True, slots=True)
class TouchContact:
    ref: int
    kind: ObjectKind
    bearing: float
    properties: Properties = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Event:
    kind: EventKind
    value: float | None = None
    properties: Properties = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Injury:
    kind: InjuryKind
    severity: float
    bleeding: float


@dataclass(frozen=True, slots=True)
class InventoryItem:
    slot: int
    kind: ObjectKind
    mass: float
    properties: Properties = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SelfState:
    health: float
    energy: float
    hydration: float
    satiety: float
    body_mass: float
    carried_mass: float
    speed: float
    orientation: float
    injuries: tuple[Injury, ...]
    inventory: tuple[InventoryItem, ...]
    equipped_slot: int | None


@dataclass(frozen=True, slots=True)
class Observation:
    tick: int
    self_state: SelfState
    vision: tuple[VisualDetection, ...]
    hearing: tuple[SoundDetection, ...]
    touch: tuple[TouchContact, ...]
    events: tuple[Event, ...]


@dataclass(frozen=True, slots=True)
class Movement:
    forward: float
    right: float
    speed: float


@dataclass(frozen=True, slots=True)
class Rotation:
    angular_speed: float


@dataclass(frozen=True, slots=True)
class Pickup:
    target_ref: int


@dataclass(frozen=True, slots=True)
class Drop:
    slot: int


@dataclass(frozen=True, slots=True)
class Equip:
    slot: int


@dataclass(frozen=True, slots=True)
class Use:
    slot: int
    target_ref: int | None = None


@dataclass(frozen=True, slots=True)
class Throw:
    slot: int
    bearing: float
    power: float


@dataclass(frozen=True, slots=True)
class Shoot:
    bearing: float


@dataclass(frozen=True, slots=True)
class Ingest:
    slot: int


@dataclass(frozen=True, slots=True)
class Manipulate:
    target_ref: int


Interaction: TypeAlias = Pickup | Drop | Equip | Use | Throw | Shoot | Ingest | Manipulate


@dataclass(frozen=True, slots=True)
class Action:
    movement: Movement | None = None
    rotation: Rotation | None = None
    rest: bool = False
    interaction: Interaction | None = None


class EpisodeEndReason(Enum):
    ELIMINATED = auto()
    TIME_LIMIT = auto()
    SIMULATION_END = auto()


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    reason: EpisodeEndReason
    survival_ticks: int


class Agent(Protocol):
    def reset(self) -> None: ...
    def act(self, observation: Observation) -> Action: ...
    def on_episode_end(self, result: EpisodeResult) -> None: ...
