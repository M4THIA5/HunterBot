"""Perception et mémoire courte construites UNIQUEMENT depuis Observation.

Aucun `ref` n'est conservé d'un tick à l'autre. Les directions mémorisées sont
stockées en angle absolu (orientation propre + bearing) puis reconverties en
bearing relatif au tick suivant grâce à `self_state.orientation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

from rl_arena.arena_api import (
    EventKind,
    InventoryItem,
    ObjectKind,
    Observation,
    SoundKind,
    VisualDetection,
)

TWO_PI = 2 * pi


def wrap(a: float) -> float:
    return (a + pi) % TWO_PI - pi


# Catégories qui bloquent physiquement le passage (ou qu'il vaut mieux éviter).
SOLID_KINDS = frozenset(
    {
        ObjectKind.WALL,
        ObjectKind.OBSTACLE,
        ObjectKind.MOVING_OBSTACLE,
        ObjectKind.MACHINE,
        ObjectKind.CONTAINER,
        ObjectKind.HAZARD,
        ObjectKind.AGENT,
    }
)
CONSUMABLE_KINDS = frozenset({ObjectKind.FOOD, ObjectKind.DRINK, ObjectKind.MEDICINE})
COMBAT_ITEM_KINDS = frozenset({ObjectKind.WEAPON, ObjectKind.AMMUNITION, ObjectKind.WEAPON_MODIFIER})
USEFUL_KINDS = CONSUMABLE_KINDS | COMBAT_ITEM_KINDS
THREAT_SOUNDS = frozenset({SoundKind.GUNSHOT, SoundKind.AGENT, SoundKind.MOVEMENT, SoundKind.IMPACT})

# Valeurs par défaut prudentes quand `properties` ne précise rien.
DEFAULT_GAIN = 0.3


def prop_float(props, key: str, default: float) -> float:
    try:
        v = props.get(key, default)
        return float(v) if v is not None else default
    except (TypeError, ValueError, AttributeError):
        return default


def door_is_closed(d: VisualDetection) -> bool | None:
    """True/False si la propriété `open` est perçue, None sinon."""
    v = d.properties.get("open") if d.properties else None
    if v is None:
        return None
    return not bool(v)


def is_blocking(d: VisualDetection) -> bool:
    if d.kind in SOLID_KINDS:
        return True
    if d.kind is ObjectKind.DOOR:
        return door_is_closed(d) is not False  # fermée ou inconnue
    if d.kind is ObjectKind.UNKNOWN:
        return d.distance < 2.0
    return False


# ----------------------------------------------------------------------
# Inventaire
# ----------------------------------------------------------------------
def items_of(inv: tuple[InventoryItem, ...], kind: ObjectKind) -> list[InventoryItem]:
    return [i for i in inv if i.kind is kind]


def equipped_item(obs: Observation) -> InventoryItem | None:
    s = obs.self_state
    if s.equipped_slot is None:
        return None
    return next((i for i in s.inventory if i.slot == s.equipped_slot), None)


def weapon_mode(item: InventoryItem | None) -> str | None:
    """'ranged', 'melee' ou None, d'après les propriétés perçues."""
    if item is None or item.kind is not ObjectKind.WEAPON:
        return None
    wt = str(item.properties.get("weapon_type", item.properties.get("label", "")))
    if "launcher" in wt or "gun" in wt or "bow" in wt:
        return "ranged"
    return "melee"


def has_ammo(obs: Observation) -> bool:
    return any(i.kind is ObjectKind.AMMUNITION for i in obs.self_state.inventory)


def is_armed(obs: Observation) -> bool:
    mode = weapon_mode(equipped_item(obs))
    if mode == "ranged":
        return has_ammo(obs)
    return mode == "melee"


def total_bleeding(obs: Observation) -> float:
    return sum(i.bleeding * i.severity for i in obs.self_state.injuries)


# ----------------------------------------------------------------------
# Mémoire courte
# ----------------------------------------------------------------------
@dataclass(slots=True)
class Memory:
    tick: int = 0
    ticks_since_damage: int = 10_000
    damage_recent: float = 0.0  # somme décroissante des dégâts récents
    # Menace (agent vu ou entendu) : direction absolue + fraîcheur.
    threat_heading: float | None = None
    threat_distance: float = 99.0
    threat_age: int = 10_000
    threat_seen: bool = False
    seen_age: int = 10_000   # ticks depuis la dernière menace vue ou touchée
    heard_gunshot: bool = False
    threat_armed: bool = False
    threat_closing: bool = False  # la menace vue s'est rapprochée depuis le tick précédent
    heard_threat: bool = False
    # Navigation.
    collisions_recent: int = 0
    last_collision_tick: int = -10_000
    escape_heading: float | None = None
    escape_ticks: int = 0
    door_cooldown: int = 0
    fight_cooldown: int = 0
    fight_best: float = 99.0
    fight_no_progress: int = 0
    fight_last_tick: int = -10_000
    wander_heading: float | None = None
    wander_ticks: int = 0
    # Ressource ciblée (pas de ref : direction absolue + distance).
    target_heading: float | None = None
    target_distance: float = 99.0
    target_kind: ObjectKind | None = None
    target_age: int = 10_000
    # Catégories temporairement ignorées après des Pickup sans effet
    # (inventaire plein, trop lourd, objet inaccessible...).
    kind_cooldown: dict = field(default_factory=dict)
    # Mémoire spatiale par odométrie (agent/spatial.py) ; désactivable pour A/B.
    use_spatial: bool = True
    # Ablation (results/spatial_ab.md) : portes et exploration vers l'inconnu dégradent
    # (dérive de l'odométrie) ; seule la mémoire des ressources est conservée.
    spatial_uses: frozenset = frozenset({"items"})
    spatial: object = None

    def heading_to_bearing(self, heading: float, orientation: float) -> float:
        return wrap(heading - orientation)

    def update(self, obs: Observation) -> None:
        s = obs.self_state
        self.tick = obs.tick
        if self.use_spatial:
            if self.spatial is None:
                from .spatial import SpatialMap

                self.spatial = SpatialMap()
            self.spatial.integrate(obs)
            self.spatial.observe(obs)
        self.ticks_since_damage += 1
        self.damage_recent *= 0.85
        self.threat_age += 1
        self.seen_age += 1
        self.target_age += 1
        self.door_cooldown = max(0, self.door_cooldown - 1)
        self.fight_cooldown = max(0, self.fight_cooldown - 1)
        self.collisions_recent = max(0, self.collisions_recent - 1) if obs.tick % 4 == 0 else self.collisions_recent

        for ev in obs.events:
            if ev.kind is EventKind.DAMAGE_RECEIVED:
                self.ticks_since_damage = 0
                self.damage_recent += float(ev.value or 0.0)
            elif ev.kind is EventKind.COLLISION:
                self.collisions_recent += 1
                self.last_collision_tick = obs.tick
            elif ev.kind is EventKind.INTERACTION and ev.properties.get("kind") == "door":
                self.door_cooldown = 30

        agents = [d for d in obs.vision if d.kind is ObjectKind.AGENT]
        seen_before = self.seen_age <= 1  # seen_age déjà incrémenté ce tick
        self.threat_seen = bool(agents)
        self.threat_closing = False
        if agents:
            t = min(agents, key=lambda d: d.distance)
            if seen_before:
                # Pas d'identité entre ticks : on compare simplement l'agent le
                # plus proche d'un tick à l'autre (bruit toléré par le seuil).
                self.threat_closing = self.threat_distance - t.distance > 0.25
            self.threat_heading = wrap(s.orientation + t.bearing)
            self.threat_distance = t.distance
            self.threat_age = 0
            self.seen_age = 0
            armed = t.properties.get("armed")
            if armed is not None:
                self.threat_armed = bool(armed)
            elif not seen_before:
                self.threat_armed = False  # trop loin pour le voir : inconnu
        else:
            # Un agent qui nous touche n'est pas forcément visible (dans le dos).
            touching = [c for c in obs.touch if c.kind is ObjectKind.AGENT]
            if touching:
                self.threat_heading = wrap(s.orientation + touching[0].bearing)
                self.threat_distance = 0.5
                self.threat_age = 0
                self.seen_age = 0

        self.heard_threat = False
        self.heard_gunshot = any(h.kind is SoundKind.GUNSHOT and h.intensity > 0.3 for h in obs.hearing)
        for snd in obs.hearing:
            if snd.kind in THREAT_SOUNDS and snd.intensity > 0.18:
                self.heard_threat = True
                if self.seen_age > 2:
                    self.threat_heading = wrap(s.orientation + snd.bearing)
                    self.threat_distance = snd.distance_estimate if snd.distance_estimate is not None else 10.0
                    self.threat_age = 1
                break

        for k in list(self.kind_cooldown):
            self.kind_cooldown[k] -= 1
            if self.kind_cooldown[k] <= 0:
                del self.kind_cooldown[k]
        if any(ev.kind is EventKind.ITEM_PICKED_UP or ev.kind is EventKind.ITEM_USED for ev in obs.events):
            self.kind_cooldown.clear()  # la place a pu changer

        if self.escape_ticks > 0:
            self.escape_ticks -= 1

    # Lecture pratique ---------------------------------------------------
    def threat_bearing(self, obs: Observation) -> float | None:
        if self.threat_heading is None:
            return None
        return self.heading_to_bearing(self.threat_heading, obs.self_state.orientation)

    def threat_active(self, horizon: int = 4) -> bool:
        """Menace vue ou touchée récemment (le son seul ne suffit pas)."""
        return self.seen_age <= horizon

    def under_attack(self, horizon: int = 4) -> bool:
        return self.ticks_since_damage <= horizon


def nearest(obs: Observation, kinds) -> VisualDetection | None:
    best = None
    for d in obs.vision:
        if d.kind in kinds and (best is None or d.distance < best.distance):
            best = d
    return best
