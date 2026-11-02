from __future__ import annotations

from math import pi
from random import Random

from .arena_api import (
    Action,
    Drop,
    Equip,
    Ingest,
    Manipulate,
    Movement,
    ObjectKind,
    Observation,
    Pickup,
    Rotation,
    Shoot,
    TICK_SECONDS,
    Use,
)


class RandomBot:
    """Simple exploratory baseline using only public observations."""

    def __init__(self, seed: int = 0):
        self._initial_seed = seed
        self.rng = Random(seed)
        self.turn_ticks = 0
        self.turn_rate = 0.0

    def reset(self) -> None:
        self.rng = Random(self._initial_seed)
        self.turn_ticks = 0
        self.turn_rate = 0.0

    def on_episode_end(self, result) -> None:
        pass

    def act(self, obs: Observation) -> Action:
        s = obs.self_state

        # Basic self-care.
        for item in s.inventory:
            if item.kind is ObjectKind.DRINK and s.hydration < 0.45:
                return Action(interaction=Ingest(item.slot))
            if item.kind is ObjectKind.FOOD and s.satiety < 0.45:
                return Action(interaction=Ingest(item.slot))
            if item.kind is ObjectKind.MEDICINE and s.health < 0.65:
                return Action(interaction=Ingest(item.slot))

        if s.equipped_slot is None:
            weapon = next((i for i in s.inventory if i.kind is ObjectKind.WEAPON), None)
            if weapon is not None:
                return Action(interaction=Equip(weapon.slot))

        # Nearby portable-looking things are attractive.
        for d in obs.vision:
            if d.distance < 0.9 and d.kind in {
                ObjectKind.FOOD,
                ObjectKind.DRINK,
                ObjectKind.MEDICINE,
                ObjectKind.WEAPON,
                ObjectKind.AMMUNITION,
                ObjectKind.WEAPON_MODIFIER,
                ObjectKind.RAW_MATERIAL,
            }:
                return Action(interaction=Pickup(d.ref))

        # Open a nearby door in front.
        for d in obs.vision:
            if d.kind is ObjectKind.DOOR and d.distance < 0.85 and abs(d.bearing) < 0.7:
                return Action(interaction=Manipulate(d.ref))

        # Turn away from close obstacles.
        close = [d for d in obs.vision if d.kind in {ObjectKind.WALL, ObjectKind.OBSTACLE, ObjectKind.MOVING_OBSTACLE} and d.distance < 1.2]
        if close:
            nearest = min(close, key=lambda d: d.distance)
            turn = -1.8 if nearest.bearing >= 0 else 1.8
            return Action(movement=Movement(0.4, 0.0, 1.2), rotation=Rotation(turn))

        if self.turn_ticks <= 0:
            self.turn_ticks = self.rng.randint(5, 18)
            self.turn_rate = self.rng.uniform(-0.8, 0.8)
        self.turn_ticks -= 1

        speed = 1.4 if s.energy < 0.35 else self.rng.uniform(1.6, 2.5)
        if s.energy < 0.15:
            return Action(rest=True)
        return Action(movement=Movement(1.0, self.rng.uniform(-0.15, 0.15), speed), rotation=Rotation(self.turn_rate))


class HunterBot:
    """Aggressive scripted baseline, still limited to public observations."""

    def __init__(self, seed: int = 0):
        self._initial_seed = seed
        self.rng = Random(seed)
        self.search_turn = 0.45
        self.last_seen_bearing: float | None = None
        self.memory_ticks = 0
        self.melee_strafe_sign = 1.0
        self.melee_strafe_ticks = 0

    def reset(self) -> None:
        self.rng = Random(self._initial_seed)
        self.search_turn = self.rng.choice([-0.55, 0.55])
        self.last_seen_bearing = None
        self.memory_ticks = 0
        self.melee_strafe_sign = self.rng.choice([-1.0, 1.0])
        self.melee_strafe_ticks = 0

    def on_episode_end(self, result) -> None:
        pass

    def act(self, obs: Observation) -> Action:
        s = obs.self_state

        # Survival takes priority when badly depleted.
        if s.health < 0.45:
            med = next((i for i in s.inventory if i.kind is ObjectKind.MEDICINE), None)
            if med is not None:
                return Action(interaction=Ingest(med.slot))
        if s.hydration < 0.30:
            drink = next((i for i in s.inventory if i.kind is ObjectKind.DRINK), None)
            if drink is not None:
                return Action(interaction=Ingest(drink.slot))
        if s.satiety < 0.30:
            food = next((i for i in s.inventory if i.kind is ObjectKind.FOOD), None)
            if food is not None:
                return Action(interaction=Ingest(food.slot))
        if s.energy < 0.12:
            return Action(rest=True)

        equipped = next((i for i in s.inventory if i.slot == s.equipped_slot), None)
        if equipped is None:
            weapon = next((i for i in s.inventory if i.kind is ObjectKind.WEAPON), None)
            if weapon is not None:
                return Action(interaction=Equip(weapon.slot))

        # Pick up tactically useful nearby objects.
        useful_order = {
            ObjectKind.WEAPON: 0,
            ObjectKind.AMMUNITION: 1,
            ObjectKind.MEDICINE: 2,
            ObjectKind.DRINK: 3,
            ObjectKind.FOOD: 4,
        }
        useful = [d for d in obs.vision if d.kind in useful_order and d.distance < 1.0]
        if useful:
            d = min(useful, key=lambda x: (useful_order[x.kind], x.distance))
            return Action(interaction=Pickup(d.ref))

        enemies = [d for d in obs.vision if d.kind is ObjectKind.AGENT]
        if enemies:
            target = min(enemies, key=lambda d: d.distance)
            self.last_seen_bearing = target.bearing
            self.memory_ticks = 6

            equipped = next((i for i in s.inventory if i.slot == s.equipped_slot), None)
            label = equipped.properties.get("label") if equipped else None
            if label == "potato_launcher" and target.distance < 8.0:
                has_ammo = any(i.kind is ObjectKind.AMMUNITION for i in s.inventory)
                if has_ammo:
                    # Keep some distance instead of walking into the target.
                    forward = -0.55 if target.distance < 2.4 else 0.35
                    strafe = 0.45 * self.melee_strafe_sign if target.distance < 3.5 else 0.0
                    return Action(
                        movement=Movement(forward, strafe, 1.5),
                        rotation=Rotation(target.bearing / TICK_SECONDS),
                        interaction=Shoot(target.bearing),
                    )
            if label == "rolling_pin" and target.distance < 1.25:
                # Melee should look like a fight, not two discs glued together.
                # Back off and circle while striking, periodically changing side.
                if self.melee_strafe_ticks <= 0:
                    self.melee_strafe_sign *= -1.0
                    self.melee_strafe_ticks = self.rng.randint(3, 7)
                self.melee_strafe_ticks -= 1
                return Action(
                    movement=Movement(-0.45, 0.75 * self.melee_strafe_sign, 1.45),
                    rotation=Rotation(target.bearing / TICK_SECONDS),
                    interaction=Use(equipped.slot, target.ref),
                )
            return Action(
                movement=Movement(1.0, 0.0, 2.6 if s.energy > 0.3 else 1.7),
                rotation=Rotation(target.bearing / TICK_SECONDS),
            )

        # Open doors rather than getting stuck on them.
        doors = [d for d in obs.vision if d.kind is ObjectKind.DOOR and d.distance < 0.9]
        if doors:
            return Action(interaction=Manipulate(min(doors, key=lambda d: d.distance).ref))

        # Very rough auditory pursuit.
        if obs.hearing:
            sound = max(obs.hearing, key=lambda x: x.intensity)
            if sound.intensity > 0.2:
                return Action(
                    movement=Movement(1.0, 0.0, 2.1),
                    rotation=Rotation(sound.bearing / TICK_SECONDS),
                )

        if self.memory_ticks > 0 and self.last_seen_bearing is not None:
            self.memory_ticks -= 1
            return Action(
                movement=Movement(1.0, 0.0, 2.0),
                rotation=Rotation(self.last_seen_bearing / TICK_SECONDS),
            )

        # Search pattern.
        if obs.tick % 12 == 0:
            self.search_turn *= -1
        return Action(movement=Movement(1.0, 0.0, 1.9), rotation=Rotation(self.search_turn))
