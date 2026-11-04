"""Options (macro-actions) et leurs contrôleurs bas niveau.

Une option est choisie par une politique (scriptée en phase 1, apprise
ensuite), puis exécutée pendant plusieurs ticks. Interface commune :

    available(obs, mem) -> bool   l'option a-t-elle un sens maintenant ?
    start(obs, mem)               initialisation à la sélection
    act(obs, mem) -> Action       action du tick courant
    done(obs, mem) -> bool        terminaison naturelle (testée avant act)

Les contrôleurs n'utilisent que l'Observation courante et la Memory ; tout
`ref` utilisé provient de l'observation du tick courant.
"""

from __future__ import annotations

from math import pi
from random import Random

from rl_arena.arena_api import (
    Action,
    Equip,
    Ingest,
    InventoryItem,
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

from .navigation import drive, steer
from .perception import (
    CONSUMABLE_KINDS,
    DEFAULT_GAIN,
    USEFUL_KINDS,
    Memory,
    door_is_closed,
    equipped_item,
    has_ammo,
    is_armed,
    items_of,
    nearest,
    prop_float,
    total_bleeding,
    weapon_mode,
    wrap,
)

QUIET_SPEED = 1.5       # sous le seuil où le déplacement devient audible
PICKUP_DISTANCE = 0.9   # distance perçue (bord de l'objet) en deçà de laquelle on tente Pickup
DOOR_DISTANCE = 0.75    # distance perçue d'une porte en deçà de laquelle on la manipule


def face(bearing: float) -> Rotation:
    return Rotation(max(-pi, min(pi, bearing / TICK_SECONDS)))


DOOR_APPROACH_TRIES = 3   # tentatives sans effet avant de s'approcher en manipulant
DOOR_GIVEUP_TRIES = 8     # puis abandon temporaire des portes
DOOR_BLOCK_TICKS = 40


def door_interaction(obs: Observation, mem: Memory) -> Action | None:
    """Ouvre une porte fermée juste devant (sans jamais refermer une porte ouverte)."""
    if mem.door_block > 0:
        return None
    target = None
    for d in obs.vision:
        if d.kind is ObjectKind.DOOR and d.distance < DOOR_DISTANCE and abs(d.bearing) < 1.0:
            if door_is_closed(d) is True:
                target = (d.ref, d.bearing)
                break
    if target is None:
        for c in obs.touch:  # le toucher ne rapporte que les portes fermées
            if c.kind is ObjectKind.DOOR:
                target = (c.ref, c.bearing)
                break
    if target is None:
        mem.door_tries = 0  # tentatives consécutives seulement
        return None
    ref, bearing = target
    if mem.use_door_giveup:
        mem.door_tries += 1
        if mem.door_tries > DOOR_GIVEUP_TRIES:
            mem.door_tries = 0
            mem.door_block = DOOR_BLOCK_TICKS
            mem.door_cooldown = DOOR_BLOCK_TICKS  # Explore cesse d'être attiré par les portes
            mem.wander_heading = wrap(obs.self_state.orientation + pi)
            mem.wander_ticks = 20
            return None
        if mem.door_tries > DOOR_APPROACH_TRIES:
            mv, _ = drive(bearing, 0.6)
            return Action(interaction=Manipulate(ref), rotation=face(bearing), movement=mv)
    return Action(interaction=Manipulate(ref), rotation=face(bearing))


def door_score(d, mem: Memory, tick: int) -> float:
    """Plus petit = plus attractif. Porte fermée : intérieur probablement inexploré ;
    bâtiment récemment visité (étiquette perçue) : moins utile d'y retourner."""
    _, closed_bonus, visited_malus, memory = mem.door_pref
    score = d.distance
    if door_is_closed(d) is True:
        score -= closed_bonus
    label = d.properties.get("building") if d.properties else None
    if label is not None and tick - mem.entered.get(label, -10**9) < memory:
        score += visited_malus
    return score


# ----------------------------------------------------------------------
# Choix de consommables (aucune valeur du kit codée en dur : on lit
# `properties` avec des valeurs par défaut prudentes).
# ----------------------------------------------------------------------
def best_consumable(obs: Observation) -> InventoryItem | None:
    s = obs.self_state
    inv = s.inventory
    bleeding = total_bleeding(obs)

    meds = items_of(inv, ObjectKind.MEDICINE)
    if bleeding > 0.0:
        bandage = next((m for m in meds if m.properties.get("medical_type") == "bandage"), None)
        if bandage is not None:
            return bandage
    heals = [m for m in meds if m.properties.get("medical_type") != "bandage"]
    if heals and s.health < 0.55:
        return max(heals, key=lambda m: prop_float(m.properties, "heal", DEFAULT_GAIN))

    candidates: list[tuple[float, InventoryItem]] = []
    for kind, level, key in ((ObjectKind.DRINK, s.hydration, "hydration"), (ObjectKind.FOOD, s.satiety, "nutrition")):
        for it in items_of(inv, kind):
            gain = prop_float(it.properties, key, DEFAULT_GAIN)
            # Ne pas gaspiller : consommer quand le gain tient dans la jauge,
            # ou d'urgence quand la jauge est basse.
            if level + gain <= 1.02 or level < 0.35:
                candidates.append((level, it))
    if candidates:
        return min(candidates, key=lambda t: t[0])[1]
    # Soin peu urgent mais jauge sinon perdue : quand la santé a assez baissé.
    if heals and s.health < 0.8:
        best = max(heals, key=lambda m: prop_float(m.properties, "heal", DEFAULT_GAIN))
        if s.health + prop_float(best.properties, "heal", DEFAULT_GAIN) <= 1.02:
            return best
    return None


def best_weapon(obs: Observation) -> InventoryItem | None:
    weapons = items_of(obs.self_state.inventory, ObjectKind.WEAPON)
    if not weapons:
        return None

    def score(w: InventoryItem) -> float:
        mode = weapon_mode(w)
        dmg = prop_float(w.properties, "damage", 0.1)
        # Arme à distance sans munition assortie : inutilisable.
        return (2.0 if (mode == "ranged" and has_ammo(obs, w)) else 1.0 if mode == "melee" else 0.0) + dmg

    return max(weapons, key=score)


def wanted_item(obs: Observation, mem: Memory) -> object | None:
    """Objet utile visible le plus intéressant (détection du tick courant)."""
    s = obs.self_state
    inv = s.inventory
    inv_kinds = [i.kind for i in inv]
    has_weapon = ObjectKind.WEAPON in inv_kinds
    weapon_labels = {i.properties.get("label") for i in inv if i.kind is ObjectKind.WEAPON}
    ranged = any(weapon_mode(i) == "ranged" for i in inv)
    ranged_types = {i.properties.get("ammo_type") for i in inv
                    if weapon_mode(i) == "ranged" and i.properties.get("ammo_type") is not None}
    best, best_score = None, 1e9
    for d in obs.vision:
        if d.kind not in USEFUL_KINDS or d.kind in mem.kind_cooldown:
            continue
        if d.kind is ObjectKind.WEAPON_MODIFIER and not has_weapon:
            continue
        if d.kind is ObjectKind.WEAPON and d.properties.get("label") is not None \
                and d.properties.get("label") in weapon_labels:
            continue  # doublon inutile (poids)
        if d.kind is ObjectKind.AMMUNITION and inv_kinds.count(ObjectKind.AMMUNITION) >= (6 if ranged else 3):
            continue
        if d.kind is ObjectKind.AMMUNITION and ranged_types and d.properties.get("ammo_type") is not None \
                and d.properties.get("ammo_type") not in ranged_types:
            continue  # munition d'une autre arme que les nôtres
        priority = 0.0
        if d.kind is ObjectKind.DRINK:
            priority = -2.0 * (1.0 - s.hydration)
        elif d.kind is ObjectKind.FOOD:
            priority = -2.0 * (1.0 - s.satiety)
        elif d.kind is ObjectKind.MEDICINE:
            priority = -1.0 * (1.0 - s.health)
        elif d.kind is ObjectKind.WEAPON:
            priority = 1.0 if has_weapon else -0.5
        else:
            priority = 1.5
        score = d.distance + 4.0 * priority
        if score < best_score:
            best, best_score = d, score
    return best


# ----------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------
class Option:
    name = "OPTION"
    max_ticks = 6

    def __init__(self) -> None:
        self.ticks = 0

    def available(self, obs: Observation, mem: Memory) -> bool:
        return True

    def start(self, obs: Observation, mem: Memory) -> None:
        self.ticks = 0

    def done(self, obs: Observation, mem: Memory) -> bool:
        return self.ticks >= self.max_ticks

    def step(self, obs: Observation, mem: Memory) -> Action:
        self.ticks += 1
        return self.act(obs, mem)

    def act(self, obs: Observation, mem: Memory) -> Action:  # pragma: no cover - abstrait
        raise NotImplementedError


class Explore(Option):
    """Avance à vitesse silencieuse, évite les obstacles, ouvre les portes,
    et se laisse attirer par les portes (accès aux bâtiments)."""

    name = "EXPLORE"
    max_ticks = 8

    def __init__(self, rng: Random) -> None:
        super().__init__()
        self.rng = rng

    def act(self, obs: Observation, mem: Memory) -> Action:
        s = obs.self_state
        door_act = door_interaction(obs, mem)
        if door_act is not None:
            return door_act

        desired = None
        ignore = None
        sp = mem.spatial
        uses = mem.spatial_uses
        if sp is not None and "items" in uses:
            # 1. Ressource nécessaire mémorisée hors de vue.
            needed = set()
            kinds = {i.kind for i in s.inventory}
            if s.hydration < 0.7 and ObjectKind.DRINK not in kinds:
                needed.add(ObjectKind.DRINK)
            if s.satiety < 0.7 and ObjectKind.FOOD not in kinds:
                needed.add(ObjectKind.FOOD)
            if s.health < 0.5 and ObjectKind.MEDICINE not in kinds:
                needed.add(ObjectKind.MEDICINE)
            lm = sp.nearest_item(needed) if needed else None
            if lm is not None:
                desired, _ = sp.bearing_to(lm.x, lm.y, s.orientation)
        if desired is None and mem.door_cooldown == 0:
            doors = [d for d in obs.vision if d.kind is ObjectKind.DOOR and d.distance < mem.door_pref[0]]
            if doors:
                d = min(doors, key=lambda x: door_score(x, mem, obs.tick))
                desired, ignore = d.bearing, d.ref
                if d.distance < 0.4 and door_is_closed(d) is not True:
                    mem.door_cooldown = 25  # on vient de franchir la porte
                    label = d.properties.get("building")
                    if label is not None:
                        mem.entered[label] = obs.tick
        if desired is None and sp is not None and "doors" in uses and mem.door_cooldown == 0:
            # 2. Porte mémorisée jamais franchie (accès probable à un bâtiment).
            door = sp.unvisited_door()
            if door is not None:
                b, dist = sp.bearing_to(door.x, door.y, s.orientation)
                if dist < 15.0:
                    desired = b
        if desired is None:
            if mem.wander_heading is None or mem.wander_ticks <= 0:
                if sp is not None and "novelty" in uses:
                    # 3. Direction la moins visitée.
                    mem.wander_heading = wrap(sp.novelty_heading(s.orientation) + self.rng.uniform(-0.3, 0.3))
                else:
                    turn, lo, hi = mem.wander
                    mem.wander_heading = wrap(s.orientation + self.rng.uniform(-turn, turn))
                mem.wander_ticks = self.rng.randint(mem.wander[1], mem.wander[2])
            mem.wander_ticks -= 1
            desired = mem.heading_to_bearing(mem.wander_heading, s.orientation)
            # Si on bute régulièrement, on change de cap.
            if obs.tick - mem.last_collision_tick <= 1:
                mem.wander_heading = wrap(s.orientation + self.rng.choice([-1, 1]) * self.rng.uniform(1.2, 2.6))
        speed = QUIET_SPEED if s.energy > 0.35 else 1.0
        mv, rot = steer(obs, desired, speed, mem, ignore_ref=ignore)
        return Action(movement=mv, rotation=rot)


class SeekItem(Option):
    """Rejoint l'objet utile visible le plus intéressant puis le ramasse."""

    name = "SEEK_ITEM"
    max_ticks = 14

    def available(self, obs: Observation, mem: Memory) -> bool:
        return wanted_item(obs, mem) is not None

    def start(self, obs: Observation, mem: Memory) -> None:
        super().start(obs, mem)
        self.lost = 0
        self.attempts = 0

    def done(self, obs: Observation, mem: Memory) -> bool:
        if any(e.kind.name == "ITEM_PICKED_UP" for e in obs.events) and self.ticks > 0:
            return True
        return self.ticks >= self.max_ticks or self.lost >= 3 or self.attempts >= 4

    def act(self, obs: Observation, mem: Memory) -> Action:
        target = wanted_item(obs, mem)
        if target is None:
            self.lost += 1
            mv, rot = steer(obs, None, 1.0, mem)
            return Action(movement=mv, rotation=rot)
        self.lost = 0
        if target.distance <= PICKUP_DISTANCE:
            self.attempts += 1
            if self.attempts >= 4:
                mem.kind_cooldown[target.kind] = 60  # Pickup sans effet : abandon
            return Action(interaction=Pickup(target.ref), rotation=face(target.bearing),
                          movement=Movement(1.0, 0.0, 0.3) if self.attempts > 1 else None)
        door_act = door_interaction(obs, mem)
        if door_act is not None:
            return door_act
        speed = min(QUIET_SPEED + 0.3, 0.6 + target.distance)
        mv, rot = steer(obs, target.bearing, speed, mem, ignore_ref=target.ref)
        return Action(movement=mv, rotation=rot)


class Consume(Option):
    name = "CONSUME"
    max_ticks = 1

    def available(self, obs: Observation, mem: Memory) -> bool:
        return best_consumable(obs) is not None

    def act(self, obs: Observation, mem: Memory) -> Action:
        item = best_consumable(obs)
        if item is None:
            return Action()
        return Action(interaction=Ingest(item.slot))


class Arm(Option):
    name = "ARM"
    max_ticks = 1

    def available(self, obs: Observation, mem: Memory) -> bool:
        w = best_weapon(obs)
        return w is not None and w.slot != obs.self_state.equipped_slot

    def act(self, obs: Observation, mem: Memory) -> Action:
        w = best_weapon(obs)
        return Action(interaction=Equip(w.slot)) if w is not None else Action()


class Rest(Option):
    name = "REST"
    max_ticks = 6

    def done(self, obs: Observation, mem: Memory) -> bool:
        return (self.ticks >= self.max_ticks and obs.self_state.energy > 0.8) or mem.under_attack(1) \
            or (mem.threat_seen and mem.threat_distance < 5.0) or self.ticks >= 40

    def act(self, obs: Observation, mem: Memory) -> Action:
        return Action(rest=True)


class Hide(Option):
    """Immobile (silencieux), tourné vers la dernière menace pour la surveiller."""

    name = "HIDE"
    max_ticks = 6

    def done(self, obs: Observation, mem: Memory) -> bool:
        return self.ticks >= self.max_ticks or mem.under_attack(1)

    def act(self, obs: Observation, mem: Memory) -> Action:
        b = mem.threat_bearing(obs)
        return Action(rotation=face(b * 0.5) if b is not None else None)


class Flee(Option):
    """S'éloigne de la menace (vue, touchée ou entendue) en évitant les obstacles."""

    name = "FLEE"
    max_ticks = 8

    def available(self, obs: Observation, mem: Memory) -> bool:
        return mem.threat_heading is not None and mem.threat_active(6)

    def done(self, obs: Observation, mem: Memory) -> bool:
        if mem.flee_hold and mem.under_attack(mem.flee_hold) and obs.self_state.energy > 0.1:
            # Frappé récemment : le poursuivant est sans doute juste derrière, hors du
            # champ de vision ; ralentir maintenant, c'est se faire rattraper.
            return False
        return self.ticks >= self.max_ticks or not mem.threat_active(4)

    def act(self, obs: Observation, mem: Memory) -> Action:
        s = obs.self_state
        b = mem.threat_bearing(obs)
        away = wrap(b + pi) if b is not None else 0.0
        if mem.serious_threat():
            # L'exploration qui suivra garde le cap de fuite au lieu de revenir vers la menace.
            mem.wander_heading = wrap(s.orientation + away)
            mem.wander_ticks = 15
        speed = 3.2 if s.energy > 0.4 else (2.2 if s.energy > 0.15 else 1.4)
        door_act = door_interaction(obs, mem)
        mv, rot = steer(obs, away, speed, mem)
        if door_act is not None and door_act.interaction is not None:
            return Action(movement=mv, rotation=rot, interaction=door_act.interaction)
        return Action(movement=mv, rotation=rot)


COUNTER_WINDOW = 2  # ticks après un coup invisible pendant lesquels on peut riposter


def counter_ready(obs: Observation, mem: Memory) -> bool:
    return mem.counter_attack and obs.tick - mem.blind_hit_tick <= COUNTER_WINDOW \
        and obs.self_state.health >= mem.counter_min_health


class Fight(Option):
    """Fait face à l'agent le plus proche et attaque avec l'arme équipée."""

    name = "FIGHT"
    max_ticks = 5

    def available(self, obs: Observation, mem: Memory) -> bool:
        if mem.fight_cooldown > 0 or not is_armed(obs):
            return False
        # Riposte : frappé à l'instant par un attaquant invisible (dans le dos).
        return any(d.kind is ObjectKind.AGENT for d in obs.vision) or counter_ready(obs, mem)

    def start(self, obs: Observation, mem: Memory) -> None:
        super().start(obs, mem)
        self.lost = 0
        # En riposte, laisser le temps de faire demi-tour (π rad/s : ~2 ticks).
        self.max_lost = 3 if counter_ready(obs, mem) else 2
        if mem.tick - mem.fight_last_tick > 3:  # nouvel engagement
            mem.fight_best, mem.fight_no_progress = 99.0, 0
        self.strafe = 1.0 if obs.tick % 2 else -1.0

    def done(self, obs: Observation, mem: Memory) -> bool:
        return self.ticks >= self.max_ticks or self.lost >= self.max_lost or not is_armed(obs) or mem.fight_cooldown > 0

    def act(self, obs: Observation, mem: Memory) -> Action:
        target = nearest(obs, {ObjectKind.AGENT})
        if target is None:
            self.lost += 1
            b = mem.threat_bearing(obs)
            return Action(rotation=face(b) if b is not None else None)
        self.lost = 0
        weapon = equipped_item(obs)
        mode = weapon_mode(weapon)
        if self.ticks % 4 == 0:
            self.strafe = -self.strafe
        if mode == "ranged":
            fwd = -0.7 if target.distance < 3.0 else (0.4 if target.distance > 7.0 else 0.0)
            interaction = Shoot(target.bearing) if abs(target.bearing) < 0.35 and target.distance < 9.0 else None
            return Action(movement=Movement(fwd, 0.6 * self.strafe, 1.8), rotation=face(target.bearing),
                          interaction=interaction)
        # Mêlée : coller la cible en tournant autour, frapper à portée.
        mem.fight_last_tick = mem.tick
        if target.distance < mem.fight_best - 0.15:
            mem.fight_best, mem.fight_no_progress = target.distance, 0
        else:
            mem.fight_no_progress += 1
        if mem.fight_no_progress >= 8 and target.distance >= 1.0:
            mem.fight_cooldown = 40  # cible inatteignable (obstacle, coincée...)
        if target.distance < 1.0 and weapon is not None:
            return Action(movement=Movement(-0.3, 0.8 * self.strafe, 1.4), rotation=face(target.bearing),
                          interaction=Use(weapon.slot, target.ref))
        mv, rot = drive(target.bearing, 2.4)
        return Action(movement=mv, rotation=rot)


OPTION_NAMES = ("EXPLORE", "FLEE", "HIDE", "REST", "SEEK_ITEM", "CONSUME", "ARM", "FIGHT")


def make_options(rng: Random) -> dict[str, Option]:
    opts: list[Option] = [Explore(rng), Flee(), Hide(), Rest(), SeekItem(), Consume(), Arm(), Fight()]
    return {o.name: o for o in opts}
