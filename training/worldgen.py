"""Générateur de cartes procédurales pour tester l'agent hors de la ville du kit.

L'arène finale aura « une carte, des distributions d'objets et une complexité
différentes ». Ce module produit des mondes variés à partir de profils :

- disposition aléatoire des bâtiments (nombre, taille, côté de la porte) ;
- proportions d'objets propres à chaque profil, dont de nouveaux objets
  (armes de mêlée et à distance avec leurs munitions, nourriture hydratante,
  boisson énergisante, aliment avarié, trousse de soins...) ;
- conditions : eau, glace, vapeur, zones dangereuses (feu, produit
  toxique), brouillard, obscurité, taille de carte, rythme de la
  physiologie.

Les nouveaux objets n'utilisent que des propriétés lues par le moteur
généralisé (`attack`, `ammo_type`, `reach`, `projectile_speed`, `energy`,
`toxicity`, `damage_per_tick`...). Rien ici n'est visible par la politique
autrement que via l'Observation.

Usage :
    python -m training.worldgen                       # liste des profils
    python -m training.worldgen --gui desert --seed 3 # visualiser une carte
    python -m training.worldgen --stats               # inventaire moyen par profil
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from random import Random
from typing import Callable

from rl_arena.arena_api import ObjectKind
from rl_arena.config import EnvironmentConfig, GenerationConfig, PhysiologyConfig
from rl_arena.generation import GeneratedWorld, _add_building, _Ids, _spawn_item
from rl_arena.geometry import Rect
from rl_arena.world import AreaEffect, RectEntity, WorldObject

AGENT_CLEARANCE = 2.0   # couloir minimal entre bâtiments (> diamètre d'un agent : pas de fente-piège)
SPAWN_CLEARANCE = 1.2   # rayon laissé libre autour des points d'apparition du moteur
SPAWN_PROTECTED = 8     # le moteur n'a besoin que d'un point par agent (8 au plus)
SPAWN_FRACTIONS = [      # mêmes fractions que TrainingEnvironment._spawn_points
    (0.50, 0.08), (0.08, 0.50), (0.50, 0.92), (0.92, 0.50), (0.30, 0.47), (0.47, 0.30),
    (0.70, 0.53), (0.53, 0.70), (0.18, 0.18), (0.82, 0.82), (0.82, 0.18), (0.18, 0.82),
]


# ----------------------------------------------------------------------
# Catalogue d'objets
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ItemTemplate:
    kind: ObjectKind
    mass: float
    props: dict
    theme: str | None = None  # bâtiment préféré quand l'objet est placé à l'intérieur


def _food(label, nutrition, mass=0.35, **extra):
    return ItemTemplate(ObjectKind.FOOD, mass, {"label": label, "nutrition": nutrition, **extra}, "shop")


def _drink(label, hydration, mass=0.5, **extra):
    return ItemTemplate(ObjectKind.DRINK, mass, {"label": label, "hydration": hydration, **extra}, "shop")


def _med(label, heal, mass=0.25, medical_type="medicine"):
    return ItemTemplate(ObjectKind.MEDICINE, mass, {"label": label, "medical_type": medical_type, "heal": heal}, "medical")


def _melee(label, damage, reach, mass):
    return ItemTemplate(ObjectKind.WEAPON, mass, {"label": label, "weapon_type": label, "attack": "melee",
                                                  "damage": damage, "reach": reach}, "industrial")


def _ranged(label, ammo, damage, speed, mass, loudness=1.0):
    return ItemTemplate(ObjectKind.WEAPON, mass, {"label": label, "weapon_type": label, "attack": "ranged",
                                                  "ammo_type": ammo, "damage": damage, "projectile_speed": speed,
                                                  "loudness": loudness}, "armory")


def _ammo(label, mass=0.18):
    return ItemTemplate(ObjectKind.AMMUNITION, mass, {"label": label, "ammo_type": label}, "armory")


CATALOG: dict[str, ItemTemplate] = {
    # Objets du kit.
    "sandwich": _food("sandwich", 0.35),
    "fruit": _food("fruit", 0.22, mass=0.30),
    "bottle": _drink("bottle", 0.40, mass=0.55),
    "water_bottle": _drink("water_bottle", 0.35),
    "bandage": _med("bandage", 0.04, medical_type="bandage"),
    "medicine": _med("medicine", 0.10, mass=0.20),
    "rolling_pin": ItemTemplate(ObjectKind.WEAPON, 0.65, {"label": "rolling_pin", "weapon_type": "rolling_pin",
                                                          "damage": 0.12}, "shop"),
    "potato_launcher": ItemTemplate(ObjectKind.WEAPON, 2.2, {"label": "potato_launcher",
                                                             "weapon_type": "potato_launcher", "damage": 0.18},
                                    "industrial"),
    "potato": _ammo("potato"),
    "reinforced_spring": ItemTemplate(ObjectKind.WEAPON_MODIFIER, 0.25, {"label": "reinforced_spring",
                                      "modifier": "projectile_speed", "factor": 1.15}, "industrial"),
    "wood": ItemTemplate(ObjectKind.RAW_MATERIAL, 0.8, {"label": "wood", "material": "wood"}),
    "metal": ItemTemplate(ObjectKind.RAW_MATERIAL, 1.2, {"label": "metal", "material": "metal"}, "industrial"),
    # Nouveaux consommables.
    "canned_food": _food("canned_food", 0.45, mass=0.5),
    "watermelon": _food("watermelon", 0.20, mass=1.6, hydration=0.25),
    "soup": _food("soup", 0.25, mass=0.6, hydration=0.15),
    "spoiled_meat": _food("spoiled_meat", 0.30, mass=0.4, toxicity=0.10, spoiled=True),
    "soda": _drink("soda", 0.25, mass=0.35, energy=0.10),
    "energy_drink": _drink("energy_drink", 0.15, mass=0.30, energy=0.35),
    "water_jug": _drink("water_jug", 0.60, mass=2.2),
    "milk": _drink("milk", 0.25, mass=0.6, nutrition=0.15),
    "first_aid_kit": _med("first_aid_kit", 0.30, mass=0.8),
    "painkiller": _med("painkiller", 0.06, mass=0.05),
    # Nouvelles armes de mêlée.
    "kitchen_knife": _melee("kitchen_knife", 0.09, 0.45, 0.3),
    "baseball_bat": _melee("baseball_bat", 0.16, 0.75, 1.0),
    "crowbar": _melee("crowbar", 0.14, 0.60, 2.0),
    # Nouvelles armes à distance et munitions assorties.
    "slingshot": _ranged("slingshot", "pebble", 0.10, 8.0, 0.4, loudness=0.4),
    "crossbow": _ranged("crossbow", "bolt", 0.26, 10.0, 2.8, loudness=0.5),
    "nail_gun": _ranged("nail_gun", "nail", 0.12, 9.0, 1.8),
    "pebble": _ammo("pebble", 0.10),
    "bolt": _ammo("bolt", 0.15),
    "nail": _ammo("nail", 0.05),
}


# ----------------------------------------------------------------------
# Profils
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MapProfile:
    name: str
    description: str
    size: tuple[float, ...] = (30.0,)
    buildings: tuple[int, int] = (3, 5)
    building_size: tuple[float, float] = (4.5, 8.0)
    building_themes: tuple[str, ...] = ("shop", "medical", "industrial", "shop", "armory")
    obstacles: tuple[int, int] = (5, 10)
    trees: tuple[int, int] = (2, 5)
    moving: tuple[int, int] = (1, 2)
    water: tuple[int, int] = (0, 1)
    steam: tuple[int, int] = (0, 1)
    hazards: tuple[int, int] = (0, 0)
    hazard_damage: tuple[float, float] = (0.004, 0.012)
    ice: tuple[int, int] = (0, 0)
    fog_probability: float = 0.3
    low_light_probability: float = 0.25
    item_count: tuple[int, int] = (12, 20)
    inside_ratio: float = 0.6            # part des objets placés dans un bâtiment
    weights: dict = field(default_factory=dict)  # nom du catalogue -> poids
    physiology: float = 1.0              # multiplicateur des besoins (soif, faim)

    def config(self, rng: Random, max_ticks: int = 3000) -> EnvironmentConfig:
        size = rng.choice(self.size)
        decay = self.physiology
        return EnvironmentConfig(
            max_ticks=max_ticks,
            map_width=size,
            map_height=size,
            generation=GenerationConfig(fog_probability=self.fog_probability,
                                        low_light_probability=self.low_light_probability),
            physiology=PhysiologyConfig(hydration_decay_per_tick=0.0006 * decay,
                                        satiety_decay_per_tick=0.0004 * decay),
        )

    def generator(self) -> Callable[[EnvironmentConfig, Random], GeneratedWorld]:
        return lambda config, rng: generate(self, config, rng)


BALANCED = {"sandwich": 3, "fruit": 3, "bottle": 3, "water_bottle": 4, "bandage": 2, "medicine": 2,
            "rolling_pin": 1, "potato_launcher": 1, "potato": 3, "reinforced_spring": 0.5,
            "wood": 2, "metal": 1}
NEW_ITEMS = {"canned_food": 1.5, "watermelon": 1, "soup": 1, "spoiled_meat": 1, "soda": 1.5, "energy_drink": 1,
             "water_jug": 0.7, "milk": 1, "first_aid_kit": 0.6, "painkiller": 1, "kitchen_knife": 1,
             "baseball_bat": 0.8, "crowbar": 0.6, "slingshot": 0.8, "crossbow": 0.5, "nail_gun": 0.5,
             "pebble": 2, "bolt": 1.2, "nail": 1.5}


def _mix(*dicts: dict, scale: dict | None = None) -> dict:
    out: dict = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + v
    for k, f in (scale or {}).items():
        out[k] = out.get(k, 0.0) * f
    return out


PROFILES: dict[str, MapProfile] = {p.name: p for p in (
    MapProfile("town", "ville aléatoire, objets du kit et nouveaux à parts égales",
               weights=_mix(BALANCED, NEW_ITEMS)),
    MapProfile("desert", "grande carte, peu de bâtiments, boissons très rares, soif x1.4, zones brûlantes",
               size=(40.0, 46.0), buildings=(2, 3), water=(0, 0), hazards=(2, 4), item_count=(8, 13),
               inside_ratio=0.4, physiology=1.4,
               weights=_mix(BALANCED, NEW_ITEMS, scale={"bottle": 0.3, "water_bottle": 0.3, "water_jug": 0.4,
                                                      "soda": 0.5, "watermelon": 2.0, "spoiled_meat": 2.0})),
    MapProfile("abundance", "beaucoup d'objets, nourriture et eau partout",
               item_count=(28, 40), inside_ratio=0.4,
               weights=_mix(BALANCED, NEW_ITEMS, scale={"water_jug": 2, "canned_food": 2, "first_aid_kit": 2})),
    MapProfile("armory", "beaucoup d'armes et de munitions, peu de soins",
               buildings=(4, 6), item_count=(18, 26),
               building_themes=("armory", "armory", "industrial", "shop"),
               weights=_mix(BALANCED, NEW_ITEMS, scale={"kitchen_knife": 3, "baseball_bat": 3, "crowbar": 3,
                                                      "slingshot": 3, "crossbow": 3, "nail_gun": 3, "pebble": 3,
                                                      "bolt": 3, "nail": 3, "potato_launcher": 2, "potato": 2,
                                                      "medicine": 0.3, "bandage": 0.3, "first_aid_kit": 0.3})),
    MapProfile("maze", "petite carte encombrée : nombreux bâtiments et obstacles, couloirs étroits",
               size=(30.0, 34.0), buildings=(6, 9), building_size=(3.5, 6.0), obstacles=(12, 18),
               trees=(4, 8), moving=(2, 4), inside_ratio=0.75, weights=_mix(BALANCED, NEW_ITEMS)),
    MapProfile("toxic", "zones toxiques, aliments avariés, soins abondants",
               hazards=(4, 7), hazard_damage=(0.006, 0.02), steam=(2, 4), fog_probability=0.6,
               weights=_mix(BALANCED, NEW_ITEMS, scale={"spoiled_meat": 4, "first_aid_kit": 3, "medicine": 2,
                                                      "painkiller": 2})),
    MapProfile("frozen", "glace et eau, brouillard fréquent, faim x1.3",
               ice=(3, 6), water=(1, 2), fog_probability=0.8, physiology=1.3,
               weights=_mix(BALANCED, NEW_ITEMS, scale={"soup": 3, "canned_food": 2})),
    MapProfile("dark", "nuit noire et brouillard permanents, vapeur",
               fog_probability=1.0, low_light_probability=1.0, steam=(2, 4), weights=_mix(BALANCED, NEW_ITEMS)),
    MapProfile("tiny", "carte 20x20 bondée, peu d'objets",
               size=(20.0, 22.0), buildings=(2, 3), building_size=(4.0, 5.5), obstacles=(3, 6), trees=(1, 2),
               item_count=(8, 12), weights=_mix(BALANCED, NEW_ITEMS)),
    MapProfile("huge", "carte 56x56 clairsemée, objets dispersés dehors",
               size=(52.0, 56.0), buildings=(5, 8), obstacles=(12, 20), trees=(6, 12), water=(1, 3),
               item_count=(18, 28), inside_ratio=0.35, weights=_mix(BALANCED, NEW_ITEMS)),
    MapProfile("new_only", "uniquement les nouveaux objets (aucun objet du kit)",
               weights=dict(NEW_ITEMS)),
)}

BUILDING_NAMES = {
    "shop": ["Epicerie", "Superette", "Boulangerie", "Cafe", "Bazar", "Marche"],
    "medical": ["Pharmacie", "Clinique", "Infirmerie"],
    "industrial": ["Garage", "Atelier", "Entrepot", "Hangar"],
    "armory": ["Armurerie", "Poste", "Caserne"],
}
RENDER_THEME = {"armory": "industrial"}  # thèmes connus du rendu Arcade


# ----------------------------------------------------------------------
# Génération
# ----------------------------------------------------------------------
def _spawn_points(W: float, H: float) -> list[tuple[float, float]]:
    return [(W * fx, H * fy) for fx, fy in SPAWN_FRACTIONS[:SPAWN_PROTECTED]]


def _rect_overlaps(a: Rect, b: Rect, margin: float) -> bool:
    return (abs(a.x - b.x) < (a.w + b.w) / 2 + margin) and (abs(a.y - b.y) < (a.h + b.h) / 2 + margin)


def _rect_near_point(r: Rect, x: float, y: float, margin: float) -> bool:
    return abs(r.x - x) < r.w / 2 + margin and abs(r.y - y) < r.h / 2 + margin


def generate(profile: MapProfile, config: EnvironmentConfig, rng: Random) -> GeneratedWorld:
    world = GeneratedWorld()
    ids = _Ids()
    W, H = config.map_width, config.map_height
    spawns = _spawn_points(W, H)
    taken: list[Rect] = []  # empreintes déjà occupées (bâtiments, obstacles, zones dures)

    def free_rect(w: float, h: float, margin: float, tries: int = 200, border: float = 1.0) -> Rect | None:
        for _ in range(tries):
            r = Rect(rng.uniform(border + w / 2, W - border - w / 2), rng.uniform(border + h / 2, H - border - h / 2), w, h)
            if any(_rect_overlaps(r, o, margin) for o in taken):
                continue
            if any(_rect_near_point(r, x, y, SPAWN_CLEARANCE) for x, y in spawns):
                continue
            return r
        return None

    # Bâtiments.
    buildings: list[tuple[Rect, str, str]] = []
    used_names: set[str] = set()
    for _ in range(rng.randint(*profile.buildings)):
        theme = rng.choice(profile.building_themes)
        w = round(rng.uniform(*profile.building_size), 1)
        h = round(rng.uniform(*profile.building_size), 1)
        r = free_rect(w, h, AGENT_CLEARANCE)
        if r is None:
            continue
        names = [n for n in BUILDING_NAMES[theme] if n not in used_names] or [f"{BUILDING_NAMES[theme][0]}{len(used_names)}"]
        label = rng.choice(names)
        used_names.add(label)
        _add_building(world, ids, cx=r.x, cy=r.y, w=w, h=h, door_side=rng.choice(["top", "bottom", "left", "right"]),
                      label=label, theme=RENDER_THEME.get(theme, theme))
        # Réserver aussi l'accès devant la porte (le rectangle est déjà réservé avec marge).
        taken.append(r)
        buildings.append((r, theme, label))

    # Obstacles rectangulaires.
    for _ in range(rng.randint(*profile.obstacles)):
        label, (w, h) = rng.choice([("bench", (1.4, 0.5)), ("crate", (1.3, 1.3)), ("pallet", (1.8, 0.8)),
                                    ("sign", (0.7, 2.2)), ("pipe", (2.0, 0.7)), ("car", (2.2, 1.2))])
        if rng.random() < 0.5:
            w, h = h, w
        r = free_rect(w, h, AGENT_CLEARANCE * 0.7)
        if r is None:
            continue
        world.rects.append(RectEntity(ids.next(), ObjectKind.OBSTACLE, r, sound_absorption=0.15,
                                      properties={"label": label}))
        taken.append(r)

    # Arbres (obstacles ronds, solides et occultants).
    for _ in range(rng.randint(*profile.trees)):
        rad = rng.uniform(0.45, 0.65)
        r = free_rect(2 * rad, 2 * rad, AGENT_CLEARANCE * 0.7)
        if r is None:
            continue
        world.objects.append(WorldObject(uid=ids.next(), kind=ObjectKind.OBSTACLE, x=r.x, y=r.y, mass=250.0,
                                         radius=rad, portable=False, solid=True, properties={"label": "tree"}))
        taken.append(r)

    # Chariots mobiles : trajectoire entièrement libre.
    for _ in range(rng.randint(*profile.moving)):
        axis = rng.choice(["x", "y"])
        amp = rng.uniform(2.0, 4.0)
        span = (2 * amp + 1.2, 1.2) if axis == "x" else (1.2, 2 * amp + 1.2)
        r = free_rect(*span, AGENT_CLEARANCE * 0.5)
        if r is None:
            continue
        world.objects.append(WorldObject(uid=ids.next(), kind=ObjectKind.MOVING_OBSTACLE, x=r.x, y=r.y, mass=80.0,
                                         radius=0.55, portable=False, solid=True,
                                         properties={"axis": axis, "origin_x": r.x, "origin_y": r.y, "amplitude": amp,
                                                     "period_ticks": rng.randint(18, 36)}))
        taken.append(r)

    # Zones (traversables) : elles évitent bâtiments et points d'apparition.
    def area(kind: ObjectKind, size: tuple[float, float], props: dict) -> None:
        r = free_rect(rng.uniform(*size), rng.uniform(*size), 0.5)
        if r is not None:
            world.areas.append(AreaEffect(ids.next(), kind, r, props))
            if kind is ObjectKind.HAZARD:
                taken.append(r)  # pas d'objet ni d'obstacle dans le feu

    for _ in range(rng.randint(*profile.water)):
        area(ObjectKind.WATER, (3.0, 7.0), {"speed_factor": 0.55, "energy_factor": 1.25})
    for _ in range(rng.randint(*profile.steam)):
        area(ObjectKind.STEAM, (2.0, 4.0), {"visibility_factor": rng.uniform(0.45, 0.65)})
    for _ in range(rng.randint(*profile.ice)):
        area(ObjectKind.ICE, (3.0, 6.0), {"label": "ice", "speed_factor": 0.65, "energy_factor": 1.15})
    for _ in range(rng.randint(*profile.hazards)):
        label = rng.choice(["fire", "toxic_spill", "broken_glass"])
        area(ObjectKind.HAZARD, (1.5, 3.5), {"label": label, "damage_per_tick": round(rng.uniform(*profile.hazard_damage), 4)})

    # Atmosphère.
    if rng.random() < config.generation.fog_probability:
        world.visibility_factor = rng.uniform(0.55, 0.82)
    if rng.random() < config.generation.low_light_probability:
        world.low_light_factor = rng.uniform(0.65, 0.90)

    # Objets.
    names = [n for n, w in profile.weights.items() if w > 0]
    weights = [profile.weights[n] for n in names]
    inner = {label: Rect(r.x, r.y, r.w - 1.2, r.h - 1.2) for r, _, label in buildings}
    target = rng.randint(*profile.item_count)
    placed = attempts = 0
    while placed < target and attempts < 40 * target:
        attempts += 1
        name = rng.choices(names, weights)[0]
        tpl = CATALOG[name]
        inside = buildings and rng.random() < profile.inside_ratio
        if inside:
            themed = [b for b in buildings if b[1] == tpl.theme] or buildings
            r, _, label = rng.choice(themed)
            box = inner[label]
            x = rng.uniform(box.x - box.w / 2, box.x + box.w / 2)
            y = rng.uniform(box.y - box.h / 2, box.y + box.h / 2)
        else:
            x, y = rng.uniform(0.5, W - 0.5), rng.uniform(0.5, H - 0.5)
            if any(_rect_near_point(o, x, y, 0.4) for o in taken):
                continue
        world.objects.append(_spawn_item(ids, tpl.kind, x, y, tpl.mass, **dict(tpl.props)))
        placed += 1
    return world


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _stats(seeds: int = 30) -> None:
    from collections import Counter

    print(f"{'profil':10s} {'taille':>6s} {'bât.':>5s} {'objets':>6s} {'boiss.':>6s} {'nourr.':>6s} {'soins':>6s} "
          f"{'armes':>6s} {'munit.':>6s} {'danger':>6s} {'glace':>6s}")
    for name, prof in PROFILES.items():
        c: Counter = Counter()
        for s in range(seeds):
            rng = Random(s)
            cfg = prof.config(rng)
            w = generate(prof, cfg, rng)
            c["size"] += cfg.map_width
            c["b"] += len(w.buildings)
            for o in w.objects:
                if o.portable:
                    c["items"] += 1
                    c[o.kind] += 1
            for a in w.areas:
                c[a.kind] += 1
        f = lambda k: c[k] / seeds  # noqa: E731
        print(f"{name:10s} {f('size'):6.0f} {f('b'):5.1f} {f('items'):6.1f} {f(ObjectKind.DRINK):6.1f} "
              f"{f(ObjectKind.FOOD):6.1f} {f(ObjectKind.MEDICINE):6.1f} {f(ObjectKind.WEAPON):6.1f} "
              f"{f(ObjectKind.AMMUNITION):6.1f} {f(ObjectKind.HAZARD):6.1f} {f(ObjectKind.ICE):6.1f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gui", metavar="PROFIL", help="visualiser une carte de ce profil avec nos agents")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stats", action="store_true")
    a = p.parse_args()
    if a.stats:
        _stats()
        return
    if a.gui:
        from rl_arena.render_arcade import run_arcade

        from training.envs import map_spec, build_env

        spec = map_spec(a.gui, a.seed)
        env = build_env(spec, student=__import__("student_agent").MyAgent(), render=True)
        env.reset(seed=a.seed)
        print(f"profil {a.gui} : {PROFILES[a.gui].description} (seed {a.seed})")
        run_arcade(env)
        return
    for name, prof in PROFILES.items():
        print(f"{name:10s} {prof.description}")


if __name__ == "__main__":
    main()
