# Contrat d'interface Agent ↔ Arène

Ce document décrit ce qui est garanti entre l'environnement d'entraînement et l'arène finale.

La définition Python faisant foi est `rl_arena/arena_api.py`.

## 1. Cycle d'un tick

Un tick représente **0,5 seconde simulée**.

À chaque tick, tous les agents encore vivants :

1. reçoivent une observation calculée à partir du même état logique du monde ;
2. choisissent une action ;
3. voient ensuite toutes les actions résolues par le moteur.

L'ordre d'appel des agents ne donne pas d'avantage physique.

## 2. Cycle de vie de l'agent

Votre classe doit exposer :

```python
class MyAgent:
    def reset(self) -> None:
        ...

    def act(self, observation: Observation) -> Action:
        ...

    def on_episode_end(self, result: EpisodeResult) -> None:
        ...
```

`reset()` est appelé au début d'un épisode.

`act()` est appelé à chaque tick tant que l'agent est vivant.

`on_episode_end()` est appelé lorsque l'épisode de l'agent se termine.

## 3. Conventions

- distances : mètres ;
- masses : kilogrammes ;
- temps : secondes ;
- angles : radians ;
- angles positifs : **sens horaire** ;
- un `bearing` est relatif à l'orientation de l'agent ;
- `bearing = 0` signifie devant ;
- `bearing = +π/2` signifie à droite ;
- `bearing = -π/2` signifie à gauche.

## 4. Observation

```python
@dataclass(frozen=True, slots=True)
class Observation:
    tick: int
    self_state: SelfState
    vision: tuple[VisualDetection, ...]
    hearing: tuple[SoundDetection, ...]
    touch: tuple[TouchContact, ...]
    events: tuple[Event, ...]
```

L'observation est **partielle**. Elle ne contient pas l'état réel du monde.

### 4.1 État propre

```python
SelfState(
    health,
    energy,
    hydration,
    satiety,
    body_mass,
    carried_mass,
    speed,
    orientation,
    injuries,
    inventory,
    equipped_slot,
)
```

`health`, `energy`, `hydration` et `satiety` sont dans `[0, 1]`.

L'inventaire contient des `InventoryItem` avec :

```python
InventoryItem(slot, kind, mass, properties)
```

Le `slot` est stable tant que l'objet est transporté.

### 4.2 Vision

```python
VisualDetection(
    ref,
    kind,
    distance,
    bearing,
    angular_size,
    properties,
)
```

Le moteur peut dégrader la vision à cause du brouillard, de la vapeur, de la lumière ou des occultations.

Vous ne recevez pas la variable « brouillard » : vous en observez seulement les conséquences.

`kind` peut être `ObjectKind.UNKNOWN` si la reconnaissance est insuffisante.

`properties` peut être vide ou partiel. N'écrivez jamais une politique qui suppose qu'une clé est toujours présente.

#### Référence `ref`

`ref` est une référence opaque valable **uniquement pendant le tick courant**.

Elle sert par exemple à écrire :

```python
Action(interaction=Pickup(target_ref=detection.ref))
```

Ne mémorisez pas `ref` comme identité durable d'un objet. Elle peut changer au tick suivant.

### 4.3 Audition

```python
SoundDetection(
    kind,
    intensity,
    bearing,
    bearing_uncertainty,
    distance_estimate,
    muffling,
)
```

Le son est une perception, pas une vérité parfaite.

La direction et la distance peuvent être fausses ou imprécises. Les murs et matériaux peuvent atténuer et étouffer le son.

`intensity` et `muffling` sont normalisés dans `[0, 1]`.

### 4.4 Toucher

```python
TouchContact(ref, kind, bearing, properties)
```

Le toucher décrit les objets effectivement au contact de l'agent.

### 4.5 Événements

Les événements décrivent les conséquences perceptibles depuis le tick précédent : dégâts, collision, objet récupéré, tir, soin, etc.

```python
Event(kind, value=None, properties={...})
```

## 5. Catégories d'objets

La grande catégorie appartient à l'énumération fermée `ObjectKind` :

```text
UNKNOWN
AGENT
WALL
OBSTACLE
MOVING_OBSTACLE
DOOR
CONTROL
MACHINE
RAW_MATERIAL
WATER
ICE
STEAM
WEAPON
WEAPON_MODIFIER
AMMUNITION
PROJECTILE
FOOD
DRINK
MEDICINE
TOOL
CONTAINER
LIGHT_SOURCE
HAZARD
```

Les détails sont portés par `properties`, lorsqu'ils sont perceptibles.

## 6. Actions

```python
Action(
    movement=None,
    rotation=None,
    rest=False,
    interaction=None,
)
```

Un déplacement, une rotation et une interaction peuvent être demandés pendant le même tick si la situation physique le permet.

### 6.1 Déplacement

```python
Movement(forward, right, speed)
```

Le repère est **local à l'agent** :

- `forward=+1` : devant ;
- `forward=-1` : derrière ;
- `right=+1` : droite ;
- `right=-1` : gauche.

`speed` est la vitesse désirée en m/s.

Le moteur reste responsable de la vitesse réellement possible, de l'énergie et des collisions.

### 6.2 Rotation

```python
Rotation(angular_speed)
```

`angular_speed` est en rad/s, positif dans le sens horaire.

### 6.3 Repos

```python
Action(rest=True)
```

Le repos favorise la récupération d'énergie. Certaines actions incompatibles peuvent alors ne produire aucun effet.

### 6.4 Interactions

```python
Pickup(target_ref)
Drop(slot)
Equip(slot)
Use(slot, target_ref=None)
Throw(slot, bearing, power)
Shoot(bearing)
Ingest(slot)
Manipulate(target_ref)
```

`Throw.power` est dans `[0, 1]`.

`Shoot.bearing` et `Throw.bearing` sont relatifs à l'orientation de l'agent.

### Actions sans effet

Une action physiquement impossible n'est pas une erreur de protocole.

Exemples : tirer sans munition, ramasser trop loin, utiliser un slot vide.

Dans ces cas, l'action correspondante ne modifie simplement pas le monde.

## 7. Règle de compatibilité

Pour être transférable vers l'arène finale, votre politique doit prendre ses décisions à partir de :

```python
rl_arena.arena_api
```

et des informations reçues dans `Observation`.

Vous pouvez modifier votre copie de l'environnement pour vos expériences, mais ne faites pas dépendre votre agent de l'état interne du moteur de test.
