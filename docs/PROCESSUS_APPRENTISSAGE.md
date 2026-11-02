# Processus d'apprentissage de `MyAgent`

Ce document décrit la démarche retenue pour entraîner par renforcement l'agent de `student_agent.py`, de la représentation de l'état jusqu'à la livraison pour le tournoi.

Rappel de l'objectif : le classement du tournoi repose **uniquement sur la durée de survie**, dans une arène finale **inconnue** qui respecte le même contrat (`rl_arena/arena_api.py`). Tout ce qui suit est orienté par deux exigences :

1. **survivre**, pas maximiser la récompense d'entraînement ;
2. **généraliser** à une carte, des objets, des adversaires et une complexité différents.

---

## 1. Diagnostic de l'environnement d'entraînement

Avant de concevoir quoi que ce soit, nous avons mesuré le comportement de l'agent trivial fourni (il avance tout droit à 1 m/s).

| Configuration | Résultat mesuré |
|---|---|
| Défaut (`max_ticks=600`, 1 `RandomBot` + 1 `HunterBot`), 30 seeds | survit **600/600 ticks** sur les 30 seeds |
| `max_ticks=4000`, 1 `RandomBot`, 10 seeds | meurt systématiquement au tick **1750** |
| `max_ticks=1500`, 1 `RandomBot` + 3 `HunterBot`, 10 seeds | survit 1500/1500 ticks sur les 10 seeds |
| Vitesse de simulation (headless, 3 agents) | ≈ 340 ticks/s, soit ≈ 2 s par épisode de 600 ticks |

Conséquences :

- **L'environnement par défaut ne discrimine pas** : un agent qui ne fait rien d'intelligent atteint la troncature. Entraîner tel quel n'apprendrait rien d'utile.
- La mort au tick 1750 s'explique par la physiologie (`config.py`) : l'hydratation perd 0,0006/tick, passe sous le seuil critique 0,15 vers le tick 1417, puis la santé perd 0,003/tick pendant ≈ 333 ticks. **Sur des épisodes longs, la gestion de l'eau et de la nourriture devient la cause de mort principale.**
- Les `HunterBot` actuels tuent peu. L'arène finale contiendra les agents des autres groupes, probablement plus dangereux : il faut **durcir l'adversité** pendant l'entraînement.

Le processus d'apprentissage commence donc par **modifier l'environnement d'entraînement** (section 6), ce que le contrat autorise explicitement.

---

## 2. Architecture de l'agent : réflexes + politique apprise sur macro-actions

Apprendre directement l'action brute (`Movement`, `Rotation`, interaction, 8 types d'interactions avec `ref`/`slot`) toutes les 0,5 s est hors de portée d'un Q-learning tabulaire : l'espace d'actions est continu et combinatoire, et une décision isolée de 0,5 s a un effet trop faible pour être créditée.

Nous structurons l'agent en trois couches :

```
Observation
   │
   ├─► 1. Perception / mémoire  ──► état discret s  (features.py)
   │
   ├─► 2. Politique apprise     ──► option o = argmax Q(s, ·)   (Q-learning)
   │
   └─► 3. Contrôleur bas niveau ──► Action concrète du tick      (options.py)
            (scripté, exécute l'option pendant plusieurs ticks)
```

- **Couche 1 — perception** : transforme l'`Observation` bruitée et partielle en un petit vecteur discret, avec une mémoire à court terme (dernière direction de menace, ticks depuis les derniers dégâts…).
- **Couche 2 — décision apprise** : choisit une **option** (macro-action) parmi une liste fixe. C'est la partie entraînée par renforcement.
- **Couche 3 — exécution** : chaque option est un petit contrôleur scripté qui produit des `Action` valides pendant quelques ticks (s'orienter, se déplacer, cibler un `ref` du tick courant, choisir un `slot`).

Ce découpage correspond au cadre des **options / semi-MDP** (Sutton & Barto, 2e éd., §17.2). Il réduit l'espace de décision à une dizaine de choix, allonge l'horizon effectif de chaque décision, et rend le comportement lisible pour la soutenance.

Des **réflexes de sécurité** non appris peuvent court-circuiter la politique dans des cas sans ambiguïté (ex. : consommer un soin si `health < 0.2` et qu'un `MEDICINE` est dans l'inventaire). Ils doivent rester rares : chaque réflexe est un choix que la politique n'apprend pas.

---

## 3. Représentation de l'état

### 3.1 Principes

- Construite **uniquement** à partir de `Observation` (et de la mémoire de l'agent), jamais de `environment.py`, `debug_state()`, positions, seeds ou carte.
- **Relative à l'agent** (distances, `bearing`) : aucune coordonnée absolue, la carte de l'arène finale est différente.
- **Robuste** : `kind` peut valoir `UNKNOWN`, `properties` peut être vide. Toute lecture de propriété passe par `.get(...)` avec une valeur par défaut.
- **Aucune constante de l'environnement de test** codée en dur (8 slots, 15 kg, taille de carte, valeurs de soin…). On raisonne en ratios et en présence/absence.
- **Petite** : en tabulaire, chaque état doit être visité assez souvent. Cible : ≲ 10⁴ états effectivement atteints.

### 3.2 Features retenues (version 1)

| Feature | Source | Discrétisation |
|---|---|---|
| Santé | `self_state.health` | 3 niveaux : `< 0.35`, `< 0.7`, sinon |
| Énergie | `self_state.energy` | 3 niveaux : `< 0.2`, `< 0.5`, sinon |
| Besoin vital | `min(hydration, satiety)` | 3 niveaux : `< 0.3`, `< 0.6`, sinon |
| Ressource en inventaire pour ce besoin | `inventory` (`DRINK`/`FOOD`) | booléen |
| Soin en inventaire | `inventory` (`MEDICINE`) | booléen |
| Menace visuelle | `vision`, `kind == AGENT` le plus proche | 4 : aucune, loin (> 8 m), moyenne (3–8 m), proche (< 3 m) |
| Côté de la menace | `bearing` de cette menace | 3 : devant, gauche, droite |
| Menace sonore | `hearing`, `GUNSHOT`/`AGENT`/`MOVEMENT` intense | booléen |
| Dégâts récents | `events`, `DAMAGE_RECEIVED` dans les 4 derniers ticks (mémoire) | booléen |
| Armé | arme équipée (+ munition si l'arme en demande) | booléen |
| Ressource utile visible | `vision` : `FOOD`/`DRINK`/`MEDICINE`/`WEAPON`/`AMMUNITION` | booléen |
| Obstacle devant | `vision` : `WALL`/`OBSTACLE`/`MOVING_OBSTACLE`/`HAZARD` proche, `|bearing| < 0.5` | booléen |

Taille théorique : 3·3·3·2·2·4·3·2·2·2·2·2 ≈ 41 000, mais de nombreuses combinaisons sont inatteignables ou rares (le côté de la menace n'a de sens que s'il y a une menace : on le fixe à « devant » sinon). En pratique quelques milliers d'états sont visités.

### 3.3 Mémoire à court terme

`ref` n'est valable que pendant le tick courant : la mémoire ne stocke **jamais** de `ref`. Elle conserve :

- `ticks_since_damage` (mis à jour depuis `events`) ;
- la dernière direction de menace connue, recalée chaque tick avec la rotation propre de l'agent (`orientation` courante − `orientation` au moment de l'observation) ;
- un compteur de « menace persistante » (menace vue/entendue pendant les N derniers ticks) pour lisser le bruit de perception ;
- l'option en cours et sa durée.

Le suivi d'un objet entre deux ticks, si nécessaire, se fait par appariement `(kind, label, distance, bearing)` avec le tick précédent, jamais par `ref`.

### 3.4 Exemple de code

```python
from rl_arena.arena_api import EventKind, ObjectKind, Observation

def bin3(x, a, b):
    return 0 if x < a else (1 if x < b else 2)

def encode_state(obs: Observation, mem) -> tuple:
    s = obs.self_state
    kinds = {i.kind for i in s.inventory}
    need = min(s.hydration, s.satiety)
    need_kind = ObjectKind.DRINK if s.hydration <= s.satiety else ObjectKind.FOOD

    agents = [d for d in obs.vision if d.kind is ObjectKind.AGENT]
    threat, side = 0, 0
    if agents:
        t = min(agents, key=lambda d: d.distance)
        threat = 3 if t.distance < 3 else (2 if t.distance < 8 else 1)
        side = 0 if abs(t.bearing) < 0.4 else (1 if t.bearing < 0 else 2)

    return (
        bin3(s.health, 0.35, 0.7),
        bin3(s.energy, 0.2, 0.5),
        bin3(need, 0.3, 0.6),
        need_kind in kinds,
        ObjectKind.MEDICINE in kinds,
        threat, side,
        mem.heard_threat,
        mem.ticks_since_damage <= 4,
        mem.armed,
        mem.useful_item_visible,
        mem.obstacle_ahead,
    )
```

---

## 4. Espace d'actions : les options

| Option | Comportement du contrôleur bas niveau | Fin de l'option |
|---|---|---|
| `EXPLORE` | avance à vitesse modérée, évite les obstacles, tourne aléatoirement | 6 ticks, ou nouvelle menace |
| `FLEE` | s'oriente à l'opposé de la menace (vue ou entendue), vitesse élevée, évite les obstacles | 6 ticks, ou menace hors de vue depuis 3 ticks |
| `HIDE` | s'arrête derrière/près d'un obstacle, reste immobile (moins de bruit) | 6 ticks, ou dégâts reçus |
| `REST` | `Action(rest=True)` sans mouvement | 4 ticks, ou menace proche/dégâts |
| `SEEK_ITEM` | se dirige vers l'objet utile visible le plus proche puis `Pickup(ref)` | objet ramassé, perdu de vue, ou 10 ticks |
| `CONSUME` | `Ingest(slot)` sur l'objet qui couvre le besoin le plus urgent (soin, boisson, nourriture) | 1 tick |
| `ARM` | `Equip(slot)` de l'arme la plus adaptée | 1 tick |
| `FIGHT` | fait face à l'agent le plus proche ; `Shoot(bearing)` ou `Use(slot, ref)` au contact, en gardant ses distances | 4 ticks, ou cible perdue |

Les interactions impossibles étant des no-ops silencieux, un contrôleur qui échoue ne casse rien ; en revanche la politique doit apprendre que, par exemple, `CONSUME` sans ressource est une perte de temps (le tick passe, la récompense de l'option est mauvaise).

Les contrôleurs bas niveau s'inspirent de `rl_arena/bots.py`, qui sont des implémentations de référence du contrat (orientation via `Rotation(bearing / TICK_SECONDS)`, ouverture de portes via `Manipulate`, etc.).

---

## 5. Récompense d'entraînement

### 5.1 Pourquoi ne pas garder la récompense par défaut telle quelle

La récompense fournie (`RewardConfig`) contient `kill = +5` et `first_pickup = +1`, qui ne correspondent pas à l'objectif du tournoi. Un agent qui sur-optimise les kills prend des risques inutiles. Nous redéfinissons la récompense **dans notre copie de l'environnement** ou recalculons la nôtre dans la boucle d'entraînement (l'agent final, lui, n'utilise aucune récompense).

### 5.2 Récompense retenue

| Terme | Valeur | Justification |
|---|---|---|
| Survie par tick | +0,1 | signal principal, aligné avec le classement |
| Mort | −20 | rend la mort nettement pire que n'importe quel gain court terme |
| Dégâts reçus | −1 × dégâts | précurseur dense de la mort |
| Collision nuisible | −0,5 | conservé |
| Kill | **0 à +1** | éliminer un adversaire supprime une menace, mais ne doit pas devenir un objectif |
| Premier pickup | **+0,2** | aide l'exploration au début, réduit ensuite (voir curriculum) |
| Shaping physiologique | potentiel Φ = 0,5·(health + min(hydration, satiety)) | `r += γΦ(s') − Φ(s)` : shaping basé sur un potentiel, qui ne change pas la politique optimale (Ng et al., 1999) |

Modification de la configuration :

```python
from rl_arena.config import EnvironmentConfig, RewardConfig

cfg = EnvironmentConfig(
    max_ticks=3000,
    reward=RewardConfig(kill=0.5, first_pickup=0.2),
)
```

Pour une option qui dure k ticks, la récompense de l'option est la somme actualisée des récompenses de chaque tick : R = Σᵢ γⁱ rᵢ.

---

## 6. Environnement d'entraînement et curriculum

### 6.1 Leviers de modification (autorisés)

- `max_ticks` : allonger les épisodes pour que la faim, la soif et l'énergie comptent (≥ 3000 ticks).
- **Adversaires** : plusieurs `HunterBot`, des `RandomBot`, des variantes plus agressives (tir à plus longue portée, poursuite au son), et des **copies figées de notre propre agent** (self-play). `max_agents = 8` par défaut, dans la limite des points d'apparition.
- `GenerationConfig` : rareté des objets (`item_count`), brouillard (`fog_probability`), faible lumière (`low_light_probability`), obstacles. Ces paramètres sont **tirés aléatoirement à chaque épisode** (domain randomization) pour forcer la généralisation.
- Taille de carte (`map_width`, `map_height`) variable.
- `PhysiologyConfig` : décroissance plus rapide de l'hydratation/satiété pour obtenir plus d'épisodes informatifs par heure de calcul.

### 6.2 Curriculum

| Étape | Environnement | Ce que l'agent doit apprendre | Critère de passage |
|---|---|---|---|
| 0 | 1 `RandomBot`, `max_ticks=3000`, objets abondants, pas de brouillard | gérer énergie, boire, manger, se reposer | survie moyenne > 2500 ticks |
| 1 | + 1–2 `HunterBot` | fuir / se cacher, s'armer | survie moyenne > 2000 ticks |
| 2 | 3–4 `HunterBot`, objets rares, brouillard et faible lumière aléatoires | décider sous perception dégradée | progression stabilisée sur 500 épisodes |
| 3 | self-play : 2–3 copies figées de nos meilleures versions + bots | robustesse face à des agents « intelligents » | ne pas régresser contre les versions précédentes |

On ne quitte une étape que lorsque le critère est atteint **sur les seeds d'évaluation**, pas sur les seeds d'entraînement. Le pourcentage d'épisodes de chaque étape peut être mélangé (ex. 20 % d'épisodes d'étapes antérieures) pour éviter l'oubli.

Les seuils ci-dessus sont des valeurs de départ, à recalibrer après avoir mesuré la survie des bots de référence placés à la place de notre agent dans chaque configuration.

---

## 7. Algorithme d'apprentissage

### 7.1 Q-learning tabulaire sur options (SMDP)

Pour chaque décision prise dans l'état s, l'option o est exécutée pendant k ticks, cumule la récompense R et aboutit à l'état s' :

```
Q(s, o) ← Q(s, o) + α · [ R + γᵏ · max_o' Q(s', o') − Q(s, o) ]
```

avec `max Q(s', ·) = 0` si l'agent est mort (`terminated`). En cas de **troncature** (`truncated`), on n'annule pas la valeur future : l'agent n'est pas mort, l'épisode a simplement été coupé (bootstrap sur Q(s', ·)).

Hyperparamètres de départ :

| Paramètre | Valeur | Remarque |
|---|---|---|
| γ (par tick) | 0,995 | horizon effectif ≈ 200 ticks ≈ 100 s simulées |
| α | 0,1 puis décroissance `1 / (1 + n(s,o))^0.6` | taux par couple visité |
| ε (ε-greedy) | 1,0 → 0,05 sur les premiers 60 % des épisodes | exploration |
| Initialisation de Q | 0 (optimiste par rapport à la mort) | favorise l'essai des options |
| Masquage d'actions | `CONSUME` sans consommable, `FIGHT` sans menace, etc. exclues | accélère l'apprentissage |

Le masquage ne repose que sur l'observation (inventaire, vision), il reste donc valide dans l'arène finale.

### 7.2 Boucle d'entraînement

On utilise le **mode de contrôle externe** (`register_agent("student")` sans contrôleur) pour récupérer `StepResult.rewards` à chaque tick, comme dans `train_agent_skeleton.py`.

```python
import random
from rl_arena.config import EnvironmentConfig
from rl_arena.environment import TrainingEnvironment
from rl_arena.bots import HunterBot, RandomBot

def run_episode(learner, seed, stage_cfg, rng):
    env = TrainingEnvironment(stage_cfg.sample_config(rng))   # domain randomization
    env.register_agent("student")                              # contrôle externe
    for i, bot in enumerate(stage_cfg.sample_opponents(rng)):  # bots + copies figées
        env.register_agent(f"opp{i}", bot)
    obs = env.reset(seed=seed)["student"]
    learner.begin_episode(obs)

    while True:
        action = learner.act(obs)                  # choisit/continue une option
        res = env.step({"student": action})
        done = res.terminated["student"]
        learner.record(res.rewards["student"], done, res.truncated)
        if done or res.truncated:
            learner.end_episode(None if done else res.observations["student"], done)
            return res.info["student"].survival_ticks
        obs = res.observations["student"]
```

`learner.record` accumule R et k pour l'option en cours ; la mise à jour Q n'a lieu que lorsque l'option se termine (ou à la fin de l'épisode).

### 7.3 Seeds

- **Entraînement** : seeds tirées dans `[0, 1 000 000)`, une seed différente par épisode.
- **Évaluation** : ensemble fixe et disjoint, par exemple `range(10_000_000, 10_000_200)`, jamais utilisé pour l'entraînement.

### 7.4 Budget de calcul

À ≈ 340 ticks/s, un épisode de 3000 ticks coûte environ 10 s. 5000 épisodes ≈ 14 h en mono-processus. Pistes :

- réduire `max_ticks` ou accélérer la physiologie aux premières étapes ;
- paralléliser l'**évaluation** avec `multiprocessing` (épisodes indépendants, politique figée) ;
- pour l'entraînement, lancer plusieurs workers qui collectent des transitions `(s, o, R, k, s', done)` avec une Q-table figée, puis fusionner les mises à jour périodiquement dans le processus principal.

### 7.5 Évolution possible : approximation de fonction

Si la table reste trop creuse (beaucoup d'états vus < 10 fois) ou si l'on veut des features continues, on passe à une approximation de Q :

- **Q linéaire** sur features continues + tile coding (Sutton & Barto, ch. 9–10), en NumPy ;
- ou un petit réseau (`sklearn.neural_network.MLPRegressor` avec `partial_fit`) entraîné sur un replay buffer, avec une copie cible mise à jour périodiquement (principe de DQN).

Attention : l'agent final doit pouvoir s'exécuter dans l'arène. Si NumPy ou scikit-learn ne sont pas garantis dans l'environnement du tournoi, on exporte les poids et on réimplémente l'inférence en Python pur.

---

## 8. Protocole d'évaluation

Évaluation **toutes les 250 épisodes d'entraînement**, politique gloutonne (ε = 0), sur les seeds d'évaluation, dans plusieurs configurations :

| Configuration | Paramètres |
|---|---|
| `default` | environnement fourni, `max_ticks` allongé à 3000 |
| `hostile` | 4 `HunterBot` + 2 copies figées de versions antérieures |
| `scarce` | `item_count=(4, 8)`, physiologie accélérée |
| `blind` | `fog_probability=1.0`, `low_light_probability=1.0` |
| `big_map` | carte 50 × 50 m |

Métriques par configuration :

- survie **moyenne**, **médiane** et **10ᵉ centile** (le classement punit les morts précoces : le 10ᵉ centile est aussi important que la moyenne) ;
- taux de troncature (épisodes survécus jusqu'au bout) ;
- **cause de mort** : combat (dégâts reçus d'un agent juste avant la mort) ou physiologie (soif/faim, saignement). Cette analyse peut utiliser `info` et `debug_state()` : elle sert au diagnostic, pas à la politique ;
- fréquence d'utilisation de chaque option (détecte une politique dégénérée, ex. `REST` en permanence).

Références de comparaison, évaluées sur les mêmes seeds : l'agent trivial fourni, `RandomBot` et `HunterBot` placés à la place de notre agent, et une **politique scriptée** qui utilise les mêmes options avec des règles écrites à la main. L'apprentissage n'a de valeur que s'il bat cette dernière.

On conserve, pour chaque évaluation, la Q-table correspondante : la version livrée est celle qui maximise un score combiné (ex. moyenne des 10ᵉ centiles sur toutes les configurations), pas forcément la dernière.

---

## 9. Livraison dans `student_agent.py`

- La Q-table est sérialisée en JSON (clé = état encodé en chaîne, valeur = liste de Q par option) dans un fichier livré avec l'agent, chargé via un chemin relatif à `__file__`.
- `MyAgent.act` : encode l'état, choisit `argmax Q` (ε = 0) parmi les options valides, délègue au contrôleur bas niveau. Pour un état jamais vu, repli sur la politique scriptée.
- `MyAgent.reset` réinitialise uniquement la mémoire de l'épisode (pas la Q-table).
- Tout `act()` est protégé par un `try/except` qui renvoie une action sûre (`Action(rest=True)` ou une fuite simple) : une exception dans l'arène finale ne doit jamais coûter l'épisode.
- Imports limités à `rl_arena.arena_api` et à la bibliothèque standard (+ nos propres modules).
- Vérifications avant livraison :

```bash
python check_student_agent.py
python run_student_agent.py --seed 7 --gui
python -m unittest discover -s tests -v
```

Organisation de fichiers proposée :

```
student_agent.py          # MyAgent : chargement de la Q-table + politique gloutonne
agent/features.py         # encode_state, mémoire courte
agent/options.py          # contrôleurs bas niveau des options
agent/q_table.json        # politique apprise (version livrée)
training/train.py         # boucle d'entraînement, curriculum, self-play
training/evaluate.py      # protocole d'évaluation, rapports
training/envs.py          # configurations et adversaires d'entraînement
```

---

## 10. Plan de travail

| Phase | Contenu | Livrable |
|---|---|---|
| 1 | Contrôleurs bas niveau des options + politique scriptée de référence | agent scripté qui bat l'agent trivial sur `hostile` et `scarce` |
| 2 | `encode_state`, mémoire courte, environnements d'entraînement durcis, protocole d'évaluation | tableau de référence des baselines |
| 3 | Q-learning SMDP, étapes 0–1 du curriculum | courbes d'apprentissage, comparaison avec le scripté |
| 4 | Étapes 2–3 (perception dégradée, self-play), réglage des hyperparamètres et de l'état | Q-table candidate |
| 5 | Sélection de la meilleure version, intégration dans `MyAgent`, tests de robustesse | agent livré |
| 6 | Préparation de la soutenance (15 min, présentation + démonstration `--gui`) | slides : diagnostic, architecture, état, récompense, courbes, analyse des morts |

---

## 11. Pièges à éviter

- Stocker un `ref` d'un tick sur l'autre.
- Lire `properties["clé"]` sans valeur par défaut, ou supposer `kind != UNKNOWN`.
- Coder en dur 8 slots, 15 kg, 30 × 30 m, les valeurs de soin ou d'hydratation des objets du kit.
- Utiliser des coordonnées absolues, la seed, `debug_state()` ou l'état interne du moteur dans la politique.
- Évaluer sur les seeds d'entraînement.
- Traiter la troncature comme une mort (valeur future mise à 0).
- Optimiser la récompense cumulée plutôt que la durée de survie : c'est toujours la survie qui tranche entre deux versions.
- Sur-apprendre contre `HunterBot` : un agent qui exploite une faiblesse précise d'un bot scripté ne survivra pas aux agents des autres groupes.

---

## 12. Résultats de la phase 1

Implémentation :

- `agent/perception.py` : mémoire courte (dégâts récents, menace vue/entendue en angle absolu, cooldowns), sans aucun `ref` conservé ;
- `agent/navigation.py` : évitement d'obstacles par rayons (chaque détection bloquante = segment perpendiculaire), dégagement vers le plus grand secteur libre après collisions répétées ;
- `agent/options.py` : les 8 options de la section 4 (`EXPLORE`, `FLEE`, `HIDE`, `REST`, `SEEK_ITEM`, `CONSUME`, `ARM`, `FIGHT`), interface `available / start / done / step` ;
- `agent/scripted.py` : `OptionRunner` (réutilisable par le Q-learning) + politique scriptée par priorités ;
- `training/envs.py`, `training/evaluate.py` : scénarios `default`, `hostile`, `scarce`, `blind`, `big_map` (`max_ticks=3000`), évaluation parallèle sur seeds disjointes.

Évaluation, 30 seeds d'évaluation par scénario (`python -m training.evaluate --agents trivial student --scenarios default hostile scarce blind big_map`) :

| Scénario | Agent | Moyenne | Médiane | 10ᵉ centile | Survivants |
|---|---|---|---|---|---|
| default | trivial | 1750 | 1750 | 1750 | 0 % |
| default | scripté | **2587** | **2663** | **1783** | 17 % |
| hostile | trivial | 1750 | 1750 | 1750 | 0 % |
| hostile | scripté | **2390** | **2392** | 1750 | 17 % |
| scarce | trivial | 1278 | 1278 | 1278 | 0 % |
| scarce | scripté | **1490** | **1311** | 1266 | 0 % |
| blind | trivial | 1750 | 1750 | 1750 | 0 % |
| blind | scripté | **2294** | **2384** | 1750 | 3 % |
| big_map | trivial | 1750 | 1750 | 1750 | 0 % |
| big_map | scripté | **2353** | **2359** | 1750 | 23 % |

Constats pour la phase 2 :

- Toutes les morts restent d'origine physiologique : 4 `HunterBot` ne suffisent pas à rendre `hostile` dangereux, il faudra des adversaires plus agressifs (bots armés d'office, self-play).
- Sur `scarce`, le 10ᵉ centile est légèrement sous l'agent trivial (petits dégâts de collision, ressources non trouvées) : l'exploration sans mémoire spatiale trouve mal les rares ressources. Piste : odométrie à partir de `orientation` et `speed` pour mémoriser portes et ressources vues.
- Piège connu : une fente entre deux bâtiments à peine plus large que le corps, combinée à un agent qui pousse, peut immobiliser l'agent jusqu'à la mort.
