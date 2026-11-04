# Démarche d'apprentissage de l'agent

Ce document retrace comment l'agent de `student_agent.py` a été construit, entraîné et évalué, phase par phase, avec les résultats obtenus et les décisions qui en ont découlé.

Objectif du projet : survivre le plus longtemps possible dans une arène finale inconnue. Le classement ne dépend que du temps de survie. Les récompenses d'entraînement ne servent qu'à guider l'apprentissage.

---

## 1. Contraintes de départ

- **Contrat public uniquement.** Le code de décision n'importe que `rl_arena.arena_api` et la bibliothèque standard. Il ne lit que l'`Observation`. Il n'utilise ni positions internes, ni seed, ni plan de la carte, ni `ref` conservée d'un tick à l'autre.
- **Observation partielle et bruitée.** Le champ de vision fait 120°, et le brouillard, la vapeur et l'obscurité dégradent la vue. Le son donne une direction et une distance approximatives. Les `properties` des objets peuvent être vides.
- **Arène finale différente.** La carte, la répartition des objets et les adversaires changent. Aucune constante de l'environnement d'entraînement n'est codée en dur : 8 emplacements, 15 kg, taille de carte, etc.
- **Physiologie.** Sans boire, l'hydratation descend de 1,0 à 0,15 en environ 1 417 ticks, puis l'agent perd de la santé. Il meurt vers le tick 1 750. Les objets sont en nombre fini et ne réapparaissent pas.

Ces contraintes ont orienté tout le travail vers une politique **robuste et généralisable**, plutôt que vers une politique qui exploite une carte précise.

---

## 2. Architecture retenue : des options plutôt que des actions brutes

Apprendre directement sur les actions élémentaires (mouvement, rotation, interaction à chaque tick de 0,5 s) poserait deux problèmes : l'espace d'actions est continu et combinatoire, et une décision a souvent un effet des centaines de ticks plus tard. L'agent raisonne donc avec des **options**, c'est-à-dire des comportements de haut niveau qui durent plusieurs ticks :

| Option | Rôle |
|---|---|
| `EXPLORE` | errer, ouvrir des portes, visiter de nouveaux bâtiments |
| `FLEE` | s'éloigner d'une menace |
| `HIDE` | se mettre à couvert |
| `REST` | récupérer de l'énergie |
| `SEEK_ITEM` | aller chercher un objet utile vu ou mémorisé |
| `CONSUME` | boire, manger ou se soigner |
| `ARM` | équiper une arme, avec munition compatible si elle tire à distance |
| `FIGHT` | attaquer au corps à corps ou à distance |

Chaque option est codée à la main dans `agent/options.py`. Elle s'appuie sur :

- `agent/perception.py` : une mémoire courte qui suit les menaces, les dégâts reçus et les coups venus de l'extérieur du champ de vision ;
- `agent/navigation.py` : l'évitement d'obstacles par rayons ;
- `agent/spatial.py` : une mémoire spatiale par odométrie.

Le **choix de l'option** est la partie apprise ou réglée.

---

## 3. Phase 1 : politique scriptée et banc d'évaluation

**Ce qui a été fait :**

- Les huit options et une politique scriptée par priorités (`agent/scripted.py`). Par exemple : santé basse avec menace proche → fuir ; besoin vital avec objet en inventaire → consommer.
- Des scénarios d'entraînement (`training/envs.py`) :
  - `default` : la carte du kit ;
  - `hostile` : des chasseurs armés ;
  - `scarce` : des ressources rares ;
  - `blind` : une perception dégradée ;
  - `big_map` : une grande carte.
- Un évaluateur parallèle (`training/evaluate.py`) sur des seeds fixes (`EVAL_SEEDS`). Il mesure la moyenne, la médiane et le 10ᵉ centile (p10) de survie, ainsi que les causes de mort.

**Métrique principale :** le **score combiné**, moyenne des p10 sur les scénarios. Le p10 récompense la régularité : éviter les morts précoces compte plus que gagner quelques ticks quand tout va bien. Dans une arène inconnue, c'est la propriété qui compte le plus.

**Résultat :** la politique scriptée bat l'agent trivial en survie moyenne sur tous les scénarios (par exemple 2 493 contre 1 750 sur `default`). Le score combiné reste proche : 1 310 contre 1 311. Sur `scarce` et `big_map`, l'agent trivial atteint un p10 élevé simplement en ne mourant jamais au combat.

---

## 4. Phase 2 : représentation de l'état et environnements durcis

Pour appliquer un apprentissage tabulaire, l'observation est réduite à un **état discret de 12 caractéristiques**, porté à 14 au run 5 (`agent/features.py`). Toutes sont relatives et calculées depuis l'observation :

- santé, énergie et besoin vital le plus bas, chacun découpé en 3 niveaux ;
- possession d'un objet couvrant le besoin, possession d'un médicament ;
- menace vue (aucune, loin, moyenne ou proche), son côté, si elle est armée et si elle se rapproche ;
- menace entendue, dégâts récents, agent armé, objet utile visible, obstacle devant.

Un **masque d'options** interdit les options impossibles : `CONSUME` sans rien à consommer, `FIGHT` sans cible, etc.

Les environnements ont été durcis :

- `AggressiveHunter`, un chasseur armé ;
- des adversaires figés issus de l'auto-apprentissage (self-play) ;
- des dotations de départ, par exemple des adversaires qui commencent armés.

Un **curriculum** à 4 étapes, avec randomisation de domaine, a été mis en place :

| Étape | Nom | Compétence visée |
|---|---|---|
| 0 | physiology | gérer énergie, boire, manger, se reposer |
| 1 | first_threats | fuir, se cacher, s'armer |
| 2 | degraded | décider sous perception dégradée, face à des adversaires armés |
| 3 | self_play | résister à des copies de l'agent lui-même |

Les épisodes d'une étape rejouent les étapes précédentes avec une probabilité de 20 %, pour limiter l'oubli.

---

## 5. Phase 3 : Q-learning SMDP sur les options

**Algorithme** (`training/train.py`, `agent/qpolicy.py`) : Q-learning tabulaire semi-markovien. Une option dure k ticks et rapporte une récompense cumulée R :

```
Q(s,o) ← Q(s,o) + α [ R + γ^k · max_{o' valide} Q(s',o') − Q(s,o) ]
```

- La valeur future vaut zéro à la mort. En cas de troncature (fin de l'épisode sans mort), on bootstrape sur Q(s',·).
- α décroît avec le nombre de visites : `max(0.02, (1+n)^-0.6)`.
- ε décroît de 1,0 à 0,05 sur les premiers 60 % de l'entraînement. 30 % des choix aléatoires suivent la politique scriptée, pour explorer des trajectoires utiles.
- Une option en cours est interrompue si la menace augmente, si l'agent subit des dégâts ou si un objet utile apparaît.
- Entraînement parallèle : des workers jouent avec une copie figée de la table, et le processus principal applique les mises à jour. Comme le Q-learning est off-policy, il tolère ce léger décalage.
- Dans un état jamais vu, l'agent se replie sur la politique scriptée.

**Récompense d'entraînement :** survie, kill +0,5, premier ramassage +0,2, collision dangereuse −0,05. S'y ajoute un **shaping par potentiel** :

```
Φ = 2·santé + 4·min(hydratation, satiété) + [boisson en stock] + [nourriture en stock]
```

Le stock en inventaire rend un ramassage immédiatement payant. Sans lui, le bénéfice réel (ne pas mourir de soif) arrive des centaines de ticks plus tard, au-delà de l'horizon effectif de γ.

**Run 1, échec instructif :** l'agent a appris à rester immobile. La pénalité de collision dominait le gain d'une ressource, et γ était trop court. Les poids ont été recalibrés et γ porté à 0,998.

**Run 2 :** l'agent apprend, mais reste derrière la politique scriptée. Sur l'étape 0, il atteint une moyenne de 1 830, contre 2 064 pour la politique scriptée.

**Run 3, a priori scripté :** l'option que choisirait la politique scriptée reçoit un bonus constant sur sa valeur Q. L'agent ne s'en écarte que si Q montre un avantage net. Il égale ou dépasse la politique scriptée sur l'étape 0 (moyenne 2 090 contre 2 064, p10 1 446 contre 1 348), mais pas sur les scénarios d'évaluation : score combiné 1 243 contre 1 310, avec une forte baisse sur `blind`.

---

## 6. Phase 4 : curriculum complet, self-play et mémoire spatiale

| Run | Changement | Résultat (score combiné ; politique scriptée ≈ 1 310–1 320) |
|---|---|---|
| 4 | étapes 0 à 2, 7 000 épisodes, a priori scripté | bat la politique scriptée sur les étapes 1 et 2 (étape 2 : 1 195 contre 907), mais 1 248 sur l'évaluation |
| 5 | +2 caractéristiques (`threat_armed`, `closing`), étape 3 en self-play, 9 000 épisodes | 1 247 ; table de 15 000 états, très clairsemée ; meilleure moyenne sur `hostile` (1 009 contre 747) |
| 6 | même plan que le run 5, sur moteur déterministe et avec mémoire des ressources | 1 233 contre 1 320 ; gros gain sur `hostile` (moyenne 998 contre 620), mais collecte de ressources plus faible ailleurs |

Paramètres du run 6 : étapes 0 à 3 avec 1 500, 2 500, 3 000 et 2 000 épisodes ; rounds de 32 épisodes ; évaluation tous les 500 épisodes sur 64 seeds ; bonus d'a priori 1,5 ; 3 copies en self-play.

**Mémoire spatiale par odométrie** : l'agent estime sa position en intégrant la vitesse perçue, son orientation et le mouvement commandé. Il mémorise ensuite les ressources et les portes. Les collisions font dériver l'estimation de 2 à 6 m sur 300 ticks. Cibler les portes et les zones non visitées a donc nui aux résultats (p10 sur `blind` : 532 contre 1 714). Seule la mémoire des ressources a été conservée : score combiné 1 320 contre 1 290 sans mémoire.

**Correctif de déterminisme du moteur** : un ensemble de chaînes était parcouru dans un ordre qui dépendait de `PYTHONHASHSEED`. Une même seed pouvait donc produire des trajectoires différentes d'un processus à l'autre. Le parcours se fait désormais dans un ordre trié, et un test de non-régression a été ajouté. Sans ce correctif, les comparaisons A/B n'étaient pas fiables.

**Agent hybride** (`agent/hybrid.py`) : la politique scriptée quand tout est calme, la Q-table dès qu'une menace est vue ou entendue. La meilleure variante, déclenchée par une menace entendue et maintenue 10 ticks, atteint 1 283 contre 1 320. Elle reste une expérience.

**Bilan de l'apprentissage par renforcement :** la Q-table apprend des réflexes de combat utiles, ce qui se voit nettement sur `hostile`. Elle généralise moins bien que les règles pour la gestion des ressources. Avec 14 caractéristiques discrètes, l'état est trop grossier pour distinguer les situations où explorer paie : l'apprentissage améliore localement, mais dégrade la régularité (p10). **Le score combiné a donc été jugé déterminant : la politique livrée est la politique scriptée, et les enseignements de l'apprentissage y ont été réintégrés à la main.**

---

## 7. Phase 5 : amélioration guidée par diagnostic

À partir de là, chaque modification suit le même cycle :

1. **Diagnostiquer** les morts avec des outils hors ligne. Ces outils lisent l'état interne du moteur, ce qui n'est permis que pour l'analyse, jamais dans la politique :
   - `training/hostile_diag.py` : les morts au combat ;
   - `training/resource_diag.py` : la collecte des ressources.
2. **Formuler une hypothèse** et l'implémenter comme une **option activable** de `Memory`, sans écraser le comportement précédent.
3. **Comparer par A/B** les variantes définies dans `training/baselines.py`, sur 50 à 200 seeds. Avec 50 seeds, le bruit atteint ±50 ticks sur le p10 ; il faut 200 seeds pour trancher.
4. **Garder par défaut** la variante seulement si le score combiné progresse. Sinon, l'option reste disponible mais désactivée.

### 7.1 Coups invisibles et engagement de fuite

**Diagnostic sur `hostile` :** les chasseurs au corps à corps frappent dans le dos, hors du champ de vision. L'agent ne comptait comme attaque que des dégâts venant d'un agent vu ou touché, donc il continuait d'explorer. De plus, il alternait `FLEE` et `EXPLORE` à chaque tick dès que la menace sortait de son champ.

**Correctifs :**

- un « coup invisible » (au moins 0,03 de dégâts en un tick) fait supposer un attaquant derrière l'agent ;
- un **engagement de fuite** empêche une fuite commencée d'être interrompue par un choix calme, et l'exploration qui suit garde le cap de fuite.

| Variante, 200 seeds | Moyenne `hostile` | Score combiné |
|---|---|---|
| sans les deux correctifs | 744 | 1 138 |
| coup invisible seul | 728 | 1 191 |
| engagement « always » | 796 | 1 242 |
| engagement « serious » | **828** | 1 161 |

La variante « serious » (n'engager la fuite que face à une menace sérieuse) a été retenue par défaut, pour sa meilleure moyenne de survie.

### 7.2 Arène à 8 agents

`training/arena_ffa.py` réunit les 8 versions de l'agent dans une même partie. Sur 100 parties, aucun gagnant net ne se dégage : les 6 premiers sont à moins de 41 ticks les uns des autres. La plupart des agents meurent de soif au tick 1 750, car 8 agents se partagent les ressources.

### 7.3 Recherche de ressources

**Diagnostic :** l'agent récupère déjà environ 72 % des boissons. Ses échecs viennent de trois causes :

- il n'entre jamais dans le bâtiment qui contient la boisson ;
- des pièges rares le bloquent : une porte hors de portée d'un millimètre, ou l'agent coincé entre des murs par un chasseur qui le pousse ;
- il oscille entre `FLEE` et `EXPLORE`.

**Correctifs :**

- s'approcher d'une porte pendant qu'il la manipule, puis l'abandonner temporairement si elle ne s'ouvre pas ;
- détecter qu'il est coincé et balayer des directions d'échappement.

Sur 200 seeds, toutes les variantes restent dans le bruit (score combiné de 1 585 à 1 588). Les correctifs sont gardés parce qu'ils suppriment des blocages réels, sans rien coûter.

### 7.4 Cartes procédurales variées (généralisation)

L'arène finale étant inconnue, `training/worldgen.py` génère des cartes selon des **profils** très différents. Le moteur d'entraînement a été étendu pour les accueillir :

- **Profils :**
  - `town` : ville ;
  - `desert` : très peu de ressources ;
  - `abundance` : ressources abondantes ;
  - `armory` : beaucoup d'armes ;
  - `maze` : labyrinthe ;
  - `toxic` : zones dangereuses ;
  - `frozen` : glace ;
  - `dark` : obscurité ;
  - `tiny`, `huge` : très petite ou très grande carte ;
  - `famine` : nourriture et boissons rares, beaucoup de matériaux inutiles.
- **Objets :** uniquement ceux du kit. Seule leur **quantité** varie d'un profil à l'autre, comme dans l'arène finale.
- **Conditions :** zones `HAZARD` qui infligent des dégâts à chaque tick, zones `ICE` qui ralentissent et coûtent de l'énergie, brouillard et faible luminosité aléatoires, taille de carte, rythme de la soif et de la faim.

**Historique :** une première version ajoutait de nouvelles armes (couteau, batte, arbalète, fronde…) et de nouveaux consommables, avec un profil `new_only` réservé à ces objets. Elle a été retirée, car l'arène finale ne contient que les objets définis au début du projet. Les résultats `maps_eval`, `combat_ab` et `counter_ab` ont été obtenus sur cette première version.

**Ce que cette version a révélé :** l'agent supposait implicitement les noms des armes du kit, et ne vérifiait pas que la munition correspondait à l'arme. La reconnaissance repose maintenant sur les propriétés de l'objet (`attack`, `ammo_type`), avec des mots-clés en repli. Une arme à distance n'est considérée comme utilisable qu'avec une munition compatible. C'est sans effet sur les objets du kit, mais plus robuste si leurs `properties` sont partielles.

### 7.5 Combats : fuite prolongée, limitation, riposte

**Diagnostic des morts précoces :** un chasseur armé rattrape l'agent, car sa vitesse (≤ 3,0) dépasse la vitesse de croisière de l'agent (2,6), et l'agent reprend l'exploration trop tôt.

| Variante, 5 cartes, 50 seeds | Score combiné |
|---|---|
| avant correctifs | 232 |
| limitation des combats seule | 215 |
| **fuite prolongée seule** (15 ticks après le dernier coup) | **266** |
| les deux | 262 |

La **fuite prolongée** est gardée et la limitation des combats est désactivée. Sur `hostile`, la moyenne atteint environ 930, contre 744 au départ. Ce chiffre vient de 50 seeds et reste à confirmer sur 200.

**Riposte** (contre-attaquer quand l'agent est armé et frappé dans le dos) : effet neutre à légèrement négatif, 455 sans riposte contre 440 avec, sur 7 scénarios. Elle se déclenche rarement, et un échange de coups à armes égales n'est pas favorable. Elle est désactivée.

---

## 8. État actuel de l'agent livré

`student_agent.MyAgent` utilise `ScriptedAgent` avec les réglages par défaut suivants :

- coups invisibles pris en compte ;
- engagement de fuite « serious » et fuite prolongée de 15 ticks ;
- mémoire spatiale des ressources par odométrie ;
- approche puis abandon des portes récalcitrantes, échappement quand l'agent est coincé ;
- reconnaissance générique des armes et des munitions ;
- toute exception levée dans `act()` est rattrapée et remplacée par `Action(rest=True)`, pour qu'une erreur ne coûte jamais un épisode.

Options présentes mais désactivées, faute de gain mesuré : limitation des combats, riposte, préférence pour les portes, errance plus rectiligne, fuite face aux menaces inoffensives.

---

## 9. Enseignements

1. **Mesurer la régularité, pas seulement la moyenne.** Le p10 a souvent contredit la moyenne : une variante peut survivre plus longtemps en moyenne tout en mourant plus souvent très tôt.
2. **Le déterminisme et un nombre suffisant de seeds sont des prérequis.** Sans eux, plusieurs « gains » observés sur 50 seeds n'étaient que du bruit.
3. **Le Q-learning tabulaire sur options apprend des réflexes, pas une stratégie.** Il bat les règles là où la réaction immédiate compte (combat), mais un état aussi grossier ne lui permet pas de planifier la collecte de ressources. L'a priori scripté évite d'oublier ce que l'on sait déjà, sans suffire à dépasser les règles.
4. **Les diagnostics hors ligne trouvent les vrais problèmes.** Les coups dans le dos et la fuite trop courte ont tous été trouvés en rejouant les morts, pas en réglant des hyperparamètres.
5. **Varier les cartes est indispensable pour généraliser.** Les cartes procédurales ont révélé des hypothèses implicites (noms d'armes, munitions) et des morts précoces que la carte du kit ne montrait pas.

---

## 10. Reproduire

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v                       # 24 tests
python -m training.train --stages 0 1 2 3 --episodes 1500 2500 3000 2000 \
       --prior-bonus 1.5 --out results/phase4_runX              # entraînement Q-learning
python -m training.evaluate --agents scripted scripted_nocounter \
       --scenarios hostile default --episodes 200 --markdown out.md  # A/B (variantes : training/baselines.py)
python -m training.hostile_diag --agent scripted --scenario hostile
python -m training.worldgen --gui town --seed 3                 # visualiser une carte procédurale
python -m training.arena_ffa --games 100 [--gui]                # arène à 8 agents
```

Les résultats détaillés sont dans `results/` : `baselines.md`, `phase*/eval_scenarios.md`, `hostile_ab.md`, `resources_ab.md`, `maps_eval.md`, `combat_ab.md` et `counter_ab.md`.
