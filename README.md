# RL Arena — Kit étudiant V1

Ce kit contient l'environnement d'entraînement du projet d'apprentissage par renforcement multi-agent.

Votre objectif final est simple : **survivre le plus longtemps possible** dans une arène inconnue respectant le même contrat d'interface que cet environnement d'entraînement.

## À retenir avant tout

Votre agent reçoit une `Observation` toutes les **0,5 seconde simulée** et retourne une `Action`.

Le moteur ne vous fournit pas un état discret. La représentation de l'état, la discrétisation, la mémoire, la Q-table ou le modèle sont **votre responsabilité**.

Le seul contrat garanti dans l'arène finale est :

```python
from rl_arena.arena_api import ...
```

Vous pouvez lire et modifier librement l'environnement d'entraînement fourni, mais **votre agent final ne doit pas dépendre des modules internes** tels que `environment.py`, `world.py` ou `generation.py` pour prendre ses décisions.

## Installation

Python **3.10 ou supérieur** est requis.

### Environnement headless seulement

Depuis le dossier du projet :

```bash
python -m pip install -e .
```

### Avec l'interface graphique Arcade

```bash
python -m pip install -e ".[graphics]"
```

Le kit cible Arcade **3.3.3**.

## Premier test en 30 secondes

Vérifiez d'abord l'installation :

```bash
python demo_headless.py
```

Puis lancez la ville :

```bash
python demo_arcade.py
```

Enfin, testez votre propre agent :

```bash
python check_student_agent.py
python run_student_agent.py
python run_student_agent.py --gui
```

Le fichier à modifier en premier est :

```text
student_agent.py
```

## Pour commencer un apprentissage

Un squelette de boucle RL externe est fourni :

```bash
python train_agent_skeleton.py
```

Le fichier montre où placer :

- votre représentation/discrétisation de l'état ;
- votre politique de choix d'action ;
- votre mise à jour Q-learning ou votre modèle.

Il ne fournit volontairement aucune discrétisation « correcte » : ce choix fait partie du projet.

## Affichage Arcade

Commandes principales :

| Touche | Fonction |
|---|---|
| `Espace` | pause / reprise |
| `N` | avancer d'un tick en pause |
| `Tab` | changer l'agent suivi |
| `V` | afficher la perception visuelle |
| `H` | afficher la perception auditive |
| `T` | afficher les contacts tactiles |
| `D` | debug moteur / comparaison perception-réalité |
| `+` / `-` | vitesse d'exécution |
| clic sur un agent | sélectionner l'agent |

Les overlays de perception sont particulièrement utiles pour comprendre ce que reçoit réellement votre politique.

## Organisation du kit

```text
student_agent.py             votre agent
run_student_agent.py         test contre les bots fournis
check_student_agent.py       vérification rapide du contrat
train_agent_skeleton.py      squelette d'entraînement RL

demo_arcade.py               démonstration graphique
demo_headless.py             démonstration rapide sans affichage

docs/CONTRAT_INTERFACE.md    référence du contrat public
docs/ENVIRONNEMENT.md        monde d'entraînement et rewards
docs/FAQ.md                  erreurs et questions fréquentes

rl_arena/arena_api.py        CONTRAT PUBLIC
rl_arena/...                 implémentation de l'environnement de test
```

## Tests du moteur

```bash
python -m unittest discover -s tests -v
```

## Important : environnement d'entraînement ≠ arène finale

L'arène finale utilisera le **même contrat d'interface**, les mêmes unités et un tick de 0,5 s, mais sa carte, ses distributions d'objets, ses situations et sa complexité pourront être différentes.

Évitez donc d'apprendre une suite de coordonnées, un seed ou une carte particulière. Cherchez plutôt une politique capable de généraliser à partir de ses perceptions.
