# FAQ étudiant

## Puis-je modifier l'environnement fourni ?

Oui. Vous pouvez modifier votre copie, ajouter des cartes, des objets ou des bots pour votre entraînement.

En revanche, votre agent destiné au tournoi doit prendre ses décisions uniquement depuis le contrat public et ses observations.

## Puis-je utiliser les positions internes de `environment.py` pendant mon apprentissage ?

Vous pouvez les utiliser pour vos propres outils de debug, mais votre politique finale ne doit pas en dépendre. Ces informations n'existeront pas dans l'arène finale.

## Pourquoi `ref` change-t-il ?

Parce qu'un `ref` sert uniquement à cibler une perception pendant le tick courant. Il ne constitue pas l'identité permanente d'un objet.

## Comment suivre un objet entre plusieurs ticks ?

C'est un problème de perception/mémoire à résoudre côté agent à partir de ses propriétés, angle, distance, mouvement estimé, etc.

## Pourquoi la visibilité n'est-elle pas donnée directement ?

Parce que l'agent doit percevoir les conséquences de l'environnement, pas recevoir ses variables cachées. Le brouillard se manifeste par une vision dégradée.

## Pourquoi un son peut-il indiquer une mauvaise direction ?

La direction reçue est une estimation perceptive. Les obstacles et matériaux peuvent augmenter son incertitude.

## Une action impossible provoque-t-elle une exception ?

Non. Elle ne produit simplement pas l'effet demandé.

## `reward` est-il présent dans `Observation` ?

Non. En apprentissage externe, il est renvoyé par `env.step()` dans `StepResult.rewards`.

## Dois-je utiliser une Q-table ?

Pas nécessairement. Le contrat ne vous impose pas la représentation ou l'algorithme. Le cours vous donne notamment les outils pour construire une solution Q-learning, mais la manière de représenter l'état fait partie du travail.

## Pourquoi le jeu peut-il sembler moins fluide qu'un jeu d'action classique ?

Les agents prennent une décision toutes les 0,5 seconde simulée. Cette fréquence fait partie du contrat. Le rendu graphique est un outil de visualisation ; l'entraînement headless exécute exactement la même logique sans attendre le temps réel.

## Comment vérifier rapidement mon agent ?

```bash
python check_student_agent.py
```

Puis :

```bash
python run_student_agent.py --gui
```

## Où est la définition exacte des types ?

Dans :

```text
rl_arena/arena_api.py
```

C'est la référence technique du contrat.
