# Environnement d'entraînement

L'environnement fourni est une **petite ville 2D torique vue du dessus**.

Il est volontairement plus simple que l'arène finale, mais expose les mêmes familles de perceptions et d'actions.

## Monde

La ville contient notamment :

- commerces et bâtiments pénétrables ;
- portes manipulables ;
- parc, arbres et bassin ;
- zone médicale ;
- zone industrielle ;
- obstacles fixes et mobiles ;
- eau et vapeur ;
- nourriture, boisson et soins ;
- matières premières ;
- rouleau à pâtisserie ;
- lance-patate et pommes de terre ;
- modificateur d'arme.

La carte est torique : sortir par un bord ramène par le bord opposé.

La génération varie avec le `seed`.

## Capteurs

La vision possède un champ limité et peut être occultée ou dégradée.

L'audition peut être atténuée et mal localisée.

Le toucher ne fonctionne qu'au contact.

Dans l'interface Arcade, les touches `V`, `H` et `T` permettent de visualiser les perceptions de l'agent sélectionné.

## Physiologie

L'agent possède :

- santé ;
- énergie ;
- hydratation ;
- satiété ;
- blessures ;
- masse transportée.

La vitesse et la charge augmentent le coût énergétique. Le repos permet de récupérer.

Les ressources alimentaires et hydriques contribuent à maintenir la récupération. Les blessures peuvent ralentir ou pénaliser l'agent.

## Inventaire

Dans l'environnement fourni, les valeurs initiales sont :

```text
8 slots
15 kg de charge maximale
```

Ces valeurs font partie de l'environnement de test, pas d'une hypothèse à coder en dur pour généraliser.

## Armes ludiques

Le rouleau à pâtisserie s'utilise au contact via `Use(slot, target_ref)`.

Le lance-patate doit être équipé puis utilisé via `Shoot(bearing)`. Il consomme une munition de type pomme de terre.

## Deux bots de référence

`RandomBot` explore, collecte et agit de façon simple.

`HunterBot` explore puis poursuit les agents qu'il perçoit et utilise les armes qu'il possède.

Ils utilisent uniquement le même contrat d'observation que vous.

Vous êtes libres d'ajouter vos propres bots et de modifier l'environnement de test pour enrichir votre entraînement.

## Rewards d'entraînement

Le tournoi final est classé sur la **durée de survie**.

L'environnement d'entraînement fournit cependant une récompense plus dense pour faciliter le RL :

```text
+ petite récompense par tick survécu
+ élimination d'un adversaire
+ première acquisition d'un objet
- dégâts reçus
- collision nuisible
- mort
```

Valeurs par défaut actuelles :

```text
survie / tick      +0.1
kill               +5.0
premier pickup     +1.0
dégâts reçus       -1.0 × dégâts
collision nuisible -0.5
mort               -20.0
```

Ces valeurs appartiennent à l'environnement d'entraînement et pourront évoluer.

Un objet ne rapporte le bonus de pickup qu'une fois par agent : `drop -> pickup` ne permet pas de farmer la récompense.

## Episode et troncature

Par défaut :

```text
max_ticks = 600
```

soit 5 minutes simulées.

`terminated[agent]` signifie que l'épisode de l'agent est terminé naturellement, typiquement par sa mort.

`truncated=True` signifie que la limite externe de l'épisode a été atteinte.

Vous pouvez modifier `max_ticks` dans votre copie de l'environnement.

## Seed et déterminisme

À configuration, seed et actions identiques, la simulation vise un résultat identique.

Utilisez plusieurs seeds pendant l'entraînement. Une politique qui mémorise une carte ou un seed n'est pas l'objectif du projet.
