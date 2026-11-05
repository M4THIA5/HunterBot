Seeds d'évaluation : 50 premières de `EVAL_SEEDS`, `max_ticks=3000`.

| Scénario | Agent | Moyenne | Médiane | 10ᵉ centile | Survivants | Morts combat | Morts physio | Kills | Pickups |
|---|---|---|---|---|---|---|---|---|---|
| default | student | 1954 | 1750 | 1750 | 4% | 0% | 96% | 0.00 | 2.5 |
| default | scripted | 2472 | 2417 | 1750 | 16% | 0% | 84% | 0.22 | 8.4 |
| default | qlearned | 2466 | 2450 | 1750 | 24% | 2% | 74% | 0.20 | 8.5 |
| hostile | student | 1206 | 1598 | 57 | 2% | 40% | 58% | 0.00 | 2.4 |
| hostile | scripted | 939 | 474 | 107 | 4% | 54% | 42% | 0.12 | 3.7 |
| hostile | qlearned | 870 | 369 | 72 | 4% | 66% | 30% | 0.16 | 3.3 |
| scarce | student | 1349 | 1278 | 1275 | 0% | 0% | 100% | 0.00 | 1.1 |
| scarce | scripted | 1510 | 1592 | 1266 | 0% | 0% | 100% | 0.20 | 4.4 |
| scarce | qlearned | 1483 | 1311 | 1276 | 0% | 0% | 100% | 0.22 | 4.3 |
| blind | student | 1937 | 1750 | 1744 | 2% | 2% | 96% | 0.00 | 2.4 |
| blind | scripted | 2108 | 2332 | 1310 | 12% | 2% | 86% | 0.38 | 6.3 |
| blind | qlearned | 2221 | 2352 | 1592 | 16% | 0% | 84% | 0.18 | 7.1 |
| big_map | student | 1960 | 1750 | 1750 | 2% | 2% | 96% | 0.00 | 1.6 |
| big_map | scripted | 2316 | 2384 | 1741 | 20% | 2% | 78% | 0.18 | 6.5 |
| big_map | qlearned | 2274 | 2366 | 1750 | 16% | 2% | 82% | 0.12 | 6.2 |

| Agent | Score combiné (moyenne des p10) |
|---|---|
| student | 1315 |
| scripted | 1235 |
| qlearned | 1288 |
