Seeds d'évaluation : 50 premières de `EVAL_SEEDS`, `max_ticks=3000`.

| Scénario | Agent | Moyenne | Médiane | 10ᵉ centile | Survivants | Morts combat | Morts physio | Kills | Pickups |
|---|---|---|---|---|---|---|---|---|---|
| default | qlearned | 2405 | 2392 | 1750 | 18% | 0% | 82% | 0.24 | 7.8 |
| default | scripted | 2568 | 2444 | 1783 | 24% | 0% | 76% | 0.32 | 9.1 |
| hostile | qlearned | 998 | 428 | 40 | 6% | 52% | 42% | 0.22 | 4.1 |
| hostile | scripted | 620 | 326 | 57 | 4% | 76% | 20% | 0.16 | 3.3 |
| scarce | qlearned | 1458 | 1311 | 1276 | 0% | 2% | 98% | 0.14 | 4.0 |
| scarce | scripted | 1535 | 1666 | 1278 | 0% | 0% | 100% | 0.22 | 4.5 |
| blind | qlearned | 2172 | 2333 | 1347 | 12% | 2% | 86% | 0.26 | 7.2 |
| blind | scripted | 2324 | 2384 | 1734 | 12% | 2% | 86% | 0.48 | 7.3 |
| big_map | qlearned | 2312 | 2352 | 1750 | 10% | 0% | 90% | 0.10 | 6.6 |
| big_map | scripted | 2373 | 2400 | 1750 | 20% | 0% | 80% | 0.22 | 6.6 |

| Agent | Score combiné (moyenne des p10) |
|---|---|
| qlearned | 1233 |
| scripted | 1320 |
