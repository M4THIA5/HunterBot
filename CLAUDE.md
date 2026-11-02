# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Student kit for a multi-agent reinforcement-learning course project ("RL Arena", French-language docs/code comments). Goal: build an agent in `student_agent.py` that **survives as long as possible** in an unseen final arena sharing the same interface contract. Tournament ranking = survival time only; training rewards are just a denser shaping signal.

## Commands

Python ≥ 3.10. No runtime dependencies for the headless core; Arcade 3.3.3 only for GUI.

**Always use the project virtualenv `.venv/`** (Python 3.14) — never the system Python or a global pip. Either `source .venv/bin/activate` first, or call `.venv/bin/python` directly. All commands below assume the venv is active (`python` = `.venv/bin/python`).

```bash
python -m pip install -e .               # headless
python -m pip install -e ".[graphics]"   # + Arcade GUI

python check_student_agent.py            # quick contract check of student_agent.MyAgent
python run_student_agent.py [--seed N] [--gui]   # one episode vs the two reference bots
python train_agent_skeleton.py           # external RL training loop skeleton
python demo_headless.py / demo_arcade.py

python -m unittest discover -s tests -v                                   # all tests
python -m unittest tests.test_core.CoreEnvironmentTests.test_reset_is_deterministic -v   # single test
```

GUI keys: Space pause, N single tick, Tab switch followed agent, V/H/T vision/hearing/touch overlays, D engine debug (perception vs reality), +/- speed.

## Hard constraint: public contract only

`rl_arena/arena_api.py` is the **only** module guaranteed in the final arena (frozen dataclasses: `Observation`, `Action`, `Movement`, `Rotation`, interactions `Pickup/Drop/Equip/Use/Throw/Shoot/Ingest/Manipulate`, `ObjectKind`, etc.). Agent decision code must import only from `rl_arena.arena_api` and use only data in `Observation`. Never make the policy depend on `environment.py`, `world.py`, `generation.py`, `debug_state()`, internal positions, seeds or map layout — they won't exist in the final arena. The training env itself may be freely modified (maps, bots, rewards, `max_ticks`) for experiments.

Contract semantics that are easy to get wrong (full spec: `docs/CONTRAT_INTERFACE.md`):
- 1 tick = 0.5 s simulated. Units: m, kg, s, rad.
- Angles positive **clockwise**; `bearing` relative to agent heading (0 = ahead, +π/2 = right). `Movement(forward, right, speed)` is in the agent's local frame.
- `ref` in detections is valid **only for the current tick** — never store it as an object identity; cross-tick tracking is the agent's job.
- `properties` dicts may be empty/partial; `kind` may be `ObjectKind.UNKNOWN`. Observation is partial and noisy (fog/steam/light degrade vision; sound bearing/distance can be wrong). Hidden variables like fog are never exposed.
- Impossible actions are silently no-ops, not errors.
- Training-env values (8 inventory slots, 15 kg carry, map size, reward weights) must not be hard-coded as assumptions.

## Architecture

- `rl_arena/environment.py` — `TrainingEnvironment`, the deterministic simulation core (no Arcade dependency). Toric 2D city (30×30 m default). Lifecycle: `register_agent(...)` (before `reset`) → `reset(seed)` returns `{agent_id: Observation}` → `step(actions)` returns `StepResult(observations, rewards, terminated, truncated, info)`; `info[aid]` is `AgentStepInfo` (survival_ticks, kills, pickups, damage...).
- **Two control modes** in `step`: an agent registered with a controller object (`register_agent("student", MyAgent())`) has its `act()` called internally on its last observation; an agent registered with `agent=None` is externally controlled — its action must be passed in the `actions` dict (used by `train_agent_skeleton.py`). An entry in `actions` overrides the controller; missing external agents get a no-op `Action()`. Rewards are only available from `StepResult.rewards`, not in `Observation`.
- Tick order inside `step`: rotations applied → interactions resolved (using refs from the previous observation) → velocities → `_simulate_physics` (substeps) → `_update_physiology` → sounds → truncation check → new observations via `_build_observation` (vision/hearing/touch with occlusion, steam, absorption).
- `rl_arena/config.py` — all tunables as dataclasses: `EnvironmentConfig` (max_ticks=600, map, substeps, max_agents=8) nesting `RewardConfig`, `GenerationConfig`, `AgentPhysicsConfig` (speed, FOV 120°, vision 12 m, hearing 20 m, pickup reach), `PhysiologyConfig`.
- `rl_arena/generation.py` — seed-based procedural city (`generate_city`); `world.py` — internal entity/state dataclasses; `geometry.py` — toric distance/vector helpers.
- `rl_arena/bots.py` — `RandomBot` and `HunterBot` (added via `add_default_bots()` as `bot_random`/`bot_hunter`); they use only the public contract and are good reference agent implementations.
- `rl_arena/render_arcade.py` — optional Arcade viewer (`run_arcade(env)`), only imported when `--gui`.

Determinism: same config + seed + actions ⇒ same trajectory (tested in `tests/test_core.py`). Train across many seeds; the final arena has a different map, item distributions and complexity, so policies must generalize from perceptions.

`SyllabusDuProjet.pdf` holds the course project requirements.
