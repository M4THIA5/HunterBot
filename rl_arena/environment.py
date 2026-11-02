from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, ceil, cos, hypot, pi, sin
from random import Random
from typing import Any

from .arena_api import (
    Action,
    Agent,
    AgentId,
    Drop,
    EpisodeEndReason,
    EpisodeResult,
    Equip,
    Event,
    EventKind,
    Ingest,
    Injury,
    InventoryItem,
    Manipulate,
    Movement,
    ObjectKind,
    Observation,
    Pickup,
    Rotation,
    SelfState,
    Shoot,
    SoundDetection,
    SoundKind,
    TICK_SECONDS,
    Throw,
    TouchContact,
    Use,
    VisualDetection,
)
from .config import EnvironmentConfig
from .generation import GeneratedWorld, generate_city
from .geometry import (
    Rect,
    bearing_from_vector,
    circle_rect_overlap,
    clamp,
    heading_vector,
    nearest_point_on_rect,
    nearest_toric_rect,
    normalize,
    point_in_rect,
    right_vector,
    segment_intersects_rect,
    shifted_rects,
    torus_distance,
    torus_vector,
    wrap_angle,
    wrap_pos,
)
from .world import (
    AgentState,
    AreaEffect,
    DoorEntity,
    InjuryState,
    InventoryEntry,
    Projectile,
    RectEntity,
    SoundEventInternal,
    WorldObject,
)


@dataclass(frozen=True, slots=True)
class AgentStepInfo:
    survival_ticks: int
    cumulative_reward: float
    kills: int
    pickups: int
    damage_received: float
    damage_inflicted: float


@dataclass(frozen=True, slots=True)
class StepResult:
    observations: dict[AgentId, Observation]
    rewards: dict[AgentId, float]
    terminated: dict[AgentId, bool]
    truncated: bool
    info: dict[AgentId, AgentStepInfo]


@dataclass(slots=True)
class _Runtime:
    controller: Agent | None = None
    state: AgentState | None = None
    cumulative_reward: float = 0.0
    kills: int = 0
    pickups: int = 0
    damage_received: float = 0.0
    damage_inflicted: float = 0.0
    picked_uids: set[int] = field(default_factory=set)
    end_notified: bool = False
    survival_ticks: int = 0


class TrainingEnvironment:
    """Deterministic multi-agent training environment.

    The simulation core has no Arcade dependency. Rendering is provided by
    :mod:`rl_arena.render_arcade` and can be attached by the demo application.
    """

    def __init__(self, config: EnvironmentConfig | None = None, render: bool = False):
        self.config = config or EnvironmentConfig()
        self.render_enabled = render
        self.tick = 0
        self.seed: int | None = None
        self.rng = Random()
        self.world: GeneratedWorld | None = None
        self._runtimes: dict[AgentId, _Runtime] = {}
        self._projectiles: list[Projectile] = []
        self._sounds: list[SoundEventInternal] = []
        self._events: dict[AgentId, list[Event]] = {}
        self._ref_maps: dict[AgentId, dict[int, int]] = {}
        self._last_observations: dict[AgentId, Observation] = {}
        self._next_uid = 1
        self._initialized = False
        self._truncated = False

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------
    def register_agent(self, agent_id: AgentId, agent: Agent | None = None) -> None:
        if self._initialized:
            raise RuntimeError("Register agents before reset().")
        if agent_id in self._runtimes:
            raise ValueError(f"Agent id already registered: {agent_id}")
        if len(self._runtimes) >= self.config.max_agents:
            raise ValueError("max_agents exceeded")
        self._runtimes[agent_id] = _Runtime(controller=agent)

    def add_default_bots(self) -> None:
        from .bots import HunterBot, RandomBot

        if "bot_random" not in self._runtimes:
            self.register_agent("bot_random", RandomBot(seed=101))
        if "bot_hunter" not in self._runtimes:
            self.register_agent("bot_hunter", HunterBot(seed=202))

    @property
    def agent_ids(self) -> tuple[AgentId, ...]:
        return tuple(self._runtimes.keys())

    def reset(self, seed: int | None = None) -> dict[AgentId, Observation]:
        self.seed = seed
        self.rng = Random(seed)
        self.tick = 0
        self._truncated = False
        self.world = generate_city(self.config, self.rng)
        self._projectiles.clear()
        self._sounds.clear()
        self._events = {aid: [] for aid in self._runtimes}
        self._ref_maps = {aid: {} for aid in self._runtimes}
        self._last_observations.clear()

        used_uids = [e.uid for e in self.world.rects + self.world.doors + self.world.objects + self.world.areas]
        self._next_uid = (max(used_uids) + 1) if used_uids else 1

        spawn_points = self._spawn_points()
        if len(self._runtimes) > len(spawn_points):
            raise ValueError("Not enough spawn points for registered agents")

        shuffled = list(spawn_points)
        self.rng.shuffle(shuffled)
        for (aid, rt), (x, y) in zip(self._runtimes.items(), shuffled):
            rt.cumulative_reward = 0.0
            rt.kills = 0
            rt.pickups = 0
            rt.damage_received = 0.0
            rt.damage_inflicted = 0.0
            rt.picked_uids.clear()
            rt.end_notified = False
            rt.survival_ticks = 0
            rt.state = AgentState(
                uid=self._alloc_uid(),
                agent_id=aid,
                x=x,
                y=y,
                orientation=self.rng.uniform(-pi, pi),
                body_mass=self.config.agent.body_mass,
                radius=self.config.agent.radius,
            )
            if rt.controller is not None:
                rt.controller.reset()

        self._initialized = True
        self._last_observations = self._observe_all()
        return dict(self._last_observations)

    def step(self, actions: dict[AgentId, Action]) -> StepResult:
        if not self._initialized or self.world is None:
            raise RuntimeError("Call reset() before step().")
        if self._truncated:
            raise RuntimeError("Episode is truncated. Call reset().")

        self._events = {aid: [] for aid in self._runtimes}
        self._sounds = []

        intents: dict[AgentId, Action] = {}
        for aid in sorted(self._runtimes):
            rt = self._runtimes[aid]
            st = rt.state
            if st is None or not st.alive:
                continue
            if aid in actions:
                intents[aid] = actions[aid]
            elif rt.controller is not None:
                intents[aid] = rt.controller.act(self._last_observations[aid])
            else:
                intents[aid] = Action()

        rewards = {aid: 0.0 for aid in self._runtimes}
        for aid, rt in self._runtimes.items():
            if rt.state is not None and rt.state.alive:
                rewards[aid] += self.config.reward.survival_per_tick
                rt.survival_ticks += 1

        # Rotation is an intention over the coming tick. We update orientation
        # first so movement and aiming are coherent for this interval.
        for aid, action in intents.items():
            st = self._runtimes[aid].state
            assert st is not None
            if action.rotation is not None and not action.rest:
                av = clamp(
                    action.rotation.angular_speed,
                    -self.config.agent.max_angular_speed,
                    self.config.agent.max_angular_speed,
                )
                st.orientation = wrap_angle(st.orientation + av * TICK_SECONDS)

        # Interactions use references from the observation that produced these
        # actions, so they are evaluated before the world advances.
        self._resolve_interactions(intents, rewards)

        # Compute desired velocities. Physical constraints remain engine-side.
        requested_speeds: dict[AgentId, float] = {}
        for aid, action in intents.items():
            st = self._runtimes[aid].state
            assert st is not None
            if not st.alive or action.rest or action.movement is None:
                st.vx = st.vy = 0.0
                st.speed = 0.0
                requested_speeds[aid] = 0.0
                continue
            vx, vy, speed = self._movement_velocity(st, action.movement)
            st.vx, st.vy, st.speed = vx, vy, speed
            requested_speeds[aid] = speed

        self._simulate_physics(intents, rewards)
        self._update_physiology(intents, requested_speeds, rewards)

        # Movement itself can be audible.
        for aid, rt in self._runtimes.items():
            st = rt.state
            if st is not None and st.alive and st.speed > 1.6:
                self._sounds.append(
                    SoundEventInternal(st.x, st.y, SoundKind.MOVEMENT, min(0.65, 0.18 + st.speed * 0.12), aid)
                )

        self.tick += 1

        if self.config.max_ticks is not None and self.tick >= self.config.max_ticks:
            self._truncated = True
            for aid, rt in self._runtimes.items():
                st = rt.state
                if st is not None and st.alive and not rt.end_notified:
                    self._notify_end(aid, EpisodeEndReason.TIME_LIMIT)

        for aid, reward in rewards.items():
            self._runtimes[aid].cumulative_reward += reward

        self._last_observations = self._observe_all()
        terminated = {
            aid: (rt.state is None or not rt.state.alive)
            for aid, rt in self._runtimes.items()
        }
        info = {
            aid: AgentStepInfo(
                survival_ticks=rt.survival_ticks,
                cumulative_reward=rt.cumulative_reward,
                kills=rt.kills,
                pickups=rt.pickups,
                damage_received=rt.damage_received,
                damage_inflicted=rt.damage_inflicted,
            )
            for aid, rt in self._runtimes.items()
        }
        return StepResult(dict(self._last_observations), rewards, terminated, self._truncated, info)

    def close(self) -> None:
        self._initialized = False

    # ------------------------------------------------------------------
    # Introspection for the renderer / teacher debug. Not agent API.
    # ------------------------------------------------------------------
    def debug_state(self) -> dict[str, Any]:
        if self.world is None:
            return {}
        return {
            "tick": self.tick,
            "world": self.world,
            "agents": {aid: rt.state for aid, rt in self._runtimes.items()},
            "projectiles": tuple(self._projectiles),
            "sounds": tuple(self._sounds),
            "observations": dict(self._last_observations),
            "stats": {
                aid: {
                    "cumulative_reward": rt.cumulative_reward,
                    "kills": rt.kills,
                    "pickups": rt.pickups,
                    "survival_ticks": rt.survival_ticks,
                }
                for aid, rt in self._runtimes.items()
            },
        }

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------
    def _movement_velocity(self, st: AgentState, movement: Movement) -> tuple[float, float, float]:
        lx, ly = normalize(clamp(movement.forward, -1, 1), clamp(movement.right, -1, 1))
        fdx, fdy = heading_vector(st.orientation)
        rdx, rdy = right_vector(st.orientation)
        dx = lx * fdx + ly * rdx
        dy = lx * fdy + ly * rdy

        injury_penalty = 1.0 - min(0.45, sum(i.severity for i in st.injuries) * 0.12)
        energy_penalty = 0.35 + 0.65 * st.energy
        water_factor = self._area_factor(st.x, st.y, ObjectKind.WATER, "speed_factor", 1.0)
        max_speed = self.config.agent.max_speed * injury_penalty * energy_penalty * water_factor
        speed = clamp(movement.speed, 0.0, max_speed)
        return dx * speed, dy * speed, speed

    def _simulate_physics(self, intents: dict[AgentId, Action], rewards: dict[AgentId, float]) -> None:
        n = max(1, self.config.physics_substeps)
        dt = TICK_SECONDS / n
        collided_static: set[AgentId] = set()
        collided_pairs: set[tuple[AgentId, AgentId]] = set()

        for sub in range(n):
            phase = self.tick + (sub + 1) / n
            self._update_moving_obstacles(phase)

            old = {
                aid: (rt.state.x, rt.state.y)
                for aid, rt in self._runtimes.items()
                if rt.state is not None and rt.state.alive
            }
            proposed: dict[AgentId, tuple[float, float]] = {}

            for aid, (ox, oy) in old.items():
                st = self._runtimes[aid].state
                assert st is not None
                nx = wrap_pos(ox + st.vx * dt, self.config.map_width)
                ny = oy
                hit = False
                if self._circle_hits_solid(nx, ny, st.radius):
                    nx = ox
                    hit = True
                ny2 = wrap_pos(oy + st.vy * dt, self.config.map_height)
                if self._circle_hits_solid(nx, ny2, st.radius):
                    ny2 = oy
                    hit = True
                ny = ny2
                proposed[aid] = (nx, ny)
                if hit:
                    collided_static.add(aid)

            # Symmetric agent/agent resolution. Pair overlaps generate
            # displacement corrections that are accumulated first and then
            # applied simultaneously, avoiding an advantage from call order.
            corrections = {aid: [0.0, 0.0] for aid in proposed}
            aids = sorted(proposed)
            for i, a in enumerate(aids):
                sa = self._runtimes[a].state
                assert sa is not None
                ax, ay = proposed[a]
                for b in aids[i + 1 :]:
                    sb = self._runtimes[b].state
                    assert sb is not None
                    bx, by = proposed[b]
                    dx, dy = torus_vector(ax, ay, bx, by, self.config.map_width, self.config.map_height)
                    dist = hypot(dx, dy)
                    min_dist = sa.radius + sb.radius
                    if dist < min_dist:
                        if dist < 1e-8:
                            # Deterministic fallback axis for exact overlap.
                            ux, uy = (1.0, 0.0) if a < b else (-1.0, 0.0)
                        else:
                            ux, uy = dx / dist, dy / dist
                        push = (min_dist - dist + 1e-4) * 0.5
                        corrections[a][0] -= ux * push
                        corrections[a][1] -= uy * push
                        corrections[b][0] += ux * push
                        corrections[b][1] += uy * push

                        # Perfectly inelastic response along the contact normal:
                        # remove only the relative velocity that keeps both
                        # bodies penetrating each other. Tangential motion is
                        # preserved, so agents can slide/circle naturally.
                        va_n = sa.vx * ux + sa.vy * uy
                        vb_n = sb.vx * ux + sb.vy * uy
                        closing_speed = va_n - vb_n
                        if closing_speed > 0.0:
                            correction_v = 0.5 * closing_speed
                            sa.vx -= ux * correction_v
                            sa.vy -= uy * correction_v
                            sb.vx += ux * correction_v
                            sb.vy += uy * correction_v
                            sa.speed = hypot(sa.vx, sa.vy)
                            sb.speed = hypot(sb.vx, sb.vy)

                        pair = (a, b)
                        if pair not in collided_pairs:
                            collided_pairs.add(pair)
                            rel = hypot(sa.vx - sb.vx, sa.vy - sb.vy)
                            if rel > 1.5:
                                damage = min(0.065, (rel - 1.5) * 0.011)
                                self._apply_damage(a, damage, b, rewards)
                                self._apply_damage(b, damage, a, rewards)
                            self._events[a].append(Event(EventKind.COLLISION, properties={"with": "agent"}))
                            self._events[b].append(Event(EventKind.COLLISION, properties={"with": "agent"}))
                            self._sounds.append(SoundEventInternal(ax, ay, SoundKind.IMPACT, min(0.7, 0.25 + rel * 0.08)))

            for aid, (nx, ny) in proposed.items():
                st = self._runtimes[aid].state
                assert st is not None
                cx, cy = corrections[aid]
                fx = wrap_pos(nx + cx, self.config.map_width)
                fy = wrap_pos(ny + cy, self.config.map_height)
                if self._circle_hits_solid(fx, fy, st.radius):
                    fx, fy = old[aid]
                st.x, st.y = fx, fy

            self._advance_projectiles(dt, rewards)

        for aid in collided_static:
            st = self._runtimes[aid].state
            if st is None or not st.alive:
                continue
            self._events[aid].append(Event(EventKind.COLLISION, properties={"with": "obstacle"}))
            rewards[aid] -= self.config.reward.harmful_collision
            self._sounds.append(SoundEventInternal(st.x, st.y, SoundKind.IMPACT, 0.28, aid))

    def _resolve_interactions(self, intents: dict[AgentId, Action], rewards: dict[AgentId, float]) -> None:
        pickup_groups: dict[int, list[AgentId]] = {}
        door_groups: dict[int, list[AgentId]] = {}
        pending_damage: list[tuple[str, float, str | None]] = []

        for aid in sorted(intents):
            action = intents[aid]
            st = self._runtimes[aid].state
            if st is None or not st.alive or action.interaction is None:
                continue
            interaction = action.interaction

            if isinstance(interaction, Pickup):
                uid = self._resolve_ref(aid, interaction.target_ref)
                if uid is not None:
                    pickup_groups.setdefault(uid, []).append(aid)

            elif isinstance(interaction, Manipulate):
                uid = self._resolve_ref(aid, interaction.target_ref)
                if uid is not None:
                    door_groups.setdefault(uid, []).append(aid)

            elif isinstance(interaction, Equip):
                if interaction.slot in st.inventory:
                    st.equipped_slot = interaction.slot
                    self._events[aid].append(Event(EventKind.ITEM_EQUIPPED, properties={"slot": interaction.slot}))

            elif isinstance(interaction, Drop):
                self._drop_slot(aid, interaction.slot)

            elif isinstance(interaction, Ingest):
                self._ingest_slot(aid, interaction.slot)

            elif isinstance(interaction, Shoot):
                self._shoot(aid, interaction)

            elif isinstance(interaction, Throw):
                self._throw(aid, interaction)

            elif isinstance(interaction, Use):
                dmg = self._use(aid, interaction)
                if dmg is not None:
                    pending_damage.append(dmg)

        # Competing pickups are resolved by the environment RNG, not by call order.
        for uid, contenders in sorted(pickup_groups.items()):
            obj = self._world_object_by_uid(uid)
            if obj is None or not obj.portable:
                continue
            eligible = [aid for aid in sorted(contenders) if self._can_pickup(aid, obj)]
            if not eligible:
                continue
            winner = self.rng.choice(eligible)
            self._pickup(winner, obj, rewards)

        # Multiple valid manipulations of the same door in one tick toggle it once.
        for uid, contenders in sorted(door_groups.items()):
            door = self._door_by_uid(uid)
            if door is None:
                continue
            eligible = [aid for aid in sorted(contenders) if self._in_reach(aid, door.x, door.y, 0.8)]
            if eligible:
                door.open = not door.open
                door.properties["open"] = door.open
                self._sounds.append(SoundEventInternal(door.x, door.y, SoundKind.MECHANISM, 0.45))
                for aid in eligible:
                    self._events[aid].append(Event(EventKind.INTERACTION, properties={"kind": "door", "open": door.open}))

        # Melee damage is accumulated first to keep simultaneous attacks possible.
        for target_id, damage, source_id in pending_damage:
            self._apply_damage(target_id, damage, source_id, rewards)

    def _pickup(self, aid: str, obj: WorldObject, rewards: dict[str, float]) -> None:
        st = self._runtimes[aid].state
        assert st is not None
        slot = self._first_free_slot(st)
        if slot is None:
            return
        if st.carried_mass + obj.mass > self.config.agent.max_carry_mass:
            return
        st.inventory[slot] = InventoryEntry(obj.uid, obj.kind, obj.mass, dict(obj.properties))
        self.world.objects.remove(obj)  # type: ignore[union-attr]
        self._events[aid].append(Event(EventKind.ITEM_PICKED_UP, properties={"slot": slot, "kind": obj.kind.name}))
        rt = self._runtimes[aid]
        rt.pickups += 1
        if obj.uid not in rt.picked_uids:
            rt.picked_uids.add(obj.uid)
            rewards[aid] += self.config.reward.first_pickup

    def _drop_slot(self, aid: str, slot: int) -> None:
        st = self._runtimes[aid].state
        assert st is not None
        entry = st.inventory.pop(slot, None)
        if entry is None:
            return
        if st.equipped_slot == slot:
            st.equipped_slot = None
        dx, dy = heading_vector(st.orientation)
        obj = WorldObject(
            uid=entry.uid,
            kind=entry.kind,
            x=wrap_pos(st.x + dx * 0.9, self.config.map_width),
            y=wrap_pos(st.y + dy * 0.9, self.config.map_height),
            mass=entry.mass,
            radius=0.20,
            portable=True,
            solid=False,
            properties=dict(entry.properties),
        )
        self.world.objects.append(obj)  # type: ignore[union-attr]
        self._events[aid].append(Event(EventKind.ITEM_DROPPED, properties={"slot": slot}))

    def _ingest_slot(self, aid: str, slot: int) -> None:
        st = self._runtimes[aid].state
        assert st is not None
        entry = st.inventory.get(slot)
        if entry is None:
            return
        used = False
        if entry.kind is ObjectKind.FOOD:
            st.satiety = clamp(st.satiety + float(entry.properties.get("nutrition", 0.25)), 0, 1)
            used = True
        elif entry.kind is ObjectKind.DRINK:
            st.hydration = clamp(st.hydration + float(entry.properties.get("hydration", 0.30)), 0, 1)
            used = True
        elif entry.kind is ObjectKind.MEDICINE:
            medical_type = entry.properties.get("medical_type")
            if medical_type == "bandage":
                for injury in st.injuries:
                    injury.bleeding *= 0.15
                used = True
            else:
                healed = float(entry.properties.get("heal", 0.08))
                before = st.health
                st.health = clamp(st.health + healed, 0, 1)
                if st.health > before:
                    self._events[aid].append(Event(EventKind.HEALED, value=st.health - before))
                used = True
        if used:
            del st.inventory[slot]
            if st.equipped_slot == slot:
                st.equipped_slot = None
            self._events[aid].append(Event(EventKind.ITEM_USED, properties={"slot": slot}))

    def _use(self, aid: str, use: Use) -> tuple[str, float, str | None] | None:
        st = self._runtimes[aid].state
        assert st is not None
        entry = st.inventory.get(use.slot)
        if entry is None:
            return None

        if entry.kind is ObjectKind.WEAPON and entry.properties.get("weapon_type") == "rolling_pin":
            if use.target_ref is None:
                return None
            uid = self._resolve_ref(aid, use.target_ref)
            target_id = self._agent_id_by_uid(uid) if uid is not None else None
            if target_id is None or target_id == aid:
                return None
            target = self._runtimes[target_id].state
            if target is None or not target.alive:
                return None
            reach = st.radius + target.radius + 0.55
            if torus_distance(st.x, st.y, target.x, target.y, self.config.map_width, self.config.map_height) <= reach:
                damage = float(entry.properties.get("damage", 0.12))
                self._events[aid].append(Event(EventKind.ITEM_USED, properties={"weapon": "rolling_pin"}))
                self._sounds.append(SoundEventInternal(target.x, target.y, SoundKind.IMPACT, 0.55, aid))
                return target_id, damage, aid

        if entry.kind is ObjectKind.WEAPON_MODIFIER:
            eq = st.inventory.get(st.equipped_slot) if st.equipped_slot is not None else None
            if eq is not None and eq.kind is ObjectKind.WEAPON:
                modifier = entry.properties.get("modifier")
                factor = float(entry.properties.get("factor", 1.0))
                if modifier == "projectile_speed":
                    eq.properties["projectile_speed_factor"] = float(eq.properties.get("projectile_speed_factor", 1.0)) * factor
                    del st.inventory[use.slot]
                    self._events[aid].append(Event(EventKind.ITEM_USED, properties={"modifier": str(modifier)}))
        elif entry.kind in {ObjectKind.FOOD, ObjectKind.DRINK, ObjectKind.MEDICINE}:
            self._ingest_slot(aid, use.slot)
        return None

    def _shoot(self, aid: str, shoot: Shoot) -> None:
        st = self._runtimes[aid].state
        assert st is not None
        if st.equipped_slot is None:
            return
        weapon = st.inventory.get(st.equipped_slot)
        if weapon is None or weapon.kind is not ObjectKind.WEAPON or weapon.properties.get("weapon_type") != "potato_launcher":
            return
        ammo_slot = next((slot for slot, item in sorted(st.inventory.items()) if item.kind is ObjectKind.AMMUNITION and item.properties.get("ammo_type") == "potato"), None)
        if ammo_slot is None:
            return
        del st.inventory[ammo_slot]
        if st.equipped_slot == ammo_slot:
            st.equipped_slot = None

        angle = wrap_angle(st.orientation + shoot.bearing)
        dx, dy = heading_vector(angle)
        speed = 6.0 * float(weapon.properties.get("projectile_speed_factor", 1.0))
        self._projectiles.append(
            Projectile(
                uid=self._alloc_uid(),
                owner_id=aid,
                x=wrap_pos(st.x + dx * (st.radius + 0.25), self.config.map_width),
                y=wrap_pos(st.y + dy * (st.radius + 0.25), self.config.map_height),
                vx=dx * speed,
                vy=dy * speed,
                radius=0.13,
                damage=float(weapon.properties.get("damage", 0.18)),
                ttl=3.0,
                properties={"label": "potato", "kind": "potato"},
            )
        )
        self._events[aid].append(Event(EventKind.SHOT_FIRED))
        self._sounds.append(SoundEventInternal(st.x, st.y, SoundKind.GUNSHOT, 1.0, aid))

    def _throw(self, aid: str, throw: Throw) -> None:
        st = self._runtimes[aid].state
        assert st is not None
        entry = st.inventory.pop(throw.slot, None)
        if entry is None:
            return
        if st.equipped_slot == throw.slot:
            st.equipped_slot = None
        angle = wrap_angle(st.orientation + throw.bearing)
        dx, dy = heading_vector(angle)
        power = clamp(throw.power, 0.0, 1.0)
        speed = (2.0 + 4.0 * power) / max(0.65, entry.mass ** 0.35)
        self._projectiles.append(
            Projectile(
                uid=self._alloc_uid(),
                owner_id=aid,
                x=wrap_pos(st.x + dx * (st.radius + 0.25), self.config.map_width),
                y=wrap_pos(st.y + dy * (st.radius + 0.25), self.config.map_height),
                vx=dx * speed,
                vy=dy * speed,
                radius=0.16,
                damage=min(0.16, 0.02 + entry.mass * power * 0.025),
                ttl=2.5,
                properties={
                    "thrown_item": True,
                    "item_uid": entry.uid,
                    "item_kind": entry.kind.name,
                    "item_mass": entry.mass,
                    "item_properties": dict(entry.properties),
                },
            )
        )

    def _advance_projectiles(self, dt: float, rewards: dict[str, float]) -> None:
        survivors: list[Projectile] = []
        for p in self._projectiles:
            p.ttl -= dt
            nx = wrap_pos(p.x + p.vx * dt, self.config.map_width)
            ny = wrap_pos(p.y + p.vy * dt, self.config.map_height)
            hit = False
            if self._circle_hits_rects(nx, ny, p.radius):
                hit = True
                self._sounds.append(SoundEventInternal(nx, ny, SoundKind.IMPACT, 0.45, p.owner_id))
            else:
                for aid, rt in self._runtimes.items():
                    st = rt.state
                    if st is None or not st.alive or aid == p.owner_id:
                        continue
                    if torus_distance(nx, ny, st.x, st.y, self.config.map_width, self.config.map_height) <= p.radius + st.radius:
                        self._apply_damage(aid, p.damage, p.owner_id, rewards)
                        self._sounds.append(SoundEventInternal(st.x, st.y, SoundKind.IMPACT, 0.55, p.owner_id))
                        hit = True
                        break
            if hit or p.ttl <= 0:
                self._retire_projectile(p, nx, ny)
            else:
                p.x, p.y = nx, ny
                survivors.append(p)
        self._projectiles = survivors

    def _retire_projectile(self, p: Projectile, x: float, y: float) -> None:
        if not p.properties.get("thrown_item") or self.world is None:
            return
        kind_name = str(p.properties.get("item_kind", "UNKNOWN"))
        try:
            kind = ObjectKind[kind_name]
        except KeyError:
            kind = ObjectKind.UNKNOWN
        obj = WorldObject(
            uid=int(p.properties.get("item_uid", self._alloc_uid())),
            kind=kind,
            x=x,
            y=y,
            mass=float(p.properties.get("item_mass", 0.5)),
            radius=0.20,
            portable=True,
            solid=False,
            properties=dict(p.properties.get("item_properties", {})),
        )
        self.world.objects.append(obj)

    def _update_physiology(self, intents: dict[str, Action], requested: dict[str, float], rewards: dict[str, float]) -> None:
        cfg = self.config.physiology
        for aid, rt in self._runtimes.items():
            st = rt.state
            if st is None or not st.alive:
                continue
            action = intents.get(aid, Action())
            st.hydration = clamp(st.hydration - cfg.hydration_decay_per_tick, 0, 1)
            st.satiety = clamp(st.satiety - cfg.satiety_decay_per_tick, 0, 1)

            speed_ratio = requested.get(aid, 0.0) / max(0.001, self.config.agent.max_speed)
            carry_factor = 1.0 + 0.65 * (st.carried_mass / max(0.001, self.config.agent.max_carry_mass))
            injury_factor = 1.0 + 0.25 * sum(i.severity for i in st.injuries)
            water_factor = self._area_factor(st.x, st.y, ObjectKind.WATER, "energy_factor", 1.0)
            movement_cost = cfg.movement_energy_cost_scale * (speed_ratio ** 2) * carry_factor * injury_factor * water_factor
            st.energy = clamp(st.energy - movement_cost, 0, 1)

            resource_factor = min(st.hydration, st.satiety)
            if action.rest and requested.get(aid, 0.0) < 0.05:
                st.energy = clamp(st.energy + cfg.rest_energy_gain_per_tick * (0.25 + 0.75 * resource_factor), 0, 1)
            elif requested.get(aid, 0.0) < 0.2:
                st.energy = clamp(st.energy + cfg.idle_energy_gain_per_tick * resource_factor, 0, 1)

            bleeding = sum(i.bleeding * i.severity for i in st.injuries) * 0.004
            if bleeding > 0:
                self._apply_damage(aid, bleeding, None, rewards)
                if not st.alive:
                    continue

            if st.hydration < cfg.low_resource_threshold or st.satiety < cfg.low_resource_threshold:
                self._apply_damage(aid, cfg.critical_resource_health_loss, None, rewards)
                if not st.alive:
                    continue

            if st.health < 1.0 and resource_factor > 0.35 and bleeding < 0.001:
                severity = sum(i.severity for i in st.injuries)
                heal = cfg.passive_heal_per_tick * resource_factor / (1.0 + severity)
                if heal > 0:
                    st.health = clamp(st.health + heal, 0, 1)

    def _apply_damage(self, target_id: str, damage: float, source_id: str | None, rewards: dict[str, float]) -> None:
        if damage <= 0:
            return
        rt = self._runtimes[target_id]
        st = rt.state
        if st is None or not st.alive:
            return
        damage = min(damage, st.health)
        st.health -= damage
        st.last_damage_source = source_id
        rt.damage_received += damage
        rewards[target_id] -= damage * self.config.reward.damage_multiplier
        self._events[target_id].append(Event(EventKind.DAMAGE_RECEIVED, value=damage))
        if source_id is not None and source_id in self._runtimes and source_id != target_id:
            self._runtimes[source_id].damage_inflicted += damage

        # Damage can generate simple injuries. Deterministic thresholds avoid
        # adding another hidden random channel in the first implementation.
        if damage >= 0.12:
            st.injuries.append(InjuryState(kind=self._injury_kind_for_damage(damage), severity=min(1.0, damage * 2.2), bleeding=0.35 if damage >= 0.16 else 0.0))

        if st.health <= 1e-9:
            self._kill_agent(target_id, source_id, rewards)

    def _kill_agent(self, aid: str, killer_id: str | None, rewards: dict[str, float]) -> None:
        rt = self._runtimes[aid]
        st = rt.state
        if st is None or not st.alive:
            return
        st.alive = False
        st.health = 0.0
        st.speed = st.vx = st.vy = 0.0
        rewards[aid] -= self.config.reward.death
        if killer_id is not None and killer_id in self._runtimes and killer_id != aid:
            rewards[killer_id] += self.config.reward.kill
            self._runtimes[killer_id].kills += 1
        self._drop_inventory_on_death(st)
        self._notify_end(aid, EpisodeEndReason.ELIMINATED)

    def _drop_inventory_on_death(self, st: AgentState) -> None:
        if self.world is None:
            return
        entries = list(st.inventory.values())
        st.inventory.clear()
        st.equipped_slot = None
        for idx, entry in enumerate(entries):
            angle = (2 * pi * idx / max(1, len(entries)))
            x = wrap_pos(st.x + cos(angle) * 0.45, self.config.map_width)
            y = wrap_pos(st.y + sin(angle) * 0.45, self.config.map_height)
            self.world.objects.append(
                WorldObject(entry.uid, entry.kind, x, y, entry.mass, 0.20, True, False, dict(entry.properties))
            )

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------
    def _observe_all(self) -> dict[str, Observation]:
        out: dict[str, Observation] = {}
        for aid in sorted(self._runtimes):
            st = self._runtimes[aid].state
            if st is not None and st.alive:
                out[aid] = self._build_observation(aid)
        return out

    def _build_observation(self, aid: str) -> Observation:
        st = self._runtimes[aid].state
        assert st is not None and self.world is not None
        ref_by_uid: dict[int, int] = {}
        uid_by_ref: dict[int, int] = {}

        def ref_for(uid: int) -> int:
            if uid not in ref_by_uid:
                ref = len(ref_by_uid) + 1
                ref_by_uid[uid] = ref
                uid_by_ref[ref] = uid
            return ref_by_uid[uid]

        vision: list[VisualDetection] = []
        for uid, kind, x, y, radius, props, rect in self._visual_candidates(aid):
            if rect is None:
                dx, dy = torus_vector(st.x, st.y, x, y, self.config.map_width, self.config.map_height)
                dist_center = hypot(dx, dy)
                distance = max(0.0, dist_center - radius)
                bearing = bearing_from_vector(dx, dy, st.orientation) if dist_center > 1e-9 else 0.0
                angular = 2 * atan2(radius, max(0.05, dist_center))
                target_x = st.x + dx
                target_y = st.y + dy
            else:
                shifted = nearest_toric_rect(st.x, st.y, rect, self.config.map_width, self.config.map_height)
                qx, qy = nearest_point_on_rect(st.x, st.y, shifted)
                dx, dy = qx - st.x, qy - st.y
                distance = hypot(dx, dy)
                bearing = bearing_from_vector(dx, dy, st.orientation) if distance > 1e-9 else 0.0
                angular = 2 * atan2(max(shifted.w, shifted.h) / 2, max(0.05, distance))
                target_x, target_y = qx, qy

            if abs(bearing) > self.config.agent.vision_fov / 2:
                continue
            local_factor = self.world.visibility_factor * self.world.low_light_factor
            local_factor *= self._steam_visibility_factor(st.x, st.y, target_x, target_y)
            effective_range = self.config.agent.vision_range * local_factor
            if distance > effective_range:
                continue
            if self._is_occluded(st.x, st.y, target_x, target_y, ignore_uid=uid):
                continue

            recognized_kind = kind if distance <= effective_range * 0.82 else ObjectKind.UNKNOWN
            recognized_props: dict[str, Any] = {}
            if distance <= effective_range * 0.50:
                for key, value in props.items():
                    if isinstance(value, (int, float, bool, str)) or value is None:
                        recognized_props[key] = value
            elif distance <= effective_range * 0.70 and "label" in props:
                recognized_props["label"] = props["label"]

            vision.append(
                VisualDetection(ref_for(uid), recognized_kind, distance, bearing, angular, recognized_props)
            )

        vision.sort(key=lambda d: (d.distance, d.bearing))

        touch: list[TouchContact] = []
        for uid, kind, x, y, props, rect in self._touch_candidates(aid):
            contact = False
            bearing = 0.0
            if rect is None:
                dx, dy = torus_vector(st.x, st.y, x, y, self.config.map_width, self.config.map_height)
                contact = hypot(dx, dy) <= st.radius + 0.25
                bearing = bearing_from_vector(dx, dy, st.orientation) if contact and hypot(dx, dy) > 1e-9 else 0.0
            else:
                shifted = nearest_toric_rect(st.x, st.y, rect, self.config.map_width, self.config.map_height)
                contact = circle_rect_overlap(st.x, st.y, st.radius + 0.02, shifted)
                if contact:
                    qx, qy = nearest_point_on_rect(st.x, st.y, shifted)
                    bearing = bearing_from_vector(qx - st.x, qy - st.y, st.orientation)
            if contact:
                clean_props = {k: v for k, v in props.items() if isinstance(v, (int, float, bool, str)) or v is None}
                touch.append(TouchContact(ref_for(uid), kind, bearing, clean_props))

        hearing = self._hearing_for(aid)
        self._ref_maps[aid] = uid_by_ref

        inv = tuple(
            InventoryItem(slot, item.kind, item.mass, {k: v for k, v in item.properties.items() if isinstance(v, (int, float, bool, str)) or v is None})
            for slot, item in sorted(st.inventory.items())
        )
        injuries = tuple(Injury(i.kind, i.severity, i.bleeding) for i in st.injuries)
        self_state = SelfState(
            health=st.health,
            energy=st.energy,
            hydration=st.hydration,
            satiety=st.satiety,
            body_mass=st.body_mass,
            carried_mass=st.carried_mass,
            speed=st.speed,
            orientation=st.orientation,
            injuries=injuries,
            inventory=inv,
            equipped_slot=st.equipped_slot,
        )
        return Observation(
            tick=self.tick,
            self_state=self_state,
            vision=tuple(vision),
            hearing=tuple(hearing),
            touch=tuple(touch),
            events=tuple(self._events.get(aid, ())),
        )

    def _hearing_for(self, aid: str) -> list[SoundDetection]:
        st = self._runtimes[aid].state
        assert st is not None
        out: list[SoundDetection] = []
        for sound in self._sounds:
            if sound.source_agent_id == aid:
                continue
            dx, dy = torus_vector(st.x, st.y, sound.source_x, sound.source_y, self.config.map_width, self.config.map_height)
            dist = hypot(dx, dy)
            if dist > self.config.agent.hearing_range:
                continue
            absorption = self._sound_absorption(st.x, st.y, st.x + dx, st.y + dy)
            intensity = sound.base_intensity * max(0.0, 1.0 - dist / self.config.agent.hearing_range) * (1.0 - 0.65 * absorption)
            if intensity <= 0.025:
                continue
            true_bearing = bearing_from_vector(dx, dy, st.orientation)
            uncertainty = 0.06 + absorption * 0.75 + (dist / self.config.agent.hearing_range) * 0.16
            perceived = wrap_angle(true_bearing + self.rng.gauss(0.0, uncertainty * 0.42))
            estimate = max(0.0, dist * (1.0 + self.rng.gauss(0.0, 0.04 + absorption * 0.14)))
            out.append(
                SoundDetection(sound.kind, clamp(intensity, 0, 1), perceived, uncertainty, estimate, clamp(absorption, 0, 1))
            )
        out.sort(key=lambda s: -s.intensity)
        return out

    # ------------------------------------------------------------------
    # Sensor geometry helpers
    # ------------------------------------------------------------------
    def _visual_candidates(self, self_id: str):
        assert self.world is not None
        for aid, rt in self._runtimes.items():
            st = rt.state
            if aid == self_id or st is None or not st.alive:
                continue
            armed = False
            if st.equipped_slot is not None and st.equipped_slot in st.inventory:
                armed = st.inventory[st.equipped_slot].kind is ObjectKind.WEAPON
            yield st.uid, ObjectKind.AGENT, st.x, st.y, st.radius, {"armed": armed}, None
        for obj in self.world.objects:
            yield obj.uid, obj.kind, obj.x, obj.y, obj.radius, obj.properties, None
        for p in self._projectiles:
            yield p.uid, ObjectKind.PROJECTILE, p.x, p.y, p.radius, {"label": p.properties.get("label", "projectile")}, None
        for rect in self.world.rects:
            yield rect.uid, rect.kind, rect.x, rect.y, 0.0, rect.properties, rect.rect
        for door in self.world.doors:
            props = dict(door.properties)
            props["open"] = door.open
            yield door.uid, ObjectKind.DOOR, door.x, door.y, 0.0, props, door.rect
        for area in self.world.areas:
            yield area.uid, area.kind, area.x, area.y, 0.0, area.properties, area.rect

    def _touch_candidates(self, self_id: str):
        assert self.world is not None
        for aid, rt in self._runtimes.items():
            st = rt.state
            if aid == self_id or st is None or not st.alive:
                continue
            yield st.uid, ObjectKind.AGENT, st.x, st.y, {}, None
        for obj in self.world.objects:
            yield obj.uid, obj.kind, obj.x, obj.y, obj.properties, None
        for rect in self.world.rects:
            yield rect.uid, rect.kind, rect.x, rect.y, rect.properties, rect.rect
        for door in self.world.doors:
            if not door.open:
                yield door.uid, ObjectKind.DOOR, door.x, door.y, {"open": False}, door.rect
        for area in self.world.areas:
            if point_in_rect(self._runtimes[self_id].state.x, self._runtimes[self_id].state.y, area.rect):  # type: ignore[union-attr]
                yield area.uid, area.kind, area.x, area.y, area.properties, area.rect

    def _is_occluded(self, x1: float, y1: float, x2: float, y2: float, ignore_uid: int | None = None) -> bool:
        assert self.world is not None
        for rect in self.world.rects:
            if rect.uid == ignore_uid or not rect.occluding:
                continue
            if self._segment_hits_toric_rect(x1, y1, x2, y2, rect.rect):
                return True
        for door in self.world.doors:
            if door.uid == ignore_uid or not door.effective_occluding:
                continue
            if self._segment_hits_toric_rect(x1, y1, x2, y2, door.rect):
                return True
        return False

    def _steam_visibility_factor(self, x1: float, y1: float, x2: float, y2: float) -> float:
        assert self.world is not None
        factor = 1.0
        for area in self.world.areas:
            if area.kind is ObjectKind.STEAM and self._segment_hits_toric_rect(x1, y1, x2, y2, area.rect):
                factor *= float(area.properties.get("visibility_factor", 0.6))
        return factor

    def _sound_absorption(self, x1: float, y1: float, x2: float, y2: float) -> float:
        assert self.world is not None
        survival = 1.0
        for rect in self.world.rects:
            if self._segment_hits_toric_rect(x1, y1, x2, y2, rect.rect):
                survival *= 1.0 - clamp(rect.sound_absorption, 0, 0.95)
        for door in self.world.doors:
            if not door.open and self._segment_hits_toric_rect(x1, y1, x2, y2, door.rect):
                survival *= 1.0 - clamp(door.sound_absorption, 0, 0.95)
        return 1.0 - survival

    def _segment_hits_toric_rect(self, x1: float, y1: float, x2: float, y2: float, rect: Rect) -> bool:
        shifted = nearest_toric_rect(x1, y1, rect, self.config.map_width, self.config.map_height)
        return segment_intersects_rect(x1, y1, x2, y2, shifted)

    # ------------------------------------------------------------------
    # Collision / world helpers
    # ------------------------------------------------------------------
    def _circle_hits_rects(self, x: float, y: float, radius: float) -> bool:
        assert self.world is not None
        for entity in self.world.rects:
            r = nearest_toric_rect(x, y, entity.rect, self.config.map_width, self.config.map_height)
            if circle_rect_overlap(x, y, radius, r):
                return True
        for door in self.world.doors:
            if door.open:
                continue
            r = nearest_toric_rect(x, y, door.rect, self.config.map_width, self.config.map_height)
            if circle_rect_overlap(x, y, radius, r):
                return True
        return False

    def _circle_hits_solid(self, x: float, y: float, radius: float) -> bool:
        if self._circle_hits_rects(x, y, radius):
            return True
        assert self.world is not None
        for obj in self.world.objects:
            if not obj.solid:
                continue
            if torus_distance(x, y, obj.x, obj.y, self.config.map_width, self.config.map_height) < radius + obj.radius:
                return True
        return False

    def _update_moving_obstacles(self, phase_tick: float) -> None:
        assert self.world is not None
        for obj in self.world.objects:
            if obj.kind is not ObjectKind.MOVING_OBSTACLE:
                continue
            amp = float(obj.properties.get("amplitude", 0.0))
            period = max(1.0, float(obj.properties.get("period_ticks", 24.0)))
            origin_x = float(obj.properties.get("origin_x", obj.x))
            origin_y = float(obj.properties.get("origin_y", obj.y))
            phase = 2 * pi * phase_tick / period
            if obj.properties.get("axis") == "y":
                obj.y = wrap_pos(origin_y + amp * sin(phase), self.config.map_height)
            else:
                obj.x = wrap_pos(origin_x + amp * sin(phase), self.config.map_width)

    def _area_factor(self, x: float, y: float, kind: ObjectKind, key: str, default: float) -> float:
        if self.world is None:
            return default
        value = default
        for area in self.world.areas:
            if area.kind is kind and point_in_rect(x, y, area.rect):
                value *= float(area.properties.get(key, 1.0))
        return value

    def _can_pickup(self, aid: str, obj: WorldObject) -> bool:
        st = self._runtimes[aid].state
        if st is None or not st.alive:
            return False
        if self._first_free_slot(st) is None:
            return False
        if st.carried_mass + obj.mass > self.config.agent.max_carry_mass:
            return False
        reach = st.radius + obj.radius + self.config.agent.pickup_reach
        return torus_distance(st.x, st.y, obj.x, obj.y, self.config.map_width, self.config.map_height) <= reach

    def _in_reach(self, aid: str, x: float, y: float, extra: float) -> bool:
        st = self._runtimes[aid].state
        if st is None or not st.alive:
            return False
        return torus_distance(st.x, st.y, x, y, self.config.map_width, self.config.map_height) <= st.radius + extra

    def _first_free_slot(self, st: AgentState) -> int | None:
        for slot in range(self.config.agent.inventory_slots):
            if slot not in st.inventory:
                return slot
        return None

    def _resolve_ref(self, aid: str, ref: int) -> int | None:
        return self._ref_maps.get(aid, {}).get(ref)

    def _world_object_by_uid(self, uid: int) -> WorldObject | None:
        if self.world is None:
            return None
        return next((o for o in self.world.objects if o.uid == uid), None)

    def _door_by_uid(self, uid: int) -> DoorEntity | None:
        if self.world is None:
            return None
        return next((d for d in self.world.doors if d.uid == uid), None)

    def _agent_id_by_uid(self, uid: int | None) -> str | None:
        if uid is None:
            return None
        for aid, rt in self._runtimes.items():
            if rt.state is not None and rt.state.uid == uid:
                return aid
        return None

    def _spawn_points(self) -> list[tuple[float, float]]:
        W, H = self.config.map_width, self.config.map_height
        points = [
            (W * 0.50, H * 0.08),
            (W * 0.08, H * 0.50),
            (W * 0.50, H * 0.92),
            (W * 0.92, H * 0.50),
            (W * 0.30, H * 0.47),
            (W * 0.47, H * 0.30),
            (W * 0.70, H * 0.53),
            (W * 0.53, H * 0.70),
            (W * 0.18, H * 0.18),
            (W * 0.82, H * 0.82),
            (W * 0.82, H * 0.18),
            (W * 0.18, H * 0.82),
        ]
        valid: list[tuple[float, float]] = []
        for p in points:
            if not self._circle_hits_solid(p[0], p[1], self.config.agent.radius):
                valid.append(p)
        return valid

    def _alloc_uid(self) -> int:
        uid = self._next_uid
        self._next_uid += 1
        return uid

    def _notify_end(self, aid: str, reason: EpisodeEndReason) -> None:
        rt = self._runtimes[aid]
        if rt.end_notified:
            return
        rt.end_notified = True
        if rt.controller is not None:
            rt.controller.on_episode_end(EpisodeResult(reason, rt.survival_ticks))

    @staticmethod
    def _injury_kind_for_damage(damage: float):
        from .arena_api import InjuryKind
        if damage > 0.20:
            return InjuryKind.WOUND
        return InjuryKind.BRUISE
