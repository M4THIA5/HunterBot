from __future__ import annotations

from dataclasses import dataclass, field
from math import pi


@dataclass(slots=True)
class RewardConfig:
    survival_per_tick: float = 0.1
    kill: float = 5.0
    first_pickup: float = 1.0
    damage_multiplier: float = 1.0
    harmful_collision: float = 0.5
    death: float = 20.0


@dataclass(slots=True)
class GenerationConfig:
    obstacle_count: tuple[int, int] = (5, 10)
    movable_obstacle_count: tuple[int, int] = (1, 2)
    item_count: tuple[int, int] = (12, 20)
    water_probability: float = 0.7
    fog_probability: float = 0.3
    low_light_probability: float = 0.25


@dataclass(slots=True)
class AgentPhysicsConfig:
    body_mass: float = 70.0
    radius: float = 0.565  # ~1 m² disk
    max_speed: float = 3.2
    max_angular_speed: float = pi
    max_carry_mass: float = 15.0
    inventory_slots: int = 8
    vision_fov: float = 2 * pi / 3  # 120°
    vision_range: float = 12.0
    hearing_range: float = 20.0
    pickup_reach: float = 0.45


@dataclass(slots=True)
class PhysiologyConfig:
    hydration_decay_per_tick: float = 0.0006
    satiety_decay_per_tick: float = 0.0004
    passive_heal_per_tick: float = 0.0015
    rest_energy_gain_per_tick: float = 0.055
    idle_energy_gain_per_tick: float = 0.012
    movement_energy_cost_scale: float = 0.028
    low_resource_threshold: float = 0.15
    critical_resource_health_loss: float = 0.003


@dataclass(slots=True)
class EnvironmentConfig:
    max_ticks: int | None = 600
    map_width: float = 30.0
    map_height: float = 30.0
    physics_substeps: int = 5
    max_agents: int = 8
    render_fps: int = 60
    reward: RewardConfig = field(default_factory=RewardConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    agent: AgentPhysicsConfig = field(default_factory=AgentPhysicsConfig)
    physiology: PhysiologyConfig = field(default_factory=PhysiologyConfig)
