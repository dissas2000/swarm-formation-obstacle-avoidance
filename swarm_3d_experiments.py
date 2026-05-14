
"""
swarm_3d_experiments.py

Расширенная 3D-симуляция для ВКР:
    p_i^des(t) = p_c(t) + R(t) Delta_i^(m(t))
    F_i = F_i^formation + F_i^obstacle + F_i^damping
    F_i^formation = F_i^pos + F_i^sep + F_i^coh

Сценарии:
    free
    single
    wide_barrier
    dense_field
    low_wall_overflight
    tall_wall_side

Добавлено: сценарий all_field_obstacles с препятствиями по всему полю.

Запуск:
    python swarm_3d_experiments.py
    python swarm_3d_experiments.py --scenario wide_barrier --show
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


COLORS = {
    "leader": "#163A5F",
    "follower1": "#2A9D8F",
    "follower2": "#4FB3A5",
    "follower3": "#76C7AE",
    "follower4": "#A8DDB5",
    "obstacle": "#AEB8C2",
    "obstacle_edge": "#5B6570",
    "low_obstacle": "#B7D7A8",
    "tall_obstacle": "#AEB8C2",
    "safety": "#E9C46A",
    "goal": "#F4A261",
    "start": "#6C757D",
    "center_path": "#D1495B",
    "boundary": "#6C757D",
    "boundary_safety": "#ADB5BD",
}

ROLES = ["leader", "follower1", "follower2", "follower3", "follower4"]


@dataclass
class ObstacleCylinder:
    x: float
    y: float
    r: float
    z_min: float = 0.0
    z_max: float = 2.2
    name: str = "obs"
    is_boundary: bool = False

    @property
    def center_xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


@dataclass
class Scenario:
    name: str
    title: str
    obstacles: List[ObstacleCylinder]
    start: np.ndarray
    goal: np.ndarray
    preferred_altitude: float = 1.20
    allow_overflight: bool = True
    description: str = ""


@dataclass
class Drone3D:
    role: str
    p: np.ndarray
    v: np.ndarray
    mass: float = 1.0
    radius: float = 0.24
    max_speed: float = 1.05
    max_force: float = 3.0
    energy: float = 0.0
    path_length: float = 0.0
    history: Optional[List[np.ndarray]] = None

    def __post_init__(self) -> None:
        self.p = self.p.astype(float)
        self.v = self.v.astype(float)
        self.history = [self.p.copy()]

    def apply_force(self, f: np.ndarray, dt: float) -> None:
        f = clamp_norm(f, self.max_force)
        self.energy += float(np.dot(f, f)) * dt
        self.v += (f / self.mass) * dt
        self.v = clamp_norm(self.v, self.max_speed)
        prev = self.p.copy()
        self.p += self.v * dt
        self.path_length += float(np.linalg.norm(self.p - prev))
        self.history.append(self.p.copy())


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    n = norm(v)
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0]) if fallback is None else fallback.copy()
    return v / n


def clamp_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = norm(v)
    if n > max_norm and n > 1e-9:
        return v / n * max_norm
    return v


def rotation_from_direction(d: np.ndarray) -> np.ndarray:
    d_xy = np.array([d[0], d[1], 0.0], dtype=float)
    ex = unit(d_xy, fallback=np.array([1.0, 0.0, 0.0]))
    ez = np.array([0.0, 0.0, 1.0])
    ey = unit(np.cross(ez, ex), fallback=np.array([0.0, 1.0, 0.0]))
    return np.column_stack([ex, ey, ez])


def horizontal_clearance(p: np.ndarray, obs: ObstacleCylinder, radius: float) -> float:
    return float(np.linalg.norm(p[:2] - obs.center_xy) - obs.r - radius)


def vertical_clearance(p: np.ndarray, obs: ObstacleCylinder, radius: float) -> float:
    if p[2] > obs.z_max:
        return float(p[2] - obs.z_max - radius)
    if p[2] < obs.z_min:
        return float(obs.z_min - p[2] - radius)
    return -radius


def cylinder_clearance_3d(p: np.ndarray, obs: ObstacleCylinder, drone_radius: float) -> float:
    """
    Приближённый зазор от сферического дрона до вертикального цилиндра.

    Если дрон находится в высотном диапазоне препятствия, главным является
    горизонтальный зазор. Если дрон выше препятствия, то даже при совпадении
    проекции на плоскость xy безопасный зазор определяется вертикальным
    расстоянием до верхней крышки цилиндра.
    """
    h = horizontal_clearance(p, obs, drone_radius)

    if obs.z_min - drone_radius <= p[2] <= obs.z_max + drone_radius:
        return h

    v = vertical_clearance(p, obs, drone_radius)

    if h < 0 and v >= 0:
        return v
    if h >= 0 and v < 0:
        return h
    if h >= 0 and v >= 0:
        return float(math.sqrt(h * h + v * v))
    return max(h, v)


def row_of_obstacles(x: float, y_values: List[float], radius: float, z_max: float, prefix: str) -> List[ObstacleCylinder]:
    return [ObstacleCylinder(x=x, y=y, r=radius, z_max=z_max, name=f"{prefix}_{i}") for i, y in enumerate(y_values)]



def grid_obstacle_field() -> List[ObstacleCylinder]:
    """
    Поле препятствий: препятствия расставлены почти по всей рабочей области.

    Идея сценария: куда бы группа ни пыталась двигаться от старта к цели,
    перед ней возникают препятствия. Сцена проверяет не один манёвр, а
    устойчивость всей схемы обхода и адаптации формации.
    """
    obstacles: List[ObstacleCylinder] = []
    xs = [-3.9, -2.9, -1.9, -0.9, 0.1, 1.1, 2.1, 3.1]
    rows = {
        -2.7: [-3.4, -1.4, 0.6, 2.6],
        -1.8: [-2.7, -0.7, 1.3, 3.3],
        -0.9: [-3.6, -1.6, 0.4, 2.4],
         0.0: [-2.9, -0.9, 1.1, 3.1],
         0.9: [-3.4, -1.4, 0.6, 2.6],
         1.8: [-2.7, -0.7, 1.3, 3.3],
         2.7: [-3.6, -1.6, 0.4, 2.4],
    }
    idx = 0
    for y, xs_row in rows.items():
        for x in xs_row:
            # Чередуем высоты: часть препятствий можно перелетать, часть нужно обходить.
            if idx % 5 == 0:
                zmax = 0.85
                r = 0.28
            elif idx % 3 == 0:
                zmax = 1.35
                r = 0.30
            else:
                zmax = 2.2
                r = 0.31
            obstacles.append(ObstacleCylinder(x=x, y=y, r=r, z_max=zmax, name=f"field_{idx}"))
            idx += 1
    return obstacles

def make_perimeter_fence(
    x_min: float = -8.0,
    x_max: float = 8.0,
    y_min: float = -6.5,
    y_max: float = 6.5,
    step: float = 0.60,
    radius: float = 0.10,
    z_max: float = 2.6,
) -> List[ObstacleCylinder]:
    """
    Ограждение по периметру рабочей области.

    Математически это такие же цилиндрические препятствия, но с флагом
    is_boundary=True. Они участвуют в расчёте безопасных расстояний и
    не дают планировщику выбирать траекторию за пределами рабочей зоны.
    """
    fence: List[ObstacleCylinder] = []

    xs = np.arange(x_min, x_max + 1e-9, step)
    ys = np.arange(y_min, y_max + 1e-9, step)

    for x in xs:
        fence.append(ObstacleCylinder(x=float(x), y=y_min, r=radius, z_max=z_max, name="fence_bottom", is_boundary=True))
        fence.append(ObstacleCylinder(x=float(x), y=y_max, r=radius, z_max=z_max, name="fence_top", is_boundary=True))

    for y in ys:
        fence.append(ObstacleCylinder(x=x_min, y=float(y), r=radius, z_max=z_max, name="fence_left", is_boundary=True))
        fence.append(ObstacleCylinder(x=x_max, y=float(y), r=radius, z_max=z_max, name="fence_right", is_boundary=True))

    return fence


def with_fence(obstacles: List[ObstacleCylinder]) -> List[ObstacleCylinder]:
    """Совместимость со старой версией: ограждение по краям больше не добавляется."""
    return obstacles



def checkerboard_obstacle_field(
    x_values: List[float],
    y_values: List[float],
    radius: float = 0.34,
    low_height: float = 0.75,
    mid_height: float = 1.35,
    high_height: float = 2.25,
    prefix: str = "checker",
) -> List[ObstacleCylinder]:
    """
    Разреженное шахматное поле препятствий.

    Визуальная цель: не перегружать рисунок, но расположить цилиндры
    по всей ширине рабочей области. Расстояние между соседними
    препятствиями оставлено достаточно большим, чтобы группа могла
    выбирать: сжатие, колонну, боковой обход или перелет низких
    препятствий.
    """
    obstacles: List[ObstacleCylinder] = []
    heights = [low_height, high_height, mid_height]

    for ix, x in enumerate(x_values):
        # Сдвиг каждого второго столбца создаёт шахматный порядок.
        y_shift = 0.55 if ix % 2 else 0.0

        for iy, y in enumerate(y_values):
            yy = y + y_shift
            if yy < -3.9 or yy > 3.9:
                continue

            # Оставляем несколько "окон", чтобы поле было проходимым,
            # но не имело прямой свободной полосы.
            if (ix, iy) in {(1, 2), (2, 0), (3, 3), (4, 1)}:
                continue

            h = heights[(ix + iy) % len(heights)]

            obstacles.append(
                ObstacleCylinder(
                    x=float(x),
                    y=float(yy),
                    r=float(radius),
                    z_max=float(h),
                    name=f"{prefix}_{ix}_{iy}",
                )
            )

    return obstacles


def uniform_checkerboard_obstacle_field(
    x_values: List[float],
    y_values: List[float],
    radius: float = 0.34,
    height: float = 2.20,
    prefix: str = "uniform",
) -> List[ObstacleCylinder]:
    """
    Разреженное шахматное поле препятствий одинаковой высоты.

    Этот сценарий нужен как контрольный эксперимент: группа не может
    пользоваться преимуществом низких препятствий, поэтому основное
    поведение должно сводиться к боковому обходу, сжатию или колонне.
    """
    obstacles: List[ObstacleCylinder] = []

    for ix, x in enumerate(x_values):
        y_shift = 0.55 if ix % 2 else 0.0

        for iy, y in enumerate(y_values):
            yy = y + y_shift
            if yy < -3.9 or yy > 3.9:
                continue

            if (ix, iy) in {(1, 2), (2, 0), (3, 3), (4, 1)}:
                continue

            obstacles.append(
                ObstacleCylinder(
                    x=float(x),
                    y=float(yy),
                    r=float(radius),
                    z_max=float(height),
                    name=f"{prefix}_{ix}_{iy}",
                )
            )

    return obstacles


def make_scenarios() -> Dict[str, Scenario]:
    start = np.array([-5.4, 0.0, 1.20])
    goal = np.array([5.4, 0.0, 1.20])
    scenarios: Dict[str, Scenario] = {}

    scenarios["free"] = Scenario(
        name="free",
        title="Свободное движение без препятствий",
        obstacles=with_fence([]),
        start=start,
        goal=goal,
        description="Базовая проверка сохранения формации.",
    )

    scenarios["single"] = Scenario(
        name="single",
        title="Одиночные препятствия",
        obstacles=with_fence([
            ObstacleCylinder(-0.5, 0.0, 0.45, z_max=2.2, name="central"),
            ObstacleCylinder(1.2, 1.15, 0.38, z_max=2.2, name="upper"),
            ObstacleCylinder(2.2, -1.0, 0.38, z_max=2.2, name="lower"),
        ]),
        start=start,
        goal=goal,
    )

    scenarios["wide_barrier"] = Scenario(
        name="wide_barrier",
        title="Широкая полоса препятствий: боковой облет",
        obstacles=with_fence(
            row_of_obstacles(-0.4, [-2.4, -1.6, -0.8, 0.0, 0.8, 1.6, 2.4], 0.34, 2.2, "barrier_a")
            + row_of_obstacles(0.55, [-2.0, -1.2, -0.4, 0.4, 1.2, 2.0], 0.34, 2.2, "barrier_b")
        ),
        start=start,
        goal=goal,
        allow_overflight=False,
        description="Препятствия распределены по ширине, группа обязана облетать их сбоку.",
    )

    scenarios["dense_field"] = Scenario(
        name="dense_field",
        title="Плотная среда препятствий",
        obstacles=with_fence([
            ObstacleCylinder(-2.6, -1.1, 0.32, z_max=2.2, name="d1"),
            ObstacleCylinder(-2.1,  0.85, 0.35, z_max=2.2, name="d2"),
            ObstacleCylinder(-1.25, -0.15, 0.42, z_max=2.2, name="d3"),
            ObstacleCylinder(-0.35,  1.35, 0.37, z_max=2.2, name="d4"),
            ObstacleCylinder( 0.35, -1.15, 0.40, z_max=2.2, name="d5"),
            ObstacleCylinder( 1.05,  0.05, 0.45, z_max=2.2, name="d6"),
            ObstacleCylinder( 1.9,   1.25, 0.34, z_max=2.2, name="d7"),
            ObstacleCylinder( 2.5,  -0.85, 0.36, z_max=2.2, name="d8"),
        ]),
        start=start,
        goal=goal,
    )

    scenarios["low_wall_overflight"] = Scenario(
        name="low_wall_overflight",
        title="Низкая полоса препятствий: перелет сверху",
        obstacles=with_fence(row_of_obstacles(0.0, [-2.8, -2.0, -1.2, -0.4, 0.4, 1.2, 2.0, 2.8], 0.34, 0.75, "low_wall")),
        start=np.array([-5.4, 0.0, 1.05]),
        goal=np.array([5.4, 0.0, 1.05]),
        preferred_altitude=1.05,
        allow_overflight=True,
        description="Препятствия по ширине, но низкие: выгоден перелет сверху.",
    )

    scenarios["tall_wall_side"] = Scenario(
        name="tall_wall_side",
        title="Высокая полоса препятствий: боковой облет",
        obstacles=with_fence(row_of_obstacles(0.0, [-2.8, -2.0, -1.2, -0.4, 0.4, 1.2, 2.0, 2.8], 0.34, 2.4, "tall_wall")),
        start=start,
        goal=goal,
        allow_overflight=False,
        description="Высокая стенка препятствий: перелет запрещен, нужен боковой облет.",
    )


    scenarios["all_field_obstacles"] = Scenario(
        name="all_field_obstacles",
        title="Разреженное шахматное поле препятствий разной высоты",
        obstacles=checkerboard_obstacle_field(
            # 6 продольных колонок и 5 поперечных уровней:
            # поле занимает всю ширину, но не перегружает визуализацию.
            x_values=[-3.6, -2.25, -0.9, 0.45, 1.8, 3.15],
            y_values=[-3.0, -1.5, 0.0, 1.5, 3.0],
            radius=0.34,
            low_height=0.70,
            mid_height=1.35,
            high_height=2.25,
            prefix="field",
        ),
        start=start,
        goal=goal,
        preferred_altitude=1.20,
        allow_overflight=True,
        description=(
            "Разреженное шахматное поле препятствий разной высоты. "
            "Препятствия расположены по всей ширине рабочей области, "
            "но расстояния между ними оставлены достаточно большими "
            "для демонстрации адаптации формации."
        ),
    )


    scenarios["uniform_field_obstacles"] = Scenario(
        name="uniform_field_obstacles",
        title="Шахматное поле препятствий одинаковой высоты",
        obstacles=uniform_checkerboard_obstacle_field(
            x_values=[-3.6, -2.25, -0.9, 0.45, 1.8, 3.15],
            y_values=[-3.0, -1.5, 0.0, 1.5, 3.0],
            radius=0.34,
            height=2.20,
            prefix="uniform_field",
        ),
        start=start,
        goal=goal,
        preferred_altitude=1.20,
        allow_overflight=False,
        description=(
            "Контрольный сценарий: шахматное поле препятствий одинаковой высоты. "
            "Перелет сверху запрещен, поэтому группа должна проходить поле "
            "за счет бокового обхода, сжатия и перестроения в колонну."
        ),
    )

    # defense_showcase: 5 зон слева направо
    # Зона 1: свободный участок
    # Зона 2: широкие препятствия по бокам — сжатие формации
    # Зона 3: узкий проход — перестроение в колонну
    # Зона 4: высокие слева/справа, низкое по центру — перелет
    # Зона 5: свободный выход
    scenarios["defense_showcase"] = Scenario(
        name="defense_showcase",
        title="Демонстрация всех режимов адаптации: сжатие → колонна → перелет",
        obstacles=with_fence([
            # --- Зона 2: сжатие формации ---
            ObstacleCylinder(-2.8,  2.1, 0.40, z_max=2.2, name="compress_r1"),
            ObstacleCylinder(-2.8, -2.1, 0.40, z_max=2.2, name="compress_l1"),
            ObstacleCylinder(-2.0,  1.6, 0.42, z_max=2.2, name="compress_r2"),
            ObstacleCylinder(-2.0, -1.6, 0.42, z_max=2.2, name="compress_l2"),
            # --- Зона 3: узкий проход для column ---
            ObstacleCylinder(-0.6,  1.5, 0.40, z_max=2.3, name="col_r_a"),
            ObstacleCylinder(-0.6,  2.3, 0.38, z_max=2.3, name="col_r_b"),
            ObstacleCylinder(-0.6, -1.5, 0.40, z_max=2.3, name="col_l_a"),
            ObstacleCylinder(-0.6, -2.3, 0.38, z_max=2.3, name="col_l_b"),
            ObstacleCylinder( 0.4,  1.4, 0.40, z_max=2.3, name="col_r_c"),
            ObstacleCylinder( 0.4,  2.2, 0.38, z_max=2.3, name="col_r_d"),
            ObstacleCylinder( 0.4, -1.4, 0.40, z_max=2.3, name="col_l_c"),
            ObstacleCylinder( 0.4, -2.2, 0.38, z_max=2.3, name="col_l_d"),
            # --- Зона 4: высокие слева/справа, низкое в центре → overflight ---
            ObstacleCylinder( 1.8,  1.8, 0.42, z_max=2.3, name="tall_r_a"),
            ObstacleCylinder( 1.8,  2.6, 0.36, z_max=2.3, name="tall_r_b"),
            ObstacleCylinder( 1.8, -1.8, 0.42, z_max=2.3, name="tall_l_a"),
            ObstacleCylinder( 1.8, -2.6, 0.36, z_max=2.3, name="tall_l_b"),
            ObstacleCylinder( 1.8,  0.0, 0.42, z_max=1.30, name="low_center"),
        ]),
        start=start,
        goal=goal,
        preferred_altitude=1.20,
        allow_overflight=True,
        description=(
            "Последовательная демонстрация всех режимов адаптации формации: "
            "зона 1 — свободное движение; "
            "зона 2 — боковые препятствия вынуждают сжать формацию; "
            "зона 3 — узкий проход требует перестроения в колонну; "
            "зона 4 — высокие препятствия по бокам и низкое по центру требуют перелета; "
            "зона 5 — свободный выход."
        ),
    )

    # vertical_escape_corridor: боковой обход закрыт высокими стенами,
    # единственный путь — перелёт над низким центральным препятствием.
    #
    # Геометрия (проверки):
    #   Центральное препятствие: z_max=1.30 → required_z = 1.30 + 0.55 = 1.85 ≤ 2.05 (overflightable)
    #   Боковые стены:           z_max=2.35 → required_z = 2.35 + 0.55 = 2.90 > 2.05 (NOT overflightable)
    #   Ожидаемый max_center_z ≈ 1.85  (из desired_alt = nearby_low_obstacle_height())
    scenarios["vertical_escape_corridor"] = Scenario(
        name="vertical_escape_corridor",
        title="Вертикальный выход: боковые проходы закрыты, только вверх",
        obstacles=with_fence([
            # Центральное низкое препятствие — единственный overflightable объект
            ObstacleCylinder( 0.0,  0.0, 0.65, z_max=1.30, name="vert_center"),
            # Левая стена (y > 0) — высокие, НЕ перелетаемые
            ObstacleCylinder(-3.5,  2.0, 0.44, z_max=2.35, name="wall_l1"),
            ObstacleCylinder(-2.0,  2.0, 0.44, z_max=2.35, name="wall_l2"),
            ObstacleCylinder(-0.5,  2.0, 0.44, z_max=2.35, name="wall_l3"),
            ObstacleCylinder( 1.0,  2.0, 0.44, z_max=2.35, name="wall_l4"),
            ObstacleCylinder( 2.5,  2.0, 0.44, z_max=2.35, name="wall_l5"),
            # Правая стена (y < 0) — высокие, НЕ перелетаемые
            ObstacleCylinder(-3.5, -2.0, 0.44, z_max=2.35, name="wall_r1"),
            ObstacleCylinder(-2.0, -2.0, 0.44, z_max=2.35, name="wall_r2"),
            ObstacleCylinder(-0.5, -2.0, 0.44, z_max=2.35, name="wall_r3"),
            ObstacleCylinder( 1.0, -2.0, 0.44, z_max=2.35, name="wall_r4"),
            ObstacleCylinder( 2.5, -2.0, 0.44, z_max=2.35, name="wall_r5"),
        ]),
        start=start,
        goal=goal,
        preferred_altitude=1.20,
        allow_overflight=True,
        description=(
            "Боковой обход закрыт двумя рядами высоких препятствий (z_max=2.35 > 2.05). "
            "Единственный допустимый путь — перелёт над центральным низким препятствием "
            "(z_max=1.30, required_z=1.85 ≤ 2.05). "
            "Ожидаемая последовательность режимов: normal → overflight → normal."
        ),
    )

    return scenarios


class AdaptiveFormationController3D:
    def __init__(self, scenario: Scenario, adaptive: bool = True, altitude_aware: bool = True):
        self.scenario = scenario
        self.obstacles = scenario.obstacles
        self.adaptive = adaptive
        self.altitude_aware = altitude_aware

        self.dt = 0.05
        self.center_speed = 0.42
        self.target_z = scenario.preferred_altitude
        self.max_altitude = 2.05
        self.overflight_margin = 0.55

        self.k_pos = 2.65
        self.k_sep = 1.9
        self.k_coh = 0.07
        self.k_obs = 0.85
        self.k_tan = 0.72
        self.k_damp = 1.18
        self.k_alt = 2.4

        self.drone_radius = 0.24
        self.d_safe = 0.64
        self.d_detect = 1.28
        self.obs_detect = 0.95
        self.obs_safe = 0.32
        self.constraint_margin = 0.10
        self.x_min, self.x_max = -7.72, 7.72
        self.y_min, self.y_max = -6.22, 6.22

        self.start_center = scenario.start.astype(float)
        self.goal_center = scenario.goal.astype(float)
        self.path = self.build_center_path()
        self.path_length = self.path_len(self.path)

    def role_offset(self, role: str, mode: str) -> np.ndarray:
        if mode == "normal":
            offsets = {
                "leader":    (0.00,  0.00, 0.00),
                "follower1": (-0.95,  0.80, 0.00),
                "follower2": (-0.95, -0.80, 0.00),
                "follower3": (-1.90,  0.40, 0.00),
                "follower4": (-1.90, -0.40, 0.00),
            }
        elif mode == "compressed":
            offsets = {
                "leader":    (0.00,  0.00, 0.00),
                "follower1": (-0.82,  0.36, 0.00),
                "follower2": (-1.64, -0.36, 0.00),
                "follower3": (-2.46,  0.32, 0.00),
                "follower4": (-3.28, -0.32, 0.00),
            }
        elif mode == "column":
            offsets = {
                "leader":    (0.00,  0.00, 0.00),
                "follower1": (-0.88,  0.22, 0.00),
                "follower2": (-1.76, -0.22, 0.00),
                "follower3": (-2.64,  0.22, 0.00),
                "follower4": (-3.52, -0.22, 0.00),
            }
        else:
            offsets = {
                "leader":    (0.00,  0.00,  0.00),
                "follower1": (-0.86,  0.28,  0.08),
                "follower2": (-1.72, -0.28, -0.06),
                "follower3": (-2.58,  0.24,  0.06),
                "follower4": (-3.44, -0.24, -0.08),
            }
        return np.array(offsets[role], dtype=float)

    def slot_position(self, center: np.ndarray, direction: np.ndarray, role: str, mode: str) -> np.ndarray:
        return center + rotation_from_direction(direction) @ self.role_offset(role, mode)

    def mode_clearance(self, center: np.ndarray, direction: np.ndarray, mode: str) -> float:
        min_c = 1e9
        for role in ROLES:
            s = self.slot_position(center, direction, role, mode)
            for obs in self.obstacles:
                if obs.is_boundary:
                    continue
                c = cylinder_clearance_3d(s, obs, self.drone_radius + 0.28)
                min_c = min(min_c, c)
        return min_c

    def nearby_low_obstacle_height(self, center: np.ndarray) -> Optional[float]:
        if not (self.altitude_aware and self.scenario.allow_overflight):
            return None
        candidates = []
        for obs in self.obstacles:
            if obs.is_boundary:
                continue
            if abs(obs.x - center[0]) < 1.35 and abs(obs.y - center[1]) < 3.6:
                required_z = obs.z_max + self.overflight_margin
                if required_z <= self.max_altitude:
                    candidates.append(required_z)
        return max(candidates) if candidates else None

    def choose_mode(self, center: np.ndarray, direction: np.ndarray) -> str:
        if not self.adaptive:
            return "normal"
        if self.nearby_low_obstacle_height(center) is not None:
            return "overflight"
        if self.mode_clearance(center, direction, "normal") > 0.14:
            return "normal"
        if self.mode_clearance(center, direction, "compressed") > 0.10:
            return "compressed"
        return "column"

    @staticmethod
    def path_len(path: List[np.ndarray]) -> float:
        return float(sum(norm(path[i + 1] - path[i]) for i in range(len(path) - 1)))

    def path_is_safe(self, path: List[np.ndarray], inflation: float) -> bool:
        if not self.obstacles:
            return True
        for a, b in zip(path[:-1], path[1:]):
            for k in range(100):
                t = k / 99
                p = (1 - t) * a + t * b
                for obs in self.obstacles:
                    if obs.is_boundary:
                        continue
                    if cylinder_clearance_3d(p, obs, self.drone_radius + inflation) < 0:
                        return False
        return True

    def build_center_path(self) -> List[np.ndarray]:
        direct = [self.start_center.copy(), self.goal_center.copy()]

        if self.scenario.name == "defense_showcase":
            # Явный путь через 5 зон: центр держится строго по оси y=0,
            # в зоне 4 поднимается на 0.30 м выше target_z для перелёта
            # через низкое центральное препятствие (z_max=0.75, margin=0.55 → нужен z≥1.30).
            lift_z = max(self.target_z, 1.30 + self.overflight_margin + 0.05)
            return [
                self.start_center.copy(),
                np.array([-4.5,  0.0, self.target_z]),   # зона 1: свободный участок
                np.array([-3.0,  0.0, self.target_z]),   # зона 2: вход в сжатие
                np.array([-1.5,  0.0, self.target_z]),   # зона 2/3: переход
                np.array([-0.1,  0.0, self.target_z]),   # зона 3: колонна
                np.array([ 0.8,  0.0, self.target_z]),   # зона 3/4: выход из прохода
                np.array([ 1.8,  0.0, lift_z]),          # зона 4: перелёт над низким центром
                np.array([ 2.6,  0.0, self.target_z]),   # зона 5: вход в свободный выход
                np.array([ 4.0,  0.0, self.target_z]),   # зона 5: свободный выход
                self.goal_center.copy(),
            ]

        if self.scenario.name in {"all_field_obstacles", "uniform_field_obstacles"}:
            # Для шахматного поля задаём опорную траекторию через поле,
            # но не заставляем центр идти по краю. Манёвры внутри поля
            # выполняются за счёт локальных сил и выбора режима.
            return [
                self.start_center.copy(),
                np.array([-4.2, -0.30, self.target_z]),
                np.array([-2.8,  0.80, self.target_z]),
                np.array([-1.4, -0.55, self.target_z]),
                np.array([ 0.2,  0.70, self.target_z]),
                np.array([ 1.7, -0.60, self.target_z]),
                np.array([ 3.1,  0.45, self.target_z]),
                np.array([ 4.3, -0.20, self.target_z]),
                self.goal_center.copy(),
            ]

        if self.scenario.name == "vertical_escape_corridor":
            # Центр идёт строго через y=0, чтобы пройти вблизи центрального
            # низкого препятствия (x=0, y=0). При |center_x| < 1.35 функция
            # nearby_low_obstacle_height() вернёт required_z=1.85 и контроллер
            # переключится в режим overflight. z-координата пути не используется
            # напрямую — она переопределяется в step() через desired_alt.
            return [
                self.start_center.copy(),
                np.array([-4.0,  0.0, self.target_z]),
                np.array([-2.0,  0.0, self.target_z]),
                np.array([ 0.0,  0.0, self.target_z]),  # зона overflight
                np.array([ 2.0,  0.0, self.target_z]),
                np.array([ 4.0,  0.0, self.target_z]),
                self.goal_center.copy(),
            ]

        inner_obstacles = [o for o in self.obstacles if not o.is_boundary]

        if self.scenario.allow_overflight and inner_obstacles:
            max_required_z = max(o.z_max + self.overflight_margin for o in inner_obstacles)
            if max_required_z <= self.max_altitude:
                xmid = float(np.mean([o.x for o in inner_obstacles]))
                over_z = max(self.target_z, max_required_z)
                over_path = [
                    self.start_center.copy(),
                    np.array([xmid - 1.2, 0.0, over_z]),
                    np.array([xmid + 1.2, 0.0, over_z]),
                    self.goal_center.copy(),
                ]
                if self.path_is_safe(over_path, inflation=0.25):
                    return over_path

        if self.path_is_safe(direct, inflation=1.10):
            return direct

        if not inner_obstacles:
            return direct

        xmin = min(o.x - o.r for o in inner_obstacles)
        xmax = max(o.x + o.r for o in inner_obstacles)
        ymin = min(o.y - o.r for o in inner_obstacles)
        ymax = max(o.y + o.r for o in inner_obstacles)

        margin_x = 1.50
        margin_y = 1.90
        y_top = ymax + margin_y
        y_bottom = ymin - margin_y

        def side_path(y: float) -> List[np.ndarray]:
            return [
                self.start_center.copy(),
                np.array([xmin - margin_x - 0.7, 0.0, self.target_z]),
                np.array([xmin - margin_x, 0.45 * y, self.target_z]),
                np.array([xmin - margin_x, y, self.target_z]),
                np.array([(xmin + xmax) / 2, y, self.target_z]),
                np.array([xmax + margin_x, y, self.target_z]),
                np.array([xmax + margin_x, 0.45 * y, self.target_z]),
                np.array([xmax + margin_x + 0.7, 0.0, self.target_z]),
                self.goal_center.copy(),
            ]

        candidates = sorted([side_path(y_top), side_path(y_bottom)], key=self.path_len)
        for p in candidates:
            if self.path_is_safe(p, inflation=0.42):
                return p
        return candidates[0]

    def sample_center(self, s: float) -> Tuple[np.ndarray, np.ndarray]:
        s = min(s, self.path_length)
        rem = s
        for a, b in zip(self.path[:-1], self.path[1:]):
            seg = norm(b - a)
            if rem <= seg:
                tau = rem / max(seg, 1e-9)
                p = (1 - tau) * a + tau * b
                d = unit(b - a)
                return p, d
            rem -= seg
        d = unit(self.path[-1] - self.path[-2])
        return self.path[-1].copy(), d

    def obstacle_force(self, drone: Drone3D, direction: np.ndarray) -> Tuple[np.ndarray, float]:
        f = np.zeros(3)
        min_clear = 1e9

        for obs in self.obstacles:
            if obs.is_boundary:
                continue
            c3 = cylinder_clearance_3d(drone.p, obs, drone.radius)
            min_clear = min(min_clear, c3)
            if c3 >= self.obs_detect:
                continue

            h_clear = horizontal_clearance(drone.p, obs, drone.radius)
            away2 = drone.p[:2] - obs.center_xy
            d = np.linalg.norm(away2)

            if obs.z_min - drone.radius <= drone.p[2] <= obs.z_max + drone.radius and d > 1e-9:
                n2 = away2 / d
                gap = max(h_clear - self.obs_safe, 0.12)
                gain = min(2.2, self.k_obs / (gap * gap))
                tangent2 = np.array([-n2[1], n2[0]])
                if np.dot(tangent2, direction[:2]) < 0:
                    tangent2 = -tangent2
                f[:2] += gain * n2 + self.k_tan * gain * tangent2

            if self.scenario.allow_overflight and h_clear < self.obs_detect:
                required_z = obs.z_max + self.overflight_margin
                if required_z <= self.max_altitude and drone.p[2] < required_z:
                    f[2] += self.k_alt * (required_z - drone.p[2])

        return f, min_clear

    def formation_force(self, drone: Drone3D, desired: np.ndarray, center: np.ndarray, drones: List[Drone3D]) -> np.ndarray:
        f_pos = self.k_pos * (desired - drone.p)

        f_sep = np.zeros(3)
        for other in drones:
            if other.role == drone.role:
                continue
            diff = drone.p - other.p
            d = norm(diff)
            if d < self.d_detect:
                direction = unit(diff)
                gap = max(d - self.d_safe, 0.12)
                gain = min(3.2, self.k_sep * (1.0 / gap - 1.0 / (self.d_detect - self.d_safe)) / (gap * gap))
                if gain > 0:
                    f_sep += gain * direction

        f_coh = self.k_coh * (center - drone.p)
        return f_pos + f_sep + f_coh

    def enforce_constraints(self, drones: List[Drone3D]) -> None:
        """
        Численная коррекция ограничений.

        После интегрирования ОДУ агент может оказаться немного внутри
        запрещённой области из-за конечного шага интегрирования. Поэтому
        выполняется проекция обратно в допустимую область с малым запасом.
        Это не меняет математическую идею модели, а стабилизирует численную
        реализацию ограничений безопасности.
        """
        margin = self.constraint_margin

        # 1) Ограждение периметра: удерживаем центры дронов внутри рабочей области.
        for drone in drones:
            drone.p[0] = float(np.clip(drone.p[0], self.x_min, self.x_max))
            drone.p[1] = float(np.clip(drone.p[1], self.y_min, self.y_max))

        # 2) Запрещённые цилиндры: либо вытолкнуть в сторону, либо поднять над низким препятствием.
        for drone in drones:
            for obs in self.obstacles:
                if obs.is_boundary:
                    continue

                required_xy = obs.r + drone.radius + margin
                xy = drone.p[:2] - obs.center_xy
                dxy = float(np.linalg.norm(xy))
                in_height_band = (obs.z_min - drone.radius - margin) <= drone.p[2] <= (obs.z_max + drone.radius + margin)

                if in_height_band and dxy < required_xy:
                    can_overfly = self.scenario.allow_overflight and (obs.z_max + self.overflight_margin <= self.max_altitude)
                    if can_overfly:
                        drone.p[2] = max(drone.p[2], obs.z_max + self.overflight_margin)
                        if drone.v[2] < 0:
                            drone.v[2] = 0.0
                    else:
                        if dxy < 1e-9:
                            direction = np.array([1.0, 0.0])
                        else:
                            direction = xy / dxy
                        drone.p[:2] = obs.center_xy + direction * required_xy

                        # Убираем компоненту скорости, направленную внутрь препятствия.
                        inward = float(np.dot(drone.v[:2], -direction))
                        if inward > 0:
                            drone.v[:2] += inward * direction

            if drone.history:
                drone.history[-1] = drone.p.copy()

        # 3) Междроновая безопасность: несколько мягких итераций разведения.
        min_dist = self.d_safe + 0.03
        for _ in range(4):
            for i, a in enumerate(drones):
                for b in drones[i + 1:]:
                    diff = a.p - b.p
                    d = norm(diff)
                    if d < min_dist:
                        direction = unit(diff, fallback=np.array([1.0, 0.0, 0.0]))
                        shift = 0.5 * (min_dist - d) * direction
                        a.p += shift
                        b.p -= shift
                        a.v *= 0.92
                        b.v *= 0.92
                        if a.history:
                            a.history[-1] = a.p.copy()
                        if b.history:
                            b.history[-1] = b.p.copy()

        # Повторно ограничиваем периметр после разведения пары.
        for drone in drones:
            drone.p[0] = float(np.clip(drone.p[0], self.x_min, self.x_max))
            drone.p[1] = float(np.clip(drone.p[1], self.y_min, self.y_max))
            if drone.history:
                drone.history[-1] = drone.p.copy()

    def compute_clearance_metrics(self, drones: List[Drone3D]) -> Tuple[float, float]:
        """Возвращает минимальный зазор до внутренних препятствий и минимальную дистанцию между дронами."""
        inner_obstacles = [o for o in self.obstacles if not o.is_boundary]
        if inner_obstacles:
            min_clearance = min(
                cylinder_clearance_3d(drone.p, obs, drone.radius)
                for drone in drones
                for obs in inner_obstacles
            )
        else:
            min_clearance = float("inf")

        min_pair_distance = min(
            norm(a.p - b.p)
            for i, a in enumerate(drones)
            for b in drones[i + 1:]
        )
        return float(min_clearance), float(min_pair_distance)

    def step(self, drones: List[Drone3D], t: float) -> Dict[str, float | str]:
        center, direction = self.sample_center(t * self.center_speed)
        mode = self.choose_mode(center, direction)

        desired_alt = self.target_z
        if mode == "overflight":
            z_req = self.nearby_low_obstacle_height(center)
            if z_req is not None:
                desired_alt = z_req

        center = center.copy()
        center[2] = desired_alt
        desired = {role: self.slot_position(center, direction, role, mode) for role in ROLES}

        total_force_norm = 0.0

        for drone in drones:
            f_form = self.formation_force(drone, desired[drone.role], center, drones)
            f_obs, _ = self.obstacle_force(drone, direction)
            f_damp = -self.k_damp * drone.v
            f = clamp_norm(f_form + f_obs + f_damp, drone.max_force)

            total_force_norm += norm(f)
            drone.apply_force(f, self.dt)

        # Важно: метрики считаются после коррекции ограничений, а не до неё.
        self.enforce_constraints(drones)

        formation_errors = [norm(drone.p - desired[drone.role]) for drone in drones]
        min_clearance, min_pair_distance = self.compute_clearance_metrics(drones)

        return {
            "time": t,
            "mode": mode,
            "center_x": center[0],
            "center_y": center[1],
            "center_z": center[2],
            "formation_error": float(np.mean(formation_errors)),
            "min_obstacle_clearance": float(min_clearance),
            "min_pair_distance": float(min_pair_distance),
            "total_energy": float(sum(d.energy for d in drones)),
            "total_path_length": float(sum(d.path_length for d in drones)),
            "avg_force_norm": float(total_force_norm / len(drones)),
        }


def make_drones(controller: AdaptiveFormationController3D) -> List[Drone3D]:
    center, direction = controller.sample_center(0.0)
    drones = []
    for role in ROLES:
        p = controller.slot_position(center, direction, role, "normal")
        drones.append(Drone3D(role=role, p=p, v=np.zeros(3)))
    return drones


def run_experiment(scenario: Scenario, adaptive: bool, out_dir: Path, make_plots: bool = True) -> Tuple[pd.DataFrame, List[Drone3D], AdaptiveFormationController3D]:
    controller = AdaptiveFormationController3D(scenario=scenario, adaptive=adaptive)
    drones = make_drones(controller)

    rows = []
    max_steps = 2200
    for k in range(max_steps):
        t = k * controller.dt
        row = controller.step(drones, t)
        rows.append(row)
        center = np.array([row["center_x"], row["center_y"], row["center_z"]])
        if norm(center - controller.goal_center) < 0.12 and k > 200:
            break

    df = pd.DataFrame(rows)
    strategy = "adaptive" if adaptive else "fixed"
    scenario_dir = out_dir / scenario.name / strategy
    scenario_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(scenario_dir / "metrics.csv", index=False)

    if make_plots:
        save_trajectory_plot(drones, scenario.obstacles, controller.path, scenario_dir / "trajectory_3d.png", f"{scenario.title} / {strategy}", scenario)
        save_top_view_plot(drones, scenario.obstacles, controller.path, scenario_dir / "top_view.png", f"{scenario.title} / {strategy}", scenario)
        save_metric_plot(df, ["formation_error", "min_pair_distance", "min_obstacle_clearance"], scenario_dir / "safety_metrics.png", f"Метрики безопасности / {scenario.name} / {strategy}")
        save_mode_plot(df, scenario_dir / "mode_timeline.png", f"Режимы движения / {scenario.name} / {strategy}")

    return df, drones, controller


def draw_cylinder(ax, obs: ObstacleCylinder, alpha: float = 0.20, safety: bool = False) -> None:
    radius = obs.r + (0.35 if safety else 0.0)
    if obs.is_boundary:
        color = COLORS["boundary_safety"] if safety else COLORS["boundary"]
        edgecolor = COLORS["boundary"]
    else:
        color = COLORS["safety"] if safety else (COLORS["low_obstacle"] if obs.z_max < 1.0 else COLORS["tall_obstacle"])
        edgecolor = COLORS["safety"] if safety else COLORS["obstacle_edge"]

    theta = np.linspace(0, 2 * np.pi, 44)
    z = np.linspace(obs.z_min, obs.z_max, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)

    x_grid = obs.x + radius * np.cos(theta_grid)
    y_grid = obs.y + radius * np.sin(theta_grid)

    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=0.08 if safety else alpha, linewidth=0, shade=True)

    xt = obs.x + radius * np.cos(theta)
    yt = obs.y + radius * np.sin(theta)
    ax.plot(xt, yt, np.full_like(theta, obs.z_max), color=edgecolor, linewidth=0.9, alpha=0.8)
    ax.plot(xt, yt, np.full_like(theta, obs.z_min), color=edgecolor, linewidth=0.7, alpha=0.5)


def draw_drone(ax, pos: np.ndarray, direction: np.ndarray, role: str) -> None:
    color = COLORS.get(role, "#333333")
    label = "L" if role == "leader" else role[-1]
    x, y, z = pos
    scale = 0.16 if role != "leader" else 0.20
    ax.scatter([x], [y], [z], color=color, s=58 if role == "leader" else 42, depthshade=True)

    d = unit(direction, fallback=np.array([1.0, 0.0, 0.0]))
    v = np.array([d[0], d[1]])
    if np.linalg.norm(v) < 1e-9:
        v = np.array([1.0, 0.0])
    v = v / np.linalg.norm(v)
    perp = np.array([-v[1], v[0]])

    a1 = np.array([x, y]) + scale * v
    b1 = np.array([x, y]) - scale * v
    a2 = np.array([x, y]) + scale * perp
    b2 = np.array([x, y]) - scale * perp

    ax.plot([a1[0], b1[0]], [a1[1], b1[1]], [z, z], color=color, linewidth=1.8)
    ax.plot([a2[0], b2[0]], [a2[1], b2[1]], [z, z], color=color, linewidth=1.8)
    for p2 in (a1, b1, a2, b2):
        ax.scatter([p2[0]], [p2[1]], [z], color=color, s=16, alpha=0.95)
    ax.text(x, y, z + 0.09, label, color=color, fontsize=8, weight="bold")


def save_trajectory_plot(drones, obstacles, center_path, out, title, scenario) -> None:
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    for obs in obstacles:
        if obs.is_boundary:
            continue
        draw_cylinder(ax, obs, safety=False)

    cp = np.array(center_path)
    ax.plot(cp[:, 0], cp[:, 1], cp[:, 2], color=COLORS["center_path"], linewidth=2.5, linestyle="--", alpha=0.9, label="center path")

    for drone in drones:
        h = np.array(drone.history)
        color = COLORS.get(drone.role, "#333333")
        ax.plot(h[:, 0], h[:, 1], h[:, 2], color=color, linewidth=1.8, alpha=0.82, label=drone.role)
        ax.scatter(h[0, 0], h[0, 1], h[0, 2], color=COLORS["start"], s=24)
        ax.scatter(h[-1, 0], h[-1, 1], h[-1, 2], color=color, marker="x", s=58)
        direction = h[-1] - h[-2] if len(h) > 1 else np.array([1.0, 0.0, 0.0])
        draw_drone(ax, h[-1], direction, drone.role)

    ax.scatter([scenario.goal[0]], [scenario.goal[1]], [scenario.goal[2]], color=COLORS["goal"], s=140, marker="*", label="goal")
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-8, 8)
    ax.set_ylim(-6.7, 6.7)
    ax.set_zlim(0, 2.7)
    ax.view_init(elev=25, azim=-62)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_top_view_plot(drones, obstacles, center_path, out, title, scenario) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))

    for obs in obstacles:
        if obs.is_boundary:
            body = plt.Circle((obs.x, obs.y), obs.r, color=COLORS["boundary"], alpha=0.65)
            ax.add_patch(body)
            continue

        safety = plt.Circle((obs.x, obs.y), obs.r + 0.35, color=COLORS["safety"], alpha=0.13)
        body = plt.Circle((obs.x, obs.y), obs.r, color=COLORS["low_obstacle"] if obs.z_max < 1.0 else COLORS["tall_obstacle"], alpha=0.5)
        ax.add_patch(safety)
        ax.add_patch(body)
        ax.text(obs.x, obs.y, f"{obs.z_max:.1f}", ha="center", va="center", fontsize=7)

    cp = np.array(center_path)
    ax.plot(cp[:, 0], cp[:, 1], color=COLORS["center_path"], linewidth=2.4, linestyle="--", label="center path")

    for drone in drones:
        h = np.array(drone.history)
        color = COLORS.get(drone.role, "#333333")
        ax.plot(h[:, 0], h[:, 1], color=color, linewidth=1.8, label=drone.role)
        ax.scatter(h[-1, 0], h[-1, 1], color=color, s=42)

    ax.scatter([scenario.start[0]], [scenario.start[1]], color=COLORS["start"], s=70, marker="o", label="start")
    ax.scatter([scenario.goal[0]], [scenario.goal[1]], color=COLORS["goal"], s=140, marker="*", label="goal")
    ax.set_title(title + " / вид сверху")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-8, 8)
    ax.set_ylim(-6.7, 6.7)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_metric_plot(df, columns, out, title) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in columns:
        ax.plot(df["time"], df[col], label=col, linewidth=2)
    ax.set_xlabel("time")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_mode_plot(df, out, title) -> None:
    mode_map = {"normal": 0, "compressed": 1, "column": 2, "overflight": 3}
    y = df["mode"].map(mode_map).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.step(df["time"], y, where="post", linewidth=2.3)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["normal", "compressed", "column", "overflight"])
    ax.set_xlabel("time")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def animate_fast(drones, obstacles, center_path, df, scenario, title) -> None:
    histories = {d.role: np.array(d.history) for d in drones}
    n_frames = min(len(df), min(len(h) for h in histories.values()))
    frames = list(range(0, n_frames, 12))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for obs in obstacles:
        if obs.is_boundary:
            continue
        draw_cylinder(ax, obs, safety=False)

    cp = np.array(center_path)
    ax.plot(cp[:, 0], cp[:, 1], cp[:, 2], color=COLORS["center_path"], linewidth=2.2, linestyle="--", alpha=0.75)

    line_map = {}
    point_map = {}
    text_map = {}

    for role, h in histories.items():
        color = COLORS.get(role, "#333333")
        line, = ax.plot([], [], [], color=color, linewidth=2.0, alpha=0.9)
        point, = ax.plot([], [], [], marker="o", markersize=7 if role == "leader" else 5, linestyle="None", color=color)
        label = "L" if role == "leader" else role[-1]
        text = ax.text(0, 0, 0, label, color=color, fontsize=8, weight="bold")
        line_map[role] = line
        point_map[role] = point
        text_map[role] = text

    ax.scatter([scenario.goal[0]], [scenario.goal[1]], [scenario.goal[2]], color=COLORS["goal"], s=130, marker="*")
    info = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
    trail_len = 45

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-8, 8)
    ax.set_ylim(-6.7, 6.7)
    ax.set_zlim(0, 2.7)
    ax.view_init(elev=25, azim=-62)

    def update(frame_idx: int):
        i0 = max(0, frame_idx - trail_len)
        for role, h in histories.items():
            seg = h[i0:frame_idx + 1]
            pos = h[frame_idx]
            line_map[role].set_data(seg[:, 0], seg[:, 1])
            line_map[role].set_3d_properties(seg[:, 2])
            point_map[role].set_data([pos[0]], [pos[1]])
            point_map[role].set_3d_properties([pos[2]])
            text_map[role].set_position((pos[0], pos[1]))
            text_map[role].set_3d_properties(pos[2] + 0.08)

        row = df.iloc[min(frame_idx, len(df) - 1)]
        info.set_text(
            f"t={row['time']:.1f}s | mode={row['mode']} | "
            f"e_form={row['formation_error']:.2f} | "
            f"d_pair={row['min_pair_distance']:.2f} | "
            f"clearance={row['min_obstacle_clearance']:.2f}"
        )
        return []

    ani = FuncAnimation(fig, update, frames=frames, interval=28, blit=False, repeat=True)
    fig._ani_ref = ani
    fig.tight_layout()
    plt.show()


def summarize(df: pd.DataFrame, scenario: str, strategy: str) -> Dict[str, object]:
    return {
        "scenario": scenario,
        "strategy": strategy,
        "mean_formation_error": df["formation_error"].mean(),
        "min_pair_distance": df["min_pair_distance"].min(),
        "min_obstacle_clearance": df["min_obstacle_clearance"].min(),
        "final_energy": df["total_energy"].iloc[-1],
        "final_path_length": df["total_path_length"].iloc[-1],
        "last_mode": df["mode"].iloc[-1],
        "max_center_z": df["center_z"].max(),
        "n_steps": len(df),
    }


def run_all(out_dir: Path, selected_scenario: Optional[str] = None, show: bool = False, make_plots: bool = True) -> None:
    scenarios = make_scenarios()

    if selected_scenario is not None:
        if selected_scenario not in scenarios:
            raise ValueError(f"Unknown scenario: {selected_scenario}. Available: {list(scenarios)}")
        scenarios = {selected_scenario: scenarios[selected_scenario]}

    summaries = []
    replay = None

    for scenario in scenarios.values():
        for adaptive in (False, True):
            strategy = "adaptive" if adaptive else "fixed"
            df, drones, controller = run_experiment(scenario, adaptive=adaptive, out_dir=out_dir, make_plots=make_plots)
            summaries.append(summarize(df, scenario.name, strategy))
            if adaptive:
                replay = (df, drones, controller, scenario)

    summary_df = pd.DataFrame(summaries)
    out_dir.mkdir(exist_ok=True)
    summary_df.to_csv(out_dir / "summary_all.csv", index=False)

    print(summary_df.to_string(index=False))
    print(f"\nSaved all results to: {out_dir.resolve()}")

    if show and replay is not None:
        df, drones, controller, scenario = replay
        animate_fast(
            drones=drones,
            obstacles=scenario.obstacles,
            center_path=controller.path,
            df=df,
            scenario=scenario,
            title=f"3D animation / {scenario.title} / adaptive",
        )


def export_defense(out_dir: Path) -> None:
    scenario = make_scenarios()["defense_showcase"]

    figures_dir = out_dir / "figures"
    metrics_dir = out_dir / "metrics"
    animations_dir = out_dir / "animations"
    logs_dir = out_dir / "logs"
    for d in (figures_dir, metrics_dir, animations_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    controller = AdaptiveFormationController3D(scenario=scenario, adaptive=True)
    drones = make_drones(controller)

    rows = []
    for k in range(2200):
        t = k * controller.dt
        row = controller.step(drones, t)
        rows.append(row)
        center = np.array([row["center_x"], row["center_y"], row["center_z"]])
        if norm(center - controller.goal_center) < 0.12 and k > 200:
            break

    df = pd.DataFrame(rows)
    saved: List[Path] = []

    traj = figures_dir / "defense_trajectory_3d.png"
    save_trajectory_plot(drones, scenario.obstacles, controller.path, traj, scenario.title, scenario)
    saved.append(traj)

    top = figures_dir / "defense_top_view.png"
    save_top_view_plot(drones, scenario.obstacles, controller.path, top, scenario.title, scenario)
    saved.append(top)

    panel = figures_dir / "defense_metrics_panel.png"
    save_metric_plot(
        df,
        ["formation_error", "min_pair_distance", "min_obstacle_clearance"],
        panel,
        "Метрики безопасности / defense_showcase / adaptive",
    )
    saved.append(panel)

    timeline = figures_dir / "defense_mode_timeline.png"
    save_mode_plot(df, timeline, "Режимы движения / defense_showcase / adaptive")
    saved.append(timeline)

    metrics_csv = metrics_dir / "defense_metrics.csv"
    df.to_csv(metrics_csv, index=False)
    saved.append(metrics_csv)

    summary_csv = metrics_dir / "defense_summary.csv"
    pd.DataFrame([summarize(df, "defense_showcase", "adaptive")]).to_csv(summary_csv, index=False)
    saved.append(summary_csv)

    log_path = logs_dir / "defense_run_log.txt"
    mode_counts = df["mode"].value_counts()
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write("scenario: defense_showcase / adaptive\n")
        fh.write(f"n_steps:               {len(df)}\n")
        fh.write(f"max_center_z:          {df['center_z'].max():.4f}\n")
        fh.write(f"min_obstacle_clearance:{df['min_obstacle_clearance'].min():.6f}\n")
        fh.write(f"min_pair_distance:     {df['min_pair_distance'].min():.6f}\n")
        fh.write(f"mean_formation_error:  {df['formation_error'].mean():.6f}\n")
        fh.write(f"final_energy:          {df['total_energy'].iloc[-1]:.4f}\n")
        fh.write(f"final_path_length:     {df['total_path_length'].iloc[-1]:.4f}\n")
        fh.write("\nmode counts:\n")
        for mode, cnt in mode_counts.items():
            fh.write(f"  {mode}: {cnt}\n")
    saved.append(log_path)

    print("\nSaved files:")
    for p in saved:
        print(f"  {p.resolve()}")


def export_defense_animation(out_dir: Path) -> None:
    import shutil
    from matplotlib.animation import PillowWriter

    scenario = make_scenarios()["defense_showcase"]
    animations_dir = out_dir / "animations"
    animations_dir.mkdir(parents=True, exist_ok=True)

    controller = AdaptiveFormationController3D(scenario=scenario, adaptive=True)
    drones = make_drones(controller)

    rows = []
    for k in range(2200):
        t = k * controller.dt
        row = controller.step(drones, t)
        rows.append(row)
        center = np.array([row["center_x"], row["center_y"], row["center_z"]])
        if norm(center - controller.goal_center) < 0.12 and k > 200:
            break

    df = pd.DataFrame(rows)
    histories = {d.role: np.array(d.history) for d in drones}
    n_frames = min(len(df), min(len(h) for h in histories.values()))

    stride = 8
    frame_indices = list(range(0, n_frames, stride))
    trail_len = 40

    # ── статические элементы (рисуются один раз) ──────────────────────────
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    for obs in scenario.obstacles:
        if not obs.is_boundary:
            draw_cylinder(ax, obs, safety=False)

    cp = np.array(controller.path)
    ax.plot(cp[:, 0], cp[:, 1], cp[:, 2],
            color=COLORS["center_path"], linewidth=2.0, linestyle="--", alpha=0.60)
    ax.scatter([scenario.goal[0]], [scenario.goal[1]], [scenario.goal[2]],
               color=COLORS["goal"], s=130, marker="*", zorder=5)

    ax.set_title(scenario.title, fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_xlim(-8, 8); ax.set_ylim(-6.7, 6.7); ax.set_zlim(0, 2.7)
    ax.view_init(elev=25, azim=-62)

    # ── динамические элементы ─────────────────────────────────────────────
    line_map: Dict[str, object] = {}
    point_map: Dict[str, object] = {}

    for role in ROLES:
        color = COLORS.get(role, "#333333")
        line, = ax.plot([], [], [], color=color, linewidth=1.8, alpha=0.85)
        point, = ax.plot([], [], [], marker="o",
                         markersize=8 if role == "leader" else 5,
                         linestyle="None", color=color, label=role)
        line_map[role] = line
        point_map[role] = point

    info = ax.text2D(
        0.02, 0.97, "",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        family="monospace",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.75},
    )

    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    def update(fi: int):
        i0 = max(0, fi - trail_len)
        for role in ROLES:
            h = histories[role]
            seg = h[i0: fi + 1]
            pos = h[fi]
            line_map[role].set_data(seg[:, 0], seg[:, 1])
            line_map[role].set_3d_properties(seg[:, 2])
            point_map[role].set_data([pos[0]], [pos[1]])
            point_map[role].set_3d_properties([pos[2]])

        row = df.iloc[min(fi, len(df) - 1)]
        info.set_text(
            f"t={row['time']:.1f}s   mode={row['mode']}\n"
            f"form_err={row['formation_error']:.3f}   "
            f"pair_dist={row['min_pair_distance']:.3f}\n"
            f"clearance={row['min_obstacle_clearance']:.3f}"
        )
        return []

    ani = FuncAnimation(fig, update, frames=frame_indices, interval=55, blit=False)

    saved: List[Path] = []

    # ── GIF (всегда) ──────────────────────────────────────────────────────
    gif_path = animations_dir / "defense_main_animation.gif"
    print("Saving GIF …")
    ani.save(str(gif_path), writer=PillowWriter(fps=15), dpi=100)
    saved.append(gif_path)

    # ── MP4 (если есть ffmpeg) ────────────────────────────────────────────
    mp4_path = animations_dir / "defense_main_animation.mp4"
    if shutil.which("ffmpeg"):
        try:
            from matplotlib.animation import FFMpegWriter
            print("Saving MP4 …")
            ani.save(str(mp4_path), writer=FFMpegWriter(fps=25, bitrate=1800), dpi=140)
            saved.append(mp4_path)
        except Exception as exc:
            print(f"Warning: MP4 export failed: {exc}")
    else:
        print("Warning: ffmpeg not found — MP4 skipped.")

    plt.close(fig)

    print("\nSaved animation files:")
    for p in saved:
        print(f"  {p.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=None, help="Run only one scenario")
    parser.add_argument("--show", action="store_true", help="Show fast interactive animation for adaptive strategy")
    parser.add_argument("--out", default="results_3d_extended", help="Output directory")
    parser.add_argument("--no-plots", action="store_true", help="Only save CSV metrics, skip PNG figures")
    parser.add_argument("--export-defense", action="store_true", help="Export defense_showcase static materials to results_defense_showcase/")
    parser.add_argument("--export-defense-animation", action="store_true", help="Export defense_showcase animation (GIF + MP4) to results_defense_showcase/animations/")
    args = parser.parse_args()

    if args.export_defense:
        export_defense(Path("results_defense_showcase"))
    elif args.export_defense_animation:
        export_defense_animation(Path("results_defense_showcase"))
    else:
        run_all(out_dir=Path(args.out), selected_scenario=args.scenario, show=args.show, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
