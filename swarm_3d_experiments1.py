
"""
swarm_3d_experiments.py

Расширенная 3D-симуляция для ВКР:
    «Моделирование группового движения в среде с препятствиями»

Модель агента (система второго порядка):
    p_dot_i = v_i
    m_i * v_dot_i = F_i

Суммарная сила управления (пять компонент):
    F_i = F_i^target + F_i^formation + F_i^obstacle + F_i^damping + F_i^height

    F_i^target   — движение к цели (притяжение к виртуальному центру, горизонтально)
    F_i^formation = F_i^pos + F_i^sep + F_i^coh
        F_i^pos  — возврат к желаемому положению в строю (смещение от центра)
        F_i^sep  — разведение агентов при малом расстоянии
        F_i^coh  — удержание связности группы
    F_i^obstacle = F_i^rep + F_i^tan
        F_i^rep  — нормальная отталкивающая компонента
        F_i^tan  — касательная компонента обхода
    F_i^damping  — демпфирование скорости
    F_i^height   — стабилизация высоты:
                   k_z * (z_des - z_i) * e_z - beta_z * v_z * e_z

Четыре режима движения:
    normal     — базовая формация
    compressed — сжатая формация
    column     — движение колонной
    overflight — перелёт низких препятствий сверху

Иерархия выбора режима (адаптивная стратегия):
    D_m(t) = min_i dist(p_i^des(t, m), O)
    normal     если D_normal(t) > d_safe
    compressed если D_compressed(t) > d_safe
    overflight если боковой обход затруднён и препятствие допускает перелёт
    column     иначе

Сценарии:
    free                  — свободное движение без препятствий
    single                — одиночные препятствия
    wide_barrier          — широкая полоса, боковой облёт
    dense_field           — плотная среда препятствий
    checkerboard_equal    — шахматное поле препятствий одинаковой высоты
    checkerboard_unequal  — шахматное поле препятствий разной высоты
    low_wall_overflight   — низкая стенка, перелёт сверху
    tall_wall_side        — высокая стенка, боковой облёт
    vertical_exit         — единственный выход через верх
    defense_showcase      — демонстрация всех четырёх режимов

Ограждение по периметру рабочей области [-6,6] x [-6,6].

Запуск:
    python swarm_3d_experiments1.py
    python swarm_3d_experiments1.py --scenario wide_barrier --show
    python swarm_3d_experiments1.py --scenario defense_showcase --show
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
    # Если задано, переопределяет автоматическое планирование пути виртуального центра.
    # Используется в showcase-сценариях для гарантированного прохода через заданные зоны.
    waypoints: Optional[List[np.ndarray]] = None


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
    """Минимальное расстояние от сферы дрона до поверхности цилиндра.

    Формула учитывает три зоны:
      - дрон выше верхней крышки: расстояние до плоского верха или ребра;
      - дрон ниже нижней крышки: симметрично;
      - дрон на уровне цилиндра: горизонтальное расстояние до боковой поверхности.
    """
    dist_2d = float(np.linalg.norm(p[:2] - obs.center_xy))
    h_surf = dist_2d - obs.r  # от центра дрона до боковой поверхности (без радиуса дрона)

    if p[2] > obs.z_max:
        vert = p[2] - obs.z_max
        # Над цилиндром: если горизонтально снаружи — евклидово до верхнего ребра,
        # если прямо над крышкой — только вертикаль.
        dist_to_surface = float(math.sqrt(h_surf * h_surf + vert * vert)) if h_surf >= 0 else vert
    elif p[2] < obs.z_min:
        vert = obs.z_min - p[2]
        dist_to_surface = float(math.sqrt(h_surf * h_surf + vert * vert)) if h_surf >= 0 else vert
    else:
        # На уровне цилиндра: только горизонтальная компонента
        return h_surf - drone_radius

    return dist_to_surface - drone_radius


def row_of_obstacles(x: float, y_values: List[float], radius: float, z_max: float, prefix: str) -> List[ObstacleCylinder]:
    return [ObstacleCylinder(x=x, y=y, r=radius, z_max=z_max, name=f"{prefix}_{i}") for i, y in enumerate(y_values)]


def make_perimeter_fence(
    x_min: float = -6.0,
    x_max: float = 6.0,
    y_min: float = -6.0,
    y_max: float = 6.0,
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
    """
    Добавляет ограждение ко всем сценариям.

    Ограждение расположено по краям области [-6,6] x [-6,6].
    Визуально оно будет видно на графиках, а в расчётах используется
    как дополнительный набор препятствий.
    """
    return obstacles + make_perimeter_fence()


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

    # --- шахматное поле одинаковой высоты ---
    cb_eq_positions = [
        (-1.8, -1.0), (-0.6, -1.0), (0.6, -1.0), (1.8, -1.0),
        (-1.2,  0.0), ( 0.0,  0.0), (1.2,  0.0),
        (-1.8,  1.0), (-0.6,  1.0), (0.6,  1.0), (1.8,  1.0),
    ]
    scenarios["checkerboard_equal"] = Scenario(
        name="checkerboard_equal",
        title="Шахматное поле препятствий одинаковой высоты",
        obstacles=with_fence([
            ObstacleCylinder(x, y, 0.28, z_max=2.2, name=f"cb_eq_{i}")
            for i, (x, y) in enumerate(cb_eq_positions)
        ]),
        start=start,
        goal=goal,
        allow_overflight=False,
        description="Препятствия одинаковой высоты в шахматном порядке: группа выбирает боковые проходы.",
    )

    # --- шахматное поле разной высоты ---
    cb_uneq = [
        (-1.8, -1.0, 2.2), (-0.6, -1.0, 0.72), (0.6, -1.0, 2.2), (1.8, -1.0, 0.72),
        (-1.2,  0.0, 0.72), (0.0,  0.0, 2.2),  (1.2,  0.0, 0.72),
        (-1.8,  1.0, 0.72), (-0.6,  1.0, 2.2), (0.6,  1.0, 0.72), (1.8,  1.0, 2.2),
    ]
    scenarios["checkerboard_unequal"] = Scenario(
        name="checkerboard_unequal",
        title="Шахматное поле препятствий разной высоты",
        obstacles=with_fence([
            ObstacleCylinder(x, y, 0.28, z_max=z, name=f"cb_uneq_{i}")
            for i, (x, y, z) in enumerate(cb_uneq)
        ]),
        start=start,
        goal=goal,
        preferred_altitude=1.20,
        allow_overflight=True,
        description="Чередование высоких и низких препятствий: группа может перелетать низкие.",
    )

    # --- единственный выход через верх (vertical exit) ---
    # Стена низких препятствий перекрывает всю ширину включая зону ограждения:
    # боковой облёт невозможен, единственный выход — перелёт сверху.
    scenarios["vertical_exit"] = Scenario(
        name="vertical_exit",
        title="Единственный выход — через верх",
        obstacles=with_fence(
            row_of_obstacles(
                0.0,
                [-4.0, -3.2, -2.4, -1.6, -0.8, 0.0, 0.8, 1.6, 2.4, 3.2, 4.0],
                0.34, 0.72, "low_wall_wide",
            )
        ),
        start=np.array([-5.4, 0.0, 1.05]),
        goal=np.array([5.4, 0.0, 1.05]),
        preferred_altitude=1.05,
        allow_overflight=True,
        description="Низкая стена перекрывает все боковые проходы — группа обязана перелетать.",
    )

    # --- демонстрационный сценарий со сменой всех четырёх режимов ---
    #
    # Геометрия коридоров (проверена аналитически):
    #   Умеренное сужение (y_wall=1.20, r=0.40):
    #     mode_clearance normal:     1.20 - 0.80 - 0.40 - 0.52 = -0.52 < 0.14  → normal fails
    #     mode_clearance compressed: 1.20 - 0.36 - 0.40 - 0.52 = -0.08 < 0.10  → compressed fails
    #     Хм, нужен wider... используем y_wall=1.40:
    #     mode_clearance normal:     1.40 - 0.80 - 0.40 - 0.52 = -0.32 < 0.14  → fails
    #     mode_clearance compressed: 1.40 - 0.36 - 0.40 - 0.52 = 0.12 > 0.10   → passes!
    #
    #   Жёсткое сужение (y_wall=1.20, r=0.40):
    #     mode_clearance compressed: 1.20 - 0.36 - 0.40 - 0.52 = -0.08 < 0.10  → fails
    #     mode_clearance column:     1.20 - 0.22 - 0.40 - 0.52 = 0.06 > 0      → passes!
    #
    #   Низкая стена (z_max=0.72, drone_z=1.20):
    #     cylinder_clearance_3d vertical: 1.20 - 0.72 - 0.52 = -0.04 < 0       → normal/compressed fail
    #     nearby_low_obstacle_height: 0.72 + 0.55 = 1.27 ≤ 2.15               → overflight!
    #
    # Маршрут waypoints проводит виртуальный центр СКВОЗЬ коридоры (не вокруг),
    # что гарантирует срабатывание нужных режимов.
    _ds_obs = [
        # умеренный коридор → compressed
        ObstacleCylinder(-2.4,  1.40, 0.40, z_max=2.2, name="moderate_top"),
        ObstacleCylinder(-2.4, -1.40, 0.40, z_max=2.2, name="moderate_bot"),
        # жёсткий коридор → column
        ObstacleCylinder(-0.6,  1.20, 0.40, z_max=2.2, name="tight_top"),
        ObstacleCylinder(-0.6, -1.20, 0.40, z_max=2.2, name="tight_bot"),
        # низкие препятствия → overflight
        ObstacleCylinder(1.5, -0.8, 0.34, z_max=0.72, name="low_left"),
        ObstacleCylinder(1.5,  0.0, 0.34, z_max=0.72, name="low_mid"),
        ObstacleCylinder(1.5,  0.8, 0.34, z_max=0.72, name="low_right"),
    ]
    scenarios["defense_showcase"] = Scenario(
        name="defense_showcase",
        title="Демонстрация всех режимов движения",
        obstacles=with_fence(_ds_obs),
        start=np.array([-5.4, 0.0, 1.20]),
        goal=np.array([5.4, 0.0, 1.20]),
        preferred_altitude=1.20,
        allow_overflight=True,
        description="Последовательная демонстрация: normal → compressed → column → overflight → normal.",
        # Принудительный маршрут сквозь коридоры гарантирует срабатывание всех режимов.
        # z при overflight = 0.72 + 0.55 = 1.27 (выше верхнего края низких препятствий).
        waypoints=[
            np.array([-5.4, 0.0, 1.20]),
            np.array([-3.8, 0.0, 1.20]),   # до умеренного коридора → normal
            np.array([-2.4, 0.0, 1.20]),   # умеренный коридор → compressed
            np.array([-0.6, 0.0, 1.20]),   # жёсткий коридор → column
            np.array([ 0.6, 0.0, 1.27]),   # подъём перед низкой стеной
            np.array([ 1.5, 0.0, 1.27]),   # над низкими препятствиями → overflight
            np.array([ 2.5, 0.0, 1.27]),   # выход из зоны низких препятствий
            np.array([ 3.5, 0.0, 1.20]),   # снижение → normal
            np.array([ 5.4, 0.0, 1.20]),
        ],
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
        self.max_altitude = 2.15
        self.overflight_margin = 0.55

        # --- коэффициенты пяти компонент силы ---
        # F_target: движение к цели (горизонтальное притяжение к виртуальному центру)
        self.k_target = 2.65
        # F_formation: F_pos + F_sep + F_coh
        self.k_pos = 2.65   # коррекция к слоту в строю (смещение от центра)
        self.k_sep = 1.9    # разведение агентов
        self.k_coh = 0.07   # связность группы
        # F_obstacle: F_rep + F_tan
        self.k_obs = 0.85   # нормальная отталкивающая компонента
        self.k_tan = 0.72   # касательная компонента обхода
        # F_damping: демпфирование скорости
        self.k_damp = 1.18
        # F_height: стабилизация высоты (k_z*(z_des-z)*e_z - beta_z*v_z*e_z)
        self.k_z = 3.0
        self.beta_z = 1.2

        self.drone_radius = 0.24
        self.d_safe = 0.64
        self.d_detect = 1.28
        self.obs_detect = 0.95
        self.obs_safe = 0.32

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
        """
        Иерархия выбора режима (ВКР §4.2):
            D_m(t) = min_i dist(p_i^des(t, m), O)
            normal     если D_normal > d_safe
            compressed если D_compressed > d_safe
            overflight если боковой обход затруднён и препятствие допускает перелёт
            column     иначе
        """
        if not self.adaptive:
            return "normal"
        if self.mode_clearance(center, direction, "normal") > 0.14:
            return "normal"
        if self.mode_clearance(center, direction, "compressed") > 0.10:
            return "compressed"
        # overflight: когда боковой обход затруднён, но препятствие достаточно низкое
        if self.scenario.allow_overflight and self.nearby_low_obstacle_height(center) is not None:
            return "overflight"
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
                    if cylinder_clearance_3d(p, obs, self.drone_radius + inflation) < 0:
                        return False
        return True

    def build_center_path(self) -> List[np.ndarray]:
        # Если waypoints заданы в сценарии — используем их напрямую (showcase-режим).
        if self.scenario.waypoints is not None:
            return [w.copy() for w in self.scenario.waypoints]

        direct = [self.start_center.copy(), self.goal_center.copy()]

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
        """F_i^obstacle = F_i^rep + F_i^tan (нормальная + касательная компоненты)."""
        f = np.zeros(3)
        min_clear = 1e9

        for obs in self.obstacles:
            c3 = cylinder_clearance_3d(drone.p, obs, drone.radius)
            min_clear = min(min_clear, c3)
            if c3 >= self.obs_detect:
                continue

            h_clear = horizontal_clearance(drone.p, obs, drone.radius)
            away2 = drone.p[:2] - obs.center_xy
            d = np.linalg.norm(away2)

            # Боковое отталкивание только если агент на уровне высоты препятствия
            if obs.z_min - drone.radius <= drone.p[2] <= obs.z_max + drone.radius and d > 1e-9:
                n2 = away2 / d
                gap = max(h_clear - self.obs_safe, 0.12)
                gain = min(2.2, self.k_obs / (gap * gap))
                # F_rep: нормальная отталкивающая компонента
                # F_tan: касательная компонента (предотвращает локальный минимум)
                tangent2 = np.array([-n2[1], n2[0]])
                if np.dot(tangent2, direction[:2]) < 0:
                    tangent2 = -tangent2
                f[:2] += gain * n2 + self.k_tan * gain * tangent2

        return f, min_clear

    def target_force(self, drone: Drone3D, center: np.ndarray) -> np.ndarray:
        """F_i^target: горизонтальное притяжение к виртуальному центру группы.

        Обеспечивает движение группы к цели. Вертикальная компонента вынесена
        в height_force, чтобы разделить горизонтальную и вертикальную динамику.
        """
        f = np.zeros(3)
        f[:2] = self.k_target * (center[:2] - drone.p[:2])
        return f

    def height_force(self, drone: Drone3D, z_desired: float) -> np.ndarray:
        """F_i^height = k_z * (z_des - z_i) * e_z - beta_z * v_{z,i} * e_z.

        Стабилизирует высоту агента. При режиме overflight z_des вычисляется
        как h_obs + h_safe (высота препятствия плюс безопасный зазор).
        """
        f = np.zeros(3)
        f[2] = self.k_z * (z_desired - drone.p[2]) - self.beta_z * drone.v[2]
        return f

    def formation_force(
        self,
        drone: Drone3D,
        center: np.ndarray,
        direction: np.ndarray,
        mode: str,
        drones: List[Drone3D],
    ) -> np.ndarray:
        """F_i^formation = F_i^pos + F_i^sep + F_i^coh.

        F_pos: коррекция к желаемому положению в строю. Вычисляется как
               k_pos * R(t) * Delta_i^m — смещение от центра формации,
               без члена «центр → агент» (он вынесен в target_force).
        F_sep: разведение агентов при слишком малом расстоянии.
        F_coh: удержание связности группы (притяжение к центру формации).
        """
        # F_pos = k_pos * R @ Delta_i^m (только геометрическое смещение в строю)
        R_offset = rotation_from_direction(direction) @ self.role_offset(drone.role, mode)
        f_pos = self.k_pos * R_offset

        # F_sep: отталкивание от соседних агентов
        f_sep = np.zeros(3)
        for other in drones:
            if other.role == drone.role:
                continue
            diff = drone.p - other.p
            d = norm(diff)
            if d < self.d_detect:
                direction_sep = unit(diff)
                gap = max(d - self.d_safe, 0.12)
                gain = min(3.2, self.k_sep * (1.0 / gap - 1.0 / (self.d_detect - self.d_safe)) / (gap * gap))
                if gain > 0:
                    f_sep += gain * direction_sep

        # F_coh: притяжение к виртуальному центру (связность группы)
        f_coh = self.k_coh * (center - drone.p)
        return f_pos + f_sep + f_coh

    def step(self, drones: List[Drone3D], t: float) -> Dict[str, float | str]:
        """Один шаг интегрирования. Суммирует все пять компонент силы:
            F_i = F_target + F_formation + F_obstacle + F_damping + F_height
        """
        center, direction = self.sample_center(t * self.center_speed)
        mode = self.choose_mode(center, direction)

        # Желаемая высота: target_z в обычных режимах, h_obs + h_safe при overflight
        desired_alt = self.target_z
        if mode == "overflight":
            z_req = self.nearby_low_obstacle_height(center)
            if z_req is not None:
                desired_alt = z_req

        center = center.copy()
        center[2] = desired_alt
        desired = {role: self.slot_position(center, direction, role, mode) for role in ROLES}

        formation_errors = []
        min_clearance = 1e9
        total_force_norm = 0.0
        max_drone_z = 0.0

        for drone in drones:
            # --- пять компонент силы ---
            f_target = self.target_force(drone, center)
            f_form   = self.formation_force(drone, center, direction, mode, drones)
            f_obs, c = self.obstacle_force(drone, direction)
            f_damp   = -self.k_damp * drone.v
            f_height = self.height_force(drone, desired_alt)

            f = clamp_norm(f_target + f_form + f_obs + f_damp + f_height, drone.max_force)

            # Клиренс считается только до реальных препятствий (не до boundary fence),
            # чтобы метрика отражала именно безопасность прохода через препятствия.
            real_obs_clearance = min(
                (cylinder_clearance_3d(drone.p, obs, drone.radius)
                 for obs in self.obstacles if not obs.is_boundary),
                default=1e9,
            )

            total_force_norm += norm(f)
            min_clearance = min(min_clearance, real_obs_clearance)
            formation_errors.append(norm(drone.p - desired[drone.role]))
            max_drone_z = max(max_drone_z, drone.p[2])
            drone.apply_force(f, self.dt)

        min_agent_distance = min(norm(a.p - b.p) for i, a in enumerate(drones) for b in drones[i + 1:])

        return {
            "time": t,
            "mode": mode,
            "center_x": center[0],
            "center_y": center[1],
            "center_z": center[2],
            "formation_error": float(np.mean(formation_errors)),
            "min_obstacle_clearance": float(min_clearance),
            "min_agent_distance": float(min_agent_distance),
            "energy": float(sum(d.energy for d in drones)),
            "path_length": float(sum(d.path_length for d in drones)),
            "max_height": float(max_drone_z),
            "avg_force_norm": float(total_force_norm / len(drones)),
        }


def make_drones(controller: AdaptiveFormationController3D) -> List[Drone3D]:
    center, direction = controller.sample_center(0.0)
    drones = []
    for role in ROLES:
        p = controller.slot_position(center, direction, role, "normal")
        drones.append(Drone3D(role=role, p=p, v=np.zeros(3)))
    return drones


def run_experiment(scenario: Scenario, adaptive: bool, out_dir: Path) -> Tuple[pd.DataFrame, List[Drone3D], AdaptiveFormationController3D]:
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

    save_trajectory_plot(drones, scenario.obstacles, controller.path, scenario_dir / "trajectory_3d.png", f"{scenario.title} / {strategy}", scenario)
    save_top_view_plot(drones, scenario.obstacles, controller.path, scenario_dir / "top_view.png", f"{scenario.title} / {strategy}", scenario)
    save_metric_plot(df, ["formation_error", "min_agent_distance", "min_obstacle_clearance"], scenario_dir / "safety_metrics.png", f"Метрики безопасности / {scenario.name} / {strategy}")
    save_metric_plot(df, ["max_height"], scenario_dir / "height_profile.png", f"Профиль высоты / {scenario.name} / {strategy}")
    save_mode_plot(df, scenario_dir / "mode_timeline.png", f"Режимы движения / {scenario.name} / {strategy}")

    return df, drones, controller


def draw_cylinder(ax, obs: ObstacleCylinder, alpha: float = 0.28, safety: bool = False) -> None:
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
        if not obs.is_boundary:
            draw_cylinder(ax, obs, safety=True)
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
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
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
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
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
    frames = list(range(0, n_frames, 8))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for obs in obstacles:
        if not obs.is_boundary:
            draw_cylinder(ax, obs, safety=True)
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
    trail_len = 80

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
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
            f"d_agent={row['min_agent_distance']:.2f} | "
            f"clearance={row['min_obstacle_clearance']:.2f} | "
            f"z_max={row['max_height']:.2f}"
        )
        return []

    ani = FuncAnimation(fig, update, frames=frames, interval=35, blit=False, repeat=True)
    fig._ani_ref = ani
    fig.tight_layout()
    plt.show()


def summarize(df: pd.DataFrame, scenario: str, strategy: str) -> Dict[str, object]:
    """Сводная таблица метрик по ВКР:
        formation_error, min_agent_distance, min_obstacle_clearance,
        energy, path_length, max_height.
    """
    return {
        "scenario": scenario,
        "strategy": strategy,
        "mean_formation_error": round(float(df["formation_error"].mean()), 4),
        "min_agent_distance": round(float(df["min_agent_distance"].min()), 4),
        "min_obstacle_clearance": round(float(df["min_obstacle_clearance"].min()), 4),
        "energy": round(float(df["energy"].iloc[-1]), 2),
        "path_length": round(float(df["path_length"].iloc[-1]), 2),
        "max_height": round(float(df["max_height"].max()), 3),
        "modes_used": ",".join(sorted(df["mode"].unique())),
        "last_mode": df["mode"].iloc[-1],
        "n_steps": len(df),
    }


def run_all(out_dir: Path, selected_scenario: Optional[str] = None, show: bool = False) -> None:
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
            df, drones, controller = run_experiment(scenario, adaptive=adaptive, out_dir=out_dir)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=None, help="Run only one scenario")
    parser.add_argument("--show", action="store_true", help="Show fast interactive animation for adaptive strategy")
    parser.add_argument("--out", default="results_3d_extended", help="Output directory")
    args = parser.parse_args()

    run_all(out_dir=Path(args.out), selected_scenario=args.scenario, show=args.show)


if __name__ == "__main__":
    main()
