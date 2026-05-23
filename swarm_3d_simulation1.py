"""
swarm_3d_simulation.py

3D-симуляция для ВКР: «Моделирование группового движения в среде с препятствиями».
Независима от ARGoS, используется для численных экспериментов и иллюстраций.

Модель агента:
    p_dot_i = v_i
    m_i * v_dot_i = F_i

Пять компонент силы управления:
    F_i = F_i^target + F_i^formation + F_i^obstacle + F_i^damping + F_i^height
    F_i^formation = F_i^pos + F_i^sep + F_i^coh
    F_i^obstacle  = F_i^rep + F_i^tan
    F_i^height    = k_z*(z_des - z_i)*e_z - beta_z*v_{z,i}*e_z

Четыре режима движения: normal, compressed, column, overflight.

Run:
    python3 swarm_3d_simulation1.py

Output:
    results_3d/*.csv
    results_3d/*.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ----------------------------- data classes -----------------------------

@dataclass
class ObstacleCylinder:
    x: float
    y: float
    r: float
    z_min: float = 0.0
    z_max: float = 2.2

    @property
    def center_xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


@dataclass
class Drone3D:
    role: str
    p: np.ndarray
    v: np.ndarray
    mass: float = 1.0
    radius: float = 0.24
    max_speed: float = 0.95
    max_force: float = 2.8
    energy: float = 0.0
    path_length: float = 0.0
    history: List[np.ndarray] | None = None

    def __post_init__(self) -> None:
        self.p = self.p.astype(float)
        self.v = self.v.astype(float)
        self.history = [self.p.copy()]

    def apply_force(self, f: np.ndarray, dt: float) -> None:
        fn = np.linalg.norm(f)
        if fn > self.max_force:
            f = f / fn * self.max_force

        a = f / self.mass
        self.energy += float(np.dot(f, f)) * dt
        self.v += a * dt

        speed = np.linalg.norm(self.v)
        if speed > self.max_speed:
            self.v = self.v / speed * self.max_speed

        prev = self.p.copy()
        self.p += self.v * dt
        self.path_length += float(np.linalg.norm(self.p - prev))
        self.history.append(self.p.copy())


# ----------------------------- math helpers -----------------------------

ROLES = ["leader", "follower1", "follower2", "follower3", "follower4"]


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    n = norm(v)
    if n < 1e-9:
        if fallback is None:
            return np.array([1.0, 0.0, 0.0])
        return fallback.copy()
    return v / n


def clamp_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = norm(v)
    if n > max_norm and n > 1e-9:
        return v / n * max_norm
    return v


def rotation_from_direction(d: np.ndarray) -> np.ndarray:
    """Matrix whose local x-axis follows the horizontal route direction."""
    d_xy = np.array([d[0], d[1], 0.0], dtype=float)
    ex = unit(d_xy, fallback=np.array([1.0, 0.0, 0.0]))
    ez = np.array([0.0, 0.0, 1.0])
    ey = np.cross(ez, ex)
    ey = unit(ey, fallback=np.array([0.0, 1.0, 0.0]))
    return np.column_stack([ex, ey, ez])


def cylinder_clearance(p: np.ndarray, obs: ObstacleCylinder, drone_radius: float) -> float:
    """Horizontal clearance from a spherical drone center to a vertical cylinder."""
    d_xy = np.linalg.norm(p[:2] - obs.center_xy)
    return float(d_xy - obs.r - drone_radius)


# ----------------------------- controller -----------------------------

class AdaptiveFormationController3D:
    def __init__(
        self,
        obstacles: List[ObstacleCylinder],
        adaptive: bool = True,
        allow_overflight: bool = False,
    ):
        self.obstacles = obstacles
        self.adaptive = adaptive
        self.allow_overflight = allow_overflight

        self.dt = 0.05
        self.center_speed = 0.34
        self.target_z = 1.20
        self.max_altitude = 2.15
        self.overflight_margin = 0.55

        # --- коэффициенты пяти компонент силы ---
        self.k_target = 2.5    # F_target: движение к цели
        self.k_pos    = 2.5    # F_formation F_pos: коррекция к слоту
        self.k_sep    = 1.8    # F_formation F_sep: разведение агентов
        self.k_coh    = 0.06   # F_formation F_coh: связность группы
        self.k_obs    = 0.9    # F_obstacle F_rep: нормальная компонента
        self.k_tan    = 0.75   # F_obstacle F_tan: касательная компонента
        self.k_damp   = 1.15   # F_damping
        self.k_z      = 3.0    # F_height: жёсткость стабилизации высоты
        self.beta_z   = 1.2    # F_height: демпфирование по высоте

        self.drone_radius = 0.24
        self.d_safe = 0.62
        self.d_detect = 1.25
        self.obs_detect = 0.92
        self.obs_safe = 0.30

        self.start_center = np.array([-5.2, 0.0, self.target_z])
        self.goal_center = np.array([5.3, 0.0, self.target_z])
        self.path = self.build_center_path()
        self.path_length = self.compute_path_length()

    # ---------- formation geometry ----------

    def role_offset(self, role: str, mode: str) -> np.ndarray:
        if mode == "normal":
            offsets = {
                "leader":    (0.00,  0.00, 0.00),
                "follower1": (-0.95,  0.78, 0.00),
                "follower2": (-0.95, -0.78, 0.00),
                "follower3": (-1.85,  0.38, 0.00),
                "follower4": (-1.85, -0.38, 0.00),
            }
        elif mode == "compressed":
            offsets = {
                "leader":    (0.00,  0.00, 0.00),
                "follower1": (-0.82,  0.34, 0.00),
                "follower2": (-1.64, -0.34, 0.00),
                "follower3": (-2.46,  0.30, 0.00),
                "follower4": (-3.28, -0.30, 0.00),
            }
        elif mode == "overflight":
            # Перелёт сверху: небольшое вертикальное разнесение агентов
            offsets = {
                "leader":    (0.00,  0.00,  0.00),
                "follower1": (-0.86,  0.28,  0.08),
                "follower2": (-1.72, -0.28, -0.06),
                "follower3": (-2.58,  0.24,  0.06),
                "follower4": (-3.44, -0.24, -0.08),
            }
        else:  # column
            offsets = {
                "leader":    (0.00,  0.00, 0.00),
                "follower1": (-0.88,  0.22, 0.00),
                "follower2": (-1.76, -0.22, 0.00),
                "follower3": (-2.64,  0.22, 0.00),
                "follower4": (-3.52, -0.22, 0.00),
            }
        return np.array(offsets[role], dtype=float)

    def slot_position(self, center: np.ndarray, direction: np.ndarray, role: str, mode: str) -> np.ndarray:
        r = rotation_from_direction(direction)
        return center + r @ self.role_offset(role, mode)

    def mode_clearance(self, center: np.ndarray, direction: np.ndarray, mode: str) -> float:
        min_clearance = 1e9
        for role in ROLES:
            s = self.slot_position(center, direction, role, mode)
            for obs in self.obstacles:
                c = cylinder_clearance(s, obs, self.drone_radius + 0.28)
                min_clearance = min(min_clearance, c)
        return min_clearance

    def nearby_low_obstacle_height(self, center: np.ndarray) -> Optional[float]:
        if not self.allow_overflight:
            return None
        candidates = []
        for obs in self.obstacles:
            if abs(obs.x - center[0]) < 1.35 and abs(obs.y - center[1]) < 3.6:
                required_z = obs.z_max + self.overflight_margin
                if required_z <= self.max_altitude:
                    candidates.append(required_z)
        return max(candidates) if candidates else None

    def choose_mode(self, center: np.ndarray, direction: np.ndarray) -> str:
        """Иерархия выбора режима: normal → compressed → overflight → column."""
        if not self.adaptive:
            return "normal"
        if self.mode_clearance(center, direction, "normal") > 0.12:
            return "normal"
        if self.mode_clearance(center, direction, "compressed") > 0.08:
            return "compressed"
        if self.allow_overflight and self.nearby_low_obstacle_height(center) is not None:
            return "overflight"
        return "column"

    # ---------- path for the virtual center ----------

    def build_center_path(self) -> List[np.ndarray]:
        # If direct path intersects inflated obstacle set, use wide detour.
        if self.direct_path_is_safe(inflation=1.15):
            return [self.start_center.copy(), self.goal_center.copy()]

        xmin = min(o.x - o.r for o in self.obstacles)
        xmax = max(o.x + o.r for o in self.obstacles)
        ymin = min(o.y - o.r for o in self.obstacles)
        ymax = max(o.y + o.r for o in self.obstacles)
        margin_x = 1.45
        margin_y = 2.0

        y_top = ymax + margin_y
        y_bottom = ymin - margin_y
        top_path = [
            self.start_center.copy(),
            np.array([xmin - margin_x - 0.6, 0.0, self.target_z]),
            np.array([xmin - margin_x, 0.45 * y_top, self.target_z]),
            np.array([xmin - margin_x, y_top, self.target_z]),
            np.array([(xmin + xmax) / 2, y_top, self.target_z]),
            np.array([xmax + margin_x, y_top, self.target_z]),
            np.array([xmax + margin_x, 0.45 * y_top, self.target_z]),
            np.array([xmax + margin_x + 0.6, 0.0, self.target_z]),
            self.goal_center.copy(),
        ]
        bottom_path = [p.copy() for p in top_path]
        for p in bottom_path:
            if abs(p[1]) > 1e-9:
                p[1] = -abs(p[1]) if p[1] > 0 else p[1]
        # Correct bottom path explicitly based on y_bottom
        bottom_path = [
            self.start_center.copy(),
            np.array([xmin - margin_x - 0.6, 0.0, self.target_z]),
            np.array([xmin - margin_x, 0.45 * y_bottom, self.target_z]),
            np.array([xmin - margin_x, y_bottom, self.target_z]),
            np.array([(xmin + xmax) / 2, y_bottom, self.target_z]),
            np.array([xmax + margin_x, y_bottom, self.target_z]),
            np.array([xmax + margin_x, 0.45 * y_bottom, self.target_z]),
            np.array([xmax + margin_x + 0.6, 0.0, self.target_z]),
            self.goal_center.copy(),
        ]
        return top_path if self.path_len(top_path) <= self.path_len(bottom_path) else bottom_path

    def direct_path_is_safe(self, inflation: float) -> bool:
        n = 120
        for k in range(n + 1):
            t = k / n
            p = (1 - t) * self.start_center + t * self.goal_center
            for obs in self.obstacles:
                if cylinder_clearance(p, obs, self.drone_radius + inflation) < 0:
                    return False
        return True

    @staticmethod
    def path_len(path: List[np.ndarray]) -> float:
        return sum(norm(path[i + 1] - path[i]) for i in range(len(path) - 1))

    def compute_path_length(self) -> float:
        return self.path_len(self.path)

    def sample_center(self, s: float) -> Tuple[np.ndarray, np.ndarray]:
        s = min(s, self.path_length)
        rem = s
        for i in range(len(self.path) - 1):
            a, b = self.path[i], self.path[i + 1]
            seg = norm(b - a)
            if rem <= seg:
                tau = rem / max(seg, 1e-9)
                p = (1 - tau) * a + tau * b
                d = unit(b - a)
                return p, d
            rem -= seg
        d = unit(self.path[-1] - self.path[-2])
        return self.path[-1].copy(), d

    # ---------- force components ----------

    def obstacle_force(
        self,
        drone: Drone3D,
        direction: np.ndarray,
        center: Optional[np.ndarray] = None,
        drones_all: Optional[List["Drone3D"]] = None,
    ) -> Tuple[np.ndarray, float]:
        """F_i^obstacle = F_i^rep + F_i^tan.

        Касательная сторона определяется по виртуальному центру группы,
        ТОЛЬКО если препятствие "общее" (видимо несколькими агентами
        одновременно). Иначе используется локальное правило — это
        даёт согласованность на длинных стенах, не теряя свободы при
        одиночных препятствиях.
        """
        f = np.zeros(3)
        min_clear = 1e9

        # Сколько агентов видят каждое препятствие
        crowd_count: Dict[int, int] = {}
        if drones_all is not None:
            for obs_idx, o in enumerate(self.obstacles):
                cnt = 0
                for d_other in drones_all:
                    if cylinder_clearance(d_other.p, o, d_other.radius) < self.obs_detect:
                        cnt += 1
                crowd_count[obs_idx] = cnt

        for obs_idx, obs in enumerate(self.obstacles):
            c = cylinder_clearance(drone.p, obs, drone.radius)
            min_clear = min(min_clear, c)
            if c < self.obs_detect:
                away2 = drone.p[:2] - obs.center_xy
                d = np.linalg.norm(away2)
                if d < 1e-9:
                    continue
                n2 = away2 / d  # локальная нормаль (для F_rep)
                gap = max(c - self.obs_safe, 0.12)
                gain = min(2.0, self.k_obs / (gap * gap))

                # F_tan: согласованная сторона если препятствие общее
                shared = (center is not None and crowd_count.get(obs_idx, 1) >= 2)
                if shared:
                    away_c = center[:2] - obs.center_xy
                    d_c = np.linalg.norm(away_c)
                    n_for_tan = away_c / d_c if d_c > 1e-9 else n2
                else:
                    n_for_tan = n2

                tangent2 = np.array([-n_for_tan[1], n_for_tan[0]])
                if np.dot(tangent2, direction[:2]) < 0:
                    tangent2 = -tangent2
                f[:2] += gain * n2 + self.k_tan * gain * tangent2
        return f, min_clear

    def target_force(self, drone: Drone3D, center: np.ndarray) -> np.ndarray:
        """F_i^target: горизонтальное притяжение к виртуальному центру."""
        f = np.zeros(3)
        f[:2] = self.k_target * (center[:2] - drone.p[:2])
        return f

    def height_force(self, drone: Drone3D, z_desired: float) -> np.ndarray:
        """F_i^height = k_z*(z_des - z_i)*e_z - beta_z*v_{z,i}*e_z."""
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
        """F_i^formation = F_i^pos + F_i^sep + F_i^coh."""
        # F_pos: геометрическое смещение в строю (без члена центр→агент)
        R_offset = rotation_from_direction(direction) @ self.role_offset(drone.role, mode)
        f_pos = self.k_pos * R_offset

        # F_sep: разведение агентов
        f_sep = np.zeros(3)
        for other in drones:
            if other.role == drone.role:
                continue
            diff = drone.p - other.p
            d = norm(diff)
            if d < self.d_detect:
                direction_sep = unit(diff)
                gap = max(d - self.d_safe, 0.12)
                gain = min(3.0, self.k_sep * (1 / gap - 1 / (self.d_detect - self.d_safe)) / (gap * gap))
                if gain > 0:
                    f_sep += gain * direction_sep

        # F_coh: связность группы
        f_coh = self.k_coh * (center - drone.p)
        return f_pos + f_sep + f_coh

    def step(self, drones: List[Drone3D], t: float) -> Dict[str, float | str]:
        """Один шаг интегрирования. Суммирует пять компонент силы:
            F_i = F_target + F_formation + F_obstacle + F_damping + F_height
        """
        center, direction_raw = self.sample_center(t * self.center_speed)

        # АДАПТИВНОЕ EMA-сглаживание direction. На прямых участках следуем быстро,
        # на резких 90° поворотах — медленно (alpha от 0.008 до 0.053). Это
        # удерживает min_agent_distance >= 0.6 при поворотах. Подобран grid-search.
        if not hasattr(self, "_direction_smoothed") or self._direction_smoothed is None:
            self._direction_smoothed = direction_raw.copy()
        else:
            cos_theta = float(
                self._direction_smoothed[0] * direction_raw[0]
                + self._direction_smoothed[1] * direction_raw[1]
            )
            cos_theta = max(-1.0, min(1.0, cos_theta))
            alpha = 0.008 + 0.045 * max(0.0, cos_theta)
            self._direction_smoothed = (
                (1 - alpha) * self._direction_smoothed + alpha * direction_raw
            )
            ds_norm = np.linalg.norm(self._direction_smoothed)
            if ds_norm > 1e-9:
                self._direction_smoothed = self._direction_smoothed / ds_norm
        direction = self._direction_smoothed

        mode = self.choose_mode(center, direction)

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
            f_target = self.target_force(drone, center)
            f_form   = self.formation_force(drone, center, direction, mode, drones)
            f_obs, c = self.obstacle_force(drone, direction, center=center, drones_all=drones)
            f_damp   = -self.k_damp * drone.v
            f_height = self.height_force(drone, desired_alt)

            f = clamp_norm(f_target + f_form + f_obs + f_damp + f_height, drone.max_force)
            total_force_norm += norm(f)
            min_clearance = min(min_clearance, c)
            formation_errors.append(norm(drone.p - desired[drone.role]))
            max_drone_z = max(max_drone_z, drone.p[2])
            drone.apply_force(f, self.dt)

        min_agent_distance = min(
            norm(a.p - b.p)
            for i, a in enumerate(drones)
            for b in drones[i + 1:]
        )
        return {
            "time": t,
            "mode": mode,
            "center_x": center[0],
            "center_y": center[1],
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


# ----------------------------- plotting -----------------------------

def plot_cylinder(ax, obs: ObstacleCylinder, zmin=0.0, zmax=2.2, resolution=32) -> None:
    theta = np.linspace(0, 2 * np.pi, resolution)
    z = np.linspace(zmin, zmax, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = obs.x + obs.r * np.cos(theta_grid)
    y_grid = obs.y + obs.r * np.sin(theta_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.25, linewidth=0)


def save_trajectory_plot(drones: List[Drone3D], obstacles: List[ObstacleCylinder], out: Path, title: str) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for obs in obstacles:
        plot_cylinder(ax, obs)
    for drone in drones:
        h = np.array(drone.history)
        ax.plot(h[:, 0], h[:, 1], h[:, 2], linewidth=2, label=drone.role)
        ax.scatter(h[0, 0], h[0, 1], h[0, 2], marker="o")
        ax.scatter(h[-1, 0], h[-1, 1], h[-1, 2], marker="x")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 2.5)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_metric_plot(df: pd.DataFrame, columns: List[str], out: Path, title: str) -> None:
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    for col in columns:
        ax.plot(df["time"], df[col], label=col)
    ax.set_xlabel("time")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


# ----------------------------- runner -----------------------------

def run_experiment(adaptive: bool, out_dir: Path) -> Tuple[pd.DataFrame, List[Drone3D]]:
    obstacles = [
        ObstacleCylinder(-1.20, -0.80, 0.33),
        ObstacleCylinder(-0.20,  0.75, 0.36),
        ObstacleCylinder( 0.85,  0.00, 0.45),
        ObstacleCylinder( 1.85,  0.90, 0.34),
        ObstacleCylinder( 2.55, -0.65, 0.32),
    ]
    controller = AdaptiveFormationController3D(obstacles, adaptive=adaptive)
    drones = make_drones(controller)

    rows = []
    max_steps = 1800
    for k in range(max_steps):
        t = k * controller.dt
        row = controller.step(drones, t)
        rows.append(row)
        center = np.array([row["center_x"], row["center_y"], controller.target_z])
        if np.linalg.norm(center - controller.goal_center) < 0.05 and k > 100:
            # allow followers to settle near final slots
            if k > max_steps * 0.45:
                break

    df = pd.DataFrame(rows)
    name = "adaptive" if adaptive else "fixed"
    df.to_csv(out_dir / f"metrics_{name}.csv", index=False)
    save_trajectory_plot(drones, obstacles, out_dir / f"trajectory_{name}.png", f"3D trajectories: {name}")
    save_metric_plot(
        df,
        ["formation_error", "min_agent_distance", "min_obstacle_clearance"],
        out_dir / f"safety_metrics_{name}.png",
        f"Safety and formation metrics: {name}",
    )
    save_metric_plot(
        df,
        ["energy", "path_length"],
        out_dir / f"energy_distance_{name}.png",
        f"Energy and path length: {name}",
    )
    save_metric_plot(
        df,
        ["max_height"],
        out_dir / f"height_profile_{name}.png",
        f"Height profile: {name}",
    )
    return df, drones


def main() -> None:
    out_dir = Path("results_3d")
    out_dir.mkdir(exist_ok=True)

    summaries = []
    for adaptive in (False, True):
        df, drones = run_experiment(adaptive, out_dir)
        summaries.append({
            "strategy": "adaptive" if adaptive else "fixed",
            "mean_formation_error": round(float(df["formation_error"].mean()), 4),
            "min_agent_distance": round(float(df["min_agent_distance"].min()), 4),
            "min_obstacle_clearance": round(float(df["min_obstacle_clearance"].min()), 4),
            "energy": round(float(df["energy"].iloc[-1]), 2),
            "path_length": round(float(df["path_length"].iloc[-1]), 2),
            "max_height": round(float(df["max_height"].max()), 3),
            "last_mode": df["mode"].iloc[-1],
        })
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSaved results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
