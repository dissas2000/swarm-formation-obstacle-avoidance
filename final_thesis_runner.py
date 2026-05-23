#!/usr/bin/env python3
"""
final_thesis_runner.py

Финальный прогон всех сценариев ВКР.
Сохраняет результаты в final_thesis_results/ с русскими подписями.

Запуск:
    python final_thesis_runner.py
    python final_thesis_runner.py --skip-animation
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless backend — до импорта pyplot

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Подключаем модуль симуляции ────────────────────────────────────────────────
# Импортируем обновлённую версию (swarm_3d_experiments1.py) с тремя
# критическими исправлениями:
#   1. Гибридный выбор стороны касательной (по центру группы для общих
#      препятствий, по позиции дрона для локальных) — устраняет встречные
#      потоки агентов в сценарии tall_wall_side.
#   2. EMA-сглаживание направления path с alpha=0.04 — предотвращает
#      слипание агентов в column на резких 90° поворотах.
#   3. F_target вынесена в явную функцию (раньше была частью F_coh)
#      — соответствует формуле 8 диплома.
#
# По умолчанию ожидается модуль swarm_3d_experiments1.py в той же папке.
# Если используется другая структура, измените _MODULE_DIR / _MODULE_NAME.
_MODULE_DIR = Path(__file__).parent
_MODULE_NAME = "swarm_3d_experiments1"
sys.path.insert(0, str(_MODULE_DIR))

# Fallback: если файл swarm_3d_experiments1.py отсутствует, пробуем старое имя
try:
    import importlib
    _sim = importlib.import_module(_MODULE_NAME)
except ImportError as exc:
    raise ImportError(
        f"[runner] Не удалось импортировать {_MODULE_NAME}.py. "
        "Проверьте, что файл находится в корне репозитория и что установлены зависимости."
    ) from exc

COLORS = _sim.COLORS
ROLES = _sim.ROLES
AdaptiveFormationController3D = _sim.AdaptiveFormationController3D
Drone3D = _sim.Drone3D
ObstacleCylinder = _sim.ObstacleCylinder
Scenario = _sim.Scenario
cylinder_clearance_3d = _sim.cylinder_clearance_3d
draw_cylinder = _sim.draw_cylinder
make_drones = _sim.make_drones
make_scenarios = _sim.make_scenarios
norm = _sim.norm
summarize = _sim.summarize
unit = _sim.unit

# ── Константы ──────────────────────────────────────────────────────────────────
OUTPUT_ROOT = Path("final_thesis_results")

SCENARIOS_TO_RUN: List[str] = [
    "free",                  # свободное движение, базовая формация
    "single",                # одиночные препятствия (tangential avoidance)
    "wide_barrier",          # широкая полоса препятствий, боковой облёт
    "dense_field",           # плотная среда препятствий
    "low_wall_overflight",   # низкая стенка, перелёт сверху (overflight)
    "tall_wall_side",        # высокая стенка, боковой облёт (требует согл. обхода)
    "checkerboard_equal",    # шахматное поле одинаковой высоты (контроль)
    "checkerboard_unequal",  # шахматное поле разной высоты
    "vertical_exit",         # единственный выход через верх (overflight ключевой)
    "defense_showcase",      # последовательная демонстрация всех 4 режимов
]

RUSSIAN_LABELS: Dict[str, str] = {
    "leader":      "Лидер",
    "follower1":   "Ведомый 1",
    "follower2":   "Ведомый 2",
    "follower3":   "Ведомый 3",
    "follower4":   "Ведомый 4",
    "center_path": "Траектория центра",
    "start":       "Старт",
    "goal":        "Цель",
}

MODE_RU: Dict[str, str] = {
    "normal":     "Норма",
    "compressed": "Сжатие",
    "column":     "Колонна",
    "overflight": "Перелёт",
}

MODE_NUM: Dict[str, int] = {
    "normal": 0, "compressed": 1, "column": 2, "overflight": 3,
}

MODE_COLORS: Dict[str, str] = {
    "normal":     "#2A9D8F",
    "compressed": "#E9C46A",
    "column":     "#F4A261",
    "overflight": "#D1495B",
}

METRIC_LABELS_RU: Dict[str, str] = {
    "formation_error":        "Ошибка формации, м",
    "min_pair_distance":      "Мин. расстояние между дронами, м",
    "min_obstacle_clearance": "Мин. зазор до препятствия, м",
}

METRIC_COLORS: List[str] = ["#2A9D8F", "#E9C46A", "#D1495B"]


# ── Симуляция ──────────────────────────────────────────────────────────────────

def run_scenario(
    scenario: Scenario,
) -> Tuple[pd.DataFrame, List[Drone3D], AdaptiveFormationController3D]:
    controller = AdaptiveFormationController3D(scenario=scenario, adaptive=True)
    drones = make_drones(controller)
    rows: List[Dict] = []
    for k in range(2200):
        t = k * controller.dt
        row = controller.step(drones, t)
        rows.append(row)
        center = np.array([row["center_x"], row["center_y"], row["center_z"]])
        if norm(center - controller.goal_center) < 0.12 and k > 200:
            break
    df = pd.DataFrame(rows)
    # Алиас для обратной совместимости с runner'ом (старое имя поля)
    # В новой версии модуля поле называется min_agent_distance.
    if "min_agent_distance" in df.columns and "min_pair_distance" not in df.columns:
        df["min_pair_distance"] = df["min_agent_distance"]
    return df, drones, controller


# ── Рисунки с русскими подписями ───────────────────────────────────────────────

def save_trajectory_3d_ru(
    drones: List[Drone3D],
    obstacles: List[ObstacleCylinder],
    center_path: List[np.ndarray],
    out: Path,
    title: str,
    scenario: Scenario,
) -> None:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Препятствия (тело + зона безопасности)
    for obs in obstacles:
        if obs.is_boundary:
            continue
        draw_cylinder(ax, obs, alpha=0.22, safety=False)
        draw_cylinder(ax, obs, safety=True)  # полупрозрачная зона

    # Траектория центра (красный пунктир)
    cp = np.array(center_path)
    ax.plot(
        cp[:, 0], cp[:, 1], cp[:, 2],
        color=COLORS["center_path"], linewidth=2.5, linestyle="--",
        alpha=0.92, label=RUSSIAN_LABELS["center_path"],
    )

    # Траектории дронов
    for drone in drones:
        h = np.array(drone.history)
        color = COLORS.get(drone.role, "#333333")
        ax.plot(h[:, 0], h[:, 1], h[:, 2],
                color=color, linewidth=1.8, alpha=0.83,
                label=RUSSIAN_LABELS[drone.role])
        ax.scatter(h[0, 0], h[0, 1], h[0, 2],
                   color=COLORS["start"], s=26, zorder=5)
        ax.scatter(h[-1, 0], h[-1, 1], h[-1, 2],
                   color=color, marker="x", s=56)

    # Старт и цель
    start = np.array(scenario.start, dtype=float)
    ax.scatter([start[0]], [start[1]], [start[2]],
               color=COLORS["start"], s=95, marker="o",
               label=RUSSIAN_LABELS["start"])
    ax.scatter([scenario.goal[0]], [scenario.goal[1]], [scenario.goal[2]],
               color=COLORS["goal"], s=170, marker="*",
               label=RUSSIAN_LABELS["goal"])

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_zlabel("z, м")
    ax.set_xlim(-8, 8)
    ax.set_ylim(-6.7, 6.7)
    ax.set_zlim(0, 2.7)
    ax.view_init(elev=25, azim=-62)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_top_view_ru(
    drones: List[Drone3D],
    obstacles: List[ObstacleCylinder],
    center_path: List[np.ndarray],
    out: Path,
    title: str,
    scenario: Scenario,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 9))

    for obs in obstacles:
        if obs.is_boundary:
            body = plt.Circle(
                (obs.x, obs.y), obs.r,
                color=COLORS["boundary"], alpha=0.60,
            )
            ax.add_patch(body)
            continue

        # Безопасная зона — очень светло-жёлтая, полупрозрачная
        safety_patch = plt.Circle(
            (obs.x, obs.y), obs.r + 0.35,
            color=COLORS["safety"], alpha=0.14,
        )
        # Тело препятствия: низкие — светло-зелёные, высокие — серые
        body_color = COLORS["low_obstacle"] if obs.z_max < 1.0 else COLORS["tall_obstacle"]
        body_patch = plt.Circle(
            (obs.x, obs.y), obs.r,
            color=body_color, alpha=0.56,
        )
        ax.add_patch(safety_patch)
        ax.add_patch(body_patch)
        ax.text(obs.x, obs.y, f"{obs.z_max:.1f}",
                ha="center", va="center", fontsize=7)

    # Траектория центра
    cp = np.array(center_path)
    ax.plot(cp[:, 0], cp[:, 1],
            color=COLORS["center_path"], linewidth=2.4, linestyle="--",
            label=RUSSIAN_LABELS["center_path"])

    # Траектории дронов
    for drone in drones:
        h = np.array(drone.history)
        color = COLORS.get(drone.role, "#333333")
        ax.plot(h[:, 0], h[:, 1], color=color, linewidth=1.8,
                label=RUSSIAN_LABELS[drone.role])
        ax.scatter(h[-1, 0], h[-1, 1], color=color, s=44)

    # Старт — серый круг; цель — оранжевая звезда
    ax.scatter([scenario.start[0]], [scenario.start[1]],
               color=COLORS["start"], s=85, marker="o",
               label=RUSSIAN_LABELS["start"])
    ax.scatter([scenario.goal[0]], [scenario.goal[1]],
               color=COLORS["goal"], s=170, marker="*",
               label=RUSSIAN_LABELS["goal"])

    ax.set_title(title + " / вид сверху", fontsize=12)
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_xlim(-8, 8)
    ax.set_ylim(-6.7, 6.7)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_metrics_panel_ru(
    df: pd.DataFrame,
    out: Path,
    title: str,
) -> None:
    cols = ["formation_error", "min_pair_distance", "min_obstacle_clearance"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.suptitle(title + " / метрики", fontsize=13, y=1.005)

    for ax, col, color in zip(axes, cols, METRIC_COLORS):
        vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        times = df["time"].loc[vals.index]
        ax.plot(times, vals, color=color, linewidth=1.8)
        ax.fill_between(times, vals, alpha=0.15, color=color)
        ax.set_ylabel(METRIC_LABELS_RU[col], fontsize=9)
        ax.grid(True, alpha=0.25)
        if len(times):
            ax.set_xlim(times.min(), times.max())

    axes[-1].set_xlabel("Время, с", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_mode_timeline_ru(
    df: pd.DataFrame,
    out: Path,
    title: str,
) -> None:
    y = df["mode"].map(MODE_NUM).fillna(0)
    mode_list = list(MODE_NUM.keys())

    fig, ax = plt.subplots(figsize=(12, 4))

    # Цветная заливка фона по режиму
    for i in range(len(df) - 1):
        m = df["mode"].iloc[i]
        c = MODE_COLORS.get(m, "#CCCCCC")
        ax.axvspan(
            df["time"].iloc[i], df["time"].iloc[i + 1],
            ymin=0, ymax=1, color=c, alpha=0.22, linewidth=0,
        )

    ax.step(df["time"], y, where="post", linewidth=2.5, color="#163A5F")
    ax.set_yticks(list(MODE_NUM.values()))
    ax.set_yticklabels([MODE_RU[m] for m in mode_list], fontsize=11)
    ax.set_xlabel("Время, с", fontsize=10)
    ax.set_title(title + " / режимы адаптации", fontsize=12)
    if len(df):
        ax.set_xlim(df["time"].min(), df["time"].max())
    ax.grid(True, axis="x", alpha=0.30)

    legend_patches = [
        mpatches.Patch(color=MODE_COLORS[m], alpha=0.55, label=MODE_RU[m])
        for m in mode_list
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ── Анимация defense_showcase ──────────────────────────────────────────────────

def save_defense_animation(
    drones: List[Drone3D],
    obstacles: List[ObstacleCylinder],
    center_path: List[np.ndarray],
    df: pd.DataFrame,
    scenario: Scenario,
    out_dir: Path,
) -> List[Path]:
    from matplotlib.animation import FuncAnimation, PillowWriter

    histories = {d.role: np.array(d.history) for d in drones}
    n_frames = min(len(df), min(len(h) for h in histories.values()))
    stride = 8
    frame_indices = list(range(0, n_frames, stride))
    trail_len = 45

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    for obs in obstacles:
        if not obs.is_boundary:
            draw_cylinder(ax, obs, alpha=0.20, safety=False)

    cp = np.array(center_path)
    ax.plot(cp[:, 0], cp[:, 1], cp[:, 2],
            color=COLORS["center_path"], linewidth=2.0, linestyle="--",
            alpha=0.65, label=RUSSIAN_LABELS["center_path"])
    ax.scatter([scenario.goal[0]], [scenario.goal[1]], [scenario.goal[2]],
               color=COLORS["goal"], s=130, marker="*")

    ax.set_title(scenario.title, fontsize=11)
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_zlabel("z, м")
    ax.set_xlim(-8, 8)
    ax.set_ylim(-6.7, 6.7)
    ax.set_zlim(0, 2.7)
    ax.view_init(elev=25, azim=-62)

    line_map: Dict[str, object] = {}
    point_map: Dict[str, object] = {}

    for role in ROLES:
        color = COLORS.get(role, "#333333")
        line, = ax.plot([], [], [], color=color, linewidth=1.8, alpha=0.85)
        point, = ax.plot([], [], [], marker="o",
                         markersize=8 if role == "leader" else 5,
                         linestyle="None", color=color,
                         label=RUSSIAN_LABELS[role])
        line_map[role] = line
        point_map[role] = point

    info = ax.text2D(
        0.02, 0.97, "", transform=ax.transAxes, fontsize=8,
        verticalalignment="top", family="monospace",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.75},
    )
    ax.legend(loc="upper right", fontsize=7)
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
            f"t={row['time']:.1f}с   режим={MODE_RU.get(row['mode'], row['mode'])}\n"
            f"ошибка={row['formation_error']:.3f}   "
            f"пара={row['min_pair_distance']:.3f}\n"
            f"зазор={row['min_obstacle_clearance']:.3f}"
        )
        return []

    ani = FuncAnimation(fig, update, frames=frame_indices, interval=55, blit=False)

    saved: List[Path] = []

    gif_path = out_dir / "defense_main_animation.gif"
    print("  Сохраняю GIF анимацию …")
    ani.save(str(gif_path), writer=PillowWriter(fps=15), dpi=95)
    saved.append(gif_path)
    print(f"  GIF сохранён: {gif_path}")

    mp4_path = out_dir / "defense_main_animation.mp4"
    if shutil.which("ffmpeg"):
        try:
            from matplotlib.animation import FFMpegWriter
            print("  Сохраняю MP4 анимацию …")
            ani.save(str(mp4_path), writer=FFMpegWriter(fps=25, bitrate=1800), dpi=130)
            saved.append(mp4_path)
            print(f"  MP4 сохранён: {mp4_path}")
        except Exception as exc:
            print(f"  Предупреждение: MP4 не сохранён: {exc}")
    else:
        print("  Предупреждение: ffmpeg не найден — MP4 пропущен.")

    plt.close(fig)
    return saved


# ── Главная функция ────────────────────────────────────────────────────────────

def main(skip_animation: bool = False) -> None:
    start_ts = datetime.now()

    # Структура директорий
    dirs: Dict[str, Path] = {
        "summary":  OUTPUT_ROOT / "summary",
        "figures":  OUTPUT_ROOT / "figures",
        "selected": OUTPUT_ROOT / "selected_for_thesis",
        "logs":     OUTPUT_ROOT / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    for sc_name in SCENARIOS_TO_RUN:
        (dirs["figures"] / sc_name).mkdir(exist_ok=True)

    log_lines: List[str] = [
        "Финальный прогон ВКР — swarm_3d_experiments",
        f"Дата: {start_ts.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
    ]
    saved_files: List[Path] = []
    all_summaries: List[Dict] = []
    results: Dict[
        str,
        Tuple[pd.DataFrame, List[Drone3D], AdaptiveFormationController3D],
    ] = {}

    scenarios = make_scenarios()

    # ── Прогон всех сценариев ──────────────────────────────────────────────────
    for sc_name in SCENARIOS_TO_RUN:
        scenario = scenarios[sc_name]
        print(f"\n{'='*57}")
        print(f"  Сценарий: {sc_name}")
        print(f"{'='*57}")

        sc_dir = dirs["figures"] / sc_name

        df, drones, controller = run_scenario(scenario)
        results[sc_name] = (df, drones, controller)

        modes_seen = set(df["mode"].unique())
        min_cl = df["min_obstacle_clearance"].replace([np.inf, -np.inf], np.nan).min()
        min_pair = df["min_pair_distance"].min()

        log_lines.append(f"\n[{sc_name}]")
        log_lines.append(f"  шагов:      {len(df)}")
        log_lines.append(f"  режимы:     {sorted(modes_seen)}")
        log_lines.append(f"  мин. зазор: {min_cl:.6f}" if np.isfinite(min_cl) else "  мин. зазор: inf (нет препятствий)")
        log_lines.append(f"  мин. пара:  {min_pair:.6f}")
        log_lines.append(f"  ср. ошибка: {df['formation_error'].mean():.6f}")

        # 3D-траектория
        traj_3d = sc_dir / "trajectory_3d.png"
        save_trajectory_3d_ru(
            drones, scenario.obstacles, controller.path,
            traj_3d, scenario.title, scenario,
        )
        saved_files.append(traj_3d)
        print(f"  [3D]  {traj_3d}")

        # Вид сверху
        top_view = sc_dir / "top_view.png"
        save_top_view_ru(
            drones, scenario.obstacles, controller.path,
            top_view, scenario.title, scenario,
        )
        saved_files.append(top_view)
        print(f"  [TOP] {top_view}")

        # Панель метрик
        metrics_panel = sc_dir / "metrics_panel.png"
        save_metrics_panel_ru(df, metrics_panel, scenario.title)
        saved_files.append(metrics_panel)
        print(f"  [MTR] {metrics_panel}")

        # График режимов (сохраняем для всех сценариев)
        mode_timeline = sc_dir / "mode_timeline.png"
        save_mode_timeline_ru(df, mode_timeline, scenario.title)
        saved_files.append(mode_timeline)
        print(f"  [MOD] {mode_timeline}")

        # Метрики CSV
        metrics_csv = sc_dir / "metrics.csv"
        df.to_csv(metrics_csv, index=False)
        saved_files.append(metrics_csv)

        all_summaries.append(summarize(df, sc_name, "adaptive"))

        cl_str = f"{min_cl:.4f}" if np.isfinite(min_cl) else "inf"
        print(f"  Шагов: {len(df)}, режимы: {sorted(modes_seen)}")
        print(f"  Мин. зазор: {cl_str}  |  Мин. пара: {min_pair:.4f}")

    # ── Summary CSV ────────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(all_summaries)

    all_metrics_csv = dirs["summary"] / "all_metrics_summary.csv"
    summary_df.to_csv(all_metrics_csv, index=False)
    saved_files.append(all_metrics_csv)

    # Краткая таблица с русскими названиями столбцов
    # В новой версии модуля имена полей: min_agent_distance (раньше min_pair_distance),
    # energy (раньше final_energy), path_length (раньше final_path_length)
    selected_cols = [
        "scenario", "mean_formation_error",
        "min_agent_distance", "min_obstacle_clearance",
        "energy", "path_length", "last_mode",
    ]
    selected_df = summary_df[selected_cols].copy()
    selected_df.columns = [
        "Сценарий", "Ср. ошибка формации",
        "Мин. дист. пары", "Мин. зазор",
        "Итог. энергия", "Итог. длина пути", "Посл. режим",
    ]
    selected_metrics_csv = dirs["summary"] / "selected_metrics_table.csv"
    selected_df.to_csv(selected_metrics_csv, index=False, encoding="utf-8-sig")
    saved_files.append(selected_metrics_csv)

    print(f"\n[SUMMARY] {all_metrics_csv}")
    print(f"[SUMMARY] {selected_metrics_csv}")

    # ── Анимация defense_showcase ──────────────────────────────────────────────
    if not skip_animation:
        print(f"\n{'='*57}")
        print("  Генерирую анимацию defense_showcase …")
        print(f"{'='*57}")
        df_def, drones_def, ctrl_def = results["defense_showcase"]
        sc_dir_def = dirs["figures"] / "defense_showcase"
        anim_files = save_defense_animation(
            drones_def, scenarios["defense_showcase"].obstacles,
            ctrl_def.path, df_def, scenarios["defense_showcase"], sc_dir_def,
        )
        saved_files.extend(anim_files)
        log_lines.append(f"\n[defense_showcase] анимация: {[str(p) for p in anim_files]}")
    else:
        print("\n  Анимация пропущена (--skip-animation).")

    # ── Копии в selected_for_thesis ────────────────────────────────────────────
    def copy_to_selected(src: Path, dst_name: str) -> None:
        if src.exists():
            dst = dirs["selected"] / dst_name
            shutil.copy2(src, dst)
            saved_files.append(dst)

    copy_to_selected(
        dirs["figures"] / "defense_showcase" / "trajectory_3d.png",
        "defense_trajectory_3d.png",
    )
    copy_to_selected(
        dirs["figures"] / "defense_showcase" / "top_view.png",
        "defense_top_view.png",
    )
    copy_to_selected(
        dirs["figures"] / "defense_showcase" / "mode_timeline.png",
        "defense_mode_timeline.png",
    )
    copy_to_selected(
        dirs["figures"] / "defense_showcase" / "metrics_panel.png",
        "defense_metrics_panel.png",
    )
    copy_to_selected(
        dirs["figures"] / "checkerboard_unequal" / "trajectory_3d.png",
        "checkerboard_unequal_3d.png",
    )
    copy_to_selected(
        dirs["figures"] / "checkerboard_equal" / "trajectory_3d.png",
        "checkerboard_equal_3d.png",
    )
    copy_to_selected(
        dirs["figures"] / "vertical_exit" / "top_view.png",
        "vertical_exit_top_view.png",
    )
    copy_to_selected(
        dirs["figures"] / "tall_wall_side" / "top_view.png",
        "tall_wall_side_top_view.png",
    )
    copy_to_selected(
        dirs["figures"] / "low_wall_overflight" / "trajectory_3d.png",
        "low_wall_overflight_3d.png",
    )

    # ── Проверки ───────────────────────────────────────────────────────────────
    print(f"\n{'='*57}")
    print("  ПРОВЕРКИ БЕЗОПАСНОСТИ")
    print(f"{'='*57}")
    log_lines.append("\n" + "=" * 60)
    log_lines.append("ПРОВЕРКИ")
    log_lines.append("=" * 60)

    all_ok = True

    # 1. min_obstacle_clearance > 0 для всех сценариев
    print("\n1. Мин. зазор до препятствия > 0:")
    log_lines.append("\n1. min_obstacle_clearance > 0:")
    for sc_name in SCENARIOS_TO_RUN:
        df_sc, _, _ = results[sc_name]
        inner_obs = [
            o for o in scenarios[sc_name].obstacles if not o.is_boundary
        ]
        if not inner_obs:
            msg = f"   {sc_name:<35}: нет препятствий  [N/A]"
            print(msg)
            log_lines.append(msg)
            continue
        min_cl = df_sc["min_obstacle_clearance"].replace([np.inf, -np.inf], np.nan).min()
        ok = bool(np.isnan(min_cl) or min_cl > 0)
        if not ok:
            all_ok = False
        status = "OK" if ok else "FAIL"
        cl_str = f"{min_cl:.4f}" if np.isfinite(min_cl) else "inf"
        msg = f"   {sc_name:<35}: {cl_str}  [{status}]"
        print(msg)
        log_lines.append(msg)

    # 2. min_pair_distance > 0.7 для defense_showcase
    print("\n2. Мин. расстояние между дронами > 0.7 (defense_showcase):")
    log_lines.append("\n2. min_pair_distance > 0.7 (defense_showcase):")
    df_def2, _, _ = results["defense_showcase"]
    min_pair_def = df_def2["min_pair_distance"].min()
    ok2 = bool(min_pair_def > 0.7)
    if not ok2:
        all_ok = False
    status2 = "OK" if ok2 else "FAIL"
    msg2 = f"   min_pair_distance = {min_pair_def:.4f}  [{status2}]"
    print(msg2)
    log_lines.append(msg2)

    # 3. Все 4 режима в defense_showcase
    print("\n3. Все режимы в defense_showcase:")
    log_lines.append("\n3. Режимы в defense_showcase:")
    required_modes = {"normal", "compressed", "column", "overflight"}
    modes_in_defense = set(df_def2["mode"].unique())
    for mode in sorted(required_modes):
        present = mode in modes_in_defense
        if not present:
            all_ok = False
        status3 = "OK" if present else "FAIL"
        msg3 = f"   {mode:<20}: [{status3}]"
        print(msg3)
        log_lines.append(msg3)

    # 4. vertical_exit: overflight присутствует, max_z ∈ [1.4, 1.9], pair > 0.7
    print("\n4. vertical_exit:")
    log_lines.append("\n4. vertical_exit:")
    df_vec, _, _ = results["vertical_exit"]

    vec_modes = set(df_vec["mode"].unique())
    ok_ov = "overflight" in vec_modes
    if not ok_ov:
        all_ok = False
    m4a = f"   overflight присутствует            : [{'OK' if ok_ov else 'FAIL'}]"
    print(m4a); log_lines.append(m4a)

    vec_max_z = df_vec["center_z"].max()
    # В новой версии модели низкие препятствия имеют z_max=0.72, overflight_margin=0.55
    # => ожидаемая высота overflight = 0.72 + 0.55 = 1.27 м (выше target_z=1.20).
    # Допуск: max_center_z в диапазоне [1.25, 1.95] м.
    ok_z = bool(1.25 <= vec_max_z <= 1.95)
    if not ok_z:
        all_ok = False
    m4b = f"   max_center_z = {vec_max_z:.3f} (exp 1.25–1.95) : [{'OK' if ok_z else 'FAIL'}]"
    print(m4b); log_lines.append(m4b)

    vec_min_cl = df_vec["min_obstacle_clearance"].replace([np.inf, -np.inf], np.nan).min()
    ok_cl = bool(np.isnan(vec_min_cl) or vec_min_cl > 0)
    if not ok_cl:
        all_ok = False
    m4c = f"   min_clearance = {vec_min_cl:.4f}              : [{'OK' if ok_cl else 'FAIL'}]"
    print(m4c); log_lines.append(m4c)

    vec_min_pair = df_vec["min_pair_distance"].min()
    ok_pair = bool(vec_min_pair > 0.7)
    if not ok_pair:
        all_ok = False
    m4d = f"   min_pair      = {vec_min_pair:.4f} (exp >0.7) : [{'OK' if ok_pair else 'FAIL'}]"
    print(m4d); log_lines.append(m4d)

    # ── Итоговая таблица ───────────────────────────────────────────────────────
    print(f"\n{'='*57}")
    print("  ИТОГОВАЯ ТАБЛИЦА МЕТРИК")
    print(f"{'='*57}")
    print(summary_df.to_string(index=False))

    # ── Лог-файл ───────────────────────────────────────────────────────────────
    end_ts = datetime.now()
    elapsed = (end_ts - start_ts).total_seconds()
    log_lines.append(f"\nВремя выполнения: {elapsed:.1f} с")
    log_lines.append(f"Итог проверок: {'ВСЕ OK' if all_ok else 'ЕСТЬ ОШИБКИ'}")
    log_lines.append(f"Всего файлов: {len(saved_files)}")
    log_lines.append("\nСписок сохранённых файлов:")
    for p in saved_files:
        log_lines.append(f"  {p}")

    log_path = dirs["logs"] / "final_run_log.txt"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    saved_files.append(log_path)

    # ── Список файлов ──────────────────────────────────────────────────────────
    print(f"\n{'='*57}")
    print("  СПИСОК СОХРАНЁННЫХ ФАЙЛОВ")
    print(f"{'='*57}")
    for p in sorted(set(saved_files)):
        print(f"  {p}")

    verdict = "ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ [OK]" if all_ok else "ВНИМАНИЕ: ЕСТЬ ОШИБКИ В ПРОВЕРКАХ [FAIL]"
    print(f"\n{verdict}")
    print(f"Время: {elapsed:.1f} с")
    print(f"Результаты: {OUTPUT_ROOT.resolve()}")


# ── Только MP4 для defense_showcase ───────────────────────────────────────────

def only_mp4() -> None:
    """
    Запускает симуляцию defense_showcase и сохраняет только MP4-анимацию.
    Ничего другого (PNG, CSV, остальные сценарии) не трогает.
    """
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    # 1. Найти ffmpeg и сообщить путь (или дать инструкцию по установке)
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        print()
        print("ffmpeg не найден в PATH.")
        print()
        print("Установите ffmpeg через conda:")
        print("    conda install -c conda-forge ffmpeg")
        print()
        print("Или через pip (Windows):")
        print("    pip install imageio[ffmpeg]")
        print()
        print("После установки повторите запуск:")
        print("    python final_thesis_runner.py --only-mp4")
        return

    print(f"ffmpeg найден: {ffmpeg_path}")
    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path

    # 2. Целевая папка
    anim_dir = OUTPUT_ROOT / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = anim_dir / "defense_main_animation.mp4"

    # 3. Запустить симуляцию defense_showcase
    print("Запуск симуляции defense_showcase ...")
    scenarios = make_scenarios()
    scenario = scenarios["defense_showcase"]
    df, drones, controller = run_scenario(scenario)
    print(f"  Симуляция завершена: {len(df)} шагов, "
          f"режимы: {sorted(set(df['mode'].unique()))}")

    # 4. Подготовить анимацию
    histories = {d.role: np.array(d.history) for d in drones}
    n_frames = min(len(df), min(len(h) for h in histories.values()))
    stride = 6                   # чуть плавнее, чем GIF
    frame_indices = list(range(0, n_frames, stride))
    trail_len = 50

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    for obs in scenario.obstacles:
        if not obs.is_boundary:
            draw_cylinder(ax, obs, alpha=0.20, safety=False)

    cp = np.array(controller.path)
    ax.plot(cp[:, 0], cp[:, 1], cp[:, 2],
            color=COLORS["center_path"], linewidth=2.0, linestyle="--",
            alpha=0.65, label=RUSSIAN_LABELS["center_path"])
    ax.scatter([scenario.goal[0]], [scenario.goal[1]], [scenario.goal[2]],
               color=COLORS["goal"], s=140, marker="*",
               label=RUSSIAN_LABELS["goal"])

    ax.set_title(scenario.title, fontsize=11)
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_zlabel("z, м")
    ax.set_xlim(-8, 8)
    ax.set_ylim(-6.7, 6.7)
    ax.set_zlim(0, 2.7)
    ax.view_init(elev=25, azim=-62)

    line_map: Dict[str, object] = {}
    point_map: Dict[str, object] = {}
    for role in ROLES:
        color = COLORS.get(role, "#333333")
        line, = ax.plot([], [], [], color=color, linewidth=1.8, alpha=0.85)
        point, = ax.plot([], [], [], marker="o",
                         markersize=8 if role == "leader" else 5,
                         linestyle="None", color=color,
                         label=RUSSIAN_LABELS[role])
        line_map[role] = line
        point_map[role] = point

    info = ax.text2D(
        0.02, 0.97, "", transform=ax.transAxes, fontsize=9,
        verticalalignment="top", family="monospace",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.80},
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
            f"t={row['time']:.1f}s   mode={MODE_RU.get(row['mode'], row['mode'])}\n"
            f"form_err={row['formation_error']:.3f}   "
            f"pair={row['min_pair_distance']:.3f}\n"
            f"clearance={row['min_obstacle_clearance']:.3f}"
        )
        return []

    ani = FuncAnimation(fig, update, frames=frame_indices, interval=40, blit=False)

    # 5. Сохранить MP4
    print(f"Сохраняю MP4 ({len(frame_indices)} кадров, 25 fps) ...")
    writer = FFMpegWriter(fps=25, bitrate=2400,
                          extra_args=["-pix_fmt", "yuv420p"])
    try:
        ani.save(str(mp4_path), writer=writer, dpi=150)
    except Exception as exc:
        print(f"\nОшибка при сохранении MP4: {exc}")
        print()
        print("Возможные причины:")
        print("  - ffmpeg не поддерживает кодек H.264 в данной сборке")
        print("  - Попробуйте: conda install -c conda-forge ffmpeg")
        plt.close(fig)
        return

    plt.close(fig)

    # 6. Вывести результат
    size_mb = mp4_path.stat().st_size / (1024 * 1024)
    print()
    print(f"MP4 сохранён:")
    print(f"  Путь:   {mp4_path.resolve()}")
    print(f"  Размер: {size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Финальный прогон экспериментов ВКР",
    )
    parser.add_argument(
        "--skip-animation",
        action="store_true",
        help="Пропустить генерацию GIF/MP4 анимации (ускоряет прогон)",
    )
    parser.add_argument(
        "--only-mp4",
        action="store_true",
        help="Только MP4 для defense_showcase (ничего другого не трогает)",
    )
    args = parser.parse_args()

    if args.only_mp4:
        only_mp4()
    else:
        main(skip_animation=args.skip_animation)
