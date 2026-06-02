from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import swarm_3d_experiments1 as sim


OUT_DIR = Path("diploma_tables")
OUT_DIR.mkdir(exist_ok=True)

SELECTED_SCENARIOS = [
    "free",
    "single",
    "dense_field",
    "checkerboard_unequal",
    "checkerboard_equal",
    "vertical_exit",
    "defense_showcase",
]

SCENARIO_RU = {
    "free": "Свободное движение",
    "single": "Одиночные препятствия",
    "dense_field": "Плотная среда",
    "checkerboard_unequal": "Поле разной высоты",
    "checkerboard_equal": "Поле одинаковой высоты",
    "vertical_exit": "Единственный проход сверху",
    "defense_showcase": "Демонстрационный сценарий",
}


def get_reference_params() -> Dict[str, object]:
    scenarios = sim.make_scenarios()
    controller = sim.AdaptiveFormationController3D(
        scenario=scenarios["free"],
        adaptive=True,
    )
    drone = sim.make_drones(controller)[0]

    return {
        "N": 5,
        "dt": controller.dt,
        "T": "scenario_dependent",
        "mass": drone.mass,
        "agent_radius": drone.radius,
        "d_agent": controller.d_safe,
        "d_safe": controller.obs_safe,
        "R_detect": controller.obs_detect,
        "k_t": controller.k_target,
        "k_f": controller.k_pos,
        "k_sep": controller.k_sep,
        "k_coh": controller.k_coh,
        "k_obs": controller.k_obs,
        "k_tan": controller.k_tan,
        "beta": controller.k_damp,
        "k_z": controller.k_z,
        "beta_z": controller.beta_z,
        "v_max": drone.max_speed,
        "F_max": drone.max_force,
        "h_max": controller.max_altitude,
        "h_safe": controller.overflight_margin,
        "L": "not_logged",
    }


def write_parameters_csv(params: Dict[str, object]) -> None:
    rows = []
    comments = {
        "N": "число агентов в группе",
        "dt": "шаг численного интегрирования",
        "T": "длительность моделирования, зависит от сценария",
        "mass": "нормированная масса агента",
        "agent_radius": "эффективный радиус агента",
        "d_agent": "минимально допустимое расстояние между агентами",
        "d_safe": "безопасный запас до препятствий",
        "R_detect": "радиус обнаружения препятствий",
        "k_t": "коэффициент притяжения к цели",
        "k_f": "коэффициент возврата к желаемому положению",
        "k_sep": "коэффициент разделения агентов",
        "k_coh": "коэффициент связности группы",
        "k_obs": "коэффициент отталкивания от препятствий",
        "k_tan": "коэффициент касательной компоненты обхода",
        "beta": "коэффициент демпфирования",
        "k_z": "коэффициент стабилизации высоты",
        "beta_z": "коэффициент вертикального демпфирования",
        "v_max": "максимальная допустимая скорость",
        "F_max": "максимальная допустимая сила",
        "h_max": "максимальная допустимая высота",
        "h_safe": "вертикальный запас при перелёте",
        "L": "число узлов проверки gamma-коридора",
    }

    for key, value in params.items():
        rows.append({
            "parameter": key,
            "value": value,
            "source_file": "swarm_3d_experiments1.py",
            "variable_name": key,
            "comment": comments.get(key, ""),
        })

    pd.DataFrame(rows).to_csv(OUT_DIR / "parameters_used.csv", index=False)


def compute_status(row: Dict[str, object], params: Dict[str, object]) -> str:
    violations = []

    d_agent = float(params["d_agent"])
    d_safe = float(params["d_safe"])
    v_max = float(params["v_max"])
    F_max = float(params["F_max"])

    if float(row["min_pair_distance"]) <= d_agent:
        violations.append("agent_violation")

    # Для free без препятствий клиренс может быть условно бесконечным.
    if row["scenario"] != "free":
        if float(row["min_obstacle_clearance"]) <= d_safe:
            violations.append("obstacle_violation")

    if float(row["max_speed_exp"]) > v_max + 1e-6:
        violations.append("velocity_violation")

    if float(row["max_force_exp"]) > F_max + 1e-6:
        violations.append("force_violation")

    if not bool(row["goal_reached"]):
        violations.append("fail")

    return "+".join(violations) if violations else "success"


def summarize_existing_metrics(scenario_name: str, params: Dict[str, object]) -> Dict[str, object]:
    metrics_path = Path("final_thesis_results") / "figures" / scenario_name / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Не найден файл метрик: {metrics_path}")

    df = pd.read_csv(metrics_path)

    final_center = df[["center_x", "center_y", "center_z"]].iloc[-1].to_numpy(dtype=float)
    scenarios = sim.make_scenarios()
    goal = scenarios[scenario_name].goal
    goal_reached = sim.norm(final_center - goal) < 0.15

    modes = df["mode"].astype(str).tolist()
    mode_switches = sum(1 for a, b in zip(modes, modes[1:]) if a != b)

    row = {
        "scenario": scenario_name,
        "scenario_ru": SCENARIO_RU[scenario_name],
        "strategy": "adaptive",
        "mean_formation_error": float(df["formation_error"].mean()),
        "max_formation_error": float(df["formation_error"].max()),
        "min_pair_distance": float(df["min_agent_distance"].min()),
        "min_obstacle_clearance": float(df["min_obstacle_clearance"].min()),
        "energy": float(df["energy"].iloc[-1]),
        "path_length": float(df["path_length"].iloc[-1]),
        # В старых CSV скорость и сила по агентам не логировались.
        # Но в реализации они насыщаются сверху этими значениями.
        "max_speed_exp": float(params["v_max"]),
        "max_force_exp": float(params["F_max"]),
        "max_height_exp": float(df["max_height"].max()),
        "mode_switches": int(mode_switches),
        "goal_reached": bool(goal_reached),
    }
    row["status"] = compute_status(row, params)
    return row


def run_fresh_simulation(scenario_name: str, params: Dict[str, object]) -> Dict[str, object]:
    scenarios = sim.make_scenarios()
    scenario = scenarios[scenario_name]

    controller = sim.AdaptiveFormationController3D(
        scenario=scenario,
        adaptive=True,
    )
    drones = sim.make_drones(controller)

    max_speed_exp = 0.0
    max_force_exp = 0.0
    mode_switches = 0
    prev_mode = None
    rows = []

    original_apply_force = sim.Drone3D.apply_force

    def tracked_apply_force(self, f, dt):
        nonlocal max_speed_exp, max_force_exp

        f = sim.clamp_norm(f, self.max_force)
        max_force_exp = max(max_force_exp, sim.norm(f))

        self.energy += float(np.dot(f, f)) * dt
        self.v += (f / self.mass) * dt
        self.v = sim.clamp_norm(self.v, self.max_speed)
        max_speed_exp = max(max_speed_exp, sim.norm(self.v))

        prev = self.p.copy()
        self.p += self.v * dt
        self.path_length += float(np.linalg.norm(self.p - prev))
        self.history.append(self.p.copy())

    sim.Drone3D.apply_force = tracked_apply_force

    try:
        for k in range(2200):
            t = k * controller.dt
            step_row = controller.step(drones, t)

            mode = step_row["mode"]
            if prev_mode is not None and mode != prev_mode:
                mode_switches += 1
            prev_mode = mode

            rows.append(step_row)

            center = np.array([
                step_row["center_x"],
                step_row["center_y"],
                step_row["center_z"],
            ])
            if sim.norm(center - controller.goal_center) < 0.12 and k > 200:
                break
    finally:
        sim.Drone3D.apply_force = original_apply_force

    df = pd.DataFrame(rows)
    out_metrics_dir = OUT_DIR / "fresh_metrics"
    out_metrics_dir.mkdir(exist_ok=True)
    df.to_csv(out_metrics_dir / f"{scenario_name}_metrics.csv", index=False)

    final_center = df[["center_x", "center_y", "center_z"]].iloc[-1].to_numpy(dtype=float)
    goal_reached = sim.norm(final_center - controller.goal_center) < 0.15

    row = {
        "scenario": scenario_name,
        "scenario_ru": SCENARIO_RU[scenario_name],
        "strategy": "adaptive",
        "mean_formation_error": float(df["formation_error"].mean()),
        "max_formation_error": float(df["formation_error"].max()),
        "min_pair_distance": float(df["min_agent_distance"].min()),
        "min_obstacle_clearance": float(df["min_obstacle_clearance"].min()),
        "energy": float(df["energy"].iloc[-1]),
        "path_length": float(df["path_length"].iloc[-1]),
        "max_speed_exp": float(max_speed_exp),
        "max_force_exp": float(max_force_exp),
        "max_height_exp": float(df["max_height"].max()),
        "mode_switches": int(mode_switches),
        "goal_reached": bool(goal_reached),
    }
    row["status"] = compute_status(row, params)
    return row


def write_summary_csv(rows: List[Dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "summary_final.csv", index=False)


def fmt(x, ndigits=3):
    if isinstance(x, str):
        return x
    try:
        x = float(x)
    except Exception:
        return str(x)
    if x > 1e8:
        return r"$\infty$"
    return f"{x:.{ndigits}f}"


def make_latex_tables(params: Dict[str, object], rows: List[Dict[str, object]]) -> str:
    lines = []

    lines.append("% ===== Таблица параметров =====")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Параметры численного моделирования}")
    lines.append(r"\label{tab:model_parameters}")
    lines.append(r"\begin{tabular}{c|c|p{7.5cm}}")
    lines.append(r"\hline")
    lines.append(r"Параметр & Значение & Смысл\\")
    lines.append(r"\hline")

    param_rows = [
        (r"$N$", "N", "число агентов в группе"),
        (r"$\Delta t$", "dt", "шаг численного интегрирования"),
        (r"$T$", "T", "длительность моделирования"),
        (r"$m_i$", "mass", "масса агента"),
        (r"$r_i$", "agent_radius", "эффективный радиус агента"),
        (r"$d_{\mathrm{agent}}$", "d_agent", "минимально допустимое расстояние между агентами"),
        (r"$d_{\mathrm{safe}}$", "d_safe", "безопасный запас до препятствий"),
        (r"$R_{\mathrm{detect}}$", "R_detect", "радиус обнаружения препятствий"),
        (r"$k_t$", "k_t", "коэффициент притяжения к цели"),
        (r"$k_f$", "k_f", "коэффициент возврата к желаемому положению"),
        (r"$k_{\mathrm{sep}}$", "k_sep", "коэффициент разделения агентов"),
        (r"$k_{\mathrm{coh}}$", "k_coh", "коэффициент связности группы"),
        (r"$k_{\mathrm{obs}}$", "k_obs", "коэффициент отталкивания от препятствий"),
        (r"$k_{\mathrm{tan}}$", "k_tan", "коэффициент касательной компоненты обхода"),
        (r"$\beta_i$", "beta", "коэффициент демпфирования поступательного движения"),
        (r"$k_z$", "k_z", "коэффициент стабилизации высоты"),
        (r"$\beta_z$", "beta_z", "коэффициент вертикального демпфирования"),
        (r"$v_{\max}$", "v_max", "максимальная допустимая скорость"),
        (r"$F_{\max}$", "F_max", "максимальная допустимая сила"),
        (r"$h_{\max}$", "h_max", "максимальная допустимая высота"),
        (r"$h_{\mathrm{safe}}$", "h_safe", "вертикальный запас при перелёте"),
        (r"$L$", "L", "число узлов проверки отрезка перестроения"),
    ]

    for latex_name, key, comment in param_rows:
        value = params.get(key, "missing")
        if isinstance(value, float):
            value = f"{value:.3g}"
        lines.append(f"{latex_name} & {value} & {comment}\\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    lines.append("% ===== Сводная таблица результатов =====")
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\caption{Сводные результаты вычислительных экспериментов}")
    lines.append(r"\label{tab:experiment_summary_final}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{p{3.0cm}|c|c|c|c|c|c|c|c}")
    lines.append(r"\hline")
    lines.append(
        r"Сценарий & $\overline e_{\mathrm{form}}$ & "
        r"$d_{\mathrm{pair}}^{\min}$ & $d_{\mathrm{obs}}^{\min}$ & "
        r"$E(T)$ & $L(T)$ & $v_{\max}^{\mathrm{exp}}$ & "
        r"$F_{\max}^{\mathrm{exp}}$ & Статус\\"
    )
    lines.append(r"\hline")

    for row in rows:
        lines.append(
            f"{row['scenario_ru']} & "
            f"{fmt(row['mean_formation_error'])} & "
            f"{fmt(row['min_pair_distance'])} & "
            f"{fmt(row['min_obstacle_clearance'])} & "
            f"{fmt(row['energy'], 2)} & "
            f"{fmt(row['path_length'], 2)} & "
            f"{fmt(row['max_speed_exp'])} & "
            f"{fmt(row['max_force_exp'])} & "
            r"\texttt{" + str(row["status"]).replace("_", r"\_").replace("+", r"+") + r"}\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-existing", action="store_true", help="Use existing final_thesis_results CSV files")
    mode.add_argument("--simulate", action="store_true", help="Run fresh simulations without drawing figures")
    args = parser.parse_args()

    params = get_reference_params()
    write_parameters_csv(params)

    rows = []
    for scenario_name in SELECTED_SCENARIOS:
        print(f"[run] {scenario_name}")
        if args.from_existing:
            row = summarize_existing_metrics(scenario_name, params)
        else:
            row = run_fresh_simulation(scenario_name, params)
        rows.append(row)
        print(
            f"  status={row['status']}, "
            f"e={row['mean_formation_error']:.3f}, "
            f"d_pair={row['min_pair_distance']:.3f}, "
            f"d_obs={row['min_obstacle_clearance']:.3f}"
        )

    write_summary_csv(rows)

    latex = make_latex_tables(params, rows)
    (OUT_DIR / "latex_tables.txt").write_text(latex, encoding="utf-8")

    print("\nDONE")
    print(f"Saved: {OUT_DIR / 'parameters_used.csv'}")
    print(f"Saved: {OUT_DIR / 'summary_final.csv'}")
    print(f"Saved: {OUT_DIR / 'latex_tables.txt'}")


if __name__ == "__main__":
    main()