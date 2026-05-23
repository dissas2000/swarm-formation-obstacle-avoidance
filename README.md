# 3D-эксперименты для ВКР

Тема: **моделирование группового движения в среде с препятствиями**.

Этот пакет содержит автономную Python-симуляцию, не зависящую от ARGoS. Она нужна для численных экспериментов, построения графиков, получения метрик и подготовки иллюстраций для диплома и презентации.

## Математическая модель

Модель агента (система второго порядка):

```text
p_dot_i = v_i
m_i * v_dot_i = F_i
```

Желаемое положение агента:

```text
p_i^des(t) = p_c(t) + R(t) * Delta_i^(m(t))
```

где `p_c(t)` — виртуальный центр, `R(t)` — матрица поворота, `Delta_i^m` — смещение в строю.

### Пять компонент силы управления

```text
F_i = F_i^target + F_i^formation + F_i^obstacle + F_i^damping + F_i^height
```

| Компонента | Формула / смысл |
|---|---|
| `F_target`   | `k_target * (p_c_xy - p_i_xy)` — горизонтальное движение к цели |
| `F_formation`| `F_pos + F_sep + F_coh` — поддержание формации |
| `F_pos`      | `k_pos * R * Delta_i^m` — коррекция к желаемому положению в строю |
| `F_sep`      | отталкивание при слишком малом расстоянии между агентами |
| `F_coh`      | притяжение к центру (удержание группы) |
| `F_obstacle` | `F_rep + F_tan` — нормальная + касательная компонента обхода |
| `F_damping`  | `-k_damp * v_i` — демпфирование скорости |
| `F_height`   | `k_z*(z_des - z_i)*e_z - beta_z*v_z*e_z` — стабилизация высоты |

При режиме `overflight`: `z_des = h_obs + h_safe`, если `h_obs + h_safe ≤ h_max`.

## Четыре режима движения

Иерархия выбора: `normal` → `compressed` → `overflight` (если есть низкое препятствие) → `column`.

| Режим | Смысл | Когда активируется |
|---|---|---|
| `normal`     | базовая формация, широкое расстановление | зазор до препятствий достаточен |
| `compressed` | сжатая формация, уменьшено поперечное расстояние | нормальная формация не помещается |
| `column`     | движение колонной, минимальная ширина | ни normal ни compressed не помещаются |
| `overflight` | перелёт низких препятствий сверху | боковой обход затруднён + препятствие достаточно низкое |

## Ограждение по периметру

Во всех сценариях добавлено ограждение по периметру области:

```text
[-6, 6] x [-6, 6]
```

Ограждение задаётся как набор цилиндрических препятствий с флагом `is_boundary=True`.

Оно нужно, чтобы:

- сцена выглядела как ограниченная рабочая область;
- планировщик не выбирал путь за пределами сцены;
- на визуализациях было понятно, где границы эксперимента.

## Установка зависимостей

Рекомендуется использовать conda или venv.

### Вариант conda

```bash
conda create -n swarm3d python=3.11
conda activate swarm3d
pip install -r requirements.txt
```

### Вариант venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск всех экспериментов

```bash
python swarm_3d_experiments.py
```

После запуска появится папка:

```text
results_3d_extended/
```

В ней будет общая таблица:

```text
summary_all.csv
```

## Запуск одного сценария

```bash
python swarm_3d_experiments.py --scenario wide_barrier
```

## Запуск одного сценария с анимацией

```bash
python swarm_3d_experiments.py --scenario wide_barrier --show
```

или:

```bash
python swarm_3d_experiments.py --scenario low_wall_overflight --show
```

## Доступные сценарии

| Сценарий | Назначение |
|---|---|
| `free`                  | свободное движение без препятствий |
| `single`                | одиночные препятствия |
| `wide_barrier`          | препятствия по всей ширине, боковой облёт |
| `dense_field`           | плотная среда препятствий |
| `checkerboard_equal`    | шахматное поле препятствий одинаковой высоты |
| `checkerboard_unequal`  | шахматное поле препятствий разной высоты |
| `low_wall_overflight`   | низкая стенка, группа перелетает сверху |
| `tall_wall_side`        | высокая стенка, группа облетает сбоку |
| `vertical_exit`         | единственный выход через верх (стена перекрывает ширину) |
| `defense_showcase`      | демонстрация всех четырёх режимов: normal→compressed→column→overflight |

## Метрики (по ВКР)

| Метрика | Смысл |
|---|---|
| `formation_error`       | средняя ошибка формации (м) |
| `min_agent_distance`    | минимальное расстояние между агентами (м) |
| `min_obstacle_clearance`| минимальный зазор до препятствий (м) |
| `energy`                | энергетическая цена: ∫‖F‖² dt (Дж·с) |
| `path_length`           | суммарная длина траекторий всех агентов (м) |
| `max_height`            | максимальная высота полёта (м) |

## Структура результатов

Для каждого сценария и стратегии создаётся папка:

```text
results_3d_extended/<scenario>/<strategy>/
```

Внутри:

| Файл | Содержание |
|---|---|
| `metrics.csv`         | все метрики по времени |
| `trajectory_3d.png`   | 3D-траектории |
| `top_view.png`        | вид сверху |
| `safety_metrics.png`  | formation_error, min_agent_distance, min_obstacle_clearance |
| `height_profile.png`  | профиль высоты max_height(t) |
| `mode_timeline.png`   | переключения режимов движения |

Сводная таблица по всем сценариям:

```text
results_3d_extended/summary_all.csv
```

## Примеры запуска

```bash
# Все сценарии (создаёт results_3d_extended/)
python swarm_3d_experiments1.py

# Один сценарий
python swarm_3d_experiments1.py --scenario defense_showcase

# Со смотрителем анимации
python swarm_3d_experiments1.py --scenario defense_showcase --show

# Демонстрация перелёта
python swarm_3d_experiments1.py --scenario vertical_exit --show

# Базовая симуляция (results_3d/)
python swarm_3d_simulation1.py
```

## Главные сценарии для диплома

| Сценарий | Что показывает |
|---|---|
| `defense_showcase`     | все четыре режима, демонстрационный сценарий |
| `vertical_exit`        | режим overflight как единственный выход |
| `checkerboard_unequal` | адаптация к разновысоким препятствиям |
| `dense_field`          | стресс-тест адаптивной стратегии |
| `wide_barrier`         | сравнение fixed vs adaptive |

## Что отправлять на анализ

После запуска можно отправить архив папки:

```text
results_3d_extended/
```

По ней можно проверить:

- где была минимальная дистанция между дронами;
- насколько безопасно пройдены препятствия;
- какие режимы включались;
- как отличаются `fixed` и `adaptive`;
- какие картинки лучше вставить в диплом.
