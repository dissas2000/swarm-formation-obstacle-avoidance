# Swarm Formation with 3D Obstacle Avoidance

**Adaptive formation control for a five-UAV group in a 3-D obstacle environment.**

Numerical experiments for a bachelor's thesis (ВКР) on multi-agent systems.

---

## Идея

Группа из 5 дронов (1 лидер + 4 ведомых) движется от старта к цели,
сохраняя формацию и адаптируясь к препятствиям в режиме реального времени.

Центр группы следует по заранее построенному опорному пути.
Каждый дрон получает желаемую позицию:

```
p_i^des(t) = p_c(t) + R(t) · Δ_i^(m(t))
```

Суммарная сила на каждый дрон:

```
F_i = F_i^formation + F_i^obstacle + F_i^damping
F_i^formation = F_i^pos + F_i^sep + F_i^coh
```

| Компонент | Смысл |
|-----------|-------|
| `F_pos`     | притяжение к слоту в формации |
| `F_sep`     | отталкивание от соседей |
| `F_coh`     | удержание связности группы |
| `F_obstacle`| реакция на цилиндрические препятствия |
| `F_damping` | демпфирование скорости |

---

## Режимы адаптации формации

Контроллер выбирает режим автоматически в зависимости от обстановки:

| Режим | Когда включается |
|-------|-----------------|
| `normal`     | нет препятствий поблизости, формация не стеснена |
| `compressed` | боковое пространство сужается, формация сжимается по ширине |
| `column`     | очень узкий проход, перестроение в одну колонну |
| `overflight` | низкое препятствие — группа перелетает сверху |

---

## Структура проекта

```
swarm_formation_obstacle_avoidance_repo/
├── swarm_3d_experiments.py      # Основной файл: модель, сценарии, контроллер
├── final_thesis_runner.py       # Финальный прогон всех сценариев для ВКР
├── requirements.txt
├── .gitignore
├── README.md
│
├── final_thesis_results/        # Финальные результаты
│   ├── summary/                 # Сводные CSV по всем сценариям
│   ├── selected_for_thesis/     # Лучшие рисунки для диплома
│   ├── animations/              # MP4/GIF анимации
│   └── logs/                   # Лог финального прогона
│
├── results_defense_showcase/    # Детальные материалы для защиты
│   ├── figures/                 # PNG-рисунки
│   ├── metrics/                 # CSV метрики
│   ├── animations/              # GIF анимации
│   └── logs/
│
└── pics/                        # Рисунки, отобранные для текста диплома
```

---

## Установка зависимостей

### conda (рекомендуется)

```bash
conda create -n dronesim python=3.10
conda activate dronesim
pip install -r requirements.txt
# для MP4: conda install -c conda-forge ffmpeg
```

### venv

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Запуск

### Один сценарий (быстро, только CSV)

```bash
python swarm_3d_experiments.py --scenario defense_showcase --no-plots
```

### Один сценарий с рисунками

```bash
python swarm_3d_experiments.py --scenario vertical_escape_corridor
```

### Все сценарии + рисунки

```bash
python swarm_3d_experiments.py
```

### Финальный прогон для ВКР (все 7 сценариев + рисунки + проверки)

```bash
python final_thesis_runner.py --skip-animation
```

### Финальный прогон с GIF/MP4 для defense_showcase

```bash
python final_thesis_runner.py
```

### Только MP4 для defense_showcase

```bash
python final_thesis_runner.py --only-mp4
```

---

## Сценарии

| Сценарий | Описание | Основные режимы |
|----------|----------|----------------|
| `free` | Свободное движение, нет препятствий | normal |
| `single` | Одиночные препятствия | normal, column |
| `dense_field` | Плотная среда препятствий | normal, column |
| `all_field_obstacles` | Шахматное поле разной высоты | normal, column, overflight |
| `uniform_field_obstacles` | Шахматное поле одинаковой высоты | normal, compressed, column |
| `defense_showcase` | Демонстрация всех 4 режимов | normal, compressed, column, overflight |
| `vertical_escape_corridor` | Боковой обход закрыт, только вверх | normal, overflight |

---

## Результаты (final_thesis_results/)

| Папка | Содержание |
|-------|-----------|
| `summary/` | `all_metrics_summary.csv` — все метрики по всем сценариям |
| `summary/` | `selected_metrics_table.csv` — краткая таблица для диплома |
| `selected_for_thesis/` | Лучшие PNG для вставки в текст ВКР |
| `animations/` | `defense_main_animation.mp4` (H.264) |
| `logs/` | `final_run_log.txt` — результаты всех проверок |

---

## Метрики

| Метрика | Смысл |
|---------|-------|
| `formation_error` | среднее отклонение дронов от желаемых позиций в формации, м |
| `min_pair_distance` | минимальное расстояние между любой парой дронов, м |
| `min_obstacle_clearance` | минимальный зазор от любого дрона до любого препятствия, м |
| `total_energy` | суммарная потраченная энергия группы (∝ ∫‖F‖² dt) |
| `total_path_length` | суммарная длина путей всех дронов, м |
| `mode` | текущий режим формации (normal/compressed/column/overflight) |
| `center_z` | высота виртуального центра группы, м |

---

## Проверки безопасности (автоматические)

`final_thesis_runner.py` в конце прогона выводит:

- `min_obstacle_clearance > 0` для каждого сценария;
- `min_pair_distance > 0.7` для defense_showcase;
- все 4 режима присутствуют в defense_showcase;
- overflight активируется в vertical_escape_corridor, `max_center_z ∈ [1.7, 1.9]`.
# swarm-formation-obstacle-avoidance
