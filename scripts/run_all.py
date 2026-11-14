"""Полный прогон: данные -> обучение -> оценка. Одна команда для всего проекта.

    python scripts/run_all.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
for _s in (sys.stdout, sys.stderr):          # корректный вывод кириллицы в Windows-консоли
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import make_sample_data
import train
import evaluate
import experiments
import make_map


def main() -> None:
    print("\n=== 1/5 Генерация данных ===")
    make_sample_data.main()
    print("\n=== 2/5 Обучение моделей ===")
    train.main()
    print("\n=== 3/5 Оценка и графики ===")
    evaluate.main()
    print("\n=== 4/5 Эксперименты (бейзлайны, абляция, разброс по seed) ===")
    experiments.main()
    print("\n=== 5/5 Интерактивная карта и ГИС-слои ===")
    make_map.main()
    print("\nГотово. Результаты — в data/outputs, модели — в data/models.")
    print("Демо CLI:   python scripts/predict.py --demo --date 2017-06-15")
    print("Streamlit открывает только реальный research-интерфейс, не это демо.")
    print("DEMO карта: data/outputs/demo/map/risk_map.html")


if __name__ == "__main__":
    from selrisk.cli import entry
    entry("run_all", main)
