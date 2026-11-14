# Обучение, сохранение модели и примеры для РКНП

Основной маршрут работает на зафиксированном реальном датасете v0.2.
Он обучает региональное presence-background ранжирование, а не вероятность
селя в конкретном ущелье. Сайт не обучает модель в браузере: читает сохранённый
JSON со scaler и коэффициентами.

## Установка

Если PR ещё не слит, используйте ветку `feat/real-dataset-v02`.
После слияния достаточно `main`. Команды выполняются из корня репозитория.

```bash
git clone https://github.com/dulattastanbay-dev/selrisk-almaty.git
cd selrisk-almaty
git switch feat/real-dataset-v02
python --version
python -m venv .venv
```

Для воспроизводимого результата используйте Python 3.14 и версии из
`requirements-research.txt`. Данные распакуются из ZIP автоматически;
при обучении не нужны API-ключи или сетевое получение погоды.

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -e '.[dev,app,dataset]' -c requirements-research.txt
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,app,dataset]" -c requirements-research.txt
```

`dataset` нужен для дополнительных тестов геоданных. Для обычного обучения
достаточно `.[dev,app]`. Если PowerShell запрещает активацию, запускайте
`.\.venv\Scripts\python.exe` вместо `python`, не меняя системную политику.

## Обучение и автоматическое сохранение

Полный маршрут обучает POWER и NOAA, сохраняет результаты и обновляет сайт:

```bash
python scripts/run_all.py
```

Обучение только POWER, без изменения сайта:

```bash
python scripts/train.py --kind power
```

Свой каталог результатов:

```bash
python scripts/train.py --kind power --output runs/my-power
```

В указанной папке, по умолчанию `data/outputs/research/`, создаются:

| Файл | Назначение |
|---|---|
| `model_power.joblib` | Полный fitted sklearn Pipeline: StandardScaler и LogisticRegression |
| `model_power.json` | Переносимая модель: порядок признаков, scaler, коэффициенты и границы train |
| `training_power.json` | Версии Python/библиотек и SHA-256 обоих сохранённых форматов |
| `report_power.json` | Выбор C, результаты train/validation/test и ограничения |
| `predictions_power.csv` | Дни, исходные 1/NULL, роль в эксперименте, баллы и статусы |

Для NOAA имена заканчиваются на `_stations`.
Повторный запуск перезаписывает одноимённые файлы в выбранной папке;
для отдельного эксперимента используйте новый `--output`.
Папка `data/outputs/` не коммитится. На GitHub Actions результаты, включая
joblib и manifest, доступны в артефакте `research-results-v02` успешного
workflow; сохранённые JSON для сайта уже версионируются в `web/data/`.
Для скачивания Actions-артефакта может понадобиться вход в GitHub.

`StandardScaler` и регрессия обучаются только на train. Параметр C выбирается
по validation proxy AP из 0.03, 0.3, 3.0. После выбора модель не дообучается
на validation или test. Train POWER: 1981–2005; validation: 2006–2012;
test: 2013–2020. Для NOAA: 1937–1973 / 1974–1981 / 1982–1991.

## Использование сохранённой модели без обучения

Переносимый JSON, безопасный декларативный формат:

```bash
python scripts/predict.py --kind power --model data/outputs/research/model_power.json --date 2013-06-06
python scripts/predict.py --kind power --model runs/my-power/model_power.json --date 2015-07-05
```

Вторая команда предполагает выполненный пример с `--output runs/my-power`.
Без `--model` команда читает `web/data/model_power.json`.
Загрузчик проверяет тип модели, источник, порядок признаков, размерности,
конечность чисел, границы и положительные масштабы; он не вызывает fit.
Эта проверка структуры не доказывает подлинность чужого файла:
для происхождения сравните SHA-256 с доверенным manifest.
Будущая дата вернёт `no_data`, а не искусственный прогноз; отсутствующие
признаки и выход за train-диапазон скрывают балл.

Полный sklearn Pipeline можно загрузить в Python:

```python
from pathlib import Path
import hashlib
import json
import joblib
from selrisk.realdata import load_regional

folder = Path("data/outputs/research")
manifest = json.loads((folder / "training_power.json").read_text(encoding="utf-8"))
path = folder / "model_power.joblib"
assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"][path.name]
# Только собственный/доверенный файл и совпадающие версии среды!
pipeline = joblib.load(path)
frame, features, _ = load_regional("power")
day = frame.loc[frame.date.eq("2013-06-06"), features]
print(pipeline.decision_function(day))
```

`decision_function` не возвращает вероятность. Прямой sklearn-вызов обходит
защиту train-диапазона; для пользовательских расчётов используйте JSON CLI.
Joblib основан на pickle: загрузка недоверенного файла может исполнить код;
межверсионная совместимость sklearn не гарантируется. Совпадение хеша
с manifest из того же недоверенного архива само по себе не делает файл
доверенным ([документация sklearn](https://scikit-learn.org/stable/model_persistence.html)).

## Обновление сайта

После обучения обоих источников в стандартную папку:

```bash
python scripts/export_web.py
npx serve . --config serve.json
```

Откройте адрес сервера с `/web/index.html`, затем раздел
«РКНП · два ретроспективных примера». Экспорт не обучает модель автоматически;
он использует стандартный `data/outputs/research`, не произвольный `--output`.
`run_all.py` уже включает экспорт, повторять его необязательно.
Для публикации GitHub Pages изменения должны попасть в обслуживаемую ветку;
наличие PR само по себе не обновляет действующий сайт.

## Два примера для защиты

Примеры строятся кодом `src/selrisk/presentation.py`, не вписываются вручную.
Порог: 90-й процентиль допустимых validation-баллов с доступным
внутридиапазонным расчётом; превышение определяется как `score >= threshold`.
Далее выбираются первые по времени test-даты выше порога, с точной
подтверждающей записью каталога, не более одной даты на год.
Это прозрачный, но всё равно ретроспективный выбор удачных случаев.

- **2013-06-06:** Ulken Almaty, очаг UA-2 (Kokshoky),
  запись `READING_276`, source FID 169.
- **2015-07-05:** Aksay, очаг A-1 (Akzhar),
  запись `READING_103`, source FID 469.

Обе записи содержатся в зафиксированном снимке
[каталога Reading](https://services1.arcgis.com/PhkL97KbkzUBf4PQ/arcgis/rest/services/Platform/FeatureServer/0).
Каталог опубликованной платформы не является независимым оперативным
подтверждением ([описание исходного исследования](https://www.nature.com/articles/s44304-024-00050-7)).
Точные баллы, порог, погодный cutoff, признаки, ID и полный test-счётчик
сохраняются в `web/data/presentation_examples.json` и показаны на сайте.
Кнопка каждого примера открывает соответствующий день в архиве POWER.
Все признаки заканчиваются до события; реальная доступность реанализа
в тот исторический момент не проверена.

Для выступления: «На отложенном периоде две зарегистрированные даты
получили балл выше фиксированного по validation порога».
Сам test уже изучался: это не новая слепая проверка и не предварительная
регистрация исследовательского протокола.
Не говорите «мы заранее предсказали эти сели», «точность 100%» или
«модель предсказала конкретный бассейн». Рядом нужно показать весь test:
попадания, невыбранные положительные даты, пропуски и выбранные дни
с неизвестным исходом. Последние не являются доказанными ложными тревогами.
Порог иллюстративный, не безопасный порог оповещения.

## Проверки

```bash
pytest -q
pytest -q data/real/v0.2/tests
python scripts/verify_export.py
```

Последняя команда сравнивает экспорт с текущим git HEAD. После осознанного
изменения алгоритма экспорт нужно проверить и закоммитить как новый эталон;
до этого расхождение с прежним HEAD ожидаемо и не должно замалчиваться.
Тесты проверяют перезагрузку Pipeline, JSON-прогноз без fit, отсутствие test
в выборе порога, источники обоих примеров и отказ от вымышленных случаев.
