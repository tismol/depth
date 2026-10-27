# Depth API

Сервис для получения карты относительной глубины по фотографии. Принимает JPEG, PNG или WebP, запускает модель на CPU и сохраняет результат в SQLite. PNG показывает глубину в оттенках серого, NPY содержит матрицу `float32` размером 224×224.

## Требования

- Linux x86_64, macOS на Apple Silicon или WSL2 с Linux x86_64.
- Python **3.12**, Git и Git LFS.

Ubuntu 24.04 / WSL2 с Ubuntu 24.04:

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv git git-lfs curl
```

macOS с установленным Homebrew:

```bash
brew install python@3.12 git git-lfs
```

## Установка

Скачать проект и веса модели:

```bash
git lfs install
git clone https://github.com/tismol/depth.git
cd depth
git lfs pull
```

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
cp .env.example .env
```

## Запуск и проверка

```bash
python -m uvicorn app_final:app --env-file .env --host 127.0.0.1 --port 8000
```

Дождаться `Application startup complete` и `Uvicorn running on http://127.0.0.1:8000`. Оставить терминал открытым. Остановка — `Ctrl+C`; для повторного запуска выполнить ту же команду в активированном окружении.

Открыть [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) в браузере на том же компьютере.

1. В `GET /health` нажать **Try it out → Execute**. Ожидается `200` и `{"status":"ok"}`.
2. В `POST /predict` нажать **Try it out**, выбрать `examples/example1.jpg` в поле `image` и нажать **Execute**. Ожидается `201` с `request_id`.
3. Скопировать `request_id` в `GET /items/{request_id}/depth.png` и выполнить запрос. Скачать результат по ссылке **Download file**. Аналогично маршрут `depth.npy` возвращает числовую матрицу. 
4. `GET /items` показывает историю.

## Маршруты

| Маршрут | Назначение |
| --- | --- |
| `GET /health` | Проверка SQLite и наличия весов |
| `GET /version` | Версия приложения |
| `POST /predict` | Изображение в поле `image`; ответ `201` с ID, именем и временем |
| `GET /items?limit=20` | Последние результаты, `limit` от 1 до 100 |
| `GET /items/{request_id}/depth.png` | Скачать карту глубины |
| `GET /items/{request_id}/depth.npy` | Скачать числовую матрицу |

Проверка через `curl` выполняется в другом терминале из корня проекта при работающем сервере:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/version
curl --fail -F image=@examples/example1.jpg http://127.0.0.1:8000/predict
curl --fail 'http://127.0.0.1:8000/items?limit=5'
```

Для скачивания заменить `ID` идентификатором из ответа `/predict`:

```bash
curl --fail http://127.0.0.1:8000/items/ID/depth.png -o example1-result.png
```

## Настройки

`.env` загружается через `--env-file`; переменные окружения имеют приоритет. После изменения настроек перезапустить сервер.

| Переменная | По умолчанию |
| --- | --- |
| `DATABASE_PATH` | `data/depth.db` |
| `MODEL_WEIGHTS_PATH` | `models/model.pt` |
| `APP_VERSION` | `1.0.0` |
| `MAX_UPLOAD_MB` | `10` МБ |

## Тесты и CI

В активированном окружении из корня проекта:

```bash
python -m pytest -v
```

Тесты проверяют `/health` и версию, настоящее предсказание и скачивание, сохранность базы данных, поврежденное изображение, неизвестный ID и лимит загрузки.

## Диагностика


| Проблема | Действие |
| --- | --- |
| `400` или `413` | Проверить файл и лимит размера |
| `404` или `422` | Проверить ID результата или значение `limit` |
| `503` | Прочитать `detail` и логи; проверить веса и доступ к файлу SQLite |
