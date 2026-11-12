import re

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import os
from pathlib import Path
from collections import Counter, defaultdict
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
# ==========================================
# КОНФІГУРАЦІЯ ШЛЯХІВ (Адаптуйте під себе)
# ==========================================
BASE_OUTPUT_DIR = Path("/home/mehanic/Downloads/news_output")
JSONL_FILE = BASE_OUTPUT_DIR / "json" / "breaking_and_top_news.jsonl"
SCREENSHOTS_DIR = BASE_OUTPUT_DIR / "screenshots"

app = FastAPI(
    title="HLS News OCR API",
    description="API для перегляду та фільтрації збагачених новинних даних",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_channel_paths(channel: str = "alHadath") -> Dict[str, Path]:
    """Повертає словник шляхів для конкретного каналу."""
    # Захист від небезпечних символів у назві каналу
    safe_ch = re.sub(r'[^a-zA-Z0-9_-]', '', channel)
    
    # 🆕 РЕГІСТРОНЕЗАЛЕЖНИЙ ПОШУК ПАПКИ
    base = None
    if (BASE_OUTPUT_DIR / safe_ch).exists():
        base = BASE_OUTPUT_DIR / safe_ch
    else:
        # Шукаємо папку, ігноруючи регістр
        for folder in BASE_OUTPUT_DIR.iterdir():
            if folder.is_dir() and folder.name.lower() == safe_ch.lower():
                base = folder
                break
        
        if not base:
            base = BASE_OUTPUT_DIR / safe_ch  # Фолбек
    
    json_dir = base / "json"
    
    # 1. Спочатку спробуємо знайти файл з точним іменем
    coords_file = json_dir / f"coordinates_{safe_ch.lower()}.jsonl"
    
    # 2.  Якщо не знайдено, шукаємо будь-який файл координат у цій папці
    if not coords_file.exists():
        matching_files = list(json_dir.glob("coordinates_*.jsonl"))
        if matching_files:
            coords_file = matching_files[0]
        else:
            coords_file = json_dir / "coordinates.jsonl"

    return {
        "news": json_dir / "breaking_and_top_news.jsonl",
        "ticker": json_dir / "ticker_news.jsonl",
        "coords": coords_file,
        "speed": json_dir / "ticker_speed.jsonl",
        "panorama": base / "json" / "debug" / "panorama",
        "ocrvis": base / "json" / "debug" / "ocr_visualization",
        "screenshots": base / "screenshots",  # ✅ Тепер правильно
        "raw_frames": base / "raw_frames",
        "headline": base / "headline_preview",
        "urgent": base / "urgent_preview"
    }
# ==========================================
# ДОПОМІЖНІ ФУНКЦІЇ
# ==========================================
def read_jsonl_reverse(channel: str, limit: int = 50, offset: int = 0, 
                       news_type: Optional[str] = None, search: Optional[str] = None) -> Dict[str, Any]:
    """
    Читає JSONL файл конкретного каналу з кінця (найновіші спочатку).
    """
    # 1. Отримуємо правильний шлях для цього каналу
    paths = get_channel_paths(channel)
    file_path = paths["news"]
    
    if not file_path.exists():
        return {"total": 0, "items": []}

    results = []
    total_matched = 0
    
    # 2. Читаємо рядки
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 3. Обробляємо з кінця (найновіші першими)
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
            
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        # 4. Застосування фільтрів (канал вже відфільтрований на рівні файлу!)
        if news_type and item.get("news_type") != news_type:
            continue
        if search and search.lower() not in item.get("original_text", item.get("text", "")).lower():
            continue

        total_matched += 1

        # 5. Пагінація
        if total_matched > offset + limit:
            break
            
        if total_matched > offset:
            results.append(item)

    return {
        "total": total_matched,
        "limit": limit,
        "offset": offset,
        "items": results
    }

# def get_available_channels() -> List[str]:
#     """Повертає список унікальних каналів з файлу."""
#     if not JSONL_FILE.exists():
#         return []
    
#     channels = set()
#     with open(JSONL_FILE, "r", encoding="utf-8") as f:
#         for line in f:
#             try:
#                 item = json.loads(line)
#                 if "channel" in item:
#                     channels.add(item["channel"])
#             except json.JSONDecodeError:
#                 continue
#     return sorted(list(channels))

# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/", response_class=HTMLResponse)
def root():
    html_file = Path(__file__).parent / "index.html"
    if html_file.exists():
        return html_file.read_text(encoding="utf-8")
    return "<h1>Файл index.html не знайдено. Покладіть його поруч з api_server.py</h1>"

@app.get("/api/news")
def get_news(
    limit: int = Query(50, ge=1, le=200, description="Кількість записів на сторінку"),
    offset: int = Query(0, ge=0, description="Зміщення для пагінації"),
    channel: str = Query("alHadath", description="Фільтр по каналу (напр. 'alHadath')"), # 🆕 Змінено на str з дефолтом
    news_type: Optional[str] = Query(None, description="Фільтр по типу: 'TOP_STORY' або 'URGENT'"),
    search: Optional[str] = Query(None, description="Пошук по тексту новини")
):
    return read_jsonl_reverse(channel=channel, limit=limit, offset=offset, news_type=news_type, search=search)

@app.get("/api/stats")
def get_stats(channel: Optional[str] = Query(None, description="Фільтр по каналу (None = всі канали)")):
    """
    Отримати статистику.
    - Якщо channel=None: агрегована статистика по ВСІХ каналах
    - Якщо channel="alHadath": тільки по цьому каналу
    """
    
    # Визначаємо, які папки сканувати
    if channel and channel != "all":
        # Тільки один канал
        channels_to_scan = [channel]
    else:
        # Всі канали (скануємо всі папки в BASE_OUTPUT_DIR)
        if not BASE_OUTPUT_DIR.exists():
            return {"error": "Базова директорія не знайдена"}
        channels_to_scan = [
            d.name for d in BASE_OUTPUT_DIR.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ]
    
    stats = {
        "total_items": 0,
        "by_channel": Counter(),
        "by_type": Counter(),
        "by_sentiment": Counter()
    }
    
    # Читаємо файли для кожного каналу
    for ch in channels_to_scan:
        paths = get_channel_paths(ch)
        if not paths["news"].exists():
            continue
            
        with open(paths["news"], "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    stats["total_items"] += 1
                    stats["by_channel"][item.get("channel", ch)] += 1
                    stats["by_type"][item.get("news_type", "Unknown")] += 1
                    
                    sentiment = item.get("enrichment", {}).get("sentiment", "unknown")
                    stats["by_sentiment"][sentiment] += 1
                    
                except json.JSONDecodeError:
                    continue
    
    return {
        "total_items": stats["total_items"],
        "by_channel": dict(stats["by_channel"]),
        "by_type": dict(stats["by_type"]),
        "by_sentiment": dict(stats["by_sentiment"])
    }

@app.get("/api/channels")
def get_channels():
    if not BASE_OUTPUT_DIR.exists():
        return {"channels": []}
    # Просто читаємо назви папок у BASE_OUTPUT_DIR
    channels = [d.name for d in BASE_OUTPUT_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')]
    return {"channels": sorted(channels)}

@app.get("/api/image/{channel}/{filename}")
def get_screenshot(channel: str, filename: str):
    """Сервінг зображень-скріншотів конкретного каналу."""
    paths = get_channel_paths(channel)  # ✅ Використовуємо нашу функцію
    safe_filename = os.path.basename(filename)
    file_path = paths["screenshots"] / safe_filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Screenshot not found: {file_path}")

    return FileResponse(file_path, media_type="image/jpeg")


@app.get("/api/diagnostic/find")
def find_diagnostic(channel: str, folder: str, prefix: str, suffix: str, timestamp: str = None):
    paths = get_channel_paths(channel)
    
    # Мапінг імені папки на ключ у словнику paths
    folder_map = {
        "urgent_preview": "urgent",
        "headline_preview": "headline",
        "raw_frames": "raw_frames",
        "ocr_visualization": "ocrvis"
    }
    
    path_key = folder_map.get(folder, folder)
    search_dir = paths.get(path_key)
    
    if not search_dir or not search_dir.exists():
        return {"filename": None}
    
    for file in search_dir.iterdir():
        if file.is_file() and file.name.startswith(prefix) and file.name.endswith(suffix):
            if timestamp and timestamp not in file.name:
                continue
            return {"filename": file.name}
            
    return {"filename": None}

# ПОТІМ загальний маршрут "/api/diagnostic/{folder}/{filename}"
@app.get("/api/diagnostic/{folder}/{channel}/{filename}")
def get_diagnostic(folder: str, channel: str, filename: str):
    """Сервінг діагностичних зображень з конкретних підпапок каналу."""
    paths = get_channel_paths(channel)
    
    # Мапінг імені папки з URL на ключ у словнику paths
    folder_map = {
        "urgent_preview": "urgent",
        "headline_preview": "headline",
        "raw_frames": "raw_frames",
        "ocr_visualization": "ocrvis"
    }
    
    path_key = folder_map.get(folder, folder)
    target_dir = paths.get(path_key)
    
    if not target_dir:
        raise HTTPException(status_code=404, detail="Папка не знайдена")
        
    safe_filename = os.path.basename(filename)
    file_path = target_dir / safe_filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Diagnostic image not found: {file_path}")

    return FileResponse(file_path, media_type="image/jpeg")


@app.get("/api/diagnostic/timeline")
def get_diagnostic_timeline(
    channel: str,
    start_time: str,
    end_time: str
):
    paths = get_channel_paths(channel)
    raw_dir = paths["raw_frames"]  # ✅ Тепер шукаємо в папці каналу
    
    if not raw_dir.exists():
        return {"frames": []}
    
    frames = []
    for file in raw_dir.iterdir():
        if file.is_file():
            match = re.search(r'_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_', file.name)
            if match:
                file_time = match.group(1)
                if start_time <= file_time <= end_time:
                    frames.append({
                        "filename": file.name,
                        "timestamp": file_time,
                        "url": f"/api/diagnostic/raw_frames/{channel}/{file.name}"
                    })
    
    frames.sort(key=lambda x: x["timestamp"])
    return {"frames": frames[:6]}



@app.get("/api/diagnostic/headline-timeline")
def get_headline_timeline(channel: str, start_time: str, end_time: str):
    paths = get_channel_paths(channel)
    headline_dir = paths["headline"]  # ✅ Тепер шукаємо в папці каналу
    
    if not headline_dir.exists():
        return {"frames": []}
    
    frames = []
    for file in headline_dir.iterdir():
        if file.is_file():
            match = re.search(r'_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_', file.name)
            if match:
                file_time = match.group(1)
                if start_time <= file_time <= end_time:
                    frames.append({
                        "filename": file.name,
                        "timestamp": file_time,
                        "url": f"/api/diagnostic/headline_preview/{channel}/{file.name}"
                    })
    
    frames.sort(key=lambda x: x["timestamp"])
    return {"frames": frames}



# ==========================================
# 🎯 НОВІ КОНФІГУРАЦІЇ ДЛЯ ТІКЕР-СЕРВІСУ
# ==========================================
#TICKER_JSONL = BASE_OUTPUT_DIR / "json" / "ticker_news.jsonl"
#PANORAMA_DIR = BASE_OUTPUT_DIR / "json" / "debug" / "panorama"

# ==========================================
# 🎯 API ДЛЯ ТІКЕР-ДАШБОРДУ
# ==========================================

@app.get("/ticker", response_class=HTMLResponse)
def ticker_dashboard():
    """Сторінка з тікер-дашбордом (візуалізація біжучого рядка)."""
    html_file = Path(__file__).parent / "ticker.html"
    if html_file.exists():
        return html_file.read_text(encoding="utf-8")
    return "<h1>Файл ticker.html не знайдено. Покладіть його поруч з api_server.py</h1>"


@app.get("/api/tickers")
def get_tickers(
    channel: str = Query(..., description="Назва каналу (напр. alHadath, alArabiya, aljazeeraArabic)"),
    limit: int = Query(500, ge=1, le=5000)
):
    """
    Повертає тікер-повідомлення конкретного каналу (найновіші першими).
    Для дашборду з панорамами та межами речень.
    
    Параметр channel обов'язковий - вкажіть назву каналу.
    """
    paths = get_channel_paths(channel)
    
    if not paths["ticker"].exists():
        return {"error": f"Файл тікерів не знайдено для каналу '{channel}'", "items": []}

    items = []
    with open(paths["ticker"], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Найновіші — першими
    items.reverse()
    return {"total": len(items), "items": items[:limit]}

@app.get("/api/coordinates")
def get_coordinates(
    channel: str = Query(..., description="Назва каналу (напр. alHadath, alArabiya, aljazeeraArabic)")
):
    """
    Повертає повний аналіз проміжків (gap_analysis), згрупований за frame.
    Для дашборду з панорамами та межами речень.
    
    Параметр channel обов'язковий - вкажіть назву каналу.
    """
    paths = get_channel_paths(channel)
    coords_file = paths["coords"]
    
    if not coords_file.exists():
        return {}
    
    try:
        with open(coords_file, "r", encoding="utf-8") as f:
            # Читаємо останні 5000 рядків, щоб не перевантажувати пам'ять
            lines = f.readlines()[-5000:] 
    except Exception as e:
        print(f"⚠️ Помилка читання {coords_file}: {e}")
        return {}

    grouped = defaultdict(lambda: {"gaps": [], "width": 2800})
    
    for line in lines:
        try:
            data = json.loads(line.strip())
            
            # Беремо ТІЛЬКИ gap_analysis, де є frame
            if data.get("type") == "gap_analysis" and "frame" in data:
                frame_key = str(data["frame"])
                
                # Передаємо ВСІ поля, які очікує фронтенд (зокрема w_right та w_left)
                grouped[frame_key]["gaps"].append({
                    "x_start": float(data.get("x_start", 0)),
                    "x_end": float(data.get("x_end", 0)),
                    "w": float(data.get("gap_width", 0)),
                    "w_right": float(data.get("w_right", 0)),
                    "w_left": float(data.get("w_left", 0)),
                    "score": float(data.get("raw_score", 0)),
                    "decision": str(data.get("decision", "UNKNOWN")),
                    # ✅ Поля для підрахунку слів (фронтенд чекає саме на них)
                    "words_before": int(data.get("words_before", 0)),
                    "words_after": int(data.get("words_after", 0)),
                    # Додаткові діагностичні поля
                    "ratio_local": float(data.get("ratio_local", 0)),
                    "vacuum_overlap": float(data.get("vacuum_overlap", 0)),
                    "temporal_bonus": int(data.get("temporal_bonus", 0)),
                    "stability_count": int(data.get("stability_count", 0))
                })
                
                if "panorama_width" in data:
                    grouped[frame_key]["width"] = float(data["panorama_width"])
                    
        except json.JSONDecodeError:
            continue
            
    return dict(grouped)

#SPEED_FILE = BASE_OUTPUT_DIR / "json" / "ticker_speed.jsonl"

@app.get("/api/speed")
def get_speed(
    channel: str = Query(..., description="Назва каналу (напр. alHadath, alArabiya, aljazeeraArabic)"),
    limit: int = Query(1000, ge=1, le=5000)
):
    """
    Метрики швидкості стрічки (ticker_speed.jsonl), найновіші першими.
    
    Параметр channel обов'язковий - вкажіть назву каналу.
    """
    paths = get_channel_paths(channel)
    
    if not paths["speed"].exists():
        return {"total": 0, "items": []}
    
    items = []
    with open(paths["speed"], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    items.reverse()
    return {"total": len(items), "items": items[:limit]}

# ==========================================
#  ДЕБАГ-ВІЗУАЛІЗАЦІЯ OCR (ocr_vis_frame_*.jpg)
# ==========================================
#OCRVIS_DIR = BASE_OUTPUT_DIR / "json" / "debug" / "ocr_visualization"

@app.get("/api/ocrvis")
def list_ocrvis(channel: str = Query("alHadath")):
    paths = get_channel_paths(channel)
    if not paths["ocrvis"].exists():
        return {"total": 0, "items": []}

    items = []
    for file in paths["ocrvis"].iterdir():
        m = re.match(r"ocr_vis_frame_(\d{6})_(\d+)\.jpg", file.name)
        if m and file.is_file():
            items.append({
                "time": m.group(1),
                "frame": int(m.group(2)),
                "filename": file.name,
                "url": f"/api/ocrvis/{channel}/{file.name}", # 🆕 Додано channel в URL
            })
    items.sort(key=lambda x: x["frame"])
    return {"total": len(items), "items": items}

@app.get("/api/ocrvis/{channel}/{filename}") # 🆕 Змінено маршрут
def get_ocrvis(channel: str, filename: str):
    paths = get_channel_paths(channel)
    safe_filename = os.path.basename(filename)
    file_path = paths["ocrvis"] / safe_filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="OCR visualization not found")
    return FileResponse(file_path, media_type="image/jpeg")



@app.get("/api/panoramas")
def list_panoramas(channel: str = Query("alHadath")):
    paths = get_channel_paths(channel)
    if not paths["panorama"].exists():
        return {"items": []}

    items = []
    for file in paths["panorama"].iterdir():
        if file.is_file() and file.name.startswith("panorama_") and file.name.endswith(".png"):
            try:
                frame_num = int(file.stem.replace("panorama_", ""))
                items.append({
                    "frame": frame_num,
                    "filename": file.name,
                    "url": f"/api/panorama/{channel}/{file.name}", # 🆕 Додано channel в URL
                    "size_kb": round(file.stat().st_size / 1024, 1)
                })
            except ValueError:
                continue
    items.sort(key=lambda x: x["frame"])
    return {"total": len(items), "items": items}

@app.get("/api/panorama/{channel}/{filename}") # 🆕 Змінено маршрут
def get_panorama(channel: str, filename: str):
    paths = get_channel_paths(channel)
    safe_filename = os.path.basename(filename)
    if not safe_filename.startswith("panorama_") or not safe_filename.endswith(".png"):
        raise HTTPException(status_code=400, detail="Некоректне ім'я файлу панорами")

    file_path = paths["panorama"] / safe_filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Панорама не знайдена")

    return FileResponse(file_path, media_type="image/png")



# ==========================================
# ТОЧКА ВХОДУ (запуск сервера)
# ==========================================
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 HLS News OCR API Server (Multi-Channel)")
    print("="*60)
    print(f"📂 Базова директорія: {BASE_OUTPUT_DIR}")
    print(f"📺 Підтримка динамічних каналів: УВІМКНЕНО")
    print("-"*60)
    print("🌐 API:        http://127.0.0.1:10000")
    print("📖 Swagger UI: http://127.0.0.1:10000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(
        "api_server:app",
        host="127.0.0.1",
        port=10000,
        reload=True,
        log_level="info"
    )