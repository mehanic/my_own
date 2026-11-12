#!/usr/bin/env python3
"""Unified HLS lower-third/news OCR engine.

One engine + per-channel JSON configuration files.
Built from the channel-specific scripts supplied with this project.

Examples:
    python3 hls_news_ocr.py --list-channels
    python3 hls_news_ocr.py --channel alArabiya
    python3 hls_news_ocr.py --channel AlMashhad --output-dir /tmp/news
    python3 hls_news_ocr.py --config channel_configs/aljazeeraEnglish.json
    python3 hls_news_ocr.py --validate-configs

    python3 hls_news_ocr.py --config channel_configs/alHadath.json
    python3 api_server.py
"""

from __future__ import annotations

import argparse
from asyncio.log import logger
import builtins
import difflib
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from collections import Counter, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from quote_capture import QuoteCapturePlugin
import stanza
from ticker_service import TickerService, ChannelProfileManager # <-- ДОДАНО: Імпорт сервісу біжучого рядка

# # 1. Видаляємо застарілий пакет
# pip uninstall pynvml -y

# # 2. Встановлюємо актуальний офіційний пакет
# pip install nvidia-ml-py

import os
import logging

os.environ["HTTPX_LOG_LEVEL"] = "WARNING"
os.environ["LOGURU_LEVEL"] = "WARNING"

logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpcore.connection").setLevel(logging.WARNING)
logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)

# Paddle/PaddleOCR environment must be set before importing Paddle.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = BASE_DIR / "channel_configs"

# Heavy OCR dependencies are intentionally loaded only when processing starts.
cv2 = None
np = None
PaddleOCR = None
models = None

log = logging.getLogger("hls_ocr")
log.setLevel(logging.DEBUG)
log.propagate = False


def _emit(*args, **kwargs):
    """Module-local print replacement: logs our messages without patching builtins.print."""
    message = " ".join(str(a) for a in args)
    if log.handlers:
        log.info(message)
    else:
        builtins.print(message, **{k: v for k, v in kwargs.items() if k in {"file", "flush"}})


print = _emit


def configure_logging(output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_filename = str(Path(output_dir) / f"news_log_{datetime.now():%Y%m%d_%H%M%S}.txt")
    
    # 🛠️ КЛЮЧОВА ЗМІНА: Налаштовуємо ROOT логер, щоб він ловив повідомлення з УСІХ модулів
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Очищаємо старі хендлери, щоб уникнути дублювання
    for handler in list(root_logger.handlers):
        try:
            handler.close()
        finally:
            root_logger.removeHandler(handler)

    # 1. Файловий хендлер (зберігає все, включаючи DEBUG з ticker_service)
    fh = logging.FileHandler(log_filename, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-16s | %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_logger.addHandler(fh)

    # 2. Консольний хендлер (тільки INFO і вище, щоб не спамити)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", 
        datefmt="%H:%M:%S"
    ))
    root_logger.addHandler(ch)

    # 3. Приглушуємо зайві бібліотеки (вони все одно йдуть у файл, але не в консоль)
    for logger_name in [
        "sentence_transformers", "qdrant_client", "urllib3", "torch", 
        "paddle", "ppocr", "ppstructure", "httpx", "requests", "stanza"
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    return log_filename

def _load_ocr_dependencies() -> None:
    global cv2, np, PaddleOCR
    if cv2 is not None:
        return
    try:
        import cv2 as _cv2
        import numpy as _np
        from paddleocr import PaddleOCR as _PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "OCR dependencies are missing. Install the same cv2/numpy/paddleocr packages "
            "that were used by the original channel scripts."
        ) from exc
    cv2, np, PaddleOCR = _cv2, _np, _PaddleOCR


def _load_qdrant_models():
    global models
    if models is None:
        from qdrant_client.http import models as _models
        models = _models
    return models


def list_config_paths(config_dir: Path = DEFAULT_CONFIG_DIR) -> list[Path]:
    return sorted(p for p in config_dir.glob("*.json") if p.name != "index.json")


def load_config(channel: str | None = None, config_path: str | None = None,
                config_dir: Path = DEFAULT_CONFIG_DIR) -> dict[str, Any]:
    if config_path:
        path = Path(config_path).expanduser().resolve()
    elif channel:
        direct = config_dir / f"{channel}.json"
        if direct.exists():
            path = direct
        else:
            wanted = re.sub(r"[^a-z0-9]", "", channel.lower())
            matches = [p for p in list_config_paths(config_dir)
                       if re.sub(r"[^a-z0-9]", "", p.stem.lower()) == wanted]
            if len(matches) != 1:
                available = ", ".join(p.stem for p in list_config_paths(config_dir))
                raise ValueError(f"Unknown channel '{channel}'. Available: {available}")
            path = matches[0]
    else:
        raise ValueError("Specify --channel or --config")

    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(path)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    required = ["id", "channel_name", "stream_url", "language", "frame", "ocr",
                "layout", "urgent_detection", "text", "dedup", "llm", "qdrant"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Config {cfg.get('id', '<unknown>')} misses: {', '.join(missing)}")
    if not cfg["stream_url"]:
        raise ValueError(f"Config {cfg['id']}: stream_url is empty")
    if cfg["layout"].get("strategy") not in {
        "fixed", "alarabiya", "alhadath", "mashhad_three", "aj2_location",
        "red_gap", "red_dark", "white_rows_araby", "white_rows_english"
    }:
        raise ValueError(f"Config {cfg['id']}: unknown layout strategy")
    if cfg["urgent_detection"].get("mode") not in {"colors", "keywords", "black_or_keywords"}:
        raise ValueError(f"Config {cfg['id']}: unknown urgent detection mode")

# ==============================================================================
# 🆕 STANZA NLP: Ініціалізація для точного розпізнавання сутностей (NER)
# ==============================================================================
def init_stanza_pipeline(lang: str):
    """Послідовно пробує набори процесорів, поки не знайде суміسний."""
    processors_to_try = [
        'tokenize, mwt, pos, ner',  # Спроба 1: Повний набір (EN, AR, TR)
        'tokenize, pos, ner',       # Спроба 2: Без MWT (UR має NER, але не має MWT)
        'tokenize, pos'             # Спроба 3: Мінімальний набір
    ]
    for processors in processors_to_try:
        try:
            return stanza.Pipeline(lang, processors=processors, use_gpu=False, verbose=False, logging_level='WARN')
        except (ValueError, FileNotFoundError) as e:
            error_msg = str(e).lower()
            if "is not known" in error_msg or "cannot load model" in error_msg:
                log.warning(f"⚠️ Процесор недоступний для '{lang}' ({processors}). Пробуємо спрощений...")
                continue
            else:
                raise e
    log.error(f"❌ Не вдалося ініціалізувати Stanza для мови '{lang}'.")
    return None

# Завантажуємо ресурси тільки для обраних мов (робиться 1 раз при старті)
log.info("📥 Завантаження моделей Stanza (en, ar, tr, ur)...")
stanza.download('en', logging_level='WARN')
stanza.download('ar', logging_level='WARN')
stanza.download('tr', logging_level='WARN')
stanza.download('ur', logging_level='WARN')

# Ініціалізуємо пайплайни
nlp_en = init_stanza_pipeline('en')
nlp_ar = init_stanza_pipeline('ar')
nlp_tr = init_stanza_pipeline('tr')
nlp_ur = init_stanza_pipeline('ur')
nlp_model = nlp_en if nlp_en else None  
log.info("✅ Stanza моделі (EN/AR/TR/UR) успішно ініціалізовано")

def get_nlp_pipeline(text: str):
    """Автоматичний вибір моделі за мовою тексту (Unicode діапазони)."""
    if not text or not text.strip():
        return nlp_en
    
    letters = sum(1 for c in text if c.isalpha())
    if letters == 0:
        return nlp_en

    urdu_chars = set("ڑٹپچژگںھے")
    urdu_count = sum(1 for c in text if c in urdu_chars)
    
    turkish_chars = set("çğıöşüÇĞİÖŞÜâîûÂÎÛ")
    turkish_count = sum(1 for c in text if c in turkish_chars)
    
    arabic_count = sum(1 for c in text if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF')
    
    if (urdu_count / letters) > 0.03:
        return nlp_ur if nlp_ur else nlp_en
    elif (turkish_count / letters) > 0.10:
        return nlp_tr if nlp_tr else nlp_en
    elif (arabic_count / letters) > 0.30:
        return nlp_ar if nlp_ar else nlp_en
    
    return nlp_en
# ==============================================================================



class HLSNewsOCR:
    VALID_LATIN_ACRONYMS_DEFAULT = {
        "US", "UK", "TV", "AI", "CNN", "BBC", "AP", "AFP", "EU", "UN",
        "NATO", "ISIS", "UAE", "KSA", "IDF", "FOX", "RT", "CGTN", "IMF",
    }
    LOCATION_DATA = {'الكويت': {'en': 'Kuwait', 'coords': (29.3759, 47.9774)}, 'العراق': {'en': 'Iraq', 'coords': (33.2232, 43.6793)}, 'أربيل': {'en': 'Erbil', 'coords': (36.1911, 44.0093)}, 'بغداد': {'en': 'Baghdad', 'coords': (33.3128, 44.3615)}, 'دير البلح': {'en': 'Deir al-Balah', 'coords': (31.4239, 34.3514)}, 'غزة': {'en': 'Gaza', 'coords': (31.5017, 34.4668)}, 'لبنان': {'en': 'Lebanon', 'coords': (33.8547, 35.8623)}, 'إيران': {'en': 'Iran', 'coords': (35.6892, 51.389)}, 'فلسطين': {'en': 'Palestine', 'coords': (31.9522, 35.2332)}, 'سوريا': {'en': 'Syria', 'coords': (33.5138, 36.2765)}, 'اليمن': {'en': 'Yemen', 'coords': (15.5527, 48.5164)}, 'مصر': {'en': 'Egypt', 'coords': (30.0444, 31.2357)}, 'الأردن': {'en': 'Jordan', 'coords': (31.9454, 35.9284)}, 'قطر': {'en': 'Qatar', 'coords': (25.2854, 51.531)}, 'السعودية': {'en': 'Saudi Arabia', 'coords': (24.7136, 46.6753)}, 'الإمارات': {'en': 'UAE', 'coords': (24.4539, 54.3773)}, 'البحرين': {'en': 'Bahrain', 'coords': (26.2285, 50.586)}, 'عمان': {'en': 'Oman', 'coords': (23.588, 58.3829)}, 'المغرب': {'en': 'Morocco', 'coords': (33.9716, -6.8498)}, 'تونس': {'en': 'Tunisia', 'coords': (36.8065, 10.1815)}, 'الجزائر': {'en': 'Algeria', 'coords': (36.7538, 3.0588)}, 'ليبيا': {'en': 'Libya', 'coords': (32.8872, 13.1913)}, 'السودان': {'en': 'Sudan', 'coords': (15.5007, 32.5599)}, 'تركيا': {'en': 'Turkey', 'coords': (39.9334, 32.8597)}, 'أفغانستان': {'en': 'Afghanistan', 'coords': (34.5553, 69.2075)}, 'باكستان': {'en': 'Pakistan', 'coords': (33.6844, 73.0479)}, 'الهند': {'en': 'India', 'coords': (28.6139, 77.209)}, 'روسيا': {'en': 'Russia', 'coords': (55.7558, 37.6173)}, 'أمريكا': {'en': 'USA', 'coords': (38.9072, -77.0369)}, 'بريطانيا': {'en': 'UK', 'coords': (51.5074, -0.1278)}, 'فرنسا': {'en': 'France', 'coords': (48.8566, 2.3522)}, 'ألمانيا': {'en': 'Germany', 'coords': (52.52, 13.405)}}
    LOCATION_MARKERS = set(LOCATION_DATA)

    def __init__(self, config: dict[str, Any], *, stream_url: str | None = None,
                 output_dir: str | None = None):
        _load_ocr_dependencies()
        self.cfg = config
        self.stream_url = stream_url or config["stream_url"]
        self.output_dir = output_dir or config.get("output_dir", "/home/mehanic/Downloads/news_output")
        self.channel_name = config["channel_name"]
        self.channel_icon = config.get("channel_icon", "📺")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        fcfg = config["frame"]
        self.frame_w = int(fcfg.get("width", 960))
        self.frame_h = int(fcfg.get("height", 540))
        self.fps = float(fcfg.get("fps", 3))
        self.sample_every = int(fcfg.get("sample_every_frames", 15))
        self.diagnostic_every = int(fcfg.get("diagnostic_every_frames", 50))
        self.stats_every = int(fcfg.get("stats_every_frames", 100))

        tcfg = config["text"]
        self.TICKER_STOPWORDS = set(tcfg.get("ticker_stopwords", []))
        self.URGENT_KEYWORDS = set(tcfg.get("urgent_keywords", []))
        self.VALID_LATIN_ACRONYMS = set(tcfg.get("valid_latin_acronyms", [])) or self.VALID_LATIN_ACRONYMS_DEFAULT

        dcfg = config["dedup"]
        self.HEADLINE_DEDUP_WINDOW_SEC = float(dcfg.get("headline_window_sec", 200))
        self.URGENT_DEDUP_WINDOW_SEC = float(dcfg.get("urgent_window_sec", 90))
        self.URGENT_COOLDOWN_SEC = float(dcfg.get("urgent_cooldown_sec", 20))
        self.HEADLINE_COOLDOWN_SEC = float(dcfg.get("headline_cooldown_sec", 15))
        self.dedup_similarity = float(dcfg.get("similarity", 0.85))

        use_gpu = self._check_gpu()
        device = "gpu" if use_gpu else "cpu"
        ocfg = config["ocr"]
        log.info(f"🔄 PaddleOCR: {self.channel_name}; device={device.upper()}; rec={ocfg['recognition_model']}")
        self.ocr = PaddleOCR(
            enable_mkldnn=False,
            text_detection_model_name=ocfg.get("detection_model", "PP-OCRv5_mobile_det"),
            text_recognition_model_name=ocfg.get("recognition_model", "arabic_PP-OCRv5_mobile_rec"),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device,
        )

        self.frame_count = 0
        self.headline_changes = 0
        self.urgent_changes = 0
        self.last_headline = ""
        self._headline_windows = deque()
        self._headline_vocab = set()
        self._urgent_windows = deque()
        self._urgent_vocab = set()
        self._last_urgent_normalized = ""
        self._last_urgent_time = 0.0
        self._last_headline_normalized = ""
        self._last_headline_time = 0.0
        self._first_seen_times: dict[str, float] = {}
        self.start_time = 0.0
        self.ffmpeg = None
        self.ffmpeg_log = str(Path(self.output_dir) / "ffmpeg.log")
        self._ffmpeg_log_file = None

        self.qdrant_enabled = False
        self.qdrant_client = None
        self.embedding_model = None
        if self.cfg.get("qdrant", {}).get("enabled", True):
            self._init_qdrant()

        # 🆕 ПІДКЛЮЧЕННЯ QUOTE CAPTURE З ПРАВИЛЬНИМИ ШЛЯХАМИ
        self.job_id = f"job_{datetime.now():%Y%m%d_%H%M%S}"
        # Переконуємося, що цільова директорія (вже з назвою каналу) існує
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        # Передаємо ВЖЕ готову папку каналу як базу. Плагін просто додасть до неї "/quote_capture"
        plugin_config = {
            "channel_name": self.channel_name,
            "output_base_dir": self.output_dir  # <-- Це має бути /home/.../alHadath
        }
        self.quote_plugin = QuoteCapturePlugin(plugin_config)
        self.quote_plugin.setup(self.stream_url, self.job_id)
        try:
            self.quote_plugin.load_models()
        except Exception as e:
            print(f"⚠️ Face models not loaded, will capture text only: {e}")

        # >>> ІНІЦІАЛІЗАЦІЯ TICKER SERVICE <<<
        profile_path = BASE_DIR / "channel_profiles_ticker.json"
        self.ticker_profile_manager = ChannelProfileManager(str(profile_path))
        
        # 🆕 Використовуємо ticker_id з конфігу, або генеруємо як запасний варіант
        safe_channel_id = self.cfg.get("ticker_id", self.channel_name.lower().replace(" ", "_"))
        
        self.ticker_service = TickerService(
            channel_id=safe_channel_id,
            profile_manager=self.ticker_profile_manager,
            output_dir=self.output_dir
        )
        self.ticker_service.init_ocr()
        # <<< КІНЕЦЬ ІНІЦІАЛІЗАЦІЇ TICKER <<<

    def _check_gpu(self) -> bool:
        try:
            import paddle
            return paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        except Exception:
            return False

    # ------------------------------- Qdrant ---------------------------------
    @property
    def collection_name(self) -> str:
        return self.cfg["qdrant"]["collection"]

    def _init_qdrant(self) -> None:
        try:
            from qdrant_client import QdrantClient
            from sentence_transformers import SentenceTransformer
            qm = _load_qdrant_models()
            qcfg = self.cfg["qdrant"]
            self.qdrant_client = QdrantClient(host=qcfg.get("host", "localhost"), port=int(qcfg.get("port", 6333)))
            self.embedding_model = SentenceTransformer(qcfg.get("embedding_model", "BAAI/bge-m3"))
            names = [c.name for c in self.qdrant_client.get_collections().collections]
            if self.collection_name not in names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qm.VectorParams(size=int(qcfg.get("vector_size", 1024)), distance=qm.Distance.COSINE),
                )
                print(f"✅ Qdrant collection created: {self.collection_name}")
                self._create_payload_indexes()
            else:
                print(f"✅ Qdrant collection: {self.collection_name}")
            self.qdrant_enabled = True
        except Exception as exc:
            print(f"⚠️ Qdrant disabled: {exc}")
            self.qdrant_enabled = False

    def _create_payload_indexes(self) -> None:
        if not self.qdrant_client:
            return
        qm = _load_qdrant_models()
        fields = [("timestamp", qm.PayloadSchemaType.FLOAT),
                  ("topic", qm.PayloadSchemaType.KEYWORD),
                  ("news_type", qm.PayloadSchemaType.KEYWORD)]
        if self.cfg.get("features", {}).get("location_markers"):
            fields.append(("locations", qm.PayloadSchemaType.KEYWORD))
        for field, schema in fields:
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name, field_name=field, field_schema=schema
                )
            except Exception as exc:
                log.debug("Payload index %s: %s", field, exc)

    def _generate_embedding(self, text: str, is_query: bool = False) -> list:
        if self.embedding_model is None:
            self._init_qdrant()
        if self.embedding_model is None:
            raise RuntimeError("Embedding model is unavailable")
        prefix = "query: " if is_query else "passage: "
        return self.embedding_model.encode(prefix + text, normalize_embeddings=True).tolist()

    def _save_to_qdrant(self, text: str, metadata: dict) -> None:
        if not self.qdrant_enabled:
            return
        try:
            qm = _load_qdrant_models()
            topic = metadata.get("topic", "Unknown")
            entities = metadata.get("entities", {})
            countries = entities.get("countries", [])
            countries_str = ", ".join(countries) if countries else "Не вказано"
            vector = self._generate_embedding(
                f"Тема: {topic}. Країни: {countries_str}. Текст: {text}", is_query=False
            )
            payload = {
                "text": text,
                "timestamp": metadata.get("timestamp", 0),
                "news_type": metadata.get("news_type", "UNKNOWN"),
                "screenshot_path": metadata.get("screenshot_path", ""),
                "time_range": metadata.get("time_range", ""),
                "channel": self.channel_name,
                "entities": entities,
                "topic": topic,
                "sentiment": metadata.get("sentiment", "neutral"),
                "locations": metadata.get("locations", []),
                "location_details": metadata.get("location_details", []),
                "is_live": metadata.get("is_live", False),
            }
            normalized = self._normalize_for_dedup(text)
            doc_id = int(hashlib.md5(normalized.encode("utf-8")).hexdigest(), 16) % (10**18)
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[qm.PointStruct(id=doc_id, vector=vector, payload=payload)],
            )
            log.debug("Saved to Qdrant: %s...", text[:40])
        except Exception as exc:
            log.warning("Qdrant save failed: %s", exc)

    def search_similar_news(self, query_text: str, top_k: int = 5, hours_back: int = 24) -> list[dict]:
        if not self.qdrant_enabled:
            self._init_qdrant()
        if not self.qdrant_enabled:
            return []
        try:
            qm = _load_qdrant_models()
            query_vector = self._generate_embedding(query_text, is_query=True)
            cutoff = (datetime.now() - timedelta(hours=hours_back)).timestamp()
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=qm.Filter(must=[qm.FieldCondition(key="timestamp", range=qm.Range(gte=cutoff))]),
                limit=top_k,
                score_threshold=float(self.cfg["qdrant"].get("score_threshold", .7)),
            )
            return [{"text": hit.payload.get("text"), "score": hit.score, **hit.payload} for hit in results]
        except Exception as exc:
            log.warning("Qdrant search failed: %s", exc)
            return []

    # ------------------------------- FFmpeg ---------------------------------
    def start_ffmpeg(self) -> None:
        self._stop_ffmpeg()
        cmd = [
            "ffmpeg", "-reconnect", "1", "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5", "-i", self.stream_url,
            "-vf", f"fps={self.fps:g},scale={self.frame_w}:{self.frame_h}",
            "-f", "image2pipe", "-pix_fmt", "bgr24", "-vcodec", "rawvideo", "-",
        ]
        self._ffmpeg_log_file = open(self.ffmpeg_log, "a", encoding="utf-8")
        self.ffmpeg = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self._ffmpeg_log_file, bufsize=10**8)
        log.info(f"🎬 FFmpeg: {self.stream_url}")

    def _stop_ffmpeg(self) -> None:
        if self.ffmpeg:
            try:
                self.ffmpeg.kill(); self.ffmpeg.wait(timeout=3)
            except Exception:
                pass
            self.ffmpeg = None
        if self._ffmpeg_log_file and not self._ffmpeg_log_file.closed:
            try:
                self._ffmpeg_log_file.flush(); self._ffmpeg_log_file.close()
            except Exception:
                pass
            self._ffmpeg_log_file = None

    # ---------------------------- Regions/layout ----------------------------
    @staticmethod
    def _crop(frame, crop: dict):
        h, w = frame.shape[:2]
        y0, y1 = int(h * crop.get("y0", 0)), int(h * crop.get("y1", 1))
        x0, x1 = int(w * crop.get("x0", 0)), int(w * crop.get("x1", 1))
        return frame[y0:y1, x0:x1]

    def extract_regions(self, frame) -> dict[str, Any]:
        strategy = self.cfg["layout"]["strategy"]
        h, w = frame.shape[:2]
        if strategy == "fixed":
            region = self._crop(frame, self.cfg["layout"]["crop"])
            return {"headline": region, "urgent": region}
        if strategy == "alarabiya":
            return {"headline": frame[int(h*.65):int(h*.80), :], "urgent": frame[int(h*.65):int(h*.92), :]}
        if strategy == "alhadath":
            y0, y1 = int(h*.80), int(h*.92)
            urgent = frame[y0:y1, :]
            headline = frame[y0:y0 + int((y1-y0)*.70), :]
            return {"headline": headline, "urgent": urgent}
        if strategy == "mashhad_three":
            return {
                "topic": frame[int(h*.66):int(h*.74), :],
                "headline": frame[int(h*.73):int(h*.88), :],
                "urgent": frame[int(h*.88):int(h*.97), :],
            }
        if strategy == "aj2_location":
            return {"location_markers": frame[int(h*.70):int(h*.72), :],
                    "headline": frame[int(h*.72):int(h*.90), :],
                    "urgent": frame[int(h*.72):int(h*.90), :]}
        if strategy == "white_rows_araby":
            y0, y1 = int(h*.78), int(h*.93)
            zone = frame[y0:y1, :]
            hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([0,0,200]), np.array([180,40,255]))
            idx = np.where(mask.mean(axis=1)/255.0 > .30)[0]
            max_end = int(h*.90)
            end = min(y0 + idx[-1] + 15, max_end) if len(idx) else max_end
            urgent = frame[y0:end, :]
            headline = frame[y0:y0 + int((end-y0)*.50), :]
            return {"headline": headline, "urgent": urgent}
        if strategy == "white_rows_english":
            y0, y1 = int(h*.70), int(h*.95)
            zone = frame[y0:y1, :]
            gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
            idx = np.where(((gray > 160).astype(np.uint8)*255).mean(axis=1)/255.0 > .15)[0]
            max_end = int(h*.92)
            if len(idx):
                start = y0 + idx[0]; end = min(y0 + idx[-1] + 15, max_end)
                if end-start >= 10:
                    return {"headline": frame[start:start+int((end-start)*.55), :], "urgent": frame[start:end, :]}
            start, end = int(h*.78), int(h*.88)
            return {"headline": frame[start:start+int((end-start)*.55), :], "urgent": frame[start:end, :]}
        if strategy == "red_gap":
            headline = frame[int(h*.72):int(h*.85), :]
            y0, y1 = int(h*.75), int(h*.92)
            hsv = cv2.cvtColor(frame[y0:y1, :], cv2.COLOR_BGR2HSV)
            m1 = cv2.inRange(hsv, np.array([0,100,80]), np.array([10,255,255]))
            m2 = cv2.inRange(hsv, np.array([170,100,80]), np.array([180,255,255]))
            dark_idx = np.where((m1|m2).mean(axis=1)/255.0 < .30)[0]
            max_end = int(h*.93)
            end = min(y0 + dark_idx[0], max_end) if len(dark_idx) else max_end
            return {"headline": headline, "urgent": frame[int(h*.68):end, :]}
        if strategy == "red_dark":
            y0, y1 = int(h*.70), int(h*.93)
            hsv = cv2.cvtColor(frame[y0:y1, :], cv2.COLOR_BGR2HSV)
            r1 = cv2.inRange(hsv, np.array([0,100,100]), np.array([10,255,255]))
            r2 = cv2.inRange(hsv, np.array([170,100,100]), np.array([180,255,255]))
            red = np.where((r1|r2).mean(axis=1)/255.0 > .30)[0]
            dark_m = cv2.inRange(hsv, np.array([0,0,0]), np.array([180,255,80]))
            dark = np.where(dark_m.mean(axis=1)/255.0 > .30)[0]
            rows = red if len(red) else dark
            max_end = int(h*.90)
            if len(rows):
                start = y0 + rows[0]; end = min(y0 + rows[-1] + 10, max_end)
                return {"headline": frame[start:start+int((end-start)*.5), :], "urgent": frame[start:end, :]}
            return {"headline": frame[int(h*.75):int(h*.83), :], "urgent": frame[int(h*.75):int(h*.90), :]}
        raise ValueError(f"Unsupported layout strategy: {strategy}")

    # ----------------------------- Image/OCR --------------------------------
    # def preprocess(self, region):
    #     profile = self.cfg["text"].get("preprocess_profile", "default")
    #     if region is None or region.size == 0:
    #         return region
    #     if profile == "english":
    #         gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    #         enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
    #         img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    #         padded = cv2.copyMakeBorder(img,20,20,60,60,cv2.BORDER_CONSTANT,value=(0,0,0))
    #         scale = 3.0 if padded.shape[0] < 85 else 2.5
    #     elif profile == "alarabiya":
    #         gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    #         gray = cv2.GaussianBlur(gray,(3,3),0)
    #         padded = cv2.copyMakeBorder(gray,15,15,40,40,cv2.BORDER_CONSTANT,value=255)
    #         scale = 3.0 if padded.shape[0] < 100 else 2.5
    #         resized = cv2.resize(padded,(int(padded.shape[1]*scale),int(padded.shape[0]*scale)),interpolation=cv2.INTER_CUBIC)
    #         return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    #     else:
    #         lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
    #         l,a,b = cv2.split(lab)
    #         cl = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8)).apply(l)
    #         img = cv2.cvtColor(cv2.merge((cl,a,b)),cv2.COLOR_LAB2BGR)
    #         padded = cv2.copyMakeBorder(img,10,10,40,40,cv2.BORDER_CONSTANT,value=(0,0,0))
    #         scale = 3.0 if padded.shape[0] < 85 else 2.5
    #     return cv2.resize(padded,(int(padded.shape[1]*scale),int(padded.shape[0]*scale)),interpolation=cv2.INTER_CUBIC)


    def preprocess(self, region):
        """Покращений препроцесинг БЕЗ перевищення лімітів PaddleOCR."""
        profile = self.cfg["text"].get("preprocess_profile", "default")
        if region is None or region.size == 0:
            return region
            
        # ВИПРАВЛЕНО: Помірний масштаб (не більше 3.0)
        scale = 3.0 if region.shape[0] < 85 else 2.5
        
        if profile == "english":
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
            img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            padded = cv2.copyMakeBorder(img, 20, 20, 60, 60, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        elif profile == "alarabiya":
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            padded = cv2.copyMakeBorder(gray, 15, 15, 40, 40, cv2.BORDER_CONSTANT, value=255)
            scale = 3.0 if padded.shape[0] < 100 else 2.5
            resized = cv2.resize(padded, (int(padded.shape[1] * scale), int(padded.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)
            return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        else:
            # Покращений контраст
            lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
            img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
            padded = cv2.copyMakeBorder(img, 20, 20, 50, 50, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # Масштабування з перевіркою максимального розміру
        new_width = int(padded.shape[1] * scale)
        new_height = int(padded.shape[0] * scale)
        
        # ОБМЕЖЕННЯ: Не більше 2000 пікселів по будь-якій стороні
        max_size = 2000
        if new_width > max_size or new_height > max_size:
            ratio = min(max_size / new_width, max_size / new_height)
            new_width = int(new_width * ratio)
            new_height = int(new_height * ratio)
        
        scaled = cv2.resize(padded, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Sharpening (залишаємо)
        kernel = np.array([[0, -1, 0], 
                           [-1, 5, -1], 
                           [0, -1, 0]])
        sharpened = cv2.filter2D(scaled, -1, kernel)
        
        return sharpened

    #непрацює нормально ще - і невідомо чи треба
    def _separate_text_blocks(self, blocks):
        """
        Розділяє блоки тексту і з'єднує їх двокрапкою, якщо між ними великий розрив.
        Наприклад: "لحج" + "أكثر من 120 ألف" -> "لحج: أكثر من 120 ألف"
        """
        if not blocks:
            return []
        
        # 1. Сортуємо по Y-координаті (вертикальна позиція)
        sorted_blocks = sorted(blocks, key=lambda b: b[0])
        groups = []
        current_group = [sorted_blocks[0]]
        
        for i in range(1, len(sorted_blocks)):
            y_diff = abs(sorted_blocks[i][0] - current_group[-1][0])
            if y_diff > 30: # Новий рядок
                groups.append(current_group)
                current_group = [sorted_blocks[i]]
            else:
                current_group.append(sorted_blocks[i])
        
        if current_group:
            groups.append(current_group)
        
        final_lines = []
        for group in groups:
            # 2. Сортуємо по X (для арабської — справа наліво, тому reverse=True)
            sorted_by_x = sorted(group, key=lambda b: b[1], reverse=True)
            
            line_parts = []
            current_part = [sorted_by_x[0]]
            
            for i in range(1, len(sorted_by_x)):
                # різниця між x_min поточного і x_max попереднього
                x_diff = abs(sorted_by_x[i][1] - current_part[-1][2])  
                
                # Якщо розрив по X більше 100 пікселів — це окремий блок (напр. мітка і текст)
                if x_diff > 100:
                    line_parts.append(current_part)
                    current_part = [sorted_by_x[i]]
                else:
                    current_part.append(sorted_by_x[i])
            
            if current_part:
                line_parts.append(current_part)
            
            # 3. ФОРМАТУВАННЯ: З'єднуємо частини через двокрапку
            part_texts = []
            for part in line_parts:
                # Збираємо слова в межах однієї частини
                words = [b[3] for b in part]
                part_texts.append(' '.join(words))
            
            # Якщо є кілька частин (наприклад, ["لحج", "أكثر من..."]), з'єднуємо їх через ": "
            if len(part_texts) > 1:
                final_lines.append(': '.join(part_texts))
            else:
                final_lines.append(part_texts[0])
        
        return final_lines
        
    def _clean_words(self, text: str) -> str:
        cleaned=[]
        fixes=self.cfg["text"].get("word_fixes",{})
        for w in text.split():
            w=w.replace("ـ","")
            for old,new in fixes.items():
                w=w.replace(old,new)
            w=re.sub(r'^%([0-9٠-٩]+)$',r'\1%',w)
            if re.search(r'^[أإآ]{2}',w) or 'أأ' in w or 'إإ' in w: continue
            if w in ('■','▪','•','●','◼','⬛'): cleaned.append('■'); continue
            if w in (':','：'): cleaned.append(':'); continue
            if re.fullmatch(r'[0-9٠-٩]+',w): cleaned.append(w); continue
            if re.fullmatch(r'[A-Za-z0-9]+',w):
                if w.upper() in self.VALID_LATIN_ACRONYMS or len(w)>2: cleaned.append(w)
                continue
            w=re.sub(r'^[0-9٠-٩]+|[0-9٠-٩]+$','',w)
            if not w: continue
            if re.match(r'^\d{1,2}:\d{2}',w): continue
            if ':' in w:
                if re.fullmatch(r'[0-9٠-٩.:：]+',w): continue
                no=w.replace(':','').replace('：','')
                if not no or re.fullmatch(r'[0-9٠-٩.]+',no): continue
                cleaned.extend([no,':']); continue
            if len(w)<=1 or (len(w)==3 and w.startswith('ال')) or w in self.TICKER_STOPWORDS: continue
            if len(w)>=4 and w[:2]=='ال' and w[2:4]=='ال': w=w[2:]
            cleaned.append(w)
        return ' '.join(cleaned)

    @staticmethod
    def _group_lines(blocks):
        if not blocks: return []
        ordered=sorted(blocks,key=lambda b:b[0])
        if len(ordered)==1: return [ordered]
        gaps=[ordered[i][0]-ordered[i-1][0] for i in range(1,len(ordered))]
        max_gap=max(gaps); split=gaps.index(max_gap)+1
        lines=[ordered] if max_gap<30 else [ordered[:split],ordered[split:]]
        return [sorted(line,key=lambda b:b[1],reverse=True) for line in lines]

    def _detect_colons(self, image):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        _,binary=cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
        eroded=cv2.erode(binary.astype(np.uint8),np.ones((3,3),np.uint8))
        n,_,stats,centroids=cv2.connectedComponentsWithStats(eroded,connectivity=8)
        dots=[]
        for i in range(1,n):
            x,y,w,h,area=stats[i]
            if 5<=w<=20 and 5<=h<=20 and area>=15: dots.append((centroids[i][0],centroids[i][1],w,h))
        out=[]; used=set()
        for i,(cx,cy,w,h) in enumerate(dots):
            if i in used: continue
            for j in range(i+1,len(dots)):
                if j in used: continue
                cx2,cy2,w2,h2=dots[j]
                if abs(cx-cx2)<12 and 10<abs(cy-cy2)<45 and abs(w-w2)<=6 and abs(h-h2)<=6:
                    out.append((int((cx+cx2)/2),int((cy+cy2)/2))); used|={i,j}; break
        return out

    @staticmethod
    def _insert_colons(line_blocks, colon_positions):
        if not colon_positions: return [b[3] for b in line_blocks]
        result=[]
        for i,(y,x_min,_,text) in enumerate(line_blocks):
            result.append(text)
            if i < len(line_blocks)-1:
                next_x_max=line_blocks[i+1][2]
                if any(next_x_max < cx < x_min and abs(cy-y)<30 for cx,cy in colon_positions): result.append(':')
        return result

    def ocr_urgent_banner(self, region) -> str:
        if region is None or region.size == 0: return ""
        processed=self.preprocess(region)
        result=self.ocr.predict(processed)
        profile=self.cfg["ocr"].get("profile","default")
        min_score=float(self.cfg["ocr"].get("min_score",.25))
        colon_positions=[] if profile=='alarabiya' else self._detect_colons(processed)
        blocks=[]; has_urgent=False
        for page in result:
            boxes=page.get('rec_boxes')
            if boxes is None: boxes=page.get('dt_boxes')
            if boxes is None: boxes=page.get('rec_polys')
            if boxes is None: boxes=page.get('dt_polys')
            for i,(txt,score) in enumerate(zip(page.get('rec_texts',[]),page.get('rec_scores',[]))):
                if score<min_score: continue
                cleaned=self._clean_words(txt.strip())
                if not cleaned: continue
                marker=any(k in cleaned or cleaned in k for k in self.URGENT_KEYWORDS if k)
                if marker: has_urgent=True
                if marker and profile!='alarabiya' and len(cleaned)<=12: continue
                y=x0=x1=0.0
                if boxes is not None and i<len(boxes):
                    try:
                        flat=np.array(boxes[i]).flatten().tolist(); ys=flat[1::2]; xs=flat[0::2]
                        if ys: y=float((min(ys)+max(ys))/2)
                        if xs: x0=float(min(xs)); x1=float(max(xs))
                    except Exception: pass
                blocks.append((y,x0,x1,cleaned))
        if not blocks: return ""
        lines=sorted(self._group_lines(blocks),key=lambda line:min(b[0] for b in line))
        line_texts=[]
        for line in lines:
            if profile=='english':
                ordered=sorted(line,key=lambda b:b[1])
            else:
                ordered=sorted(line,key=lambda b:b[1],reverse=True)
            words=[b[3] for b in ordered] if profile=='alarabiya' else self._insert_colons(ordered,colon_positions)
            line_texts.append(' '.join(words))
        full=re.sub(r'\s+',' ',' '.join(line_texts)).strip()
        if has_urgent:
            prefix='BREAKING: ' if self.cfg['language']=='en' else 'عاجل: '
            if not full.upper().startswith('BREAKING:') and not full.startswith('عاجل:'): full=prefix+full
        return full

    # --------------------------- Text filtering -----------------------------
    def _is_urgent_garbage(self, text: str) -> bool:
        profile=self.cfg['text'].get('garbage_profile','arabic')
        if profile=='english':
            if not text or len(text.strip())<5: return True
            if re.search(r'[:|]+\s*[a-zA-Z0-9]+\s*[:|]+',text): return True
            clean=re.sub(r'\b(BREAKING|URGENT|LIVE|JUST IN|DEVELOPING|ALERT)\b','',text,flags=re.I)
            clean=re.sub(r'[:\-–—]+',' ',clean).strip()
            return len([w for w in clean.split() if re.search(r'[A-Za-z0-9]',w) and len(w)>1])<2
        if profile=='alarabiya':
            if not text or len(text.strip())<10: return True
            ar=[w for w in text.split() if re.search(r'[\u0600-\u06FF]',w)]
            if len(ar)>=3: return False
            if len(ar)<2:
                if text.lower().strip() in ['alarabiya','al arabia','sky news','cnn','rt']: return True
                if re.search(r'[A-Za-z]',text) and not ar: return True
            return False
        checked=text
        if profile=='branding_ar':
            checked=re.sub(r'\bSHARQIYA\s*NEWS\b','',checked,flags=re.I).strip()
            clean=re.sub(r'عاجل|:|-','',checked).strip()
            if len([w for w in clean.split() if re.search(r'[\u0600-\u06FF]',w)])<3: return True
            if re.search(r'[A-Za-z]',checked):
                parts=re.findall(r'[A-Za-z0-9]+',checked)
                if any(len(p)>2 and p.upper() not in self.VALID_LATIN_ACRONYMS for p in parts): return True
            return bool(re.search(r'[:|]+\s*[a-zA-Z0-9]+\s*[:|]+',checked))
        if re.search(r'[A-Za-z]',checked):
            # Preserve the original strict Arabic behavior while allowing configured acronyms.
            parts=re.findall(r'[A-Za-z0-9]+',checked)
            if any(p.upper() not in self.VALID_LATIN_ACRONYMS for p in parts): return True
        if re.search(r'[:|]+\s*[a-zA-Z0-9]+\s*[:|]+',checked): return True
        clean=re.sub(r'عاجل|:|-','',checked).strip()
        return len([w for w in clean.split() if re.search(r'[\u0600-\u06FF]',w)])<3

    def _prefilter_text(self, text: str) -> bool:
        """True means keep. Implements the Extra News prefilter when configured."""
        if self.cfg['text'].get('prefilter')!='extra_news': return True
        s=text.strip()
        if len(s)<15: return False
        digits=re.sub(r'[^\d٠-٩]','',s)
        if len(digits)>len(s)*.70: return False
        return len([w for w in s.split() if re.search(r'[\u0600-\u06FF]',w)])>=3

    def _is_pure_identification(self, text: str) -> bool:
        if not self.cfg['text'].get('pure_identification_filter'): return False
        if not text or len(text.strip())<10: return False
        words=text.strip().split()
        verbs={'have','has','is','are','was','were','kill','strike','attack','say','report','launch'}
        if 2<=len(words)<=4 and all(w[0].isupper() for w in words if w) and not any(w.lower() in verbs for w in words):
            return True
        indicators=['have','has','is','are','was','were','been','being','kill','strike','attack','launch','report','say','tell','after','during','amid','as','in','on','at','least','more','people','killed','injured','dead','forces','military','government','official']
        lower=text.lower()
        return not any(x in lower for x in indicators)

    def _is_urgent(self, text: str, region) -> bool:
        ucfg=self.cfg['urgent_detection']; mode=ucfg.get('mode','keywords')
        keyword_hit=any(k in text for k in self.URGENT_KEYWORDS if k)
        if mode=='keywords': return keyword_hit
        if mode=='black_or_keywords':
            gray=cv2.cvtColor(region,cv2.COLOR_BGR2GRAY)
            ratio=float(np.sum(gray<int(ucfg.get('black_threshold',50))))/(region.shape[0]*region.shape[1])
            return keyword_hit or ratio>float(ucfg.get('min_ratio',.30))
        hsv=cv2.cvtColor(region,cv2.COLOR_BGR2HSV); total=region.shape[0]*region.shape[1]
        for rule in ucfg.get('colors',[]):
            count=0
            for r in rule.get('ranges',[]):
                lo=np.array(r[:3]); hi=np.array(r[3:]); count += cv2.countNonZero(cv2.inRange(hsv,lo,hi))
            if count/total > float(rule.get('min_ratio',1.0)): return True
        return False

    @staticmethod
    def _clean_time_and_timezone(text: str) -> str:
        if not text: return text
        text=re.sub(r'\b\d{1,2}:\d{2}\b','',text)
        text=re.sub(r'توقيت\s+\w+','',text)
        for p in ['غرينتش','مكة','القاهرة','بغداد','الرياض','دبي','الكويت','GMT','UTC','EET','AST']:
            text=re.sub(r'\b'+p+r'\b','',text)
        text=re.sub(r'\b(?:الساعة|بتوقيت|محلي)\b','',text)
        text=re.sub(r'\s+',' ',text).strip(); return re.sub(r'\s+([:.،،])',r'\1',text)

    def _has_color_background(self, region: np.ndarray) -> bool:
        """
        Перевіряє, чи має регіон кольорове тло (не чорне/біле/сіре).
        Повертає True якщо це новинний банер (синій, червоний тощо).
        """
        if region is None or region.size == 0:
            return False
            
        # Конвертуємо в HSV для аналізу кольору
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Середня насиченість (saturation)
        avg_saturation = np.mean(s)
        
        # Середня яскравість (value)
        avg_brightness = np.mean(v)
        
        # Якщо насиченість висока (>30) — це кольорове тло
        # Чорно-біле/сіре тло має низьку насиченість (<15)
        return avg_saturation > 30 and 50 < avg_brightness < 200

    def _remove_channel_branding(self, text: str, region: np.ndarray = None) -> str:
        """
        Розумне видалення брендингу каналу.
        Відрізняє брендинг (الحدث | عاجل) від підмета речення (الحدث ترصد مخلفات...).
        """

           # ПРАВИЛО 0: Якщо є кольорове тло — це контент, НЕ видаляємо нічого!
        if region is not None and self._has_color_background(region):
            return text  # Залишаємо текст повністю
        
        if not self.cfg['text'].get('remove_branding', True):
            return text
            
        brands = self.cfg['text'].get('branding_words', [])
        if not brands:
            return text

        original_text = text
        text = text.strip()

        for brand in brands:
            brand = brand.strip()
            if not brand:
                continue
                
            # Перевіряємо, чи текст починається з назви бренду
            if text.startswith(brand):
                after_brand = text[len(brand):]
                
                # ПРАВИЛО 1: Якщо після бренду йде роздільник (|, •, :, -, /) — це брендинг. Видаляємо.
                if re.match(r'^\s*[\|\•\:\-\–—\/\\\s]+', after_brand):
                    text = re.sub(r'^' + re.escape(brand) + r'\s*[\|\•\:\-\–—\/\\\s]*', '', text).strip()
                    continue
                
                # ПРАВИЛО 2: Якщо після бренду йде текст, і загальна довжина значно більша за довжину бренду.
                # Це означає, що бренд є підметом речення (напр. "الحدث ترصد مخلفات..."). НЕ видаляємо.
                # Поріг +15 символів гарантує, що ми не чіпаємо повноцінні фрази.
                if len(text) > len(brand) + 15:
                    continue  # Це речення, залишаємо як є
                
                # ПРАВИЛО 3: Якщо текст короткий і майже дорівнює бренду — це просто логотип/водяний знак.
                if len(text) <= len(brand) + 5:
                    text = text.replace(brand, '', 1).strip()
                    continue

        # ПРАВИЛО 4: Стандартна фільтрація слів бренду з середини рядка (для бігучих рядків)
        style = self.cfg['text'].get('branding_style', 'words')
        preserve_lines = style in {'exact_lines'} or self.cfg.get('id') == 'AlMashhad'
        lines = text.split('\n') if preserve_lines else [text]
        out = []
        
        brand_names_lower = [b.lower() for b in brands if b]
        
        for line in lines:
            words = []
            for word in line.split():
                low = word.lower().strip()
                # Видаляємо слово тільки якщо воно точно збігається з брендом
                if low in brand_names_lower:
                    continue
                words.append(word)
            if words:
                out.append(' '.join(words))
                
        result = ('\n' if preserve_lines else ' ').join(out).strip()
        
        # Захист: якщо після всіх маніпуляцій текст став порожнім, повертаємо оригінал
        if not result and original_text:
            return original_text
            
        return result

    @staticmethod
    def _fix_aj_colons(text: str) -> str:
        if not text: return text
        patterns=[r'(مراسل الجزيرة)\s+',r'(ديوان رئيس الوزراء الإسرائيلي)\s+',r'(ديوان رئيس الوزراء اللبناني)\s+',r'(وكالة الأنباء)\s+\w+\s+',r'(مصدر أمني لبناني)\s+',r'(الجيش اللبناني)\s+',r'(وزير الدفاع الإسرائيلي)\s+',r'(الرئيس اللبناني)\s+',r'(الصليب الأحمر|الصليب الاحمر|الصليب الأحمرة)\s+',r'(ترامب)\s+',r'(زيلينسكي)\s+',r'(رويترز)\s+',r'(فانس)\s+',r'(وزير الخزانة الأمريكي)\s+',r'(وزير النفط الإيراني)\s+',r'(المتحدثة|لمتحدثة)\s+باسم\s+الحكومة\s+الإيرانية\s+',r'(سرايا القدس)\s+']
        for pat in patterns:
            if re.search(pat,text) and ': ' not in text and ' :' not in text: text=re.sub(pat,r'\1: ',text)
        return text

    def _normalize_for_dedup(self, text: str) -> str:
        if not text: 
            return ''
            
        if self.cfg['language'] == 'en':
            text = text.lower()
            text = re.sub(r'[:\-–—.,،\s]+', ' ', text)
            text = re.sub(r'\b(breaking|urgent|live|just in|alert)\b', '', text)
            return ' '.join(text.split()).strip()
        
        # 1. Нормалізація діакритики (tashkeel) та варіацій літер
        text = (text.replace('ـ', '')
                    .replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
                    .replace('ى', 'ي').replace('ة', 'ه'))
                    
        # 2. Видалення пунктуації та слів-маркерів "терміново"
        text = (re.sub(r'[:\-–—.,،]+', '', text)
                    .replace('عاجل', '')
                    .replace('عاجلة', ''))
        text = text.replace(' ', '')
        return text.strip()

    # ---------------------------- Enrichment --------------------------------
    def _enrich_with_llm(self, text: str) -> dict:
        fallback={"entities":{"countries":[],"cities":[],"persons":[],"companies":[],"religions":[]},"topic":"Unknown","sentiment":"neutral"}
        lcfg=self.cfg['llm']
        if not lcfg.get('enabled',True): return fallback
        profile=lcfg.get('profile','default')
        if profile=='english': intro='Analyze the following news text.'
        elif profile=='mashhad': intro='Analyze the following Arabic news text. The text has 3 lines: [Topic], [Headline], [Urgent News]. Extract entities ONLY from the text. DO NOT invent names that are not in the text.'
        else: intro='Analyze the following Arabic news text.'
        prompt=f"""{intro}\nReturn STRICTLY valid JSON without markdown formatting.\n\nText: \"{text}\"\n\nRequired JSON structure:\n{{\"entities\":{{\"countries\":[],\"cities\":[],\"persons\":[],\"companies\":[],\"religions\":[]}},\"topic\":\"One word topic in English\",\"sentiment\":\"negative or neutral or positive\"}}"""
        try:
            import requests
            r=requests.post(lcfg.get('url','http://localhost:11434/api/generate'),json={'model':lcfg.get('model','qwen2.5:3b'),'prompt':prompt,'stream':False,'format':'json'},timeout=float(lcfg.get('timeout_sec',30)))
            r.raise_for_status(); raw=r.json().get('response','{}'); raw=re.sub(r'^```json\s*|\s*```$','',raw,flags=re.M)
            return json.loads(raw)
        except Exception as exc:
            log.warning('LLM enrichment failed: %s',exc); return fallback

    def _save_enriched_json(self, text: str, enriched: dict, news_type: str, screenshot_path: str, time_range: str) -> None:
        d=Path(self.output_dir)/'json'; d.mkdir(parents=True,exist_ok=True)
        record={'timestamp':datetime.now().isoformat(),'channel':self.channel_name,'lang':self.cfg['language'],'news_type':news_type,'original_text':text,'screenshot':screenshot_path,'time_range':time_range,'enrichment':enriched}
        with (d/'breaking_and_top_news.jsonl').open('a',encoding='utf8') as f: f.write(json.dumps(record,ensure_ascii=False)+'\n')

    # ------------------------------ Location --------------------------------
    def extract_location_from_markers(self, region):
        if region is None or region.size==0: return [],False,[]
        result=self.ocr.predict(self.preprocess(region)); locations=[]; live=False; details=[]
        for page in result:
            for txt,score in zip(page.get('rec_texts',[]),page.get('rec_scores',[])):
                if score<.5: continue
                s=txt.strip()
                if s=='مباشر': live=True; continue
                if s in self.LOCATION_DATA:
                    locations.append(s); data=self.LOCATION_DATA[s]; details.append({'arabic':s,'english':data['en'],'coords':data['coords']})
        return locations,live,details

    def search_news_by_location(self, location_ar: str, top_k: int=10, hours_back: int=24) -> list[dict]:
        if not self.qdrant_enabled: return []
        try:
            qm=_load_qdrant_models(); cutoff=(datetime.now()-timedelta(hours=hours_back)).timestamp()
            results,_=self.qdrant_client.scroll(collection_name=self.collection_name,scroll_filter=qm.Filter(must=[qm.FieldCondition(key='locations',match=qm.MatchValue(value=location_ar)),qm.FieldCondition(key='timestamp',range=qm.Range(gte=cutoff))]),limit=top_k,with_payload=True,with_vectors=False)
            return [dict(x.payload) for x in results]
        except Exception as exc: log.warning('Location search failed: %s',exc); return []


    def _extract_entities_stanza(self, text: str) -> dict:
        """Швидке та точне витягування сутностей через Stanza замість LLM."""
        fallback = {"entities": {"countries": [], "cities": [], "persons": [], "companies": [], "religions": []}, "topic": "Unknown", "sentiment": "neutral"}
        if not text or nlp_model is None:
            return fallback
            
        try:
            # 1. Автоматично вибираємо мову (AR, TR, UR або EN)
            pipeline = get_nlp_pipeline(text)
            if pipeline is None:
                return fallback
                
            # 2. Обробляємо текст
            doc = pipeline(text)
            
            entities = {"countries": [], "cities": [], "persons": [], "companies": [], "religions": []}
            
            # 3. Збираємо сутності
            for ent in doc.ents:
                ent_type = ent.type  # PERSON, ORG, GPE, LOC тощо
                ent_text = ent.text.strip()
                
                if ent_type in ["PERSON"]:
                    if ent_text not in entities["persons"]: entities["persons"].append(ent_text)
                elif ent_type in ["GPE", "LOC"]:
                    if ent_text not in entities["cities"]: entities["cities"].append(ent_text) # GPE часто є містом/країною
                elif ent_type in ["ORG"]:
                    if ent_text not in entities["companies"]: entities["companies"].append(ent_text)
                    
            return {
                "entities": entities,
                "topic": "News", # Stanza не визначає тему, залишаємо базове або додаємо просту логіку
                "sentiment": "neutral"
            }
        except Exception as e:
            log.warning(f"⚠️ Помилка Stanza NER: {e}")
            return fallback

    def search_news_by_location_en(self, location_en: str, top_k: int=10, hours_back: int=24) -> list[dict]:
        ar=next((k for k,v in self.LOCATION_DATA.items() if v['en'].lower()==location_en.lower()),None)
        return self.search_news_by_location(ar,top_k,hours_back) if ar else []

    def get_location_heatmap(self, hours_back: int=24) -> list[dict]:
        if not self.qdrant_enabled: return []
        try:
            qm=_load_qdrant_models(); cutoff=(datetime.now()-timedelta(hours=hours_back)).timestamp()
            results,_=self.qdrant_client.scroll(collection_name=self.collection_name,scroll_filter=qm.Filter(must=[qm.FieldCondition(key='timestamp',range=qm.Range(gte=cutoff))]),limit=1000,with_payload=True,with_vectors=False)
            counts=Counter(x for news in results for x in news.payload.get('locations',[]))
            return [{'arabic':ar,'english':self.LOCATION_DATA.get(ar,{}).get('en',ar),'coords':self.LOCATION_DATA.get(ar,{}).get('coords',(0,0)),'count':count} for ar,count in counts.most_common()]
        except Exception as exc: log.warning('Heatmap failed: %s',exc); return []

    # ---------------------------- Main pipeline -----------------------------
    def _ocr_frame_text(self, regions: dict[str, Any]) -> str:
        if self.cfg['layout']['strategy']=='mashhad_three':
            topic=self.ocr_urgent_banner(regions['topic']).strip(); headline=self.ocr_urgent_banner(regions['headline']).strip(); urgent=self.ocr_urgent_banner(regions['urgent']).strip()
            lines=[]
            if topic: lines.append(topic)
            if headline: lines.append(headline)
            if urgent: lines.append(urgent if 'عاجل' in urgent else 'عاجل: '+urgent)
            return '\n'.join(lines)
        return self.ocr_urgent_banner(regions['urgent'])

    # def _write_header_files(self) -> tuple[str,str,str]:
    #     files=[str(Path(self.output_dir)/x) for x in ('headlines.txt','all_news.txt','urgent.txt')]
    #     for fn in files:
    #         with open(fn,'w',encoding='utf8') as f:
    #             f.write(f"# 📺 {self.channel_icon} {self.channel_name}\n# Потік: {self.stream_url}\n# Початок: {datetime.now()}\n\n")
    #     return files[0],files[1],files[2]

    # def save_to_file(self, filename: str, text: str, region_type: str) -> None:
    #     with open(filename,'a',encoding='utf8') as f:
    #         f.write(f"\n{'='*80}\n📺 {self.channel_icon} | {datetime.now():%Y-%m-%d %H:%M:%S}\n {region_type}\n📰 {text}\n{'='*80}\n")

    def _save_diagnostics(self, frame, regions: dict[str,Any]) -> None:
        mean = float(np.mean(frame))
        std = float(np.std(frame))
        if not ((20 < mean < 235) and std > 15):
            print(f"⚠️ damaged frame: mean={mean:.1f}, std={std:.1f}")
            return
        
        # Безпечна назва каналу (замінюємо пробіли та слеші на _)
        safe_channel = self.channel_name.replace(" ", "_").replace("/", "_")
        # Повна дата і час у форматі РРРР-ММ-ДД_ГГ-ХВ-СС
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # 1. Збереження сирого кадру в окрему папку
        raw_dir = Path(self.output_dir) / "raw_frames"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_filename = f"frame_{self.frame_count}_{safe_channel}_{timestamp}_raw.jpg"
        cv2.imwrite(str(raw_dir / raw_filename), frame)
        
        # 2. Збереження прев'ю регіонів (urgent_preview, headline_preview, тощо)
        for name, region in regions.items():
            if region is None or region.size == 0:
                continue
            
            # Створюємо папку саме для цього типу прев'ю (напр. ~/Downloads/news_output/urgent_preview)
            region_dir = Path(self.output_dir) / f"{name}_preview"
            region_dir.mkdir(parents=True, exist_ok=True)
            
            # Формуємо ім'я файлу: frame_50_Al_Hadath_2026-10-22_15-30-45_urgent_preview.jpg
            filename = f"frame_{self.frame_count}_{safe_channel}_{timestamp}_{name}_preview.jpg"
            
            try:
                cv2.imwrite(str(region_dir / filename), self.preprocess(region))
            except Exception as exc:
                log.debug('preview %s failed: %s', name, exc)

    def _update_windows(self, is_urgent: bool, banner_text: str, now: float) -> None:
        windows=self._urgent_windows if is_urgent else self._headline_windows
        seconds=self.URGENT_DEDUP_WINDOW_SEC if is_urgent else self.HEADLINE_DEDUP_WINDOW_SEC
        windows.append((now,set(banner_text.split())))
        while windows and now-windows[0][0]>seconds: windows.popleft()
        vocab=set()
        for _,words in windows: vocab |= words
        if is_urgent: self._urgent_vocab=vocab
        else: self._headline_vocab=vocab; self.last_headline=banner_text

    def process_stream(self) -> None:
        self.start_time=time.time(); print(f"\n📺 Start: {self.channel_name}\n📁 Output: {self.output_dir}/")
        self.start_ffmpeg()
        frame_size=self.frame_w*self.frame_h*3
        try:
            while True:
                raw=self.ffmpeg.stdout.read(frame_size)
                if not raw or len(raw)<frame_size:
                    print('⚠️ stream interrupted; reconnecting...'); self._stop_ffmpeg(); time.sleep(3); self.start_ffmpeg(); continue
                frame=np.frombuffer(raw,dtype=np.uint8).reshape((self.frame_h,self.frame_w,3)); self.frame_count+=1
                regions=self.extract_regions(frame)

                 # >>> ОБРОБКА БІГУЧОГО РЯДКА (ПАРАЛЕЛЬНО, кожен 10-й кадр для економії) <<<
                if self.frame_count % 3 == 0:
                    try:
                        self.ticker_service.process_frame(frame=frame, timestamp=time.time(), frame_index=self.frame_count)
                    except Exception as ticker_err:
                        log.debug(f"⚠️ Помилка Ticker Service: {ticker_err}")
                # <<< КІНЕЦЬ ОБРОБКИ БІГУЧОГО РЯДКА <<<

                if self.diagnostic_every and self.frame_count%self.diagnostic_every==0: self._save_diagnostics(frame,regions)
                if self.frame_count%self.sample_every==0:
                    banner=self._ocr_frame_text(regions)
                    if not banner.strip() or not self._prefilter_text(banner) or self._is_urgent_garbage(banner) or self._is_pure_identification(banner):
                        continue
                    urgent=self._is_urgent(banner,regions['urgent']); news_type='URGENT' if urgent else 'TOP_STORY'; now=time.time()
                    clean_for_dedup=self._clean_time_and_timezone(banner); normalized=self._normalize_for_dedup(clean_for_dedup)
                    last=self._last_urgent_normalized if urgent else self._last_headline_normalized
                    last_time=self._last_urgent_time if urgent else self._last_headline_time
                    cooldown=self.URGENT_COOLDOWN_SEC if urgent else self.HEADLINE_COOLDOWN_SEC
                    duplicate=bool(normalized and last and difflib.SequenceMatcher(None,normalized,last).ratio()>self.dedup_similarity and now-last_time<cooldown)
                    
                    
                    if duplicate: continue
                    if urgent: self.urgent_changes+=1; print(f"\n🔴 URGENT #{self.urgent_changes}:")
                    else: self.headline_changes+=1; print(f"\n🔵 TOP STORY #{self.headline_changes}:")

                    # >>> КРОК 1: Спочатку визначаємо регіони, щоб уникнути помилок <<<
                    target_region = regions.get('headline')
                    main_region = regions.get('urgent')  # Регіон, звідки взято основний banner

                    # >>> КРОК 2: Очищаємо основний текст (для Qdrant та JSONL) <<<
                    banner_cleaned = self._clean_time_and_timezone(banner)
                    # Передаємо main_region, щоб перевірити тло основного банера
                    clean = self._fix_aj_colons(self._remove_channel_branding(banner_cleaned, main_region))

                    # >>> КРОК 3: Виклик плагіна збереження цитат (ТІЛЬКИ HEADLINE) <<<
                    if hasattr(self, 'quote_plugin') and self.quote_plugin is not None:
                        quote_only_text = ""
                        
                        if target_region is not None and target_region.size > 0:
                            # 3.1. Спочатку розпізнаємо текст з цього регіону (РОЗКОМЕНТОВАНО!)
                            quote_only_text = self.ocr_urgent_banner(target_region)
                            
                            # 3.2. Потім очищаємо його від брендингу, передаючи target_region для перевірки тла
                            quote_only_text = self._remove_channel_branding(quote_only_text, target_region)
                        
                        # 3.3. Передаємо в плагін тільки "чистий" текст цитати
                        self.quote_plugin.analyze_frame(
                            frame=frame,
                            frame_analysis={
                                'frame_idx': self.frame_count,
                                'start_time_ms': int(now * 1000),
                                'text': quote_only_text,
                                'is_urgent': urgent
                            },
                            video_path=self.stream_url,
                            original_frame=frame
                        )

                    # 1. Збереження скріншоту новини з читабельним іменем
                    screenshots = Path(self.output_dir) / 'screenshots'
                    screenshots.mkdir(parents=True, exist_ok=True)
                    
                    safe_channel = self.channel_name.replace(" ", "_").replace("/", "_")
                    time_str = datetime.fromtimestamp(now).strftime("%Y-%m-%d_%H-%M-%S")
                    
                    # Формат: news_Al_Hadath_2026-10-23_00-19-21_120.jpg
                    shot_filename = f"news_{safe_channel}_{time_str}_{self.frame_count}.jpg"
                    shot = str(screenshots / shot_filename)
                    cv2.imwrite(shot, frame)

                    # 2. Збереження urgent_preview для ЦЬОГО Ж кадру новини
                    urgent_dir = Path(self.output_dir) / "urgent_preview"
                    urgent_dir.mkdir(parents=True, exist_ok=True)
                    urgent_filename = f"news_{self.frame_count}_{safe_channel}_{time_str}_urgent_preview.jpg"
                    
                    if 'urgent' in regions and regions['urgent'] is not None and regions['urgent'].size > 0:
                        try:
                            cv2.imwrite(str(urgent_dir / urgent_filename), self.preprocess(regions['urgent']))
                        except Exception as exc:
                            log.debug('urgent preview save failed: %s', exc)

                    first=self._first_seen_times.setdefault(normalized,now); tr=f"{datetime.fromtimestamp(first):%H:%M:%S} - {datetime.now():%H:%M:%S}"
                    locations=[]; details=[]; live=False
                    if self.cfg.get('features',{}).get('location_markers') and 'location_markers' in regions:
                        locations,live,details=self.extract_location_from_markers(regions['location_markers'])
                    #print(f"   {clean}"); enriched=self._enrich_with_llm(clean)
                    print(f"   {clean}")
                    # Використовуємо Stanza замість LLM для швидкості та точності
                    #enriched = self._extract_entities_stanza(clean)

                     # 1. Отримуємо точні сутності від Stanza (без галюцинацій)
                    stanza_entities = self._extract_entities_stanza(clean)
                    
                    # 2. Отримуємо тему та сентимент від LLM (Ollama)
                    llm_result = self._enrich_with_llm(clean)
                    
                    # 3. Об'єднуємо найкраще з обох світів:
                    enriched = llm_result.copy()  # Беремо topic і sentiment з LLM
                    enriched["entities"] = stanza_entities["entities"]  # Підміняємо сутності на точні від Stanza

                    #systemctl status ollama.service
                    # АБО, якщо хочете залишити LLM для теми/сентименту, але взяти сутності зі Stanza:
                    # llm_result = self._enrich_with_llm(clean)
                    # stanza_entities = self._extract_entities_stanza(clean)
                    # llm_result["entities"] = stanza_entities["entities"] # Замінюємо галюцинації LLM на факти Stanza
                    # enriched = llm_result
                    self._save_enriched_json(clean,enriched,news_type,shot,tr)
                    meta={'timestamp':now,'news_type':news_type,'screenshot_path':shot,'time_range':tr,'entities':enriched.get('entities',{}),'topic':enriched.get('topic','Unknown'),'sentiment':enriched.get('sentiment','neutral'),'locations':locations,'location_details':details,'is_live':live}
                    self._save_to_qdrant(clean,meta)
                    if urgent:
                        self._last_urgent_normalized = normalized
                        self._last_urgent_time = now
                    else:
                        self._last_headline_normalized = normalized
                        self._last_headline_time = now
                    
                    self._update_windows(urgent, banner, now)
                if self.stats_every and self.frame_count%self.stats_every==0:
                    print(f"📊 frame={self.frame_count}, top={self.headline_changes}, urgent={self.urgent_changes}, uptime={int((time.time()-self.start_time)//60)} min")
                time.sleep(.01)
        except KeyboardInterrupt:
            print('\n⏹️ stopped by user')
        except Exception as exc:
            print(f"❌ {exc}"); traceback.print_exc()
        finally:
             # >>> ЗБЕРЕЖЕННЯ ЗАЛИШКІВ TICKER ПЕРЕД ВИХОДОМ <<<
            if hasattr(self, 'ticker_service') and self.ticker_service is not None:
                log.info("💾 Збереження залишкових повідомлень Ticker...")
                self.ticker_service._save_completed_messages()
            # <<< КІНЕЦЬ ЗБЕРЕЖЕННЯ TICKER <<<

            self._stop_ffmpeg(); print(f"✅ Done. frames={self.frame_count}, top={self.headline_changes}, urgent={self.urgent_changes}"); logging.shutdown()


def main(argv: list[str] | None=None) -> int:
    ap=argparse.ArgumentParser(description='Unified HLS News OCR')
    src=ap.add_mutually_exclusive_group()
    src.add_argument('--channel',help='config id, e.g. alArabiya')
    src.add_argument('--config',help='path to a channel JSON config')
    ap.add_argument('--config-dir',default=str(DEFAULT_CONFIG_DIR))
    ap.add_argument('--output-dir',help='override output_dir from config')
    ap.add_argument('--stream-url',help='override stream_url from config')
    ap.add_argument('--no-llm',action='store_true',help='disable Ollama enrichment for this run')
    ap.add_argument('--no-qdrant',action='store_true',help='disable Qdrant for this run')
    ap.add_argument('--list-channels',action='store_true')
    ap.add_argument('--validate-configs',action='store_true')
    args=ap.parse_args(argv); cdir=Path(args.config_dir)
    if args.list_channels:
        for p in list_config_paths(cdir):
            c=json.loads(p.read_text(encoding='utf8')); builtins.print(f"{p.stem:24} {c.get('channel_name','')}  {c.get('stream_url','')}")
        return 0
    if args.validate_configs:
        bad=0
        for p in list_config_paths(cdir):
            try: validate_config(json.loads(p.read_text(encoding='utf8'))); builtins.print(f"OK  {p.name}")
            except Exception as exc: bad+=1; builtins.print(f"ERR {p.name}: {exc}")
        return 1 if bad else 0
    if not args.channel and not args.config: ap.error('specify --channel or --config')
    cfg=load_config(args.channel,args.config,cdir)
    if args.no_llm: cfg['llm']['enabled']=False
    if args.no_qdrant: cfg['qdrant']['enabled']=False
    out=args.output_dir or cfg.get('output_dir','/../../Downloads/news_output')
    log_file=configure_logging(out); print(f"📝 Log: {log_file}"); print(f"⚙️ Config: {cfg['_config_path']}")
    HLSNewsOCR(cfg,stream_url=args.stream_url,output_dir=out).process_stream(); return 0


if __name__=='__main__':
    raise SystemExit(main())
