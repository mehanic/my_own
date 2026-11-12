import logging
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import cv2
import numpy as np

FrameAnalysis = Dict[str, Any]
PluginResult = Optional[Dict]

# 1. Замінюємо відсутній логер на стандартний Python logging
logger = logging.getLogger("quote_capture")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)

# 2. Додаємо функцію часу локально
def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# 3. Безпечний імпорт розпізнавання облич
try:
    from services.analysis.face_recognizer import FaceRecognizer
except ImportError:
    FaceRecognizer = None

class QuoteCapturePlugin:
    """
    Plugin for capturing full-frame screenshots when quotes are detected.
    Масштабований для збереження в папку конкретного каналу.
    """

    def __init__(self, config: dict):
        self.config = config
        self.face_recognizer = None
        self.captured_quotes: List[Dict] = []
        self.quotes_dir: Optional[Path] = None
        self.current_video_path: str = ""
        self.current_job_id: str = ""
        
        # 🆕 Отримуємо назву каналу та базову директорію з конфігу
        # За замовчуванням використовуємо шлях, який ви показали
        self.channel_name = self.config.get("channel_name", "unknown_channel")
        base_output = self.config.get("output_base_dir", "/home/mehanic/Downloads/news_output")
        self.output_base_dir = Path(base_output)
        
        self.quote_counter = 0
        self.urgent_counter = 0

    def load_models(self) -> None:
        if FaceRecognizer is not None:
            try:
                self.face_recognizer = FaceRecognizer(
                    tolerance=0.45,
                    model="VGG-Face",
                    min_face_confidence=0.70,
                    unknown_clustering_threshold=0.45,
                    detector_backend=os.getenv("FACE_DETECTOR_BACKEND", "retinaface")
                )
            except Exception as e:
                logger.warning(f"Face recognizer skipped: {e}")

    def setup(self, video_path: str, job_id: str) -> None:
        """Initialize the plugin for a new video."""
        self.current_video_path = video_path
        self.current_job_id = job_id
        self.quote_counter = 0
        self.urgent_counter = 0
        self.captured_quotes = []
        
        if self.face_recognizer:
            self.face_recognizer.reset_unknown_registry()

        # 🆕 ВИПРАВЛЕНО: output_base_dir ВЖЕ є папкою каналу (напр. .../alHadath).
        # Ми просто додаємо "quote_capture", щоб уникнути подвійної вкладеності та пробілів у назвах.
        self.quotes_dir = Path(self.output_base_dir) / "quote_capture"
        
        # Створюємо директорії (parents=True гарантує створення всіх вкладених папок)
        self.quotes_dir.mkdir(parents=True, exist_ok=True)
        (self.quotes_dir / "full_frames").mkdir(exist_ok=True)
        (self.quotes_dir / "metadata").mkdir(exist_ok=True)

        logger.info(f"✅ QuoteCapture ініціалізовано для каналу: '{self.channel_name}'")
        logger.info(f"📂 Абсолютний шлях збереження: {self.quotes_dir.resolve()}")

        self._cleanup_previous_run()

    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_analysis: FrameAnalysis,
        video_path: str,
        original_frame: np.ndarray
    ) -> FrameAnalysis:
        
        quote_text = frame_analysis.get('text', '').strip()
        is_urgent = frame_analysis.get('is_urgent', False)
        frame_idx = frame_analysis.get('frame_idx', 0)
        timestamp_ms = int(frame_analysis.get('start_time_ms', 0))
        
        # Зберігаємо, якщо є змістовний текст
        should_capture = bool(quote_text and len(quote_text) > 15)
        
        if should_capture:
            self._capture_full_frame(
                frame=original_frame,
                frame_idx=frame_idx,
                timestamp_ms=timestamp_ms,
                quote_text=quote_text,
                is_urgent=is_urgent,
                video_path=video_path
            )
            frame_analysis['quote_captured'] = True
        else:
            frame_analysis['quote_captured'] = False
            
        return frame_analysis

    def _capture_full_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_ms: int,
        quote_text: str,
        is_urgent: bool,
        video_path: str,
        speakers: List[str] = None,
        faces: List[Dict] = None,
        job_id: str = None
    ) -> Optional[Dict]:
        speakers = speakers or []
        faces = faces or []
        job_id = job_id or self.current_job_id
        
        try:
            dt = datetime.fromtimestamp(timestamp_ms / 1000)
            time_str = dt.strftime("%Y-%m-%d_%H-%M-%S")
            
            # Безпечна назва для імені файлу
            safe_channel = self.channel_name.replace(" ", "_").replace("/", "_")
            quote_type = "urgent" if is_urgent else "quote"
            
            if is_urgent:
                self.urgent_counter += 1
                counter = self.urgent_counter
            else:
                self.quote_counter += 1
                counter = self.quote_counter
            
            unique_hash = hashlib.md5(
                f"{job_id}{frame_idx}{timestamp_ms}".encode()
            ).hexdigest()[:6]
            
            filename = f"quotes_{counter}_{safe_channel}_{time_str}_{quote_type}_preview.jpg"
            image_path = self.quotes_dir / "full_frames" / filename
            
            # Збереження кадру
            cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            quote_id = f"quote_{job_id}_{frame_idx}_{timestamp_ms}"
            
            metadata = {
                "quote_id": quote_id,
                "job_id": job_id,
                "video_path": video_path,
                "video_name": Path(video_path).name,
                "channel": self.channel_name,
                "frame_index": frame_idx,
                "timestamp_ms": timestamp_ms,
                "timestamp_iso": dt.isoformat(),
                "formatted_time": format_duration(timestamp_ms / 1000),
                "quote_type": "URGENT" if is_urgent else "QUOTE",
                "text": quote_text,
                "speakers": speakers,
                "faces_detected": len(faces),
                "face_details": [
                    {
                        "name": f['name'],
                        "confidence": f.get('confidence', 0),
                        "emotion": f.get('emotion_label', 'unknown')
                    }
                    for f in faces
                ],
                "image_file": filename,
                "image_path": str(image_path),
                "image_hash": hashlib.md5(frame.tobytes()).hexdigest(),
                "frame_dimensions": {
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0])
                },
                "captured_at": datetime.now().isoformat()
            }
            
            # Збереження метаданих
            metadata_path = self.quotes_dir / "metadata" / f"{filename.replace('.jpg', '.json')}"
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
            
            # Додаємо в загальний список результатів
            self.captured_quotes.append(metadata)
            
            logger.debug(f"Saved quote: {filename} in {self.quotes_dir}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to capture quote at frame {frame_idx}: {e}")
            return None

    def get_results(self) -> PluginResult:
        return self.captured_quotes

    def get_summary(self) -> Dict:
        if not self.captured_quotes:
            return {
                "total_quotes_captured": 0,
                "urgent_quotes": 0,
                "regular_quotes": 0,
                "unique_speakers": [],
                "quotes_directory": str(self.quotes_dir)
            }
        
        urgent_count = sum(1 for q in self.captured_quotes if q['quote_type'] == 'URGENT')
        all_speakers = set()
        for quote in self.captured_quotes:
            all_speakers.update(quote.get('speakers', []))
        
        return {
            "total_quotes_captured": len(self.captured_quotes),
            "urgent_quotes": urgent_count,
            "regular_quotes": len(self.captured_quotes) - urgent_count,
            "unique_speakers": list(all_speakers),
            "quotes_directory": str(self.quotes_dir),
            "first_quote_time": self.captured_quotes[0]['timestamp_iso'] if self.captured_quotes else None,
            "last_quote_time": self.captured_quotes[-1]['timestamp_iso'] if self.captured_quotes else None,
            "quotes_details": [
                {
                    "quote_id": q['quote_id'],
                    "text": q['text'][:100] + "..." if len(q['text']) > 100 else q['text'],
                    "speakers": q['speakers'],
                    "timestamp": q['timestamp_iso'],
                    "image_path": q['image_path']
                }
                for q in self.captured_quotes
            ]
        }

    def _cleanup_previous_run(self) -> None:
        """Clean up quotes from previous failed run for THIS specific job/channel."""
        removed = 0
        
        for json_file in (self.quotes_dir / "metadata").glob("*.json"):
            try:
                metadata = json.loads(json_file.read_text())
                if metadata.get("job_id") == self.current_job_id:
                    json_file.unlink()
                    image_file = (self.quotes_dir / "full_frames" / metadata.get("image_file", ""))
                    if image_file.exists():
                        image_file.unlink()
                    removed += 1
            except Exception as e:
                logger.warning(f"Cleanup failed for {json_file}: {e}")
                json_file.unlink(missing_ok=True)
        
        if removed:
            logger.info(f"Cleaned {removed} quotes from previous job {self.current_job_id}")

    def cleanup(self) -> None:
        if self.face_recognizer:
            self.face_recognizer.reset_unknown_registry()
        self.captured_quotes = []
        self.quote_counter = 0
        self.urgent_counter = 0

    def cleanup_models(self) -> None:
        try:
            if self.face_recognizer:
                self.face_recognizer.reset_unknown_registry()
                del self.face_recognizer
                self.face_recognizer = None
        except Exception as e:
            logger.error(f"Failed to cleanup QuoteCapturePlugin models: {e}")