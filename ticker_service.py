#!/usr/bin/env python3
"""
Ticker Service — окремий сервіс для витягування біжучого рядка.
Працює паралельно з основним детектором меж.
"""
import os
import tempfile
import cv2
import json
import time
import logging
import numpy as np
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, List, Tuple
import threading
import queue
from collections import deque
import re
from ticker_assembler import TickerAssembler, TickerFragment
from ticker_flip import FlipTickerHandler

try:
    from ticker_symbol_map import TickerSymbolMapper
    TSM_AVAILABLE = True
except ImportError:
    TSM_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning("🟥 ticker_symbol_map не знайдено. Використовуватиметься старий алгоритм.")

logger = logging.getLogger(__name__)

class ChannelProfileManager:
    """Управління профілями каналів."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.profiles: Dict = {}
        self.load_profiles()
    
    def load_profiles(self):
        """Завантажує профілі з JSON."""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.profiles = data
                logger.info(f"📥 Завантажено {len(self.profiles)} профілів каналів")
        else:
            logger.warning(f"⚠️ Файл профілів не знайдено: {self.config_path}")
            self.profiles = {}
    
    def get_profile(self, channel_id: str) -> Optional[Dict]:
        """Отримує профіль каналу."""
        return self.profiles.get(channel_id)
    
    def update_profile(self, channel_id: str, roi: Dict):
        """Оновлює ROI профілю."""
        if channel_id in self.profiles:
            self.profiles[channel_id]['ticker_roi'] = roi
            self.profiles[channel_id]['version'] += 1
            self._save_profiles()
    
    def _save_profiles(self):
        """Безпечне збереження профілів, захищене від конфліктів при записі."""
        # 1. Записуємо в тимчасовий файл
        fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(self.config_path), suffix='.tmp')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(self.profiles, f, ensure_ascii=False, indent=2)
        
        # 2. Атомарно замінюємо старий файл новим (це миттєва операція на рівні ОС)
        os.replace(temp_path, self.config_path)

class TickerService:
    """
    Основний сервіс для витягування біжучого рядка.
    """
    
    def __init__(self, channel_id: str, profile_manager: ChannelProfileManager, output_dir: str):
        self.channel_id = channel_id
        self.profile_manager = profile_manager
        self.output_dir = Path(output_dir) / "json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        

        # Ловить логи з root-логера (включаючи ticker_assembler) і пише у JSONL.
        # Import тут, а не на рівні модуля: без coordinates_tap.py сервіс лише
        # попередить і працюватиме далі (top-level import прибраний саме тому).
        self.coordinates_tap = None
        try:
            from coordinates_tap import install_coordinates_tap

              # 🆕 ЗМІНА: формуємо безпечне ім'я каналу та динамічну назву файлу
            channel_safe_name = str(self.channel_id).replace(" ", "").replace("_", "").lower()
            coords_filename = f"coordinates_{channel_safe_name}.jsonl"
            
            self.coordinates_tap = install_coordinates_tap(
                output_dir=self.output_dir,
                logger=logging.getLogger(),
                filename=coords_filename  # 🆕 Тепер файл унікальний для каналу
            )
            logger.info(f"✅ COORD-TAP: логи координат записуються у {coords_filename}")


            # self.coordinates_tap = install_coordinates_tap(
            #     output_dir=self.output_dir,
            #     logger=logging.getLogger(),  # Root logger для перехоплення всього
            #     filename="coordinates.jsonl"
            # )
        except Exception as _exc:
            logger.warning("⚠️ COORD-TAP: не встановлено (%s) — coordinates.jsonl вимкнено", _exc)
        
        self.profile = profile_manager.get_profile(channel_id)
        if not self.profile:
            raise ValueError(f"Профіль каналу {channel_id} не знайдено!")

        self.excluded_patterns = self.profile.get('excluded_patterns', [])
        if self.excluded_patterns:
            logger.info(f"🚫 Завантажено {len(self.excluded_patterns)} виключень для {channel_id}: {self.excluded_patterns}")

         # 🆕 Завантажуємо нові параметри з профілю
                        # 1. Базовий напрямок
        self.ticker_scroll = self.profile.get('ticker_scroll', 'right_to_left')
        self.is_rtl = (self.ticker_scroll == 'right_to_left')
        
        self.clock_side = self.profile.get('clock_side', 'right')
        self.fade_zone_px = self.profile.get('fade_zone_px', 110)
         # 🆕 допуск відхилення fade від профілю (було жорстко 0.25 → 84 проходило)
        # допуск відхилення fade від профілю; 0.0 = жорсткий пінг усіх кадрів
        self.fade_anchor_tol = min(0.5, max(0.0, float(self.profile.get('fade_anchor_tol', 0.10))))

         # примусово завантажуємо параметри зшивання з JSON це додав для Aljazeera
        self.stitch_expected_dx = self.profile.get('stitch_expected_dx', 106)
        self.stitch_dx_tolerance = self.profile.get('stitch_dx_tolerance', 15)
        self.stitch_dy_max_corr = self.profile.get('stitch_dy_max_corr', 12.0)
        self.stitch_dy_dead_zone = self.profile.get('stitch_dy_dead_zone', 0.3)
        self.min_gap_width = self.profile.get('min_gap_width', 50)
        self.min_speed_px_sec = self.profile.get('min_speed_px_sec', 20.0) 
        self.sep_isolation_px = float(self.profile.get('sep_isolation_px', 20.0))
        self.sep_squares = bool(self.profile.get('sep_squares', False))
        self.min_speed_px_sec = self.profile.get('min_speed_px_sec', 20.0)
        self.sep_vertical_lines = bool(self.profile.get('sep_vertical_lines', False))

        self.color_separator_detection = bool(self.profile.get('color_separator_detection', False))
        self.color_sep_dev = float(self.profile.get('color_sep_dev', 40.0))

        self.split_whole_word = bool(self.profile.get('split_whole_word', False))

        self.sep_circles = bool(self.profile.get('sep_circles', False))
        self.sep_dots = bool(self.profile.get('sep_dots', False))  # для •

        self.ticker_band = self.profile.get('ticker_band', None)   # 🆕 v6: пінінг смуги, якщо авто не влаштує
                

        logger.info(
            "FADE CONFIG: channel=%s fade_zone_px=%d tol=%.2f (коридор %.0f..%.0fpx)",
            self.channel_id, self.fade_zone_px, self.fade_anchor_tol,
            (1 - self.fade_anchor_tol) * self.fade_zone_px,
            (1 + self.fade_anchor_tol) * self.fade_zone_px,
        )



        logger.info(
            f"📋 Налаштування каналу: scroll={self.ticker_scroll}, clock={self.clock_side}, "
            f"fade_zone={self.fade_zone_px}px, stitch_dx={self.stitch_expected_dx}±{self.stitch_dx_tolerance}, "
            f"dy_max_corr={self.stitch_dy_max_corr}, dy_dead_zone={self.stitch_dy_dead_zone}"
        )
        
        #logger.info(f"📋 Налаштування каналу: scroll={self.ticker_scroll}, clock={self.clock_side}, fade_zone={self.fade_zone_px}px, stitch_dx={self.stitch_expected_dx}±{self.stitch_dx_tolerance}")       # Ініціалізація збирача
        self.assembler = TickerAssembler(channel=channel_id)

            # 🆕 Крок 2: Ініціалізація TSM (Піксельний аналізатор меж речень)
        self.tsm_mapper = None
        if TSM_AVAILABLE:
            self.tsm_mapper = TickerSymbolMapper()
            
            # Спробуємо завантажити збережений шаблон роздільника з профілю каналу
            tsm_profile = self.profile.get('separator_template')
            if tsm_profile:
                self.tsm_mapper.set_profile(tsm_profile)
                logger.info(f"✅ TSM: Завантажено профіль роздільника з конфігу для {self.channel_id}")
            else:
                logger.info(f"✅ TSM: Ініціалізовано (шаблон буде вивчено автоматично під час роботи)")

        
        # Черга кадрів для обробки
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Статистика
        self.frames_processed = 0
        self.ocr_calls = 0
        self.messages_detected = 0
        self.ticker_count = 0
        
        # PaddleOCR (ініціалізується окремо)
        self.ocr = None

        #Буфер кадрів та змінні для Stitching
        self.frame_buffer = deque(maxlen=15)  # Зберігаємо останні  кадрів ROI
        self.panorama = None
        self.last_sentence_end_x = 0
        self.frame_counter = 0
        self.speed_stats: Dict = {}
        self.speed_history = deque(maxlen=40)  # ковзне вікно для медіани

         # ═══ E1 (ЕТАП 1 EDGE DETECTION): телеметрія зшивання для детекції країв ═══
        self._last_seam_xs: List[int] = []    # позиції швів (у координатах зшитої панорами)
        self._last_panorama_meta: Dict = {}   # телеметрія останньої зшитої панорами
        self._edge_prev_meta: Dict = {}       # телеметрія попередньої ПРИЙНЯТОЇ панорами
        self._last_fade_crop_px: int = 0      # скільки _remove_left_gray_bands зрізала зліва
        self._last_stitch_ts: float = 0.0     # час останнього стику (для dt між панорамами)
        self._dup_zone_px: float = 0.0        # 🆕 C-DROP: ширина зони дублювання
        self._dup_start_x: Optional[float] = None   # 🆕 C-DROP: її межа в координатах OCR

        # ═══ TGT: Temporal Gap Tracking — підтримка меж у часі (телеметрія G1) ═══
        # значення — [count, last_seq]: лічильник + остання панорама з цим
        # ключем, щоб можна було ЧЕСНО витискати застарілі ключі за віком.
        self._gap_history: Dict[str, List[int]] = {}  # ключ="tail|head" → [к-ть, вік]
        self._gap_seq: int = 0            # 🆕v2: номер панорами з аналізом
        self._gap_history_threshold = 3   # мінімум панорам поспіль для бونусу
        self._gap_history_bonus = 20      # бонус до raw_score (60 → 80 = STRONG)
        self._gap_history_max_age = 10    # панорам без оновлення → видалити ключ


        self._pending_seam_fragments: List[Dict] = []  # PendingSeamFragment класу B — чекають продовження
        self._seam_match_log: List[Dict] = []          # історія спроб збігу (діагностика;

        self._fade_history = deque(maxlen=40)
        # E2: 'continue' склеює без спільних літер — за логами 4/4 хибні
        self.seam_allow_continue = bool(self.profile.get('seam_allow_continue', False))
        self._content_change_reported = False
                # ── П0: стан тікера (пауза / зміна контенту) ──────────────────
        self._pause_frames = 0                 # заморожених кадрів підряд
        self._pause_reported = False           # подію «ПАУЗА» емітуємо 1 раз
        self._content_change_reported = False  # «ЗМІНА» — 1 раз до відновлення руху
        #  П5: історія емісій для пост-фільтра дублів
        self.emitted_norm: List[str] = []
        self.EMITTED_KEEP = 30        # скільки останніх емісій пам'ятати

                # 🆕 FLIP: Ініціалізація ізольованого обробника вертикального тікера
        self.ticker_motion = self.profile.get('ticker_motion', 'horizontal_scroll')
        self.flip_handler = None




        # Знак для математичних розрахунків зсуву (dx)
        # Для RTL (араби) очікуваний dx зазвичай додатний (в залежності від вашої математики)
        # Для LTR (турки) - протилежний
        self.dx_sign = 1 if self.is_rtl else -1

        # Сторона, де текст ЗНИКАЄ (старий контент, зона затухання/фейду)
        # RTL: зникає зліва | LTR: зникає справа
        self.exit_edge = 'left' if self.is_rtl else 'right'

        # Сторона, де текст З'ЯВЛЯЄТЬСЯ на екрані (новий контент)
        # RTL: з'являється справа | LTR: з'являється зліва
        self.entry_edge = 'right' if self.is_rtl else 'left'

        # Зона фейду (завжди прив'язана до виходу тексту за край екрана)
        self.fade_side = self.profile.get('fade_side', self.exit_edge)

        # Мова OCR
        self.ocr_lang = self.profile.get('ocr_lang', 'ar' if self.is_rtl else 'tr')

        # Напрямок тексту для асемблера та сортування блоків
        self.text_direction = 'rtl' if self.is_rtl else 'ltr'

        

        if self.ticker_motion == "vertical_flip":
            # ГОЛОВНА ЗМІНА: передаємо 'self' (весь сервіс), а не self.ocr!
            self.flip_handler = FlipTickerHandler(
                parent_service=self,
                profile=self.profile,
                assembler=self.assembler,
                excluded_patterns=self.excluded_patterns,
                save_callback=self._save_completed_messages
            )
            logger.info(f"🔄 Активовано режим VERTICAL_FLIP для каналу {channel_id}")
        else:
            logger.info(f"➡️ Активовано стандартний режим HORIZONTAL_SCROLL для каналу {channel_id}")



        
        logger.info(f"✅ Ticker Service ініціалізовано для {channel_id}")
        

    def init_ocr(self):
        """Ініціалізація PaddleOCR для Ticker Service з оптимізацією детектора DBNet."""
        try:
            from paddleocr import PaddleOCR

            ocr_lang = getattr(self, 'ocr_lang', 'ar')
            self.ocr = PaddleOCR(
                #lang='ar',
                lang=ocr_lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device="cpu",
                enable_mkldnn=False,
                det_db_thresh=0.2,       # Знижуємо з 0.3: допомагає детектору бачити слабкі контрасти (напр., тіні між словами)
                det_db_box_thresh=0.35,   # Знижуємо з 0.5/0.6: запобігає відсіюванню дуже коротких слів або частин слів
                det_db_unclip_ratio=1.5  # Зменшуємо з дефолтних 1.5-2.0: робить bounding box "щільнішим" навколо тексту, 
                                         # що запобігає злиттю двох сусідніх слів в один гігантський прямокутник
            )
            logger.info(f"✅ Ticker PaddleOCR успішно ініціалізовано для мови '{ocr_lang}' (v3.x style + DBNet оптимізація)")
        except Exception as e:
            logger.error(f"❌ Помилка ініціалізації Ticker PaddleOCR: {e}")
            raise

    def _sort_blocks_by_direction(self, blocks: List[Dict]) -> List[Dict]:
        """
        🆕 КРОК 5.1: Універсальне сортування OCR-блоків залежно від напрямку читання.
        
        RTL (арабські канали): текст читається СПРАВА НАЛІВО
            → блоки сортуються за СПАДАННЯМ x_min (reverse=True)
            → перший блок у списку = найправіший (початок речення)
        
        LTR (турецькі канали): текст читається ЗЛІВА НАПРАВО
            → блоки сортуються за ЗРОСТАННЯМ x_min (reverse=False)
            → перший блок у списку = найлівіший (початок речення)
        """
        if not blocks:
            return blocks
        
        # Використовуємо вже існуючу змінну з Кроку 1
        reverse = self.is_rtl  # True для арабів, False для турків
        
        blocks.sort(key=lambda item: item["x_min"], reverse=reverse)
        return blocks

    #це "довгострокова пам'ять"  сервісу.
    #TickerSymbolMapper (TSM) вперше успішно знаходить роздільник (наприклад, крапку • або логотип каналу), він створює його математичний шаблон (матрицю 16x16 пікселів).

    def _save_separator_template(self):
        """
        🆕 Крок 5: Зберігає вивчений шаблон роздільника (TSM) у channel_profiles.json.
        Викликається тільки тоді, коли TSM дійсно знайшов і вивчив новий якісний шаблон.
        """
        if getattr(self, 'tsm_mapper', None) and self.tsm_mapper.profile:
            # Додаємо або оновлюємо ключ у словнику профілів (нова назва)
            self.profile_manager.profiles[self.channel_id]['separator_template'] = self.tsm_mapper.profile
            # Зберігаємо зміни на диск
            self.profile_manager._save_profiles()
            logger.info(f"💾 TSM: Шаблон роздільника '{self.tsm_mapper.profile.get('name')}' збережено в конфігу для {self.channel_id}")
    
    def extract_ticker_roi(self, frame: np.ndarray) -> np.ndarray:
        """
        Витягує область біжучого рядка з кадру.
        """
        h, w = frame.shape[:2]
        roi = self.profile['ticker_roi']
        
        y1 = int(h * roi['y_start'])
        y2 = int(h * roi['y_end'])
        x1 = int(w * roi['x_start'])
        x2 = int(w * roi['x_end'])
        
        return frame[y1:y2, x1:x2]

    def _calculate_frame_diff(self, frame1, frame2):
        """Повертає відсоток змін між двома кадрами (0.0 до 1.0)"""
        if frame1 is None or frame2 is None:
            return 1.0
        
        # Конвертація у відтінки сірого для швидкості
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Абсолютна різниця
        diff = cv2.absdiff(gray1, gray2)
        
        # Порогова обробка (відсікаємо шум стиснення відео)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Відсоток змінених пікселів
        changed_ratio = np.count_nonzero(thresh) / thresh.size
        return float(changed_ratio)

    def _is_excluded_block(self, text: str) -> bool:
        """
        Перевіряє, чи блок тексту складається ПОВНІСТЮ з виключених патернів.
        Наприклад: "www.alhadath.net" або "@Alhadath - Youtube:"
        """
        if not self.excluded_patterns:
            return False
        
        cleaned = text.strip()
        # Видаляємо всі виключені патерни
        for pattern in self.excluded_patterns:
            cleaned = cleaned.replace(pattern, '').strip()
            cleaned = cleaned.replace('-', '').strip()  # Видаляємо залишкові дефіси
        
        # Якщо після видалення залишилося менше 2 символів — це був "сміттєвий" блок
        return len(cleaned) < 2

    def _get_ink_profile_and_gaps(self, img: np.ndarray) -> tuple:
        """
        Універсальний розрахунок профілю чорнила + ДІАГНОСТИКА ширин порожнеч.
        Працює ідентично для ROI та Панорами.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        roi_y_start = int(h * 0.20)
        roi = gray[roi_y_start:, :]
        
        # Профіль "чорнила" (для фіолетової лінії) — залишаємо як був
        # 🔑 НАДІЙНЕ ВИЗНАЧЕННЯ ПОЛЯРНОСТІ ТА СПРАВЖНЬОГО ФОНУ
        roi_i = roi.astype(np.int16)
        p_low = np.percentile(roi_i, 10, axis=1)
        p_high = np.percentile(roi_i, 90, axis=1)
        med = np.median(roi_i)
        
        # Темний текст на світлому фоні: медіана ближча до СВІТЛОГО краю
        dark_on_light = (med - p_low).mean() > (p_high - med).mean()
        
        if dark_on_light:
            base = np.percentile(roi_i, 80, axis=1)          # фон = світлий рівень
            ink_map = np.clip(base[:, None] - roi_i, 0, 255).astype(np.uint8)
        else:
            base = np.percentile(roi_i, 20, axis=1)          # фон = темний рівень
            ink_map = np.clip(roi_i - base[:, None], 0, 255).astype(np.uint8)
        
        ink = ink_map.sum(axis=0).astype(np.float32)
        kernel = np.ones(21, np.float32) / 21.0   # ширше згладжування: літери слова зливаються
        profile = np.convolve(ink, kernel, mode='same')

        
        # 🔬 ДІАГНОСТИКА: міряємо ШИРИНИ ВСІХ порожніх ділянок
        # (std-метод: стовпець без літер = однорідний = std ≈ 0; інверсій не буває)
        col_std = np.std(roi.astype(np.float32), axis=0)
        x = 0
        runs = []
        while x < w:
            if col_std[x] < 20.0:
                x2 = x
                while x2 < w and col_std[x2] < 20.0:
                    x2 += 1
                if (x2 - x) > 15:   # ігноруємо мікро-щілини між літерами
                    runs.append((x, x2, x2 - x))
                x = x2
            else:
                x += 1
        
        logger.info(f"📏 ПОРОЖНІ ДІЛЯНКИ (start-end-ширина): {runs}")
        
        # Поки що маркери ставимо за правилом >=100px і не біля країв
        gaps = [(a, b) for a, b, c in runs if c >= 50 and a > 5 and b < w - 5]
        
        return profile, gaps


    
    def _debug_check_dx_sign(self, img1: np.ndarray, img2: np.ndarray) -> None:
        """
        Діагностує знак dx на двох сусідніх кадрах.
        Порівнює phaseCorrelate з очікуваним напрямком
        на основі зображення, зсунутого штучно.
        """
        if img1 is None or img2 is None:
            logger.warning("DX TEST: один із кадрів None")
            return

        if img1.shape != img2.shape:
            logger.warning("DX TEST: різні розміри кадрів")
            return

        dx = self._get_horizontal_shift(img1, img2)

        logger.info("DX TEST: реальна пара кадрів: dx=%s", dx)

        # Контрольний тест: створюємо відомий зсув вправо.
        test = img1.copy()
        known_shift = max(5, img1.shape[1] // 10)

        # Зсуваємо вміст вправо, не використовуючи циклічне прокручування.
        shifted = np.zeros_like(test)
        shifted[:, known_shift:] = test[:, :-known_shift]

        measured = self._get_horizontal_shift(test, shifted)

        logger.info(
            "DX TEST: штучний рух вправо на %d px -> measured dx=%s",
            known_shift,
            measured,
        )

        if measured is None:
            logger.warning(
                "DX TEST: контрольний зсув не визначено; "
                "перевірте текстуру ROI та response"
            )
        elif measured > 0:
            logger.info("DX TEST: знак відповідає контракту dx > 0")
        else:
            logger.warning(
                "DX TEST: знак протилежний контракту; "
                "потрібно інвертувати знак або змінити порядок аргументів"
            )


    def _draw_ink_overlay(self, img: np.ndarray, profile: np.ndarray, gaps: List[tuple], label: str = "", timestamp: float = None) -> np.ndarray:
        """
        Універсальне малювання: фіолетова лінія + зелені маркери "++".
        Додає вертикальну мітку зліва на розширеній смузі.
        """
        h, w = img.shape[:2]
        
        # 🔑 РОЗШИРЮЄМО ЗОБРАЖЕННЯ ЗЛІВА НА 100px ДЛЯ МІТКИ
        if label:
            label_width = 100
            # Створюємо нове зображення з додатковою смугою зліва
            extended_img = np.zeros((h, w + label_width, 3), dtype=img.dtype)
            extended_img[:, label_width:] = img  # Копіюємо оригінал праворуч
            # Заповнюємо ліву смугу темно-сірим фоном
            extended_img[:, :label_width] = (30, 30, 30)
            debug_img = extended_img
        else:
            debug_img = img.copy()
            label_width = 0
        
        # Малюємо фіолетову лінію профілю
        if profile is not None and len(profile) == w and profile.max() > 0:
            pmax = max(float(profile.max()), 1.0)
            scale = (h // 3) / pmax
            step = max(1, len(profile) // 500)
            # Зсуваємо точки на label_width вправо
            pts = [(x + label_width, h - 1 - int(profile[x] * scale)) for x in range(0, len(profile), step)]
            
            if len(pts) > 1:
                cv2.polylines(debug_img, [np.array(pts, np.int32)], False, (255, 0, 255), 2)
                cv2.line(debug_img, (label_width, h - 1), (w + label_width, h - 1), (255, 0, 255), 1)
                
                # ФІЛЬТРИ: прибираємо хибні маркери ДО нумерації
                MIN_MARKER_WIDTH = 10   # справжня межа між новинами ширша 
                EDGE_MARGIN = 5          # вакууми біля країв кадру = обрізаний текст, не межа

                valid_gaps = [
                    (vs, ve) for (vs, ve) in gaps
                    if (ve - vs) >= MIN_MARKER_WIDTH
                    and vs > EDGE_MARGIN
                    and ve < w - EDGE_MARGIN
                ]

                # Зелені маркери тільки для ВАЛІДНИХ пробілів
                for vac_idx, (vac_start, vac_end) in enumerate(valid_gaps):
                    x1 = int(vac_start + label_width)
                    x2 = int(vac_end + label_width)

                    x1 = max(0, min(x1, debug_img.shape[1] - 1))
                    x2 = max(0, min(x2, debug_img.shape[1] - 1))

                    if x2 <= x1:
                        continue

                    center_x = (x1 + x2) // 2

                    # Компактна зона маркера внизу зображення
                    marker_h = min(8, h - 1)
                    y_top = h - 1 - marker_h
                    y_bottom = h - 1

                    cv2.rectangle(debug_img, (x1, y_top), (x2, y_bottom), (0, 255, 0), 1)
                    cv2.line(debug_img, (center_x, y_top), (center_x, y_bottom), (0, 255, 0), 1)

                    text = f"++{vac_idx + 1}"
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    text_x = max(0, min(center_x - text_w // 2, debug_img.shape[1] - text_w))
                    text_y = max(text_h, y_top - 2)
                    cv2.putText(debug_img, text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    
        # 🔑 ВЕРТИКАЛЬНИЙ НАПИС МІТКИ ЗЛІВА
        if label:
            # Малюємо текст вертикально знизу вгору
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            color = (0, 255, 255)  # Жовтий
            
            # Розміщуємо вертикально по центру лівої смуги
            text_x = label_width // 2
            text_y_start = h - 20
            
            # Малюємо кожну букву окремо вертикально
            for i, char in enumerate(label):
                y_pos = text_y_start - (i * 25)
                if y_pos > 10:
                    cv2.putText(debug_img, char, (text_x - 10, y_pos), font, font_scale, color, thickness)

                # 🔑 ДОДАЄМО ЧАСОВУ МІТКУ ЗНИЗУ ЛІВОРУЧ
        if timestamp is not None:
            time_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
            cv2.putText(debug_img, time_str, (label_width + 5, h - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return debug_img

    def _detect_left_fade(self, roi: np.ndarray) -> int:
        """
        X-координата першого ЧИТАБЕЛЬНОГО стовпця (ширина fade-зони зліва).
        🆕 Медіанне згладжування: якщо детекція раптово «провалилась»
        (артефакт здався читабельним) — беремо медіану останніх 30 кадрів.
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h = gray.shape[0]
        band = gray[int(h * 0.20):, :]
        col_std = np.std(band.astype(np.float32), axis=0)
        readable = col_std >= 15.0

        if not readable.any():
            fade = gray.shape[1]          # кадр порожній — читабельного нема
        else:
            fade = int(np.argmax(readable))

        # 🆕 ЗАХИСТ ВІД "ПРОВАЛІВ" (артефактів стиснення)
        if len(self._fade_history) > 5:  # Чекаємо, поки накопичиться хоча б трохи даних
            med = float(np.median(self._fade_history))
            # Якщо канал зазвичай має fade >= 60, а цей кадр раптом показує < 60% від медіани
            if med >= 60 and fade < (med * 0.6):
                fade = int(med)  # Ігноруємо поточний "шум" і беремо стабільне значення

        self._fade_history.append(fade)
        
        # Запобіжник: fade не може бути більшим за половину ширини кадру
        return min(fade, gray.shape[1] // 2)


    def _detect_right_fade(self, roi: np.ndarray) -> int:
        """
        Ширина НЕЧИТАБЕЛЬНОЇ зони СПРАВА (для LTR каналів, де текст зникає вправо).
        Використовується для турецьких каналів.
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h = gray.shape[0]
        band = gray[int(h * 0.20):, :]
        col_std = np.std(band.astype(np.float32), axis=0)
        readable = col_std >= 20.0

        # 🆕 Перевертаємо масив, щоб шукати з правого краю
        rev_readable = readable[::-1]
        
        if not rev_readable.any():
            fade = gray.shape[1]          # кадр порожній
        else:
            fade = int(np.argmax(rev_readable))  # кількість нечитабельних px справа

        # 🆕 ЗАХИСТ ВІД "ПРОВАЛІВ" (ідентичний лівому)
        if len(self._fade_history) > 5:
            med = float(np.median(self._fade_history))
            if med >= 60 and fade < (med * 0.6):
                fade = int(med)

        self._fade_history.append(fade)
        return min(fade, gray.shape[1] // 2)

    def _remove_left_gray_bands(self, panorama: np.ndarray) -> np.ndarray:
        """
        Видаляє fade-зону зліва на основі параметрів каналу.
        """
        if panorama is None or panorama.shape[1] < 50:
            self._last_fade_crop_px = 0 
            return panorama
        
        gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        roi = gray[int(h * 0.20):, :]
        col_std = np.std(roi.astype(np.float32), axis=0)
        
        # 🆕 Використовуємо fade_zone_px з профілю каналу
        max_fade = self.fade_zone_px
        
        if col_std[0] < 20.0:
            x2 = 0
            while x2 < w and x2 < max_fade and col_std[x2] < 20.0:
                x2 += 1
            
            if x2 > 5:
                logger.debug(f"Видалено fade-зону: {x2}px (ліміт: {max_fade}px)")
                self._last_fade_crop_px = int(x2)  # скільки зрізано зліва
                return panorama[:, x2:]
            
        self._last_fade_crop_px = 0
        return panorama


    def _remove_right_gray_bands(self, panorama: np.ndarray) -> np.ndarray:
        """
        Видаляє fade-зону СПРАВА на основі параметрів каналу.
        Дзеркальна логіка до _remove_left_gray_bands — для турецьких (LTR) каналів,
        де старий контент зникає в правому кутку екрана.
        """
        if panorama is None or panorama.shape[1] < 50:
            self._last_fade_crop_px = 0
            return panorama
        
        gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        roi = gray[int(h * 0.20):, :]
        col_std = np.std(roi.astype(np.float32), axis=0)
        
        # 🆕 Використовуємо fade_zone_px з профілю каналу
        max_fade = self.fade_zone_px
        
        # 🆕 ЛОГІКА СПРАВА: перевіряємо ОСТАННІЙ стовпець
        if col_std[-1] < 20.0:
            x2 = 0
            # Ідемо з правого краю вліво, шукаючи перший читабельний стовпець
            while x2 < w and x2 < max_fade and col_std[-(x2 + 1)] < 20.0:
                x2 += 1
            
            if x2 > 5:
                logger.debug(f"Видалено fade-зону СПРАВА: {x2}px (ліміт: {max_fade}px)")
                self._last_fade_crop_px = int(x2)  # скільки зрізано справа
                # 🆕 ОБРІЗАЄМО СПРАВА (не зліва!)
                return panorama[:, :-x2] if x2 > 0 else panorama
        
        self._last_fade_crop_px = 0
        return panorama

    
    def _get_horizontal_shift(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        last_dx: int = 0,
    ) -> Optional[int]:
        """
        Оцінює горизонтальний зсув між сусідніми кадрами.

        Контракт для цього методу:
            dx > 0 — вміст img2 перемістився вправо відносно img1.
            dx < 0 — вміст img2 перемістився вліво відносно img1.

        Повертає None, якщо оцінка ненадійна.
        """

        if img1 is None or img2 is None:
            return None

        if img1.size == 0 or img2.size == 0:
            return None

        if img1.shape[:2] != img2.shape[:2]:
            logger.warning(
                "ROI мають різні розміри: %s і %s",
                img1.shape,
                img2.shape,
            )
            return None

        if img1.ndim == 2:
            gray1 = img1
        elif img1.ndim == 3 and img1.shape[2] == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        elif img1.ndim == 3 and img1.shape[2] == 4:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2GRAY)
        else:
            logger.warning("Непідтримуваний формат img1: %s", img1.shape)
            return None

        if img2.ndim == 2:
            gray2 = img2
        elif img2.ndim == 3 and img2.shape[2] == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        elif img2.ndim == 3 and img2.shape[2] == 4:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2GRAY)
        else:
            logger.warning("Непідтримуваний формат img2: %s", img2.shape)
            return None

        gray1 = cv2.GaussianBlur(gray1, (3, 3), 0)
        gray2 = cv2.GaussianBlur(gray2, (3, 3), 0)

        h, w = gray1.shape

        if h < 8 or w < 16:
            logger.warning(
                "ROI замалий для оцінки зсуву: width=%d height=%d",
                w,
                h,
            )
            return None

        f1 = np.asarray(gray1, dtype=np.float32)
        f2 = np.asarray(gray2, dtype=np.float32)

        window = cv2.createHanningWindow(
            (w, h),
            cv2.CV_32F,
        )

        try:
            shift, response = cv2.phaseCorrelate(f1, f2, window)
        except cv2.error as exc:
            logger.warning("Phase correlation failed: %s", exc)
            return None

        raw_dx = float(shift[0])
        dy = float(shift[1])
        response = float(response)

        if not (
            np.isfinite(raw_dx)
            and np.isfinite(dy)
            and np.isfinite(response)
        ):
            logger.warning(
                "Некоректний результат phase correlation: "
                "dx=%.2f dy=%.2f response=%.3f",
                raw_dx,
                dy,
                response,
            )
            return None

        if abs(dy) > max(3.0, h * 0.10):
            logger.warning(
                "Відхилено через вертикальний зсув: "
                "dx=%.2f dy=%.2f response=%.3f",
                raw_dx,
                dy,
                response,
            )
            return None

        if abs(raw_dx) > w * 0.5:
            logger.warning(
                "Відхилено надмірний зсув: "
                "dx=%.2f width=%d response=%.3f",
                raw_dx,
                w,
                response,
            )
            return None

        if response < 0.10:
            logger.warning(
                "Низька надійність phase correlation: "
                "dx=%.2f response=%.3f",
                raw_dx,
                response,
            )
            return None

        dx = int(round(raw_dx))

        if dx == 0:
            logger.debug(
                "Зсув після округлення дорівнює нулю: "
                "raw_dx=%.3f response=%.3f",
                raw_dx,
                response,
            )
            return None

        logger.debug(
            "Оцінка зсуву: dx=%d raw_dx=%.2f dy=%.2f "
            "response=%.3f last_dx=%d",
            dx,
            raw_dx,
            dy,
            response,
            last_dx,
        )

        return dx


    def _create_panorama(self) -> tuple[Optional[np.ndarray], List[Dict]]:

            #вибираємо правильний метод залежно від напрямку
            if not self.is_rtl:
                # LTR (турецькі канали) — окрема ізольована функція
                return self._create_panorama_ltr()
            # ── налаштування зшивання (перекриваються атрибутами сервісу) ──
            # 🆕 КРОК 3.3: Знаковане очікування залежно від напрямку
            # Для RTL (араби): додатне (рух вправо в координатах кадру)
            # Для LTR (турки): від'ємне (рух вліво в координатах кадру)
            raw_expected_dx = int(getattr(self, "stitch_expected_dx", 106))
            expected_dx = self.dx_sign * raw_expected_dx
            dx_tolerance = int(getattr(self, "stitch_dx_tolerance", 15))
            dy_dead_zone = float(getattr(self, "stitch_dy_dead_zone", 0.3))
            dy_ema_alpha = float(getattr(self, "stitch_dy_ema", 0.4))
            dy_max_corr = float(getattr(self, "stitch_dy_max_corr", 12.0))
            min_response = float(getattr(self, "stitch_min_response", 0.0))
            # 🔧 П1: поріг довіри до виміру пропущеного кадру. Нище — «контент
            # змінився» (графіка/перемотка), вище — «рух реальний, просто інший».
            trust_response = float(getattr(self, "stitch_trust_response", 0.30))

            if not self.frame_buffer:
                return None, []


             # ════════════════════════════════════════════════════════════════
            # 🔧 П3: САНІТАРІЯ FADE v3 (З ПІНГОМ ДО ПРОФІЛЮ)
            #  • Профіль каналу (fade_zone_px) є головним якіром.
            #  • Медіані буфера довіряємо ТІЛЬКИ якщо вона в коридорі ±25% від профілю.
            #  • Якщо медіана "злетіла" (артефакти) — жорсткий пінг до значення профілю.
            # ════════════════════════════════════════════════════════════════
            #raw_fades = [f.get("fade", 0) for f in self.frame_buffer]
            raw_fades = [int(f.get("fade_raw", f.get("fade", 0))) for f in self.frame_buffer]
            n = len(raw_fades)
            med_fade = float(np.median(raw_fades)) if n >= 3 else None

            profile_fade = int(getattr(self, "fade_zone_px", 0) or 0)
            anchor = None
            fixed_n = 0

            if profile_fade >= 60:
                # Перевіряємо, чи медіана знаходиться в допустимому коридорі (±25%)
                if n >= 3 and med_fade is not None and abs(med_fade - profile_fade) <= self.fade_anchor_tol * profile_fade:
                    anchor = int(med_fade)  # Медіана стабільна і поруч з профілем — довіряємо їй
                else:
                    # Медіана "перевернулась" або буфер ще не прогрітий — пінгуємо до профілю
                    anchor = profile_fade
                    if n >= 3:
                        logger.info(
                            "📌 FADE ANCHOR: медіана %.1f поза коридором профілю %d → застосовано пінг",
                            med_fade, profile_fade
                        )
            elif n >= 3 and med_fade is not None and med_fade >= 60:
                # Фолбек: якщо профіль не задано (0), але медіана стабільна
                anchor = int(med_fade)
            else:
                # Останній захист: жорсткий дефолт, якщо все інше зламалося
                anchor = 108

            # Застосовуємо якір до всіх кадрів, що виходять за межі розумного відхилення
            if anchor is not None:
                lo = (1.0 - self.fade_anchor_tol) * anchor
                hi = (1.0 + self.fade_anchor_tol) * anchor
                for f, fr in zip(self.frame_buffer, raw_fades):
                    if fr < lo or fr > hi:
                        if "fade_raw" not in f:
                            f["fade_raw"] = int(fr)
                        f["fade"] = anchor
                        fixed_n += 1

            logger.info(
                "🧼 FADE SANITY: n=%d anchor=%d fixed=%d "
                "raw[min=%d max=%d med=%s std=%.1f]",
                n,
                anchor,
                fixed_n,
                min(raw_fades) if raw_fades else 0,
                max(raw_fades) if raw_fades else 0,
                f"{np.median(raw_fades):.0f}" if raw_fades else "—",
                float(np.std(raw_fades)) if raw_fades else 0.0,
            )

            first = self.frame_buffer[0]
            fade_0 = first.get("fade", 0)
            base_img = first["image"]

            if base_img is None or base_img.shape[1] - fade_0 < 50:
                return None, []

            panorama = base_img[:, fade_0:].copy()
            left_edge_x = fade_0
            # frozen_run = 0   # 🆕 П2: заморожених кадрів у циклі
            # n_anom = 0       # 🆕 П2: пропусків (живий, не з shifts)
            shifts: List[Dict] = []

            # валідне вертикальне вікно у координатах кадру-еталона (кадру 0)
            base_h = base_img.shape[0]
            valid_top, valid_bottom = 0, base_h
            cumulative_dy = 0.0
            ema_dy = 0.0  # згладжений dy: персистентний дрейф vs шум
            # 🆕 МОНІТОРИНГ FADE ZONE
            fade_values = [fade_0]
            crop_widths = []    # 🆕 МОНІТОРИНГ ШИРИНИ ДОДАНИХ ДІЛЯНОК
            intended_widths = []  # 🔧 П2: скільки КОДИ мав дати цей стик (pending+shift)

            if len(self.frame_buffer) < 2:
                return panorama, shifts

            logger.info("Побудова панорами з %d кадрів", len(self.frame_buffer))

            # ════════════════════════════════════════════════════════════════
            # 🔧 П1: КНИЖКУВАННЯ ПРОПУСКІВ.
            #  pending_shift  — px контенту, що з'явився у ПРОПУЩЕНИХ кадрах;
            #                   наступний зшитий кадр забере їх одним crop'ом
            #                   (математика: crop = [fade_i : fade_ост_зшитого + Σрухів],
            #                    телескопічно точно, перевірено симуляцією).
            #  РОЗРИВ        — якщо пропущений кадр має resp < trust_response або
            #                   рух У ІНШИЙ БІК — контент змінився (графіка/перемотка).
            #                   Далі не шиємо: хвіст ловить наступна панорама
            #                   (контент їде лише ~105px/с і ще ~7с лишається на екрані).
            # ════════════════════════════════════════════════════════════════
            pending_shift = 0
            skipped_anom = 0      # 🔧 П2: живий лічильник (раніше n_anom завжди 0)
            skipped_none = 0      # 🔧 П2: зсув взагалі не вдалось виміряти
            frozen_frames = 0     # 🔧 П1: кадрів із застиглою стрічкою (dx≈0)
            stitched_frames = 0
            content_break = False
             #ініціалізація індексу відновлення після розриву
            self._resume_from = None

            for i in range(1, len(self.frame_buffer)):
                img1 = self.frame_buffer[i - 1]["image"]
                img2 = self.frame_buffer[i]["image"]
                fade_i = self.frame_buffer[i].get("fade", 0)
                fade_values.append(fade_i)

                dt = (self.frame_buffer[i]["timestamp"]
                    - self.frame_buffer[i - 1]["timestamp"])

                # 🔧 П1: технічні пропуски (нема кадру / форма / висота) тепер
                # теж книжкуються: контент між ними РУХАВСЯ, голо continue
                # робив у панорамі дірку на expected_dx.
                tech_skip = (img1 is None or img2 is None
                            or img1.shape != img2.shape
                            or img2.shape[0] != base_h)
                if tech_skip:
                    pending_shift += expected_dx
                    skipped_none += 1
                    if dt > 0:
                        shifts.append({
                            "dx": int(expected_dx),
                            "dt": dt,
                            "dy": 0.0, "dy_ema": round(float(ema_dy), 3),
                            "response": 0.0, "dx_anomaly": False,
                            "v_offset": 0, "fade_zone": int(fade_i),
                            "skipped": True,
                        })
                    continue

                result = self._get_horizontal_shift_with_dy(img1, img2)
                if result is None:
                    logger.warning("Пропуск кадру %d: зсув не визначено", i)
                    pending_shift += expected_dx
                    skipped_none += 1
                    if dt > 0:
                        shifts.append({
                            "dx": int(expected_dx),
                            "dt": dt,
                            "dy": 0.0, "dy_ema": round(float(ema_dy), 3),
                            "response": 0.0, "dx_anomaly": False,
                            "v_offset": 0, "fade_zone": int(fade_i),
                            "skipped": True,
                        })
                    continue

                dx, dy, response = result
                h, w = img2.shape[:2]
                #shift = abs(int(dx))
                shift = int(round(abs(dx))) # S1: було abs(int(dx)) — обрізання ~1px на кожному шві
                if abs(dx) < 2:
                    frozen_frames += 1
                    continue
                # ── контроль аномалій горизонтального зсуву ──
                dx_anomaly = abs(dx - expected_dx) > dx_tolerance

                if dx_anomaly:
                    logger.warning(
                        "🟧 ПРОПУСК ЗШИВАННЯ: frame=%d dx=%d "
                        "(очікується %d±%d), dy=%.2f response=%.3f",
                        i, int(dx), expected_dx, dx_tolerance, dy, response,
                    )
                    skipped_anom += 1  # 🔧 П2

                    if dt > 0:
                        shifts.append({
                            "dx": 0 if (dx * expected_dx < 0
                                        or response < trust_response)
                                else int(dx),
                            "dt": dt,
                            "dy": round(float(dy), 3),
                            "dy_ema": round(float(ema_dy), 3),
                            "response": round(float(response), 3),
                            "dx_anomaly": True,
                            "v_offset": 0, "fade_zone": int(fade_i),
                            "skipped": True,
                        })

                    # 🔧 П1: класифікація пропуску
                    if dx * expected_dx < 0 or response < trust_response:
                        # Стрибок у протилежний бік або зовсім чужий кадр:
                        # контент НЕ безперервний. Шити далі = гібридне речення.
                        content_break = True
                        logger.warning(
                            "🟪 РОЗРИВ КОНТЕНТУ: кадр %d (dx=%d, resp=%.2f) — "
                            "панорама обрізана після кадру %d, хвіст забере "
                            "наступна панорама",
                            i, int(dx), response, i - 1,
                        )

                                                # ═══ В1 ═══
                        if not getattr(self, "_content_change_reported", False):
                            self._content_change_reported = True
                            zone_text = ""
                            try:
                                res = self.ocr.predict(img2)   # ЩО прийшло у смугу
                                parts = [t.strip() for page in res
                                         for t, s in zip(page.get('rec_texts', []),
                                                         page.get('rec_scores', []))
                                         if float(s) > 0.5 and len(t.strip()) > 1]
                                zone_text = ' '.join(parts)[:100]
                            except Exception:
                                zone_text = ""
                            self._emit_system_event(
                                "content_changed",
                                f"новий контент у смузі: {zone_text or 'OCR пусто'}")
                        # ═══ кінець В1 ═══
                        self._resume_from = i
                        break


                    # Довірений «великий» крок (напр. dx=212 = загублений кадр
                    # джерела, dx=16 = гальмування): контент існує — книжкуємо.
                    pending_shift += shift
                    continue

                if shift <= 0 or shift >= w:
                    continue
                # 🔧 П1: перекриття з ОСТАННІМ ЗШИТИМ кадром, а не з сусідом
                total_advance = pending_shift + shift
                overlap = w - total_advance
                if overlap < int(w * 0.25):
                    content_break = True
                    logger.warning(
                        "🟪 РОЗРИВ: перекриття з останнім зшитим кадром %dpx "
                        "(<25%%) — накопичено забагато пропущеного руху, обрізаю",
                        overlap,
                    )
                    # запам'ятовуємо індекс розриву
                    self._resume_from = i
                    break
                if response < min_response:
                    # 🔧 П1: і тут книжкуємо — інакше піднятий min_response
                    # поверне дірки (за дефолтом 0.0 гілка недосяжна)
                    pending_shift += shift
                    skipped_none += 1
                    continue

                # ── накопичення дрейфу: мертва зона за EMA + стеля ──
                ema_dy = dy_ema_alpha * dy + (1.0 - dy_ema_alpha) * ema_dy
                if abs(ema_dy) >= dy_dead_zone:
                    cumulative_dy = max(
                        -dy_max_corr, min(dy_max_corr, cumulative_dy + ema_dy)
                    )
                offset = int(round(cumulative_dy))

                if dx > 0:
                    # 🔧 П1: математика з урахуванням pending_shift
                    seam_x = left_edge_x + pending_shift + shift
                    
                    # ПАТЧ 2: Відступ +3px від межі градієнта.
                    # Прибирає тонкі вертикальні артефакти H.264 на стику.
                    # Втрачені 3px контенту будуть безпечно захоплені наступним кадром.
                    start_x = fade_i + 3  
                    
                    end_x = min(seam_x, w)

                    if end_x - start_x < 5:
                        left_edge_x = min(seam_x, w)
                        pending_shift = 0
                        continue

                    new_part = img2[:, start_x:end_x].copy()
                    direction, side = "right", "left"
                else:
                    start_x = w - shift
                    end_x = w
                    new_part = img2[:, start_x:end_x].copy()
                    direction, side = "left", "right"

                # ── РАХУЄМО ФАКТИЧНУ ШИРИНУ ДІЛЯНКИ ──
                current_crop_width = end_x - start_x
                crop_widths.append(current_crop_width)
                intended_widths.append(total_advance)   # 🔧 П2

                # ── 🔑 ВИРІВНЮВАННЯ БЕЗ ЧОРНИХ РЯДКІВ (замість roll) ──
                prev_top, prev_bottom = valid_top, valid_bottom
                if offset > 0:
                    valid_bottom = min(valid_bottom, h - offset)
                elif offset < 0:
                    valid_top = max(valid_top, -offset)

                if valid_bottom - valid_top < max(10, base_h // 3):
                    # вікно вироджується — скидаємо корекцію: безпека над усе
                    logger.warning(
                        "Кадр %d: dy-корекція скинута (вікно %dpx замале)",
                        i, valid_bottom - valid_top,
                    )
                    valid_top, valid_bottom = prev_top, prev_bottom
                    offset = 0
                    cumulative_dy = 0.0

                new_part = new_part[valid_top + offset: valid_bottom + offset, :]
                panorama = panorama[valid_top - prev_top:
                                    valid_bottom - prev_top, :]

                if dx > 0:
                    panorama = cv2.hconcat([new_part, panorama])
                    left_edge_x = start_x
                else:
                    panorama = cv2.hconcat([panorama, new_part])
                    

                if dt > 0:
                    shifts.append({
                        "dx": int(dx),
                        "dt": dt,
                        "dy": round(float(dy), 3),
                        "dy_ema": round(float(ema_dy), 3),
                        "response": round(float(response), 3),
                        "dx_anomaly": dx_anomaly,
                        "v_offset": offset,
                        "fade_zone": int(fade_i),
                        "skipped": False,                # 🔧 П1
                        "pending_px": int(pending_shift)  # 🔧 П1: скільки
                                                        # пропущеного забрав цей стик
                    })

                self._last_stitch_ts = float(self.frame_buffer[i]["timestamp"])  #час останнього стику
                pending_shift = 0     # 🔧 П1: витрачено
                stitched_frames += 1
                self._content_change_reported = False   # ═══ В2: рух є → наступна зміна знову емітується

                logger.info(
                    "Stitch frame=%d dx=%d dy=%.2f cum_dy=%.2f off=%d "
                    "crop=[%d:%d] panorama=%dx%d",
                    i, int(dx), dy, cumulative_dy, offset,
                    start_x, end_x, panorama.shape[1], panorama.shape[0],
                )

            # ── якість панорами ──
            if panorama.shape[1] > 0:
                readable_ratio = max(
                    0.0, min(100.0,
                            (panorama.shape[1] - fade_0)
                            / panorama.shape[1] * 100))
                logger.info(
                    "🟩 ЯКІСТЬ ПАНОРАМИ: читабельно=%.1f%%, fade_zone=%dpx, "
                    "total_width=%dpx, height=%dpx%s",
                    readable_ratio, fade_0, panorama.shape[1], panorama.shape[0],
                    "  [🟪 ОБРІЗАНА НА РОЗРИВІ]" if content_break else "",
                )

            # 🔧 П2: статистика FADE — по СИРИХ значеннях (до санітарії).
            # Раніше std рахувався по вже виправлених — і «FADE INSTABILITY»
            # не бачив тих викидів, які санітарія щойно з'їла.
            fade_arr = np.array(raw_fades)
            fade_min = int(np.min(fade_arr))
            fade_max = int(np.max(fade_arr))
            fade_std = float(np.std(fade_arr))

            if shifts:
                avg_response = float(
                    np.mean([s.get("response", 0.0) for s in shifts]))
                max_dy = max(abs(s.get("dy", 0.0)) for s in shifts)
                last_ema = shifts[-1].get("dy_ema", 0.0)
                n_anom = sum(1 for s in shifts if s.get("dx_anomaly"))  # 🔧 П2: тепер живий

                logger.info(
                    " ЗШИВАННЯ: зшито=%d, пропущено=%d (аномалій=%d, "
                    "невиміряних=%d, заморожено=%d), response=%.3f, "
                    "макс |dy|=%.2fpx, EMA dy=%.2fpx, накопичена корекція=%.1fpx",
                    stitched_frames,
                    skipped_anom + skipped_none,
                    skipped_anom, skipped_none, frozen_frames,
                    avg_response, max_dy, last_ema, cumulative_dy,
                )

            # 🆕 ЛОГУВАННЯ FADE ZONE
            logger.info(
                "📏 FADE ZONE: min=%dpx, max=%dpx, std=%.1fpx (профіль: %dpx, "
                "сирове/до санітарії)",
                fade_min, fade_max, fade_std, self.fade_zone_px
            )

            # 🆕 РОЗРАХУНОК ТА ЛОГ CROP WIDTH
            if crop_widths:
                crop_arr = np.array(crop_widths)
                intended_arr = np.array(intended_widths, dtype=float)
                crop_min = int(np.min(crop_arr))
                crop_max = int(np.max(crop_arr))
                crop_mean = float(np.mean(crop_arr))

                # 🔧 П2: аномалія = відхилення від ВЛАСНОГО наміченого стику
                # (pending+shift), а не від expected_dx. Інакше чесний широкий
                # crop після dx=212 (відновлення 212px контенту) лічився б
                # «аномалією». Справжні аномалії (стрибки fade) лишаються видимими.
                anomalies = int(np.sum(
                    (crop_arr < intended_arr * 0.8) |
                    (crop_arr > intended_arr * 1.2)
                ))

                logger.info(
                    "🟪 CROP WIDTH: min=%dpx, max=%dpx, mean=%.1fpx "
                    "(намічено: %.1fpx, аномалій: %d)",
                    crop_min, crop_max, crop_mean,
                    float(np.mean(intended_arr)) if len(intended_arr) else 0.0,
                    anomalies
                )

                if anomalies > 0:
                    logger.warning(
                        "⚠️ CROP ANOMALIES: %d кадрів мають ширину, що не "
                        "відповідає наміченому стику (перевірте fade)!", anomalies
                    )

            # 🆕 РОЗРАХУНОК ТА ЛОГ ВЕРТИКАЛЬНОГО ДРЕЙФУ (DY)
            if shifts:
                dy_raw_values = [s["dy"] for s in shifts]
                # 🔧 П2: РАНІШЕ тут читалось поле "dy_applied_step", якого НІХТО
                # ніколи не писав → applied_sum завжди 0.0 → хибний алерт
                # «DY DEAD ZONE TOO HIGH» на КОЖНІЙ панорамі (14/14 у прогоні).
                # Тепер — по фактичних офсетах зшитих кадрів.
                offs = [s.get("v_offset", 0.0)
                        for s in shifts if not s.get("skipped")]
                dy_applied_values = (
                    [offs[0]] + [b - a for a, b in zip(offs, offs[1:])]
                    if offs else []
                )

                dy_raw_sum = float(sum(dy_raw_values))
                dy_applied_sum = float(sum(dy_applied_values))
                dy_cum_max = float(max(abs(s.get("v_offset", 0.0)) for s in shifts))

                logger.info(
                    "↕️ VERTICAL DRIFT: raw_sum=%.1fpx, applied_sum=%.1fpx, max_cum=%.1fpx",
                    dy_raw_sum, dy_applied_sum, dy_cum_max
                )

                # Алерт: є дрейф, але корекція ДІЙСНО не застосовується
                if abs(dy_raw_sum) > 3.0 and abs(dy_applied_sum) < 1.0:
                    logger.warning(
                        "⚠️ DY DEAD ZONE TOO HIGH: Виявлено дрейф (%.1fpx), але корекція ігнорується. "
                        "Рекомендується зменшити stitch_dy_dead_zone.", dy_raw_sum
                    )

                # 🔧 П2: алерт лише коли корекція ВБИЛАСЯ У СТЕЛЮ (раніше поріг 5
                # при стелі 8 гарантовано спрацьовував 13/14 разів — шум, а не сигнал)
                if dy_cum_max >= dy_max_corr:
                    logger.warning(
                        "🟥 HIGH DY CORRECTION: корекція досягла стелі %.1fpx. "
                        "Ризик обрізання тексту зверху/знизу.", dy_cum_max
                    )

            #  АЛЕРТ, ЯКЩО FADE ЗАСИЛЬНО «СТРИБАЄ» (тепер бачить і те, що з'їла санітарія)
            if fade_std > 15.0:
                logger.warning(
                    "🟥 FADE INSTABILITY: std=%.1fpx занадто високий! "
                    "Можлива причина: артефакти H.264 або зміна оформлення каналу.",
                    fade_std
                )

            logger.info(
                "Панорама: width=%d height=%d shifts=%d cum_dy=%.1fpx",
                panorama.shape[1], panorama.shape[0], len(shifts), cumulative_dy,
            )

            seam_xs, acc = [], 0
            for cw in crop_widths:
                acc += cw
                seam_xs.append(acc)


            stitched_shifts = [s for s in shifts if not s.get("skipped")]
            m = len(crop_widths)
            seam_v_offs = []
            for j in range(m):
                k = m - j - 1
                seam_v_offs.append(
                    int(stitched_shifts[k].get("v_offset", 0))
                    if 0 <= k < len(stitched_shifts) else None
                )

            base_ts = float(self.frame_buffer[0].get("timestamp", 0.0) or 0.0)
            last_stitch_ts = (
                float(getattr(self, "_last_stitch_ts", 0.0) or 0.0)
                if stitched_frames > 0 else base_ts
            )

            self._last_seam_xs = seam_xs
            self._last_panorama_meta = {
                "panorama_id": self.frame_counter,
                "stitched_w": int(panorama.shape[1]),
                "n_frames": len(self.frame_buffer),
                "stitched": int(stitched_frames),
                "content_break": bool(content_break),
                "pending_px": int(pending_shift),
                "resume_from": self._resume_from,
                "base_ts": base_ts,
                "last_stitch_ts": last_stitch_ts,
                "last_crop_w": int(crop_widths[-1]) if crop_widths else 0,
                "seam_v_offs": seam_v_offs,
            }

            return panorama, shifts



    def _create_panorama_ltr(self) -> tuple[Optional[np.ndarray], List[Dict]]:
        """
        🆕 КРОК 3.3: Зшивання панорами для LTR (турецьких) каналів.
        Повністю ізольована логіка — не впливає на арабські канали.
        
        LTR особливості:
        - Текст їде зліва направо
        - Новий контент з'являється ЗЛІВА, старий зникає СПРАВА
        - Панорама будується додаванням нових шматків СПРАВА
        - expected_dx ВІД'ЄМНИЙ (протилежний до RTL)
        """
        # ── налаштування зшивання ──
        # 🆕 Для LTR знак expected_dx протилежний
        expected_dx = -int(getattr(self, "stitch_expected_dx", 106))
        dx_tolerance = int(getattr(self, "stitch_dx_tolerance", 15))
        dy_dead_zone = float(getattr(self, "stitch_dy_dead_zone", 0.3))
        dy_ema_alpha = float(getattr(self, "stitch_dy_ema", 0.4))
        dy_max_corr = float(getattr(self, "stitch_dy_max_corr", 12.0))
        min_response = float(getattr(self, "stitch_min_response", 0.0))
        trust_response = float(getattr(self, "stitch_trust_response", 0.30))

        if not self.frame_buffer:
            return None, []

        # ═══ САНІТАРІЯ FADE (ідентична до RTL) ═══
        raw_fades = [f.get("fade", 0) for f in self.frame_buffer]
        n = len(raw_fades)
        med_fade = float(np.median(raw_fades)) if n >= 3 else None
        profile_fade = int(getattr(self, "fade_zone_px", 0) or 0)
        anchor = None
        fixed_n = 0

        if profile_fade >= 60:
           # if n >= 3 and med_fade is not None and abs(med_fade - profile_fade) <= 0.10 * profile_fade:
            if n >= 3 and med_fade is not None and abs(med_fade - profile_fade) <= self.fade_anchor_tol * profile_fade:
                anchor = int(med_fade)
            else:
                anchor = profile_fade
                if n >= 3:
                    logger.info(
                        "📌 FADE ANCHOR (LTR): медіана %.1f поза коридором профілю %d → пінг",
                        med_fade, profile_fade
                    )
        elif n >= 3 and med_fade is not None and med_fade >= 60:
            anchor = int(med_fade)
        else:
            anchor = 108

        if anchor is not None:
            lo = (1.0 - self.fade_anchor_tol) * anchor
            hi = (1.0 + self.fade_anchor_tol) * anchor
            for f, fr in zip(self.frame_buffer, raw_fades):
                if fr < lo or fr > hi:
                    if "fade_raw" not in f:
                        f["fade_raw"] = int(fr)
                    f["fade"] = anchor
                    fixed_n += 1

        logger.info(
            "🧼 FADE SANITY (LTR): n=%d anchor=%d fixed=%d "
            "raw[min=%d max=%d med=%s std=%.1f]",
            n, anchor, fixed_n,
            min(raw_fades) if raw_fades else 0,
            max(raw_fades) if raw_fades else 0,
            f"{np.median(raw_fades):.0f}" if raw_fades else "—",
            float(np.std(raw_fades)) if raw_fades else 0.0,
        )

        # ═══ ІНІЦІАЛІЗАЦІЯ ПАНОРАМИ (LTR) ═══
        first = self.frame_buffer[0]
        fade_0 = first.get("fade", 0)
        base_img = first["image"]

        if base_img is None or base_img.shape[1] - fade_0 < 50:
            return None, []

        # 🆕 LTR: обрізаємо fade СПРАВА (текст зникає вправо)
        base_w = base_img.shape[1]
        panorama = base_img[:, :base_w - fade_0].copy()
        right_edge_x = panorama.shape[1]  # правий "живий" край панорами

        shifts: List[Dict] = []
        base_h = base_img.shape[0]
        valid_top, valid_bottom = 0, base_h
        cumulative_dy = 0.0
        ema_dy = 0.0
        fade_values = [fade_0]
        crop_widths = []
        intended_widths = []

        if len(self.frame_buffer) < 2:
            return panorama, shifts

        logger.info("Побудова LTR-панорами з %d кадрів", len(self.frame_buffer))

        pending_shift = 0
        skipped_anom = 0
        skipped_none = 0
        frozen_frames = 0
        stitched_frames = 0
        content_break = False
        self._resume_from = None

        for i in range(1, len(self.frame_buffer)):
            img1 = self.frame_buffer[i - 1]["image"]
            img2 = self.frame_buffer[i]["image"]
            fade_i = self.frame_buffer[i].get("fade", 0)
            fade_values.append(fade_i)

            dt = (self.frame_buffer[i]["timestamp"]
                - self.frame_buffer[i - 1]["timestamp"])

            # Технічні пропуски
            tech_skip = (img1 is None or img2 is None
                        or img1.shape != img2.shape
                        or img2.shape[0] != base_h)
            if tech_skip:
                pending_shift += abs(expected_dx)  # LTR: pending завжди додатний
                skipped_none += 1
                if dt > 0:
                    shifts.append({
                        "dx": int(expected_dx),
                        "dt": dt,
                        "dy": 0.0, "dy_ema": round(float(ema_dy), 3),
                        "response": 0.0, "dx_anomaly": False,
                        "v_offset": 0, "fade_zone": int(fade_i),
                        "skipped": True,
                    })
                continue

            result = self._get_horizontal_shift_with_dy(img1, img2)
            if result is None:
                logger.warning("Пропуск кадру %d: зсув не визначено", i)
                pending_shift += abs(expected_dx)
                skipped_none += 1
                if dt > 0:
                    shifts.append({
                        "dx": int(expected_dx),
                        "dt": dt,
                        "dy": 0.0, "dy_ema": round(float(ema_dy), 3),
                        "response": 0.0, "dx_anomaly": False,
                        "v_offset": 0, "fade_zone": int(fade_i),
                        "skipped": True,
                    })
                continue

            dx, dy, response = result
            h, w = img2.shape[:2]
            shift = int(round(abs(dx)))

            # 🛡️ КРОК 0: ЗАХИСТ ВІД ПОМИЛКОВОГО РОЗРИВУ (Заморожені кадри / Шум)
            # Якщо зсув 0 або 1 піксель — це не розрив і не реверс, а статика або шум H.264.
            # Ми МАЄМО перевірити це ДО множення на expected_dx, інакше для LTR (де expected_dx < 0)
            # шумовий dx=1 дасть (1 * -106 < 0) і помилково тригерить "РОЗРИВ КОНТЕНТУ".
            if abs(dx) < 2:
                frozen_frames += 1
                # Опціонально: можна записати у shifts для телеметрії, але для панорами пропускаємо
                continue

            # Контроль аномалій
            dx_anomaly = abs(dx - expected_dx) > dx_tolerance

            if dx_anomaly:
                logger.warning(
                    "🟧 ПРОПУСК ЗШИВАННЯ (LTR): frame=%d dx=%d "
                    "(очікується %d±%d), dy=%.2f response=%.3f",
                    i, int(dx), expected_dx, dx_tolerance, dy, response,
                )
                skipped_anom += 1

                if dt > 0:
                    shifts.append({
                        "dx": 0 if (dx * expected_dx < 0 or response < trust_response) else int(dx),
                        "dt": dt,
                        "dy": round(float(dy), 3),
                        "dy_ema": round(float(ema_dy), 3),
                        "response": round(float(response), 3),
                        "dx_anomaly": True,
                        "v_offset": 0, "fade_zone": int(fade_i),
                        "skipped": True,
                    })

                if dx * expected_dx < 0 or response < trust_response:
                    content_break = True
                    logger.warning(
                        "🟪 РОЗРИВ КОНТЕНТУ (LTR): кадр %d (dx=%d, resp=%.2f)",
                        i, int(dx), response,
                    )
                    if not getattr(self, "_content_change_reported", False):
                        self._content_change_reported = True
                        zone_text = ""
                        try:
                            res = self.ocr.predict(img2)
                            parts = [t.strip() for page in res
                                    for t, s in zip(page.get('rec_texts', []),
                                                    page.get('rec_scores', []))
                                    if float(s) > 0.5 and len(t.strip()) > 1]
                            zone_text = ' '.join(parts)[:100]
                        except Exception:
                            zone_text = ""
                        self._emit_system_event("content_changed", f"новий контент: {zone_text or 'OCR пусто'}")
                    self._resume_from = i
                    break

                # if abs(dx) < 2:
                #     frozen_frames += 1
                #     continue

                pending_shift += shift
                continue

            if shift <= 0 or shift >= w:
                continue

            total_advance = pending_shift + shift
            overlap = w - total_advance
            if overlap < int(w * 0.25):
                content_break = True
                logger.warning(
                    "🟪 РОЗРИВ (LTR): перекриття %dpx (<25%%)", overlap
                )
                self._resume_from = i
                break

            if response < min_response:
                pending_shift += shift
                skipped_none += 1
                continue

            # Вертикальний дрейф
            # Вертикальний дрейф
            ema_dy = dy_ema_alpha * dy + (1.0 - dy_ema_alpha) * ema_dy
            if abs(ema_dy) >= dy_dead_zone:
                cumulative_dy = max(-dy_max_corr, min(dy_max_corr, cumulative_dy + ema_dy))
            offset = int(round(cumulative_dy))

            # 🆕 LTR: ЄДИНА ПРАВИЛЬНА ЛОГІКА ЗШИВАННЯ (КРОК 1: Визначення кропу)
            if dx < 0:
                # Нормальний рух для LTR (dx від'ємний)
                # Панорама росте ВПРАВО, новий контент додається СПРАВА
                end_x = w - fade_i - 3
                if end_x < 0:
                    end_x = w  # захист від виродженого fade

                # Ліва межа нового кропу: край зшитого контенту, перекадрований у поточний кадр
                start_x = max(0, right_edge_x - pending_shift - shift)

                if end_x - start_x < 5:
                    right_edge_x = max(0, min(start_x, w))
                    pending_shift = 0
                    continue

                new_part = img2[:, start_x:end_x].copy()
            else:
                # Аномальний рух для LTR (стрибок у протилежний бік)
                start_x = 0
                end_x = shift
                new_part = img2[:, start_x:end_x].copy()

            current_crop_width = end_x - start_x
            crop_widths.append(current_crop_width)
            intended_widths.append(total_advance)

            # Вирівнювання без чорних рядків (КРОК 2: Вертикальна корекція)
            prev_top, prev_bottom = valid_top, valid_bottom
            if offset > 0:
                valid_bottom = min(valid_bottom, h - offset)
            elif offset < 0:
                valid_top = max(valid_top, -offset)

            if valid_bottom - valid_top < max(10, base_h // 3):
                logger.warning("Кадр %d: dy-корекція скинута (LTR)", i)
                valid_top, valid_bottom = prev_top, prev_bottom
                offset = 0
                cumulative_dy = 0.0

            # Застосовуємо вертикальний зсув до вже визначеного new_part
            new_part = new_part[valid_top + offset: valid_bottom + offset, :]
            panorama = panorama[valid_top - prev_top: valid_bottom - prev_top, :]

            # 🆕 LTR: ЄДИНА ПРАВИЛЬНА ЛОГІКА ЗШИВАННЯ (КРОК 3: Конкатенація)
            if dx < 0:
                # Додаємо СПРАВА
                panorama = cv2.hconcat([panorama, new_part])
                right_edge_x = end_x  # ✅ Оновлюємо край у координатах кадру!
            else:
                # Аномальний рух: додаємо ЗЛІВА
                panorama = cv2.hconcat([new_part, panorama])
                right_edge_x = max(0, right_edge_x - shift)

            if dt > 0:
                shifts.append({
                    "dx": int(dx),
                    "dt": dt,
                    "dy": round(float(dy), 3),
                    "dy_ema": round(float(ema_dy), 3),
                    "response": round(float(response), 3),
                    "dx_anomaly": dx_anomaly,
                    "v_offset": offset,
                    "fade_zone": int(fade_i),
                    "skipped": False,
                    "pending_px": int(pending_shift)
                })

            self._last_stitch_ts = float(self.frame_buffer[i]["timestamp"])
            pending_shift = 0
            stitched_frames += 1
            self._content_change_reported = False

            logger.info(
                "Stitch LTR frame=%d dx=%d dy=%.2f cum_dy=%.2f off=%d "
                "crop=[%d:%d] panorama=%dx%d",
                i, int(dx), dy, cumulative_dy, offset,
                start_x, end_x, panorama.shape[1], panorama.shape[0],
            )

        # ── якість панорами ──
        if panorama.shape[1] > 0:
            readable_ratio = max(
                0.0, min(100.0,
                        (panorama.shape[1] - fade_0)
                        / panorama.shape[1] * 100))
            logger.info(
                "🟩 ЯКІСТЬ LTR-ПАНОРАМИ: читабельно=%.1f%%, fade_zone=%dpx, "
                "total_width=%dpx, height=%dpx%s",
                readable_ratio, fade_0, panorama.shape[1], panorama.shape[0],
                "  [🟪 ОБРІЗАНА]" if content_break else "",
            )

        # Статистика FADE
        fade_arr = np.array(raw_fades)
        fade_min = int(np.min(fade_arr))
        fade_max = int(np.max(fade_arr))
        fade_std = float(np.std(fade_arr))

        if shifts:
            avg_response = float(np.mean([s.get("response", 0.0) for s in shifts]))
            max_dy = max(abs(s.get("dy", 0.0)) for s in shifts)
            last_ema = shifts[-1].get("dy_ema", 0.0)
            n_anom = sum(1 for s in shifts if s.get("dx_anomaly"))

            logger.info(
                " ЗШИВАННЯ LTR: зшито=%d, пропущено=%d (аномалій=%d, "
                "невиміряних=%d, заморожено=%d), response=%.3f, "
                "макс |dy|=%.2fpx, EMA dy=%.2fpx, накопичена корекція=%.1fpx",
                stitched_frames,
                skipped_anom + skipped_none,
                skipped_anom, skipped_none, frozen_frames,
                avg_response, max_dy, last_ema, cumulative_dy,
            )

        logger.info(
            "📏 FADE ZONE (LTR): min=%dpx, max=%dpx, std=%.1fpx (профіль: %dpx)",
            fade_min, fade_max, fade_std, self.fade_zone_px
        )

        # Crop width статистика
        if crop_widths:
            crop_arr = np.array(crop_widths)
            intended_arr = np.array(intended_widths, dtype=float)
            crop_min = int(np.min(crop_arr))
            crop_max = int(np.max(crop_arr))
            crop_mean = float(np.mean(crop_arr))
            anomalies = int(np.sum(
                (crop_arr < intended_arr * 0.8) | (crop_arr > intended_arr * 1.2)
            ))

            logger.info(
                "🟪 CROP WIDTH (LTR): min=%dpx, max=%dpx, mean=%.1fpx "
                "(намічено: %.1fpx, аномалій: %d)",
                crop_min, crop_max, crop_mean,
                float(np.mean(intended_arr)) if len(intended_arr) else 0.0,
                anomalies
            )

        # Vertical drift статистика
        if shifts:
            dy_raw_values = [s["dy"] for s in shifts]
            offs = [s.get("v_offset", 0.0) for s in shifts if not s.get("skipped")]
            dy_applied_values = (
                [offs[0]] + [b - a for a, b in zip(offs, offs[1:])]
                if offs else []
            )
            dy_raw_sum = float(sum(dy_raw_values))
            dy_applied_sum = float(sum(dy_applied_values))
            dy_cum_max = float(max(abs(s.get("v_offset", 0.0)) for s in shifts))

            logger.info(
                "↕️ VERTICAL DRIFT (LTR): raw_sum=%.1fpx, applied_sum=%.1fpx, max_cum=%.1fpx",
                dy_raw_sum, dy_applied_sum, dy_cum_max
            )

        if fade_std > 15.0:
            logger.warning(
                "🟥 FADE INSTABILITY (LTR): std=%.1fpx занадто високий!", fade_std
            )

        logger.info(
            "Панорама LTR: width=%d height=%d shifts=%d cum_dy=%.1fpx",
            panorama.shape[1], panorama.shape[0], len(shifts), cumulative_dy,
        )

        # ═══ E1 ТЕЛЕМЕТРІЯ (LTR) ═══
        # Для LTR шви рахуються як ПРЕФІКСНІ суми (протилежно до RTL)

        w_base = int(panorama.shape[1]) - sum(crop_widths) # Ширина базового кадру
        seam_xs, acc = [], w_base
        for cw in crop_widths:
            acc += cw
            seam_xs.append(acc)

        stitched_shifts = [s for s in shifts if not s.get("skipped")]
        m = len(crop_widths)
        seam_v_offs = []
        for j in range(m):
            seam_v_offs.append(
                int(stitched_shifts[j].get("v_offset", 0))
                if j < len(stitched_shifts) else None
            )

        base_ts = float(self.frame_buffer[0].get("timestamp", 0.0) or 0.0)
        last_stitch_ts = (
            float(getattr(self, "_last_stitch_ts", 0.0) or 0.0)
            if stitched_frames > 0 else base_ts
        )

        self._last_seam_xs = seam_xs
        self._last_panorama_meta = {
            "panorama_id": self.frame_counter,
            "stitched_w": int(panorama.shape[1]),
            "n_frames": len(self.frame_buffer),
            "stitched": int(stitched_frames),
            "content_break": bool(content_break),
            "pending_px": int(pending_shift),
            "resume_from": self._resume_from,
            "base_ts": base_ts,
            "last_stitch_ts": last_stitch_ts,
            "last_crop_w": int(crop_widths[-1]) if crop_widths else 0,
            "seam_v_offs": seam_v_offs,
            "direction": "ltr",  # 🆕 маркер для дебагу
        }

        return panorama, shifts

    def _analyze_black_region_location(self, panorama: np.ndarray) -> str:
        """
        Визначає локалізацію чорних/темних артефактів на панорамі.
        Повертає рядок-діагноз для швидкого виправлення.
        """
        if panorama is None or panorama.size == 0:
            return "none"

        gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if h == 0 or w == 0:
            return "none"

        # Шукаємо майже чорні пікселі (поріг 10, щоб не плутати з темним текстом/фоном)
        black_mask = gray < 10

        # 1. Перевірка КРАЇВ (Лівий/Правий) - проблема fade_zone або неповного кадру
        edge_width = max(10, w // 20)  # Перевіряємо крайні 5% ширини
        left_ratio = np.mean(black_mask[:, :edge_width]) if w > edge_width else 0
        right_ratio = np.mean(black_mask[:, -edge_width:]) if w > edge_width else 0

        if left_ratio > 0.7:
            return "left_edge (fade_zone?)"
        if right_ratio > 0.7:
            return "right_edge (incomplete frame?)"

        # 2. Перевірка ВЕРХУ/НИЗУ - проблема dy корекції або ROI
        edge_height = max(5, h // 20)
        top_ratio = np.mean(black_mask[:edge_height, :]) if h > edge_height else 0
        bottom_ratio = np.mean(black_mask[-edge_height:, :]) if h > edge_height else 0

        if top_ratio > 0.7:
            return "top_edge (dy_crop?)"
        if bottom_ratio > 0.7:
            return "bottom_edge (dy_crop?)"

        # 3. Пошук ВЕРТИКАЛЬНИХ СТИКІВ (Найчастіша проблема тікера)
        # Рахуємо частку чорних пікселів у кожному стовпці
        col_darkness = np.mean(black_mask, axis=0) 
        
        # Стовпець вважається "чорною лінією", якщо > 80% його висоти чорне
        black_cols = np.where(col_darkness > 0.8)[0]

        if len(black_cols) > 0:
            # Групуємо сусідні чорні стовпці, щоб знайти центр стику
            groups = []
            current_group = [black_cols[0]]
            for x in black_cols[1:]:
                if x - current_group[-1] <= 3:  # Дозволяємо мікро-розриви
                    current_group.append(x)
                else:
                    groups.append(current_group)
                    current_group = [x]
            groups.append(current_group)

            # Знаходимо найширший стик (найбільш виражений артефакт)
            widest_seam = max(groups, key=len)
            center_x = int(np.mean(widest_seam))
            return f"seam_at_{center_x}px"

        # 4. Якщо чорнота розпорошена (загальне затемнення)
        total_ratio = np.mean(black_mask)
        if total_ratio > 0.4:
            return "scattered_dark (bad ROI?)"
        
        return "none"
    
    


    def _get_horizontal_shift_with_dy(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> Optional[tuple[float, float, float]]:
        """
        🔑 Повертає (dx, dy, response) або None.
        Тепер — ті самі гейти, що й у _get_horizontal_shift (DX TEST):
        GaussianBlur 3×3, вікно Ханна, відсів NaN, |dy|, |dx|>w/2, response<0.10.
        Різниця одна: dx==0 НЕ відкидаємо (заморожена стрічка — легітимний
        вимір, його класифікує _create_panorama).
        """
        try:
            g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
            g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2

            if g1.shape[:2] != g2.shape[:2]:
                return None
            h, w = g1.shape
            if h < 8 or w < 16:
                return None

            g1 = cv2.GaussianBlur(g1, (3, 3), 0)
            g2 = cv2.GaussianBlur(g2, (3, 3), 0)

            window = cv2.createHanningWindow((w, h), cv2.CV_32F)
            (dx, dy), response = cv2.phaseCorrelate(
                np.float32(g1), np.float32(g2), window
            )
            dx, dy, response = float(dx), float(dy), float(response)

            if not (np.isfinite(dx) and np.isfinite(dy)
                    and np.isfinite(response)):
                return None
            if abs(dy) > max(3.0, h * 0.10):
                return None
            if abs(dx) > w * 0.5:
                return None
            if response < 0.10:
                return None
            return dx, dy, response
        except cv2.error:
            return None




    def _compute_speed_from_shifts(
        self,
        shifts: List[Dict],
        roi_width: int,
    ) -> Dict:
        """
        Чиста арифметика: рахує швидкість стрічки з уже обчислених зсувів.
        Без phase correlation і без I/O.
        """
        if not shifts:
            return {}

        total_px = sum(abs(s["dx"]) for s in shifts)
        total_dt = sum(s["dt"] for s in shifts)

        if total_dt <= 0 or total_px == 0:
            return {}

        px_per_sec = total_px / total_dt

        # 🆕 ковзне медіанне згладжування (стійке до викидів)
        self.speed_history.append(px_per_sec)
        smoothed = float(np.median(self.speed_history))

        full_pass = (roi_width / smoothed) if smoothed > 0 else 0.0

        # 🆕 sanity-check: фізично можливі межі для 1080p
        sane = self.min_speed_px_sec <= smoothed <= 3000.0
        if not sane:
            logger.warning(
                "⚠️ Підозріла швидкість стрічки: %.1f пікс/сек "
                "(min=%.1f, raw=%.1f, samples=%d)",
                smoothed, self.min_speed_px_sec, px_per_sec, len(self.speed_history),
            )

        return {
            "pixels_per_sec": round(smoothed, 2),
            "raw_pixels_per_sec": round(px_per_sec, 2),
            "full_pass_time_sec": round(full_pass, 2),
            "roi_width": roi_width,
            "samples": len(shifts),
            "smoothed_samples": len(self.speed_history),
            "sane": sane,
        }

    def _log_speed_metric(self, timestamp: float, stats: Dict):
        """
        Пише один рядок метрики у ticker_speed.jsonl.
        """
        if not stats:
            return
        metrics_file = self.output_dir / "ticker_speed.jsonl"
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "frame_timestamp": timestamp,
            "frame_counter": self.frame_counter,
            "channel": self.channel_id,
            **stats,
        }
        try:
            with open(metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error("Помилка запису ticker_speed.jsonl: %s", e)


    def _expected_prev_overlap_px(self) -> Dict:
        """
        E1 · ЕТАП 1: очікувана ширина зони дублювання на правому краю поточної
        панорами — контент, який ЦЯ панорама захопила своїм базовим кадром, а
        ПОПЕРЕДНЯ (та, що дійшла до OCR) вже захопила своїм останнім стиком.

        Геометрія (виходить із руху стрічки, а не з відсотків ширини):
        за час dt між останнім стиком попередньої панорами і базовим кадром
        поточної стрічка проїжджає v·dt px, тож із видимого вікна кадру
        (w_кадру − fade) дублюється все, крім проїжджого:
            overlap = span − v·dt,  обмежено [0, span].
        dt = 0 (нормальний кінець: базовий кадр наступної панорами — ТОЙ САМИЙ
        кадр, що зашито останнім стиком) дає overlap = span — уся базова
        частина нової панорами дублює хвіст попередньої (це і є T-зона).
        """
        span = 0.0
        try:
            frame_w = int(self.frame_buffer[0]["image"].shape[1])
            span = max(0.0, float(frame_w - int(getattr(self, "fade_zone_px", 0) or 0)))
        except Exception:
            span = 0.0

        prev = getattr(self, "_edge_prev_meta", None) or {}
        info = {
            "overlap_px": 0.0,
            "span": span,
            "dt": None,
            "speed": None,
            "prev_id": prev.get("panorama_id"),
            "note": "",
        }

        last_ts = float(prev.get("last_stitch_ts", 0.0) or 0.0)
        if not prev or last_ts <= 0.0:
            info["note"] = "no_prev"      # перша панорама: зони дублювання немає
            return info
        if span <= 0.0:
            info["note"] = "no_span"
            return info

        try:
            base_ts = float(self.frame_buffer[0].get("timestamp", 0.0) or 0.0)
        except Exception:
            base_ts = 0.0
        if base_ts <= 0.0:
            info["note"] = "no_base_ts"
            return info

        dt = max(0.0, base_ts - last_ts)
        speed = float((self.speed_stats or {}).get("pixels_per_sec", 0.0) or 0.0)
        info["dt"] = dt
        info["speed"] = speed

        if speed <= 0.0:
            # швидкість невідома → консервативно вся зона: краще перебільшити
            # C, ніж пропустити дублікат
            info["overlap_px"] = span
            info["note"] = "speed_unknown"
        else:
            info["overlap_px"] = max(0.0, min(span, span - speed * dt))
            info["note"] = "ok"
        return info


    def _detect_edge_words(self, raw_blocks, panorama, timestamp):
        """
        E1 · ЕТАП 1: EDGE WORD DETECTION — лише спостереження, БЕЗ змін тексту.
        raw_blocks не мутує; групування, фрагменти й асамблер не чіпає.

        Класифікація країв (замість одного edge_side):
          A — блок перетинає ВНУТРІШНІЙ шов панорами. Пікселі слова повні
              (кропи телескопічні, дублікатів усередині панорами немає), ризик —
              артефакт H.264/dy на лінії шва, через який OCR розщеплює слово
              на 2 блоки (ознака: від'ємний gap_to_next);
          B — блок торкається ЛІВОГО краю: найновіший контент, хвіст слова
              ще не прийшов — продовження прийде в наступній панорамі.
              b_reason: break (розрив контенту) | pending (не зшиті px зліва)
              | stream (нормальний кінець панорами);
          C — блок у зоні очікуваного перекриття з ПОПЕРЕДНЬОЮ панорамою
              (правий край): дублікат → ризик хибної «новизни» в ПРАВИЛАХ 1–3.

        Усі величини детерміновані: позиції швів — суфіксні суми crop_widths
        (хук у _create_panorama), unconsumed pending_px, зона C — швидкість
        стрічки × dt між панорамами (_expected_prev_overlap_px).
        """
        if not getattr(self, "edge_detection_enabled", True):
            return
        if not raw_blocks or panorama is None:
            return

        meta = getattr(self, "_last_panorama_meta", None) or {}
        if not meta:
            logger.warning(
                "EDGE: телеметрія панорами відсутня — детекцію пропущено"
            )
            return

        fade_crop = int(getattr(self, "_last_fade_crop_px", 0) or 0)
        w_ocr = int(panorama.shape[1])

        # Страховка від застарілої телеметрії (ранній return у
        # _create_panorama): зшита ширина − fade_crop має дорівнювати ширині
        # панорами, яку бачить OCR.
        stitched_w = int(meta.get("stitched_w", 0) or 0)
        if stitched_w > 0 and stitched_w - fade_crop != w_ocr:
            logger.warning(
                "EDGE: телеметрія не відповідає панорамі "
                "(stitched=%d − fade=%d ≠ %d) — детекцію пропущено",
                stitched_w, fade_crop, w_ocr,
            )
            return

        seams_raw = getattr(self, "_last_seam_xs", None) or []
        # переводимо шви в систему координат OCR (enhanced = зшита − fade_crop)
        #seams = [float(x) - fade_crop for x in seams_raw]

        x_shift = fade_crop if self.is_rtl else 0
        seams = [float(x) - x_shift for x in seams_raw]
        v_offs = list(meta.get("seam_v_offs", []) or [])

        prox = float(getattr(self, "edge_proximity_px", 8.0))
        tol = float(getattr(self, "seam_cross_tol_px", 6.0))
        ov = self._expected_prev_overlap_px()
        overlap = float(ov.get("overlap_px", 0.0))

        stats = {"A": 0, "B": 0, "C": 0}
        reasons = {"break": 0, "pending": 0, "stream": 0}
        b_words = []   # 🆕 E2: блоки класу B → PendingSeamFragment ( популяція після SUMMARY)

        for idx, b in enumerate(raw_blocks):
            x_min = float(b["x_min"])
            x_max = float(b["x_max"])
            ys = b["box"][:, 1]
            try:
                y_min = float(ys.min())
                y_max = float(ys.max())
            except Exception:
                y_min = y_max = 0.0

            # шви, які потрапляють у bbox із допуском tol
            crossed = [
                (seams[j], v_offs[j] if j < len(v_offs) else None)
                for j in range(len(seams))
                if x_min - tol < seams[j] < x_max + tol
            ]

            # cls = None
            # reason = None
            # if crossed:
            #     cls = "A"
            # elif x_min <= prox:
            #     cls = "B"
            #     if meta.get("content_break"):
            #         reason = "break"
            #     elif int(meta.get("pending_px", 0) or 0) > 0:
            #         reason = "pending"
            #     else:
            #         reason = "stream"
            # # ВИЗНАЧЕННЯ КЛАСУ C (Зона дублікатів)
            # elif overlap > 0.0 and x_max >= w_ocr - overlap:
            #     cls = "C"
            # else:
            #     continue


            cls = None
            reason = None
            if crossed:
                cls = "A"
            # 🆕 КРОК 5.2: Клас B залежить від напрямку панорами
            elif self.is_rtl and x_min <= prox:
                # RTL (араби): живий вхід = ЛІВИЙ край
                cls = "B"
                if meta.get("content_break"):
                    reason = "break"
                elif int(meta.get("pending_px", 0) or 0) > 0:
                    reason = "pending"
                else:
                    reason = "stream"
            elif not self.is_rtl and x_max >= w_ocr - prox:
                # LTR (турки): живий вхід = ПРАВИЙ край
                cls = "B"
                if meta.get("content_break"):
                    reason = "break"
                elif int(meta.get("pending_px", 0) or 0) > 0:
                    reason = "pending"
                else:
                    reason = "stream"
            # 🆕 КРОК 5.2: Клас C (зона дублікатів) залежить від напрямку
            elif self.is_rtl and overlap > 0.0 and x_max >= w_ocr - overlap:
                # RTL: дублікати на ПРАВОМУ краю
                cls = "C"
            elif not self.is_rtl and overlap > 0.0 and x_min <= overlap:
                # LTR: дублікати на ЛІВОМУ краю
                cls = "C"
            else:
                continue

            stats[cls] += 1
            if reason:
                reasons[reason] += 1

            if cls == "B":   # 🆕 E2: підозрілий фрагмент (bbox+reason) — піде у буфер
                b_words.append((b, reason, y_min, y_max))

            nb = raw_blocks[idx + 1] if idx + 1 < len(raw_blocks) else None
            gap_next = (
                f"{x_min - float(nb['x_max']):+.0f}px" if nb is not None else "—"
            )

            if cls == "A":
                extra = (
                    f" seams_crossed={len(crossed)} "
                    f"nearest_seam={crossed[0][0]:.0f}"
                    f"(dy={crossed[0][1]})"
                )
            elif cls == "B":
                extra = f" reason={reason}"
            else:
                extra = f" in_dup_zone={w_ocr - x_max:.0f}/{overlap:.0f}px"

            logger.info(
                "🔍 EDGE WORD DETECTION: class=%s text=%r conf=%.2f "
                "bbox=[%.0f..%.0f x %.0f..%.0f] d_left=%.0f d_right=%.0f "
                "gap_to_next=%s%s status=SUSPECTED_INCOMPLETE",
                cls, b["text"], float(b.get("score", 0.0)),
                x_min, x_max, y_min, y_max,
                x_min, w_ocr - x_max,
                gap_next, extra,
            )

        b_part = (
            f"{stats['B']}(break={reasons['break']},"
            f"pending={reasons['pending']},stream={reasons['stream']})"
            if stats["B"] else "0"
        )
        logger.info(
            "🧭 EDGE SUMMARY: pan=%s A_seam=%d B_entry=%s C_overlap=%d | "
            "w=%d fade_crop=%d seams=%d | dup_zone=%s dt_prev=%s v=%s | "
            "break=%s pending_px=%s [%s]",
            meta.get("panorama_id", "—"),
            stats["A"], b_part, stats["C"],
            w_ocr, fade_crop, len(seams),
            (f"{w_ocr - overlap:.0f}..{w_ocr}" if overlap > 0.0 else "—"),
            (f"{ov['dt']:.2f}с" if ov.get("dt") is not None else "—"),
            (f"{ov['speed']:.0f}px/с" if ov.get("speed") is not None else "—"),
            "Yes" if meta.get("content_break") else "No",
            meta.get("pending_px", "—"),
            ov.get("note", "—"),
        )

        # E2: популяція буфера PendingSeamFragment (класи B) ПЕРЕД тим,
        # як телеметрія цієї панорами стане «попередньою»
        self._populate_pending_seam_fragments(
                b_words, meta, ov, panorama, timestamp, fade_crop, raw_blocks,
            )

        # телеметрія ЦІЄЇ панорами стає «попередньою» для наступної;
        # оновлюється лише коли панорама реально дійшла до OCR
        #self._edge_prev_meta = dict(meta)

        self._dup_zone_px = overlap
        self._dup_start_x = (w_ocr - overlap) if self.is_rtl else overlap
        self._edge_prev_meta = dict(meta)






    # ══════════════════════════════════════════════════════════════════
    # E2 · ЕТАП 2: CROSS-PANORAMA MATCHING — ДІАГНОСТИЧНИЙ РЕЖИМ.
    # Шукає продовження обрізаного слова (клас B з E1) у наступній панорамі.
    # КОНТРАКТ: НЕ змінює текст/групи/фрагменти/асамблер, НЕ мутує
    # raw_blocks; лише веде лог і буфер (саме тому патч оборотний).
    # Після валідації на реальних логах цей же код стане ядром
    # автоматичного відновлення (буде додано поле merged_text).
    # ══════════════════════════════════════════════════════════════════

    def _populate_pending_seam_fragments(self, b_words, meta, ov, panorama, timestamp, fade_crop, raw_blocks=None):
        """E2: класи B цієї панорами → PendingSeamFragment (структура з плану
        етапу 2: text/timestamp/panorama_id/edge/confidence/bbox/is_incomplete).
        Викликається зсередини _detect_edge_words ПЕРЕД переносом телеметрії
        в _edge_prev_meta — геометрія панорами ще доступна як «поточна».
        E2v2: додатково зберігає ctx_prev — старші (у читанні) сусідні блоки
        ТОГО САМОГО речення на панорамі-джерелі. Без них однословний фрагмент
        ('بجديه') неможливо перевірити як частину цілісного речення: атрибуції
        повторюються у різних елементів ('أمين مجلس الأمن القومي' у двох
        сусідніх), і лише контекст + геометрія розрізняють їх."""
        if not getattr(self, "cross_pano_matching_enabled", True):
            return
        self._pending_seam_fragments = []   # E2: буфер однопанорамний —
        # класи B попередньої панорами, яким не знайшли продовження, більше
        # ніколи не знайдуть його (контент уже виїхав) — тримати їх без сенсу.

        for b, reason, y_min, y_max in b_words:
            text = str(b.get("text", "") or "").strip()
            if not text:
                continue
            self._pending_seam_fragments.append({
                "text": text,
                "timestamp": float(timestamp),
                "panorama_id": meta.get("panorama_id"),
                "pan_w": int(panorama.shape[1]),
                #"edge": "left",               # у цій схемі зшивання обрізаний
                "edge": "left" if self.is_rtl else "right",
                # хвіст живе на лівому краю (x=0 — живий вхід)
                "confidence": float(b.get("score", 0.0)),
                "bbox": (float(b["x_min"]), float(b["x_max"]), y_min, y_max),
                "is_incomplete": True,
                "b_reason": reason,           # break | pending | stream
                # геометрія зони пошуку на НАСТУПНІЙ панорамі (dup zone):
                "dup_zone": (
                    f"{float(ov.get('overlap_px', 0.0)):.0f}px"
                    if float(ov.get('overlap_px', 0.0)) > 0.0 else None
                ),
                "dt_prev": ov.get("dt"),
                "speed_px_s": ov.get("speed"),
                # 🆕 E2v2: «попередні слова» речення на панорамі-джерелі —
                # старші сусідні блоки тому ж рядку, у порядку читання
                # (від найстарішого до фрагмента)
                "ctx_prev": self._seam_ctx_prev_blocks(b, raw_blocks, y_min, y_max),
            })

    def _seam_ctx_prev_blocks(self, frag_b, raw_blocks, y_min, y_max):
        """E2v2: сусіди фрагмента правіше від нього (старші в читанні) —
        це початок того самого речення на панорамі-джерелі. Ланцюг
        обривається на зазорі > 50px (сепаратор/інший елемент) або на
        лімітах слів/пікселів. Однорядковість — за суворим y-перекриттям."""
        out: List[Dict] = []
        if not raw_blocks:
            return out

        # 5. Надійне отримання Y фрагмента
        try:
            fy0 = float(min(p[1] for p in frag_b["box"]))
            fy1 = float(max(p[1] for p in frag_b["box"]))
        except Exception:
            fy0 = float(frag_b.get("y_min", y_min))
            fy1 = float(frag_b.get("y_max", y_max))

        fx1 = float(frag_b.get("x_max", 0.0))
        fx0 = float(frag_b.get("x_min", 0.0))

        # 6. "Бігучий" Y-діапазон ланцюга для запобігання дрейфу по похилій панорамі
        chain_y0, chain_y1 = fy0, fy1

        cand = []
        for b in raw_blocks:
            if b is frag_b:
                continue
            t = str(b.get("text", "") or "").strip()
            if not t:
                continue

            # 5. Надійне отримання X. Якщо дефолт 0.0, пробуємо взяти з box
            bx0 = float(b.get("x_min", 0.0))
            bx1 = float(b.get("x_max", 0.0))
            if bx0 == 0.0 and bx1 == 0.0:
                try:
                    bx0 = float(min(p[0] for p in b["box"]))
                    bx1 = float(max(p[0] for p in b["box"]))
                except Exception:
                    continue  # Пропускаємо "битий" блок без координат

            join_px = getattr(self, "_seam_ctx_join_px", 12.0)
            # Фільтр напрямку: RTL шукає правіше, LTR шукає лівіше
            if self.is_rtl:
                if bx0 < fx1 - join_px or bx1 <= fx1:
                    continue  # лівіше або перетинає — не старший сусід
            else:
                if bx0 >= fx0 or bx1 > fx0 + join_px:
                    continue  # правіше або перетинає — не старший сусід

            # 1. Надійне отримання Y кандидата. Якщо box зламан, беремо y_min/y_max
            try:
                by0 = float(min(p[1] for p in b["box"]))
                by1 = float(max(p[1] for p in b["box"]))
            except Exception:
                by0 = float(b.get("y_min", 0.0))
                by1 = float(b.get("y_max", 1e9))
                # 1. Якщо і це не спрацювало (залишились дефолти) — пропускаємо, щоб не зламати перевірку рядка
                if by0 == 0.0 and by1 == 1e9:
                    continue

            # 6. Суворий Y-перевірка: перекриття має бути >= 50% від мінімальної висоти
            # Порівнюємо з chain_y (усім ланцюгом), а не лише з фрагментом, щоб уникнути дрейфу
            overlap = min(by1, chain_y1) - max(by0, chain_y0)
            min_height = min(by1 - by0, chain_y1 - chain_y0)
            if min_height <= 0 or overlap < 0.5 * min_height:
                continue  # інший рядок або занадто великий перекіс

            cand.append((bx0, bx1, t, by0, by1))

        # 2. Правильне сортування: для LTR сортуємо за bx1 (правий край), щоб ближчий сусід був першим
        if self.is_rtl:
            cand.sort(key=lambda c: c[0])  # bx0 зростання (рух праворуч)
        else:
            cand.sort(key=lambda c: c[1], reverse=True)  # bx1 спадання (рух ліворуч)

        chain_x = fx1 if self.is_rtl else fx0
        start_x = chain_x
        current_n_words = 0

        gap_px = getattr(self, "_seam_ctx_gap_px", 50.0)
        max_blocks = getattr(self, "_seam_ctx_max_blocks", 6)
        max_words = getattr(self, "_seam_ctx_max_words", 10)
        max_px = getattr(self, "_seam_ctx_max_px", 900.0)

        for bx0, bx1, t, by0, by1 in cand:
            # Розрахунок зазору у бік руху ланцюга
            gap = (bx0 - chain_x) if self.is_rtl else (chain_x - bx1)

            if gap > gap_px:
                break  # зазор сепаратора — речення закінчилось

            # 3. Захист від дублікатів OCR (той самий текст вже в ланцюгу)
            if any(c["text"] == t for c in out):
                continue

            # 4. РОЗРАХУНОК ПРОСПЕКТИВНИХ ЛІМІТІВ ДО ДОДАВАННЯ
            prospective_words = current_n_words + len(t.split())
            # 3. Монотонність якоря: запобігає "відкату" назад при вкладених дублях детекції
            prospective_chain_x = max(chain_x, bx1) if self.is_rtl else min(chain_x, bx0)
            prospective_px_dist = abs(prospective_chain_x - start_x)

            if len(out) >= max_blocks or prospective_words >= max_words or prospective_px_dist > max_px:
                break  # Ліміт вичерпано, не додаємо цей блок

            # Додаємо лише якщо всі ліміти дотримані
            out.append({"text": t, "x_min": bx0, "x_max": bx1})
            current_n_words = prospective_words
            chain_x = prospective_chain_x  # Оновлюємо якір монотонно

            # 6. Звужуємо "бігучий" Y-діапазон до перетину, щоб наступні блоки перевірялися суворіше
            chain_y0 = max(chain_y0, by0)
            chain_y1 = min(chain_y1, by1)

        # Повертаємо від найстарішого до фрагмента (порядок читання)
        out.reverse()
        return out
    
    # def _seam_head_candidates(self, panorama):
    #     """E2: ширина зони пошуку на новій панорамі (для класу B попередньої —
    #     це найстаріший «свіжий» контент, одразу лівіше dup zone).
    #     dup_zone — ГЕОМЕТРИЧНА ОЦІНКА (span − v·dt), а не вимір; помилка в
    #     десятки пікселів, тому зона — смуга навколо межі, а не точка."""
    #     if panorama is None:
    #         return 0.0, 200.0
    #     w = int(panorama.shape[1])
    #     ov = float((self._expected_prev_overlap_px() or {}).get("overlap_px", 0.0))
    #     dup_start = max(0.0, w - ov) if ov > 0.0 else 0.0
    #     # слово (100–150px) в обидва боки від оціненої межі dup zone
    #     zone_w = min(w, dup_start + 150.0)
    #     return dup_start, zone_w

    def _seam_head_candidates(self, panorama):
        """🆕 КРОК 5.2: Зона пошуку залежить від напрямку."""
        if panorama is None:
            return 0.0, 200.0
        w = int(panorama.shape[1])
        ov = float((self._expected_prev_overlap_px() or {}).get("overlap_px", 0.0))
        
        if self.is_rtl:
            # RTL: dup zone на ПРАВОМУ краю, живий вхід ЗЛІВА
            dup_start = max(0.0, w - ov) if ov > 0.0 else 0.0
            zone_w = min(w, dup_start + 150.0)
        else:
            # LTR: dup zone на ЛІВОМУ краю, живий вхід СПРАВА
            dup_start = min(float(w), ov) if ov > 0.0 else float(w)
            zone_w = min(float(w), dup_start + 150.0)
        
        return dup_start, zone_w

    def _norm_word_e2(self, w: str) -> str:
        """E2: нормалізація слова (універсальна для обох мов)."""
        try:
            return self._normalize_text(w)  # 🆕 Заміна
        except Exception:
            return str(w or "").strip()


    def _fragment_tail_word(self, pf) -> str:
        """
        Найновіше слово фрагмента (обрізане).
        ✅ FIX: той самий фільтр літер, що і в strip / s_words — інакше
        число в кінці фрагмента зсувало верифікацію на одне слово.
        """
        norm = self._norm_word_e2(str(pf.get("text", "") or ""))
        words = [w for w in norm.split()
                if any(self._is_target_lang_char(ch) for ch in w)]
        return words[-1] if words else ""

    # def _candidate_head_word(self, b) -> str:
    #     """
    #     Найстаріше слово блоку-кандидата.
    #     Універсально для RTL та LTR: це завжди перше слово в розбитому списку.
    #     """
    #     norm = self._norm_word_e2(str(b.get("text", "") or ""))
    #     words = norm.split()
    #     return words[0] if words else ""   # ✅ ЗАВЖДИ перше слово (голова)

    def _candidate_head_word(self, b) -> str:
        norm = self._norm_word_e2(str(b.get("text", "") or ""))
        words = [w for w in norm.split()
                if any(self._is_target_lang_char(ch) for ch in w)]
        return words[0] if words else ""

    @staticmethod
    def _better_hit(cur, new):
        """E2: пріоритет патернів prefix > overlap > continue; при рівних
        залишаємо поточний (перший у читанні = старіший = ближчий до стику)."""
        order = {"prefix": 0, "overlap": 1, "continue": 2}
        if cur is None:
            return new
        if order[new["kind"]] < order[cur["kind"]]:
            return new
        return cur

    def _match_pending_fragment(self, tail_w, candidates, dup_blocks):
        """E2: L2 (текст) + L3 (надійність) для одного фрагмента; L1 (геометрія)
        — у виклику (кандидати вже відфільтровані смугою стику). Три патерни
        (який з них реальний — і має показати діагностичний запуск на логах):
          prefix   — OCR нової панорами прочитав ПОВНЕ слово, і воно
                     починається з tail: фрагмент вміщений у блок;
          overlap  — блок починається з КІНЦЯ tail (k>=2 спільних літер):
                     дубльована частина розділена по межі панорам;
          continue — блок НЕ містить жодної літери tail: чисте продовження.
                     Найризикованіший (саме він у прикладі 'المفاو'+'ضات'):
                     приймається ЛИШЕ з перехресною перевіркою — у dup zone
                     має бути ДУБЛІКАТ фрагмента (останнє слово ~ tail,
                     fuzzy <=1), і кандидат має прилягати до дубліката
                     (розріз слова по межі зон), а не бути окремим словом
                     після пробілу. Без цих двох умов — жодних «правдоподібних
                     слів» (заборона з плану етапу 2).
        Повертає hit-dict або None (фрагмент залишається окремим)."""
        if len(tail_w) < 3:          # L3a: ≤2 літери — прийменник/артикль
            return None
        best = None
        for b in candidates:
            w_c = self._candidate_head_word(b)
            if not w_c:
                continue
            # (a) prefix: повне слово починається з tail і ДОВШЕ за нього
            if w_c.startswith(tail_w) and len(w_c) > len(tail_w):
                added = w_c[len(tail_w):]
                if 1 <= len(added) <= 8:      # L3b: «додавання» не ціле слово
                    best = self._better_hit(best, {
                        "kind": "prefix", "ratio": 1.0, "k": 0,
                        "tail": tail_w, "head": w_c, "added": added,
                        "merged": w_c, "block": b,
                    })
                    continue
            # (b) overlap: кандидат починається з кінця tail (k>=2)
            k = 0
            for kk in range(min(len(tail_w) - 1, len(w_c) - 1), 1, -1):
                if w_c[:kk] == tail_w[-kk:]:
                    k = kk
                    break
            if k >= 2:
                added = w_c[k:]
                if 2 <= len(added) <= 8:
                    best = self._better_hit(best, {
                        "kind": "overlap", "ratio": round(k / max(1, len(tail_w)), 2),
                        "k": k, "tail": tail_w, "head": w_c, "added": added,
                        "merged": tail_w + added, "block": b,
                    })
                    continue
        # (c) continue: чисте продовження — тільки через дублікат у dup zone
        if best is None and getattr(self, "seam_allow_continue", False):
            for db in dup_blocks:
                dup_last = self._fragment_tail_word({"text": str(db.get("text", "") or "")})
                if not dup_last or not self._lev1(dup_last, tail_w):
                    continue      # у dup zone нема дубліката ЦЬОГО фрагмента
                
                # 🆕 ПАТЧ 13: інверсія gap для LTR
                if self.is_rtl:
                    d_edge = float(db["x_min"])
                else:
                    d_edge = float(db["x_max"])
                
                for b in candidates:
                    if self.is_rtl:
                        gap = d_edge - float(b["x_max"])
                    else:
                        gap = float(b["x_min"]) - d_edge
                    
                    if not (-12.0 <= gap <= 10.0):
                        continue   # не прилягає → це окреме слово після пробілу
                    w_c = self._candidate_head_word(b)
                    if len(w_c) < 2 or len(tail_w) + len(w_c) > 12:
                        continue   # L3c: надто коротке/довге слово
                    hit = {
                        "kind": "continue", "ratio": 0.0, "k": 0,
                        "tail": tail_w, "head": w_c, "added": w_c,
                        "merged": tail_w + w_c, "block": b,
                        "dup_text": str(db.get("text", "") or ""),
                        "dup_last": dup_last, "dup_gap": round(gap, 1),
                    }
                    return hit    # перша (старіша) прилягаюча пара — у читанні
        return best

    # ══════════════════════════════════════════════════════════════════
    # 🆕 E2v2 · FULL-SENTENCE MATCHING: продовження шукається ЦІЛИМ РЕЧЕННЯМ
    # до його кінця, а не тільки головним словом блоку. Причина (реальний
    # відміт 'بجديه' на панорамі 58): завершене слово обрізаного фрагмента
    # живе на новій панорамі в КІНЦІ свого блока (останнє слово елемента =
    # найновіше = лівіше від стику) і НІКОЛИ не стає головним словом — v1
    # (head/tail) його систематично не бачить. Крім того, атрибуції
    # повторюються ('أمين مجلس الأمن القومي' у двох сусідніх елементів) і
    # букви на стику можуть не співпасти через зсув — розрізняє все це лише
    # перевірка ЦІЛОГО речення + геометричний якір, а не перше слово.
    # Все ще ДІАГНОСТИКА: лог/свертка без мутацій (контракт E2 незмінний).
    # ══════════════════════════════════════════════════════════════════

    def _seam_strip_words(self, raw_blocks, x_lo, x_hi):
        """E2v2: УСІ слова смуги стику на поточній панорамі в порядку
        читання (від найстарішого/правого до найновішого/лівого) з
        приблизним x кожного слова. RTL: слова розкладаються по блоках
        пропорційно до довжини (похибка ±десятки px — тому якір має допуск
        ±200px, а не точку). Слова без арабських літер пропускаються."""
        words: List[Dict] = []
        # blocks = sorted(
        #     (b for b in (raw_blocks or [])
        #      if not (float(b.get("x_max", 0.0)) < x_lo
        #              or float(b.get("x_min", 0.0)) > x_hi)),
        #     key=lambda b: -float(b.get("x_min", 0.0)),   # правіше = старіше

        # )

        # 🆕 КРОК 5.1: Універсальне сортування для E2v2
        sorted_blocks = sorted(
            (b for b in (raw_blocks or [])
            if not (float(b.get("x_max", 0.0)) < x_lo
                    or float(b.get("x_min", 0.0)) > x_hi)),
            key=lambda b: float(b.get("x_min", 0.0)),
            reverse=self.is_rtl  # True для RTL (правіше=старіше), False для LTR
        )
        blocks = sorted_blocks

        for bi, b in enumerate(blocks):
            norm = self._norm_word_e2(str(b.get("text", "") or ""))
            # ws = [w for w in norm.split()
            #       if any("\u0600" <= ch <= "\u06FF" for ch in w)]

            # 🆕 КРОК 6.4: Універсальний фільтр літер
            ws = [w for w in norm.split()
                if any(self._is_target_lang_char(ch) for ch in w)]
            if not ws:
                continue
            bx0 = float(b.get("x_min", 0.0))
            bx1 = float(b.get("x_max", 0.0))
            width = max(1.0, bx1 - bx0)
            total = sum(len(w) + 1 for w in ws)
            cum = 0.0
            for w in ws:
                f = (len(w) + 1) / total
                # центр слова: правий край блоку мінус частка (RTL)
                #cx = bx1 - (cum + f / 2.0) * width
                if self.is_rtl:
                    # RTL: слова розкладаються справа наліво
                    cx = bx1 - (cum + f / 2.0) * width
                else:
                    # LTR: слова розкладаються зліва направо
                    cx = bx0 + (cum + f / 2.0) * width
                cum += f
                words.append({
                    "word": w, "x": cx, "x_min": bx0, "x_max": bx1,
                    "block_i": bi, "block": b,
                })
        return words

    def _match_pending_fragment_full(self, pf, raw_blocks, dup_start, w_ocr):
        """E2v2: зіставлення ЦІЛИМ реченням. Три рівні, як у v1:
        L1' геометричний якір — tail-слово очікується біля bbox.x_min +
            dup_start (рівно стільки нового контенту додалось лівіше від
            попереднього краю — це і є зсув між панорамами); допуск
            ±200px, бо dup_start — оцінка (span − v·dt), не вимір.
        L2' текст — tail шукається серед УСІХ слів смуги (не тільки
            головних!); попередні слова речення (ctx_prev + власні слова
            фрагмента) звіряються від якоря в бік старіших, по lev1 кожне.
        L3' надійність — повтори tail у смузі фіксуються (repeats), але
            розрізняє їх геометрія; «правдоподібних слів» не вигадуємо;
            вперед («до кінця речення») — лише ПОПЕРЕДЖЕННЯ: end_state,
            continuation і split показують погане розпізнавання на стику
            та розділеність, але не голосують за збіг.
        Повертає (hit, None) або (None, причина-рядок) — причина йде в лог."""
        tail_w = self._fragment_tail_word(pf)
        if len(tail_w) < 3:          # L3a: ≤2 літери — прийменник/артикль
            return None, "tail<3"
        # слова ЦІЛОГО речення: ctx_prev (старіші) + власний текст фрагмента
        s_words: List[str] = []

        # 🆕 КРОК 6.4: Універсальний фільтр літер
        for c in (pf.get("ctx_prev") or []):
            s_words.extend(
                w for w in self._norm_word_e2(str(c.get("text", "") or "")).split()
                if any(self._is_target_lang_char(ch) for ch in w))
        s_words.extend(
            w for w in self._norm_word_e2(str(pf.get("text", "") or "")).split()
            if any(self._is_target_lang_char(ch) for ch in w))
        if not s_words:
            return None, "порожній фрагмент"
        # strip = self._seam_strip_words(
        #     raw_blocks,
        #     max(0.0, dup_start - getattr(self, "_seam_full_left_px", 500.0)),
        #     min(float(w_ocr), dup_start + getattr(self, "_seam_full_right_px", 700.0)),
        # )
        # ✅ FIX: вікно смуги дзеркалиться. Старі слова (які верифікуємо)
#         в RTL праворуч від стику, в LTR — ліворуч.
        new_px = getattr(self, "_seam_full_left_px", 500.0)    # бік нового контенту
        old_px = getattr(self, "_seam_full_right_px", 700.0)   # бік старого контенту
        if self.is_rtl:
            lo, hi = dup_start - new_px, dup_start + old_px
        else:
            lo, hi = dup_start - old_px, dup_start + new_px
        strip = self._seam_strip_words(raw_blocks, max(0.0, lo), min(float(w_ocr), hi))

        if not strip:
            return None, "смуга стику порожня"
        #x_exp = float(pf["bbox"][0]) + dup_start
        # if self.is_rtl:
        #     x_exp = float(pf["bbox"][0]) + dup_start
        # else:
        #     # Для LTR хвіст = правий край bbox (bbox[1])
        #     x_exp = float(pf["bbox"][1]) - pf.get("pan_w", w_ocr) + dup_start

        bbox = pf["bbox"]
        if self.is_rtl:
            x_exp = float(bbox[0]) + dup_start
        else:
            x_exp = float(bbox[1]) - float(pf.get("pan_w", w_ocr)) + dup_start
          # 🆕 СТРАХОВКА: якір фізично поза панорамою = зламана геометрія/телеметрія
        if not (-300.0 <= x_exp <= w_ocr + 300.0):
            logger.warning("🟥 E2 ANCHOR OUT OF RANGE: x_exp=%.0f (w=%d, pan_w=%s, dup=%.0f)",
                           x_exp, w_ocr, pf.get("pan_w"), dup_start)

        # L1'+L2': якорі — tail-подібні слова (lev1 або завершення +1..3
        # літер, тобто слово доїхало) у межах ±200px від очікуваного x
        # ✅ FIX: турецька аглютинація — хвіст доростає більше ніж на 3 літери
#         (İstanbul → İstanbul'dakilerden)
        max_grow = 3 if (self.is_rtl or len(tail_w) < 4) else getattr(self, "_seam_tail_grow_ltr", 8)
        anchors = []
        tail_like = 0
        nearest_dx = None
        for i, sw in enumerate(strip):
            w = sw["word"]
            # like = (self._lev1(w, tail_w)
            #         or (w.startswith(tail_w) and 1 <= len(w) - len(tail_w) <= 3))
            like = (self._lev1(w, tail_w)
                    or (w.startswith(tail_w) and 1 <= len(w) - len(tail_w) <= max_grow))
            if not like:
                continue
            tail_like += 1
            dx = sw["x"] - x_exp
            if nearest_dx is None or abs(dx) < abs(nearest_dx):
                nearest_dx = dx
            if abs(dx) <= getattr(self, "_seam_full_anchor_px", 200.0):
                anchors.append(i)
        if not anchors:
            return None, (
                f"якорів немає (tail-подібних={tail_like}, "
                f"найближчий {nearest_dx:+.0f}px від exp {x_exp:.0f})"
                if nearest_dx is not None else "tail-подібних слів немає"
            )
        # верифікація ЦІЛОГО речення від кожного якоря в бік старіших
        need = len(s_words) - 1
        best = None



        # for i in anchors:
        #     verified = 0
        #     diffs = []
        #     si = need - 1          # слово перед tail (старіше)
        #     j = i - 1
        #     while si >= 0 and j >= 0:
        #         if self._lev1(strip[j]["word"], s_words[si]):
        #             ok1 = strip[j]["word"] == s_words[si]
        #             verified += 1
        #             diffs.append((s_words[si], "exact" if ok1 else "lev1"))
        #             si -= 1
        #             j -= 1
        #         else:
        #             diffs.append((s_words[si], f"miss:{strip[j]['word']}"))
        #             break
        #     ok = (need == 0) or (verified >= max(1, need - 1))



        for i in anchors:
            verified = 0
            diffs = []
            status_by_j = {}   # ✅ FIX: статус слова за індексом у strip
            j_lo = i           # найстаріший підтверджений індекс у strip
            tol = 1            # ✅ FIX: 1 допуск на OCR-склейку / розрив / зайве слово
            si = need - 1
            j = i - 1
            while si >= 0 and j >= 0:
                w_strip, w_sent = strip[j]["word"], s_words[si]
                if self._fuzzy_eq(w_strip, w_sent):
                    st = "exact" if w_strip == w_sent else "lev1"
                    verified += 1
                    diffs.append((w_sent, st))
                    status_by_j[j] = st
                    j_lo = j
                    si -= 1
                    j -= 1
                    continue
                if tol:
                    # OCR склеїв два слова речення в одне
                    if si >= 1 and self._fuzzy_eq(w_strip, s_words[si - 1] + w_sent):
                        verified += 2
                        diffs.append((s_words[si - 1] + "+" + w_sent, "merged"))
                        status_by_j[j] = "merged"
                        j_lo = j
                        si -= 2
                        j -= 1
                        tol = 0
                        continue
                    # OCR розбив слово на два
                    if j >= 1 and self._fuzzy_eq(strip[j - 1]["word"] + w_strip, w_sent):
                        verified += 1
                        diffs.append((w_sent, "split"))
                        status_by_j[j] = status_by_j[j - 1] = "split"
                        j_lo = j - 1
                        si -= 1
                        j -= 2
                        tol = 0
                        continue
                    # зайве сміттєве слово в смузі
                    if j >= 1 and self._fuzzy_eq(strip[j - 1]["word"], w_sent):
                        diffs.append((w_strip, "skip"))
                        status_by_j[j] = "skip"
                        j -= 1
                        tol = 0
                        continue
                diffs.append((w_sent, f"miss:{w_strip}"))
                break
            # ✅ FIX: не можна вимагати більше слів, ніж є в смузі до якоря
            need_eff = min(need, i)
            ok = (need == 0) or (verified >= max(1, need_eff - 1))
            if not ok:
                continue
            cand = {"anchor_i": i, "verified": verified, "diffs": diffs,
                   "status_by_j": status_by_j, "j_lo": j_lo,
                   "dx": strip[i]["x"] - x_exp}
            if best is None or (cand["verified"], -abs(cand["dx"])) > \
                    (best["verified"], -abs(best["dx"])):
                best = cand
        if best is None:
            return None, f"речення не підтверджене (verified<={max(0, need - 1)}/{need})"

        i = best["anchor_i"]
        j_lo = best["j_lo"]
        anchor_sw = strip[i]
        tail_status = ("exact" if anchor_sw["word"] == tail_w
                    else "completed" if anchor_sw["word"].startswith(tail_w)
                    else "lev1")
        # розділеність: скількома блоками лежить підтверджене речення
        span_blocks = {strip[k]["block_i"] for k in range(j_lo, i + 1)}
        split_gap = None
        if len(span_blocks) > 1:
            if self.is_rtl:
                gaps = [round(strip[k - 1]["x_min"] - strip[k]["x_max"], 1)
                        for k in range(j_lo + 1, i + 1)
                        if strip[k]["block_i"] != strip[k - 1]["block_i"]]
            else:
                gaps = [round(strip[k]["x_min"] - strip[k - 1]["x_max"], 1)
                        for k in range(j_lo + 1, i + 1)
                        if strip[k]["block_i"] != strip[k - 1]["block_i"]]
            split_gap = gaps[-1] if gaps else None
        # слово на стику: найближче до dup_start серед підтверджених
        seam_i = min(range(j_lo, i + 1),
                    key=lambda k: abs(strip[k]["x"] - dup_start))
        # ✅ FIX: статус за індексом у strip (раніше ключ — слово речення)
        seam_status = (tail_status if seam_i == i
                    else best["status_by_j"].get(seam_i, "exact"))
        # «до кінця речення»: вперед від tail до сепаратора чи краю смуги
        fwd = []
        end_state = "strip_end"
        sep_gap = None
        j = i + 1
        while j < len(strip) and len(fwd) < getattr(self, "_seam_preview_words", 6):
            sw = strip[j]
            if sw["block_i"] != strip[j - 1]["block_i"]:
                # 🆕 ПАТЧ 12d: інверсія gap для LTR
                if self.is_rtl:
                    gap = round(strip[j - 1]["x_min"] - sw["x_max"], 1)
                else:
                    gap = round(sw["x_min"] - strip[j - 1]["x_max"], 1)
                if gap >= getattr(self, "_seam_sep_gap_px", 30.0):
                    end_state = "separator"
                    sep_gap = gap
                    break        # наступний ЕЛЕМЕНТ — тільки прев'ю, без склеювання
            fwd.append(sw["word"])
            j += 1
        else:
            if j >= len(strip):
                end_state = "strip_end"
            elif len(fwd) >= getattr(self, "_seam_preview_words", 6):
                end_state = "budget"
        # scope = "word" if need == 0 else "sentence"
        # merged = (anchor_sw["word"] if need == 0 else " ".join(s_words))

        scope = "word" if need == 0 else "sentence"
        # ✅ FIX: у речення йде ДОРОЩЕНИЙ tail, а не обрізаний
        merged = (anchor_sw["word"] if need == 0
                else " ".join(s_words[:-1] + [anchor_sw["word"]]))
        return {
            "kind": "full", "scope": scope,
            "tail": tail_w, "tail_completed": anchor_sw["word"],
            "tail_status": tail_status,
            "head_block": anchor_sw["block"],
            "merged": merged,
            "verified": best["verified"], "need": need,
            "word_diffs": best["diffs"],
            "anchor_x": round(anchor_sw["x"], 1), "x_exp": round(x_exp, 1),
            "shift_px": round(best["dx"], 1),
            "repeats": tail_like,
            "split_blocks": len(span_blocks), "split_gap": split_gap,
            "seam_word": strip[seam_i]["word"],
            "seam_status": seam_status,
            "end_state": end_state, "sep_gap": sep_gap,
            "continuation": " ".join(fwd),
            "conf_new": float(anchor_sw["block"].get("score", 0.0) or 0.0),
        }, None

    def _cross_panorama_match(self, raw_blocks, panorama, timestamp):
        """E2 · ЕТАП 2: зіставлення буфера _pending_seam_fragments (клас B
        ПОПЕРЕДНЬОЇ панорами) з блоками ПОТОЧНОЇ. ДІАГНОСТИЧНИЙ РЕЖИМ:
        лише лог + _seam_match_log; текст не відновлюється, асамблер не
        чіпається (перевірка якості збігу без ризику хибних слів).
        Двоступенево: v1 (head/tail, патерни prefix/overlap/continue),
        за промахом — v2 ЦІЛИМ реченням до кінця (геометричний якір
        tail-слова + перевірка попередніх слів + прев'ю до сепаратора)."""
        if not getattr(self, "cross_pano_matching_enabled", True):
            return
        pending = list(getattr(self, "_pending_seam_fragments", None) or [])
        if not pending:
            return
        cur_pan = (self._last_panorama_meta or {}).get("panorama_id", "—")
        if not raw_blocks or panorama is None:
            # контент не прийшов зовсім (пауза/графіка): фрагмент залишається
            # в буфері до наступної панорами з контентом
            logger.info(
                "⬜ CROSS-PANO: pan=%s блоків немає — %d фрагмент(ів) залишаються в буфері",
                cur_pan, len(pending),
            )
            return

        w_ocr = int(panorama.shape[1])
        dup_start, zone_w = self._seam_head_candidates(panorama)
        # L1 геометрія: смуга стику — по слову (±150px) навколо межі dup zone
        lo = max(0.0, dup_start - 150.0)
        candidates = [
            b for b in raw_blocks
            if not (float(b["x_max"]) < lo or float(b["x_min"]) > zone_w)
        ]
        # дублікати хвоста попередньої панорами — блоки правіше межі dup zone
        #dup_blocks = [b for b in raw_blocks if float(b["x_min"]) >= dup_start - 25.0]
                # 🆕 КРОК 5.2: Дублікати хвоста попередньої панорами залежать від напрямку
        if self.is_rtl:
            # RTL: дублікати знаходяться ПРАВІШЕ (більші X) від межі dup_start
            dup_blocks = [b for b in raw_blocks if float(b["x_min"]) >= dup_start - 25.0]
        else:
            # LTR: дублікати знаходяться ЛІВІШЕ (менші X) від межі dup_start
            dup_blocks = [b for b in raw_blocks if float(b["x_max"]) <= dup_start + 25.0]
        if not candidates:
            logger.info(
                "⬜ CROSS-PANO: pan=%s у смузі пошуку [%.0f..%.0f] блоків немає "
                "(dup-блоків: %d)",
                cur_pan, lo, zone_w, len(dup_blocks),
            )
            return

        n_hit = 0
        remaining = []
        for pf in pending:
            tail_w = self._fragment_tail_word(pf)
            best = self._match_pending_fragment(tail_w, candidates, dup_blocks)
            if best is not None:
                b = best["block"]
                logger.info(
                    "🔗 CROSS-PANO MATCH: '%s' + '%s' → '%s' | kind=%s k=%s "
                    "added=%r | смуга=[%.0f..%.0f] | head_bbox=[%.0f..%.0f] "
                    "conf=%.2f | %s | MODE=DIAGNOSTIC (текст не змінено)",
                    pf["text"], str(b.get("text", "")), best["merged"],
                    best["kind"], best.get("k"), best["added"],
                    lo, zone_w, float(b["x_min"]), float(b["x_max"]),
                    float(b.get("score", 0.0)),
                    (f"dup≈{best['dup_last']!r} gap={best.get('dup_gap')}px"
                     if best["kind"] == "continue" else "—"),
                )
                self._seam_match_log.append({
                    "ts": float(timestamp),
                    "prev_pan": pf.get("panorama_id"),
                    "cur_pan": (self._last_panorama_meta or {}).get("panorama_id"),
                    "fragment": pf["text"],
                    "tail": best["tail"],
                    "head_block": str(b.get("text", "")),
                    "head": best["head"],
                    "merged": best["merged"],
                    "kind": best["kind"],
                    "k": best.get("k"),
                    "added": best["added"],
                    "dup_text": best.get("dup_text"),
                    "dup_gap": best.get("dup_gap"),
                    "edge": pf.get("edge"),
                    "b_reason": pf.get("b_reason"),
                    "conf_new": float(b.get("score", 0.0)),
                    "mode": "diagnostic",
                })
                n_hit += 1
                continue
            # 🆕 E2v2: v1 (head/tail) промахнувся — пробуємо ЦІЛИМ реченням
            full, why = self._match_pending_fragment_full(
                pf, raw_blocks, dup_start, w_ocr)
            if full is None:
                # немає збігу → фрагмент залишається окремим (план етапу 2);
                # лишаємо в буфері ДО спроби наступної панорами з контентом
                remaining.append(pf)
                logger.info(
                    "⬜ CROSS-PANO: панорама %s: фрагмент %r (tail=%r) без "
                    "продовження (кандидатів: %d, dup: %d; full: %s)",
                    cur_pan, pf["text"], tail_w, len(candidates), len(dup_blocks),
                    why or "—",
                )
                continue
            fb = full["head_block"]
            n_exact = sum(1 for _, st in full["word_diffs"] if st == "exact")
            n_lev1 = sum(1 for _, st in full["word_diffs"] if st == "lev1")
            logger.info(
                "🔗 CROSS-PANO FULL: '%s' → речення підтверджене ЦІЛИКОМ: "
                "'%s' | verified=%d/%d (exact=%d, lev1=%d) | tail@%.0f "
                "(exp %.0f, зсув %+.0f) | repeats=%d | split=%d блок(и) "
                "gap=%s | стик=%r(%s) | до кінця речення: %s·%r | conf=%.2f "
                "| MODE=DIAGNOSTIC (текст не змінено)",
                pf["text"], full["merged"],
                full["verified"], full["need"], n_exact, n_lev1,
                full["anchor_x"], full["x_exp"], full["shift_px"],
                full["repeats"], full["split_blocks"],
                (f"{full['split_gap']}px" if full.get("split_gap") is not None else "—"),
                full["seam_word"], full["seam_status"],
                full["end_state"], full["continuation"],
                full["conf_new"],
            )
            self._seam_match_log.append({
                "ts": float(timestamp),
                "prev_pan": pf.get("panorama_id"),
                "cur_pan": (self._last_panorama_meta or {}).get("panorama_id"),
                "fragment": pf["text"],
                "tail": full["tail"],
                "head_block": str(fb.get("text", "")),
                "merged": full["merged"],
                "kind": full["kind"],
                "scope": full["scope"],
                "tail_completed": full["tail_completed"],
                "tail_status": full["tail_status"],
                "verified": full["verified"],
                "need": full["need"],
                "word_diffs": full["word_diffs"],
                "anchor_x": full["anchor_x"],
                "x_exp": full["x_exp"],
                "shift_px": full["shift_px"],
                "repeats": full["repeats"],
                "split_blocks": full["split_blocks"],
                "split_gap": full.get("split_gap"),
                "seam_word": full["seam_word"],
                "seam_status": full["seam_status"],
                "end_state": full["end_state"],
                "sep_gap": full.get("sep_gap"),
                "continuation": full["continuation"],
                "edge": pf.get("edge"),
                "b_reason": pf.get("b_reason"),
                "conf_new": full["conf_new"],
                "mode": "diagnostic",
            })
            n_hit += 1

        # E2: знайдене продовження «використано» — matched фрагменти покидають
        # буфер одразу (повторний матч на 3-й панорамі — хибний); unmatched
        # чекають наступної панорами з контентом. При вдалому detect нижче
        # буфер усе одно буде замінено класами B цієї панорами.
        if n_hit:
            self._pending_seam_fragments = remaining
            logger.info(
                "⬜ CROSS-PANO SUMMARY: панорама %s: збігів=%d з %d фрагментів",
                cur_pan, n_hit, len(pending),
            )

    def _has_dot_in_gap(self, panorama: np.ndarray, x_start: int, x_end: int, y_start: int, y_end: int) -> bool:
        """
        Перевіряє наявність крапки/маркера в зоні пробілу.
        Шукає компактну темну пляму, яка займає 5-35% площі вирізу.
        """
        # 🔑 КОНВЕРТУЄМО ВСІ КООРДИНАТИ В INT
        x_start = int(round(x_start))
        x_end = int(round(x_end))
        y_start = int(round(y_start))
        y_end = int(round(y_end))
        
        # Перевірка на вихід за межі
        if x_start < 0:
            x_start = 0
        if x_end > panorama.shape[1]:
            x_end = panorama.shape[1]
        if y_start < 0:
            y_start = 0
        if y_end > panorama.shape[0]:
            y_end = panorama.shape[0]
        
        # Перевірка на вироджений випадок
        if x_end <= x_start or y_end <= y_start:
            return False
        
        # Перетворюємо на відтінки сірого, якщо панорама кольорова
        if len(panorama.shape) == 3:
            gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        else:
            gray = panorama
        
        # Обрізаємо тільки зону поточного пробілу
        gap_region = gray[y_start:y_end, x_start:x_end]
        if gap_region.size == 0:
            return False
        
        #  FIX P1-6: полярність-агностик. Білий маркер на темному
        # (Al Jazeera / Al Ghad) дає |dev| так само, як темний на світлому.
        g = gap_region.astype(np.float32)
        row_med = np.median(g, axis=1, keepdims=True)
        ink_mask = np.abs(g - row_med) > 25.0
        ink_ratio = float(ink_mask.mean())
        if not (0.05 < ink_ratio < 0.35):
            return False
        # компактна пляма, а не хвости літер по краях проміжку
        cols = np.where(ink_mask.any(axis=0))[0]
        return bool(cols.size and (cols[-1] - cols[0] + 1) <= 0.6 * g.shape[1])


    def _run_tsm_analysis(self, panorama: np.ndarray) -> tuple[list, list]:
        """
        🆕 Інтеграція ticker_symbol_map. 
        Примітка: полярність (темний/світлий фон) тепер автоматично обробляється 
        всередині методу extract_ink модуля TSM.
        """
        W = panorama.shape[1]
        img_for_tsm = panorama.copy()

        # 🆕 ДІАГНОСТИКА: вимірюємо вхідні параметри
        gray = cv2.cvtColor(img_for_tsm, cv2.COLOR_BGR2GRAY)
        median_brightness = float(np.median(gray))
        logger.info(f"🔍 TSM ДІАГНОСТИКА: медіана яскравості={median_brightness:.1f}, висота={panorama.shape[0]}px")

        #ПОЛЯРНІСТЬ: extract_ink очікує ТЕМНЕ чорнило на СВІТЛОМУ фоні.
        # Al Araby = білий текст на темно-червоному → інвертуємо.
        if median_brightness < 110.0:
            img_for_tsm = cv2.bitwise_not(img_for_tsm)
            logger.info(f"🔄 TSM: Застосовано інверсію (темний фон)")
        
        # 🔄 Адаптація під LTR (Турецькі канали) залишається, бо TSM заточений під RTL
        ltr = not self.is_rtl
        if ltr:
            img_for_tsm = cv2.flip(img_for_tsm, 1)  # 1 = horizontal flip
            
        #Запуск TSM
        try:
            tsm_result, new_profile = self.tsm_mapper.analyze_image(img_for_tsm)
            
            # 🆕 ДІАГНОСТИКА: що повернув TSM
            boundaries = tsm_result.get('boundaries', [])
            wide_gaps = tsm_result.get('wide_gaps', [])
            logger.info(f"🔍 TSM РЕЗУЛЬТАТ: boundaries={len(boundaries)}, wide_gaps={len(wide_gaps)}")

            if boundaries:
                for i, b in enumerate(boundaries[:5]):  # Збільшено до 5
                    score = b.get('confidence', 0.0)
                    logger.info(f"   boundary #{i}: x0={b['x0']}, x1={b['x1']}, width={b['x1']-b['x0']}px, score={score:.2f}")
                        
            if wide_gaps:
                for i, wg in enumerate(wide_gaps[:3]):
                    logger.info(f"   wide_gap #{i}: x0={wg['x0']}, x1={wg['x1']}, width={wg['x1']-wg['x0']}px")

            if new_profile:
                logger.info(f"💡 TSM: Вивчено новий шаблон роздільника: {new_profile.get('name')}")
                self._save_separator_template()

        except Exception as e:
            logger.warning(f"⚠️ TSM аналіз не вдався: {e}. Фолбек на старий алгоритм.")
            g, _, k = self._detect_sentence_gaps(panorama)
            return g, k

        # 4. 🟩 Конвертація координат назад у систему панорами (без змін)
        gaps = []
        gap_kinds = []
        
        for b in tsm_result.get('boundaries', []):
            fx0, fx1 = b['x0'], b['x1']
            if ltr:
                rx0 = W - fx1
                rx1 = W - fx0
            else:
                rx0, rx1 = fx0, fx1
                
            rx0 = max(0, int(rx0))
            rx1 = min(W, int(rx1))
            if rx1 > rx0:
                gaps.append((rx0, rx1))
                gap_kinds.append('dot')
                
        for wg in tsm_result.get('wide_gaps', []):
            fx0, fx1 = wg['x0'], wg['x1']
            if ltr:
                rx0 = W - fx1
                rx1 = W - fx0
            else:
                rx0, rx1 = fx0, fx1
                
            rx0 = max(0, int(rx0))
            rx1 = min(W, int(rx1))
            if rx1 > rx0 and (rx1 - rx0) >= self.min_gap_width:
                is_dup = any(rx0 >= g[0] - 10 and rx1 <= g[1] + 10 for g in gaps)
                if not is_dup:
                    gaps.append((rx0, rx1))
                    gap_kinds.append('vacuum')
                    
        combined = sorted(zip(gaps, gap_kinds), key=lambda x: x[0][0])
        gaps = [c[0] for c in combined]
        gap_kinds = [c[1] for c in combined]
        
        logger.info(f"✅ TSM знайшов {len(gaps)} меж (dot={gap_kinds.count('dot')}, vacuum={gap_kinds.count('vacuum')})")
        return gaps, gap_kinds

    
    
    def _detect_sentence_gaps(self, panorama: np.ndarray) -> tuple[List[tuple], Optional[np.ndarray], List[str]]:

        if panorama is None or panorama.shape[1] < 100:
            return [], None, []
        
        gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        roi_y_start = int(h * 0.20)
        roi = gray[roi_y_start:, :]
        h_roi = roi.shape[0]
        
        if h_roi < 10:
             return [], None, []
        
        # ==========================================
        # 🆕 КРОК 0: Одразу рахуємо ink_map, щоб він був доступний для Секції 1
        # ==========================================
        median_brightness = np.median(roi)
        row_bg = np.percentile(roi, 20, axis=1).astype(np.int16)
        flat = roi.astype(np.int16) - row_bg[:, None]
        if median_brightness > 128:
            ink_map = np.clip(-flat, 0, 255).astype(np.uint8)
        else:
            ink_map = np.clip(flat, 0, 255).astype(np.uint8)
        
        # 🔑 УНІФІКОВАНИЙ std-метод
        col_std = np.std(roi.astype(np.float32), axis=0)
        gaps = []
        gap_kinds = []  
                
        # ==========================================
        # 1. ПОШУК ПОРОЖНІХ ДІЛЯНОК (вакууми)
        # ==========================================
        runs = []
        x = 0
        while x < w:
            if col_std[x] < 20.0:
                x2 = x
                while x2 < w and col_std[x2] < 20.0:
                    x2 += 1
                runs.append((x, x2))
                x = x2
            else:
                x += 1
                
        # базові широкі вакууми з перевіркою чорнила
        for a, b in runs:
            width = b - a
            # ✅ ВИПРАВЛЕНО: 35 замість 60, щоб не відсікати робочі 39-40px межі
            if width >= max(self.min_gap_width, 35) and a > 30 and b < w - 30:
                # Додаткова перевірка: чи дійсно там немає чорнила?
                # Сума пікселів чорнила в зоні має бути мінімальною
                ink_in_gap = ink_map[:, a:b].sum()
                
                # Розрахуємо площу зони, щоб поріг був адаптивним до висоти рядка
                gap_area = (b - a) * h_roi
                # Якщо чорнило займає менше 5% площі зони - це справжній пробіл
                if ink_in_gap < (gap_area * 0.05):  
                    gaps.append((a, b))
                    gap_kinds.append("vacuum")

        # ==========================================
        # 1b. 🆕 V2 ВАКУУМИ «як у Hadath»: широкий злитий пробіл = межа
        # ==========================================
        row_med = np.median(roi.astype(np.float32), axis=1, keepdims=True)
        ink_mask = np.abs(roi.astype(np.float32) - row_med) > 40.0

        runs = []
        x = 0
        while x < w:
            if col_std[x] < 20.0:
                x2 = x
                while x2 < w and col_std[x2] < 20.0:
                    x2 += 1
                if (x2 - x) >= 6:
                    runs.append([x, x2])
                x = x2
            else:
                x += 1

        merged_runs = []
        for a, b in runs:
            # 🔑 ЗБІЛЬШЕНО до 45px, щоб охопити крапку (7-8px) + поля (16+16px ≈ 40px загального розриву)
            if merged_runs and (a - merged_runs[-1][1]) <= 30:
                pb, pa = merged_runs[-1][1], a
                h_frac = float(ink_mask[:, pb:pa].mean(axis=0).max()) if pa > pb else 1.0
                # 🔑 ПІДНЯТО до 0.65 для надійнішого розпізнавання крапки як "низького" об'єкта
                if h_frac <= 0.65:
                    merged_runs[-1][1] = b
                else:
                    merged_runs.append([a, b])
            else:
                merged_runs.append([a, b])

        for a, b in merged_runs:
            # 🔑 МІНІМАЛЬНА ШИРИНА 30px гарантує, що ми ловимо саме злиті 39-41px, а не випадкові пробіли
            if (b - a) >= max(self.min_gap_width, 30) and a > 30 and b < w - 30:
                if (a, b) not in gaps:
                    gaps.append((a, b))
                    gap_kinds.append("vacuum")
        
        # ==========================================
        # 2. ПОШУК ІЗОЛЬОВАНИХ РОЗДІЛЮВАЧІВ (•, ▪, |)
        # ==========================================
        symbol_cols = np.where(col_std >= 20.0)[0]

        if len(symbol_cols) > 0:
            symbol_groups = []
            current_group = [symbol_cols[0]]
            
            for col in symbol_cols[1:]:
                if col - current_group[-1] <= 2:
                    current_group.append(col)
                else:
                    symbol_groups.append(current_group)
                    current_group = [col]
            symbol_groups.append(current_group)
            
            # Шукаємо ІЗОЛЬОВАНІ вертикальні розділювачі
            for group in symbol_groups:
                width = len(group)
                start_x = group[0]
                end_x = group[-1] + 1
                
                if 1 <= width <= 30:
                    if start_x > 40 and end_x < w - 40:
                        
                        # ✅ ВИПРАВЛЕНО: Знаходимо НАЙБЛИЖЧУ групу зліва
                        left_candidates = [g[-1] for g in symbol_groups if g[-1] < start_x]
                        left_dist = start_x - max(left_candidates) if left_candidates else start_x
                        
                        # Для правої сторони break залишаємо, бо список відсортований і перший знайдений = найближчий
                        right_dist = w - end_x
                        for g in symbol_groups:
                            if g[0] > end_x:
                                right_dist = g[0] - end_x
                                break
                        
                        if left_dist > 25 and right_dist > 25:
                            gaps.append((start_x, end_x))
                            gap_kinds.append("dot")  # 🆕
                            shape_name = "лінія" if width <= 3 else ("квадрат" if width <= 15 else "крапка/блок")
                            logger.debug(f"ЗНАЙДЕНО РОЗДІЛЮВАЧ ({shape_name}): x={start_x}-{end_x} (width={width}px, L={left_dist}px, R={right_dist}px)")
            # ==========================================
            # 🆕 2.2 ВЕРТИКАЛЬНІ ЛІНІЇ-РОЗДІЛЮВАЧІ (Al Ghad)
            # ==========================================
            if getattr(self, "sep_vertical_lines", False):
                v_symbol_cols = np.where(col_std >= 15.0)[0]
                if len(v_symbol_cols) > 0:
                    v_symbol_groups = []
                    v_current_group = [v_symbol_cols[0]]
                    for col in v_symbol_cols[1:]:
                        if col - v_current_group[-1] <= 1:
                            v_current_group.append(col)
                        else:
                            v_symbol_groups.append(v_current_group)
                            v_current_group = [col]
                    v_symbol_groups.append(v_current_group)
                    
                    for group in v_symbol_groups:
                        width = len(group)
                        start_x = group[0]
                        end_x = group[-1] + 1
                        
                        if 1 <= width <= 5:
                            if start_x > 20 and end_x < w - 20:
                                # ✅ ВИПРАВЛЕНО
                                left_candidates = [g[-1] for g in v_symbol_groups if g[-1] < start_x]
                                left_dist = start_x - max(left_candidates) if left_candidates else start_x
                                
                                right_dist = w - end_x
                                for g in v_symbol_groups:
                                    if g[0] > end_x:
                                        right_dist = g[0] - end_x
                                        break
                                
                                if left_dist > 8 and right_dist > 8:
                                    gaps.append((start_x, end_x))
                                    gap_kinds.append("dot")  # 🆕
                                    logger.debug(f"🟡 ВЕРТИКАЛЬНА ЛІНІЯ (Al Ghad): x={start_x}-{end_x} (width={width}px, L={left_dist}px, R={right_dist}px)")
            # ==========================================
            # 🆕 2.1 КВАДРАТИ-РОЗДІЛЮВАЧІ (■) посередині тексту
            # ==========================================
            if getattr(self, "sep_squares", False):
                for group in symbol_groups:
                    width = len(group)
                    start_x = group[0]
                    end_x = group[-1] + 1
                    
                    if 3 <= width <= 15:
                        if start_x > 20 and end_x < w - 20:
                            # ✅ ВИПРАВЛЕНО
                            left_candidates = [g[-1] for g in symbol_groups if g[-1] < start_x]
                            left_dist = start_x - max(left_candidates) if left_candidates else start_x
                            
                            right_dist = w - end_x
                            for g in symbol_groups:
                                if g[0] > end_x:
                                    right_dist = g[0] - end_x
                                    break
                            
                            if left_dist > 8 and right_dist > 8:
                                gaps.append((start_x, end_x))
                                gap_kinds.append("dot")  # 🆕
                                logger.debug(f"🟦 КВАДРАТ-РОЗДІЛЮВАЧ: x={start_x}-{end_x} (width={width}px, L={left_dist}px, R={right_dist}px)")
                            else:
                                logger.debug(f"🟧 КВАДРАТ ВІДХИЛЕНО (ізоляція): x={start_x}-{end_x} (L={left_dist}px R={right_dist}px)")
            # ==========================================
            # 🆕 2.2 КРУГЛІ РОЗДІЛЮВАЧІ (• / ○) — Al Sharqiya, Al Qahera News
            # ==========================================
            if getattr(self, "sep_circles", False):
                isolation_px = int(getattr(self, "sep_isolation_px", 12))
                
                for group in symbol_groups:
                    width = len(group)
                    start_x = group[0]
                    end_x = group[-1] + 1
                    
                    if 3 <= width <= 25:
                        if start_x > 20 and end_x < w - 20:
                            # ✅ ВИПРАВЛЕНО (замінено на надійний варіант)
                            left_candidates = [g[-1] for g in symbol_groups if g[-1] < start_x]
                            left_dist = start_x - max(left_candidates) if left_candidates else start_x
                            
                            right_dist = w - end_x
                            for g in symbol_groups:
                                if g[0] > end_x:
                                    right_dist = g[0] - end_x
                                    break
                            
                            if left_dist > isolation_px and right_dist > isolation_px:
                                gaps.append((start_x, end_x))
                                gap_kinds.append("dot")  # 🆕
                                logger.debug(f"⭕ КРУГЛИЙ РОЗДІЛЮВАЧ: x={start_x}-{end_x} (width={width}px, L={left_dist}px, R={right_dist}px)")
                            else:
                                logger.debug(f"🟧 КРУГЛИЙ ВІДХИЛЕНО (мала ізоляція): x={start_x}-{end_x} (L={left_dist}px, R={right_dist}px, потрібно >{isolation_px}px)")
            # ==========================================
            # 3. 🆕 КОЛЬОРОВІ РОЗДІЛЮВАЧІ
            # ==========================================
            if getattr(self, "color_separator_detection", False):
                band = panorama[roi_y_start:, :, :].astype(np.float32)
                col_mean = band.mean(axis=0)
                col_uni = band.std(axis=0).mean(axis=1)
                uni_mask = col_uni < 20.0
                if uni_mask.sum() > 10:
                    bg = np.median(col_mean[uni_mask], axis=0)
                    dev = np.linalg.norm(col_mean - bg, axis=1)
                    dev_thr = self.color_sep_dev
                    x_lo = int(0.03 * w)
                    x_hi = int(0.85 * w)
                    solid = uni_mask & (dev > dev_thr)
                    x = x_lo
                    while x < x_hi:
                        if solid[x]:
                            x2 = x
                            while x2 < x_hi and solid[x2]:
                                x2 += 1
                            width = x2 - x
                            if 2 <= width <= 40:
                                gaps.append((x, x2))
                                gap_kinds.append("dot")
                                logger.debug(f"🟥 КОЛЬОРОВИЙ РОЗДІЛЮВАЧ: x={x}-{x2} (width={width}px, dev={dev[x:x2].max():.0f})")
                            x = x2
                        else:
                            x += 1
            
        # ==========================================
        # 4. ПОШУК КРАПОК-РОЗДІЛЮВАЧІВ (•)
        # ==========================================
        symbol_cols = np.where(col_std >= 20.0)[0]
        
        if len(symbol_cols) > 0:
            symbol_groups = []
            current_group = [symbol_cols[0]]
            
            for col in symbol_cols[1:]:
                if col - current_group[-1] <= 2:
                    current_group.append(col)
                else:
                    symbol_groups.append(current_group)
                    current_group = [col]
            symbol_groups.append(current_group)
            
            for group in symbol_groups:
                width = len(group)
                start_x = group[0]
                end_x = group[-1] + 1
                
                if 5 <= width <= 14:
                    if start_x > 50 and end_x < w - 50:
                        # ✅ ВИПРАВЛЕНО
                        left_candidates = [g[-1] for g in symbol_groups if g[-1] < start_x]
                        left_dist = start_x - max(left_candidates) if left_candidates else start_x
                        
                        right_dist = w - end_x
                        for g in symbol_groups:
                            if g[0] > end_x:
                                right_dist = g[0] - end_x
                                break
                        
                        if left_dist > 14 and right_dist > 14:
                            gaps.append((start_x, end_x))
                            gap_kinds.append("dot")
                            logger.debug(f"крапка-розділювач: x={start_x}-{end_x} (width={width}px, L={left_dist}px, R={right_dist}px)")

        # ==========================================
        # 5. 🆕 ЗЛИТТЯ: [вакуум] + [•] + [вакуум] = ОДНА межа
        # ==========================================

        if gaps:
            # Створюємо список (gap, kind) для сортування
            gaps_with_kinds = list(zip(gaps, gap_kinds))
            gaps_with_kinds.sort(key=lambda x: x[0][0])  # сортування за start_x
            
            merged = [list(gaps_with_kinds[0][0])]
            merged_kinds = [gaps_with_kinds[0][1]]
            
            for (gs, ge), kind in gaps_with_kinds[1:]:
                if gs - merged[-1][1] <= 20:
                    merged[-1][1] = max(merged[-1][1], ge)
                    # Якщо хоча б одна частина "dot" → результат "dot"
                    if kind == "dot":
                        merged_kinds[-1] = "dot"
                else:
                    merged.append([gs, ge])
                    merged_kinds.append(kind)
            
            gaps = [(a, b) for a, b in merged]
            gap_kinds = merged_kinds
        
        # ==========================================
        # Профіль чорнила (для візуалізації)
        # ==========================================

        
        ink = ink_map.sum(axis=0).astype(np.float32)
        kernel = np.ones(11, np.float32) / 11.0
        profile = np.convolve(ink, kernel, mode='same')
        
        logger.info(f"✅ Знайдено пробілів: {len(gaps)} (dot={gap_kinds.count('dot')}, vacuum={gap_kinds.count('vacuum')})")
        for i, ((a, b), kind) in enumerate(zip(gaps, gap_kinds)):
            logger.info(f"   Пробіл #{i+1}: {a}-{b} ({b-a}px) [{kind}]")
        
        return gaps, profile, gap_kinds
    

    def _is_valid_word_split(self, text: str, split_pos: int) -> bool:
        """
        Перевіряє, чи можна ділити текст у позиції split_pos.
        Не дозволяє ділити всередині слова (немає пробілу/розділювача).
        """
        if split_pos <= 0 or split_pos >= len(text):
            return False
        char_before = text[split_pos - 1]
        char_after = text[split_pos]
        
        # 🆕 КРОК 6.5: Мовно-залежні розділювачі
        if self.ocr_lang == 'tr':
            # Турецькі розділювачі (латиниця + кома)
            valid_separators = {' ', ',', '.', '•', '|', '▪', '—', ':', ';'}
        else:
            # Арабські розділювачі (арабська кома ،)
            valid_separators = {' ', '،', '.', '•', '|', '▪', '—'}
        
        if char_before in valid_separators or char_after in valid_separators:
            return True
        return False

    def _inner_gaps(self, x0: float, x1: float, gaps: List[tuple]) -> List[tuple]:
        """✅ FIX: тільки строго внутрішні gap'и + злиття перекриттів."""
        inner = sorted(
            (max(x0, float(gs)), min(x1, float(ge)))
            for gs, ge in gaps
            if x0 < gs and ge < x1 and ge > gs
        )
        # Злиття перекриттів (інакше total_gap завищений)
        merged = []
        for gs, ge in inner:
            if merged and gs <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], ge))
            else:
                merged.append((gs, ge))
        return merged

    def _split_block_by_gaps(self, blk: dict, gaps: List[tuple]) -> List[List[dict]]:
        # 🆕 КРОК 5.3: Роутер — вибираємо правильний метод залежно від напрямку
        if not self.is_rtl:
               return self._split_block_by_gaps_ltr(blk, gaps)
        text = blk.get("text", "")
        x0, x1 = float(blk["x_min"]), float(blk["x_max"])
        span = x1 - x0
        L = len(text)
        if L < 4 or span <= 0:
            return [[blk]]

        # Фільтруємо gap'и всередині блоку
        #inner = sorted((gs, ge) for gs, ge in gaps if x0 - 2 <= gs and ge <= x1 + 2)
        # Фільтруємо gap'и всередині блоку
        # inner = sorted((max(x0, float(gs)), min(x1, float(ge))) for gs, ge in gaps if x0 - 2 <= gs and ge <= x1 + 2 and ge > gs)
        # inner = [(gs, ge) for gs, ge in inner if ge > gs]
        inner = self._inner_gaps(x0, x1, gaps)
        if not inner:
            return [[blk]]
        total_gap = sum(ge - gs for gs, ge in inner)
        eff = span - total_gap
        if eff <= 10:
            return [[blk]]

        def chars_right_of(x: float) -> float:
            g = 0.0
            for gs, ge in inner:
                if gs >= x:
                    g += ge - gs
                elif ge > x:
                    g += ge - x
            ink = (x1 - x) - g
            return max(0.0, min(float(L), ink / eff * L))

        whole_word = bool(getattr(self, "split_whole_word", False))
        cuts = []        # символьні індекси розрізів
        cut_gap = {}     # символьний індекс -> (gs, ge) геометричний якір

        # Для RTL обробляємо gap'и справа наліво
        for gs, ge in reversed(inner):
            i = int(round(chars_right_of(ge)))
            i = max(1, min(L - 1, i))
            # if cuts and i - cuts[-1] < 3:
            #     continue
            # Snap ±2 до пробілу
            snapped = False
            for d in (0, 1, -1, 2, -2):
                j = i + d
                if 0 <= j < L and text[j] == " ":
                    i = j
                    snapped = True
                    break
            # Fallback «ціле слово»
            if not snapped and whole_word:
                a = text.rfind(" ", max(0, i - 15), i)
                b = text.find(" ", i, min(L, i + 15))
                if a > 0 and (b == -1 or (i - a) <= (b - i)):
                    i = a
                elif b != -1 and b < L - 1:
                    i = b

            # ==========================================
            # 🆕 ВСТАВИТИ НОВУ ЛОГІКУ З БЕЗПЕЧНИМ ВІДСТУПОМ ТУТ
            # ==========================================
            gap_width = ge - gs
            # if gap_width < 15:
            #     # Зсуваємо точку розрізу трохи вліво, щоб не "відрізати" 
            #     # хвости арабських літер (мінімум 3 символи)
            #     safe_margin_chars = max(3, int(L * 0.05)) 
            #     i = max(safe_margin_chars, i - 2)

            # Стало (✅ FIX: тільки якщо НЕ прив'язались до пробілу):
            if gap_width < 15 and not snapped:
                safe_margin_chars = max(3, int(L * 0.05))
                i = min(L - 1, max(safe_margin_chars, i - 2)) 
            # Перевірка валідності
            if hasattr(self, '_is_valid_word_split'):
                if not self._is_valid_word_split(text, i):
                    logger.warning(f"⚠️ ПРОПУСК РОЗДІЛЕННЯ: позиція {i}")
                    continue
            if any(abs(i - c) < 3 for c in cuts):
                continue
            if i in cut_gap:          # колізія після snap — не дублюємо якір
                continue
            cuts.append(i)
            cut_gap[i] = (gs, ge)     # 🔑 геометричний якір цього розрізу

        if not cuts:
            return [[blk]]

        bounds = [0] + sorted(cuts) + [L]

        # y-діапазон беремо з оригінального box
        try:
            ys = np.asarray(blk["box"], dtype=np.float32)[:, 1]
            y_min, y_max = float(ys.min()), float(ys.max())
        except Exception:
            y_min, y_max = 0.0, 30.0

        frags = []
        for k in range(len(bounds) - 1):
            part = text[bounds[k]:bounds[k + 1]].strip()
            if not part:
                return [[blk]]

            # 🔑 ТОЧНА ГЕОМЕТРІЯ (RTL):
            #  правий край фрагмента = gs gap-а, що ЙОГО ОБМЕЖУЄ СПРАВА
            #  (тобто gap на cuts[k]); для k=0 — це x1 батька.
            #  лівий край = ge gap-а ліворуч (cuts[k+1]); для останнього — x0.
            if bounds[k] == 0:
                r_edge = x1
            else:
                r_edge = cut_gap[bounds[k]][0]          # gs
            if bounds[k + 1] == L:
                l_edge = x0
            else:
                l_edge = cut_gap[bounds[k + 1]][1]      # ge

            # Страховка від виродження (якір загубився/перекриття)
            # if l_edge >= r_edge - 2:
            #     l_edge = min(l_edge, r_edge - 2)
            #     l_edge = max(x0, l_edge)

            if l_edge >= r_edge - 2:
                l_edge = max(x0, r_edge - 2)
                r_edge = max(r_edge, min(x1, l_edge + 2))

            box = np.array(
                [[l_edge, y_min], [r_edge, y_min],
                [r_edge, y_max], [l_edge, y_max]],
                dtype=np.float32,
            )
            frags.append([{
                "text": part,
                "score": float(blk.get("score", 0.0)),
                "x_min": float(l_edge),     # 🆕 реальний лівий край
                "x_max": float(r_edge),     # 🆕 реальний правий край
                "box": box,                 # 🆕 перебудований box
                "split_from_parent": True,  # мітка для діагностики
            }])

        if len(frags) < 2:
            return [[blk]]
        logger.debug(
            f"️ SPLIT: блок [{x0:.0f}..{x1:.0f}] → {len(frags)} фрагментів, "
            f"cuts={cuts}, geo=["
            + ", ".join(f"{f[0]['x_min']:.0f}..{f[0]['x_max']:.0f}" for f in frags)
            + "]"
        )
        return frags

    def _split_block_by_gaps_ltr(self, blk: dict, gaps: List[tuple]) -> List[List[dict]]:
        """
        🆕 КРОК 5.3: Розрізання блоків для LTR (турецьких) каналів.
        Дзеркальна логіка до RTL версії.
        
        LTR особливості:
        - Текст читається ЗЛІВА НАПРАВО
        - text[0] = найлівіший символ (біля x0)
        - text[L-1] = найправіший символ (біля x1)
        - chars_left_of(x) рахує символи ЗЛІВА від x
        - Ітерація по gaps: ЗЛІВА НАПРАВО (пряма, НЕ reversed)
        """
        text = blk.get("text", "")
        x0, x1 = float(blk["x_min"]), float(blk["x_max"])
        span = x1 - x0
        L = len(text)
        if L < 4 or span <= 0:
            return [[blk]]
        
        # Фільтруємо gap'и всередині блоку (ідентично до RTL)
        # inner = sorted(
        #     (max(x0, float(gs)), min(x1, float(ge)))
        #     for gs, ge in gaps
        #     if x0 - 2 <= gs and ge <= x1 + 2 and ge > gs
        # )
        # inner = [(gs, ge) for gs, ge in inner if ge > gs]
        # ✅ FIX: тільки строго внутрішні gap'и + злиття перекриттів
        inner = self._inner_gaps(x0, x1, gaps)
        if not inner:
            return [[blk]]
        
        total_gap = sum(ge - gs for gs, ge in inner)
        eff = span - total_gap
        if eff <= 10:
            return [[blk]]

        # 🆕 LTR: рахує символи ЗЛІВА від x (дзеркально до chars_right_of)
        def chars_left_of(x: float) -> float:
            """Кількість символів тексту ЗЛІВА від координати x."""
            g = 0.0
            for gs, ge in inner:
                if ge <= x:
                    g += ge - gs      # gap повністю зліва від x
                elif gs < x:
                    g += x - gs       # gap частково зліва від x
            ink = (x - x0) - g        # чорнило зліва від x
            return max(0.0, min(float(L), ink / eff * L))

        whole_word = bool(getattr(self, "split_whole_word", False))
        cuts = []        # символьні індекси розрізів
        cut_gap = {}     # символьний індекс -> (gs, ge) геометричний якір

        # 🆕 LTR: обробляємо gap'и ЗЛІВА НАПРАВО (пряма ітерація)
        for gs, ge in inner:  # НЕ reversed!
            # 🆕 LTR: використовуємо gs (початок gap), а не ge
            i = int(round(chars_left_of(gs)))
            i = max(1, min(L - 1, i))
            
            # Snap ±2 до пробілу (ідентично до RTL)
            snapped = False
            for d in (0, 1, -1, 2, -2):
                j = i + d
                if 0 <= j < L and text[j] == " ":
                    i = j
                    snapped = True
                    break
            
            # Fallback «ціле слово» (ідентично до RTL)
            if not snapped and whole_word:
                a = text.rfind(" ", max(0, i - 15), i)
                b = text.find(" ", i, min(L, i + 15))
                if a > 0 and (b == -1 or (i - a) <= (b - i)):
                    i = a
                elif b != -1 and b < L - 1:
                    i = b
            
            # Безпечний відступ для вузьких gap'ів
            # gap_width = ge - gs
            # if gap_width < 15:
            #     safe_margin_chars = max(3, int(L * 0.05))
            #     i = max(safe_margin_chars, i - 2)

            # ✅ FIX: тільки якщо НЕ прив'язались до пробілу — інакше зсув
            #  переносив розріз з пробілу всередину слова
            gap_width = ge - gs
            if gap_width < 15 and not snapped:
                safe_margin_chars = max(3, int(L * 0.05))
                i = min(L - 1, max(safe_margin_chars, i - 2))
            
            # Перевірка валідності
            if hasattr(self, '_is_valid_word_split'):
                if not self._is_valid_word_split(text, i):
                    logger.warning(f"🟥 ПРОПУСК РОЗДІЛЕННЯ (LTR): позиція {i}")
                    continue
            
            if any(abs(i - c) < 3 for c in cuts):
                continue
            if i in cut_gap:          # колізія після snap — не дублюємо якір
                continue
            
            cuts.append(i)
            cut_gap[i] = (gs, ge)     # 🔑 геометричний якір цього розрізу

        if not cuts:
            return [[blk]]

        bounds = [0] + sorted(cuts) + [L]
        
        # y-діапазон беремо з оригінального box
        try:
            ys = np.asarray(blk["box"], dtype=np.float32)[:, 1]
            y_min, y_max = float(ys.min()), float(ys.max())
        except Exception:
            y_min, y_max = 0.0, 30.0

        frags = []
        for k in range(len(bounds) - 1):
            part = text[bounds[k]:bounds[k + 1]].strip()
            if not part:
                return [[blk]]
            
            # 🔑 ТОЧНА ГЕОМЕТРІЯ (LTR): ДЗЕРКАЛЬНА до RTL
            #  лівий край фрагмента = ge gap-а, що ЙОГО ОБМЕЖУЄ ЗЛІВА
            #  (тобто gap на cuts[k]); для k=0 — це x0 батька.
            #  правий край = gs gap-а праворуч (cuts[k+1]); для останнього — x1.
            if bounds[k] == 0:
                l_edge = x0
            else:
                l_edge = cut_gap[bounds[k]][1]          # ge (кінець gap зліва)
            
            if bounds[k + 1] == L:
                r_edge = x1
            else:
                r_edge = cut_gap[bounds[k + 1]][0]      # gs (початок gap справа)
            
            # Страховка від виродження (якір загубився/перекриття)
            # if l_edge >= r_edge - 2:
            #     r_edge = max(r_edge, l_edge + 2)
            #     r_edge = min(x1, r_edge)
            # ✅ FIX: гарантуємо l_edge < r_edge навіть біля краю блоку
            if l_edge >= r_edge - 2:
                r_edge = min(x1, l_edge + 2)
                l_edge = max(x0, r_edge - 2)
            
            box = np.array(
                [[l_edge, y_min], [r_edge, y_min],
                [r_edge, y_max], [l_edge, y_max]],
                dtype=np.float32,
            )
            frags.append([{
                "text": part,
                "score": float(blk.get("score", 0.0)),
                "x_min": float(l_edge),     # 🆕 реальний лівий край
                "x_max": float(r_edge),     # 🆕 реальний правий край
                "box": box,                 # 🆕 перебудований box
                "split_from_parent": True,  # мітка для діагностики
            }])

        if len(frags) < 2:
            return [[blk]]
        
        logger.debug(
            f"🟪 SPLIT LTR: блок [{x0:.0f}..{x1:.0f}] → {len(frags)} фрагментів, "
            f"cuts={cuts}, geo=["
            + ", ".join(f"{f[0]['x_min']:.0f}..{f[0]['x_max']:.0f}" for f in frags)
            + "]"
        )
        return frags

    def _analyze_gap_scores(
        self,
        raw_blocks: List[dict],
        gaps: List[tuple],
        panorama: Optional[np.ndarray] = None
        #panorama: np.ndarray = None
    ) -> List[dict]:

        n = len(raw_blocks)
        if n < 2:
            return []

         #v2(TGT): чергова панорама з аналізом — годинник для віку ключів
        self._gap_seq += 1
        #v2(TGT): ключ рахується ≤1 разу за панораму, інакше 2-3 пари
        # з однаковими крайніми словами в одній панорамі домотали б поріг
        # за один кадр
        _tgt_seen = set()

        # Префіксні лічильники слів: words до/після пари i
        words_cnt = [len(b["text"].split()) for b in raw_blocks]
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + words_cnt[i]

        pair_gaps = []
        for i in range(1, n):
            if self.is_rtl:
                pair_gaps.append(raw_blocks[i - 1]["x_min"] - raw_blocks[i]["x_max"])
            else:
                pair_gaps.append(raw_blocks[i]["x_min"] - raw_blocks[i - 1]["x_max"])

        # Локальна медіана МАЛИХ проміжків = типовий міжсловний відступ
        # саме цієї панорами (медіана стійка до великих проміжків-меж)
        small = [g for g in pair_gaps if 0 < g <= 20]
        local_median = float(np.median(small)) if small else 1.0
        local_median = max(local_median, 1.0)

        analyses = []
        for i in range(1, n):
            # 🆕 ПАТЧ 14b: інверсія ролей блоків для LTR
            if self.is_rtl:
                right_b = raw_blocks[i - 1]   # RTL: блок ДО проміжку (праворуч)
                left_b = raw_blocks[i]        # RTL: блок ПІСЛЯ проміжку (ліворуч)
            else:
                right_b = raw_blocks[i]       # LTR: блок ДО проміжку (праворуч)
                left_b = raw_blocks[i - 1]    # LTR: блок ПІСЛЯ проміжку (ліворуч)
            
            g = pair_gaps[i - 1]
            
            w_right = max(1.0, right_b["x_max"] - right_b["x_min"])
            w_left = max(1.0, left_b["x_max"] - left_b["x_min"])

            r_min = g / min(w_right, w_left) if g > 0 else 0.0
            r_avg = g / ((w_right + w_left) / 2.0) if g > 0 else 0.0
            r_loc = g / local_median if g > 0 else 0.0

            vacuum_ratio = 0.0
            vacuum_run_w = 0.0
            right_center = (right_b["x_min"] + right_b["x_max"]) / 2.0
            left_center = (left_b["x_min"] + left_b["x_max"]) / 2.0

            for vs, ve in gaps:
                if g > 0:
                    gs, ge = left_b["x_max"], right_b["x_min"]
                    ov = max(0.0, min(ve, ge) - max(vs, gs))
                    if ov > 0:
                        ratio = ov / g
                        if ratio > vacuum_ratio:
                            vacuum_ratio = ratio
                            vacuum_run_w = float(ve - vs)
                else:
                    sep_center = (vs + ve) / 2.0
                    if left_center <= sep_center <= right_center:
                        vacuum_ratio = 1.0
                        vacuum_run_w = float(ve - vs)
                        break

            gap_x_start = min(left_b["x_max"], right_b["x_min"])
            gap_x_end = max(left_b["x_max"], right_b["x_min"])
            #gap_width = max(0.0, gap_x_end - gap_x_start)
            gap_width = max(0.0, float(g))
            
            has_dot = False
            if panorama is not None and hasattr(self, '_has_dot_in_gap'):
                h_pan = panorama.shape[0]
                y_start = int(h_pan * 0.15)
                y_end = int(h_pan * 0.85)
                has_dot = self._has_dot_in_gap(panorama, gap_x_start, gap_x_end, y_start, y_end)

            # --- ЛОГІКА БАЛІВ ---
            vacuum_s = round(40.0 * vacuum_ratio, 1)
            local_s = 25 if r_loc >= 8 else 15 if r_loc >= 4 else 8 if r_loc >= 2 else 0
            avg_s = 20 if r_avg >= 0.25 else 12 if r_avg >= 0.12 else 6 if r_avg >= 0.06 else 0
            min_s = 10 if r_min >= 0.8 else 5 if r_min >= 0.4 else 2 if r_min >= 0.2 else 0

            words_before = prefix[i]
            words_after = prefix[n] - prefix[i]
            text_s = 5 if (words_before >= 3 and words_after >= 3) else (2 if (words_before >= 1 and words_after >= 1) else 0)

            # Базовий скор без TGT бонусу
            raw_score = vacuum_s + local_s + avg_s + min_s + text_s
            dot_bonus = 15 if has_dot else 0

            # ═══ TGT: часовий бонус за стабільність проміжку ═══
            right_tail = self._norm_word_e2(right_b["text"]).split()
            left_head = self._norm_word_e2(left_b["text"]).split()
            
            if self.is_rtl:
                right_key = right_tail[-1] if right_tail else ""
                left_key = left_head[0] if left_head else ""
            else:
                right_key = right_tail[0] if right_tail else ""
                left_key = left_head[-1] if left_head else ""

            gap_key = f"{right_key}|{left_key}"
            temporal_bonus = 0

            if g > 0 and right_key and left_key:
                if gap_key not in _tgt_seen:
                    _tgt_seen.add(gap_key)
                    rec = self._gap_history.get(gap_key)
                    if rec:
                        rec[0] += 1
                        rec[1] = self._gap_seq
                    else:
                        self._gap_history[gap_key] = [1, self._gap_seq]
                
                rec = self._gap_history.get(gap_key)
                if rec and g >= 30 and rec[0] >= self._gap_history_threshold:
                    temporal_bonus = self._gap_history_bonus
            else:
                self._gap_history.pop(gap_key, None)
            # ═══ кінець TGT ═══

            # ==========================================
            # ФІНАЛЬНИЙ РОЗРАХУНОК ОЦІНКИ ТА РІШЕННЯ (ЄДИНИЙ БЛОК)
            # ==========================================
            final_score = raw_score + dot_bonus + temporal_bonus
            
            # РЯТУВАЛЬНИЙ БОНУС: вузька чиста крапка-маркер (•) без бічних пробілів
            if g <= 15 and vacuum_ratio >= 0.9 and vacuum_run_w <= 15:
                final_score = max(final_score, 70)

            # Визначення статусу межі на основі ПОВНОЇ фінальної оцінки

            if gap_width >= 60 or (gap_width >= 35 and has_dot):
                decision = "STRONG_BOUNDARY"
            elif final_score >= 80:
                decision = "STRONG_BOUNDARY"
            elif final_score >= 30 or gap_width >= 40 or (gap_width >= 25 and has_dot):
                decision = "SUSPICIOUS"
            else:
                decision = "NORMAL"

            # if gap_width >= 60 or (gap_width >= 35 and has_dot):
            #     decision = "STRONG_BOUNDARY"
            # elif gap_width >= 40 or (gap_width >= 25 and has_dot):
            #     decision = "PROBABLE_BOUNDARY"
            # elif final_score >= 80:
            #     decision = "STRONG_BOUNDARY"
            # elif final_score >= 60:
            #     decision = "PROBABLE_BOUNDARY"
            # elif final_score >= 30:
            #     decision = "SUSPICIOUS"
            # else:
            #     decision = "NORMAL"

            analyses.append({
                "pair_index": i,
                "x_start": round(float(left_b["x_max"]), 1),
                "x_end": round(float(right_b["x_min"]), 1),
                "gap_width": round(float(g), 1),
                "w_right": round(float(w_right), 1),
                "w_left": round(float(w_left), 1),
                "ratio_min": round(r_min, 3),
                "ratio_avg": round(r_avg, 3),
                "local_median_gap": round(local_median, 1),
                "ratio_local": round(r_loc, 1),
                "vacuum_overlap": round(vacuum_ratio, 2),
                "vacuum_run_width": round(vacuum_run_w, 1),
                "words_before": words_before,
                "words_after": words_after,
                "scores": {
                    "vacuum": vacuum_s, "local": local_s, "avg": avg_s,
                    "min": min_s, "text": text_s,
                },
                "has_dot": has_dot,        
                "dot_bonus": dot_bonus,
                "final_score": round(final_score, 1),
                "raw_score": round(raw_score + dot_bonus, 1), # Щоб словник мав валідне значення
                "decision": decision,
                "gap_key": gap_key,
                "temporal_bonus": temporal_bonus,
                "stability_count": self._gap_history.get(gap_key, [0, 0])[0],
            })

            logger.info(
                f"GAP ANALYSIS: frame={self.frame_counter} x_start={gap_x_start:.1f} x_end={gap_x_end:.1f} | "
                f"G={gap_width:.1f} | W_r={w_right:.1f} W_l={w_left:.1f} | "
                f"r_min={r_min:.2f} r_avg={r_avg:.2f} r_loc={r_loc:.1f}(med={local_median:.1f}) | "
                f"vac={vacuum_ratio:.2f}(run={vacuum_run_w:.1f}) | words={words_after}/{words_before} | "
                f"vac_s={vacuum_s} loc_s={local_s} avg_s={avg_s} min_s={min_s} txt_s={text_s} "
                f"dot={'YES' if has_dot else 'NO'}(+{dot_bonus}) -> score={final_score:.1f} [{decision}]"
            )

        return analyses

    def _log_gap_score_summary(
        self,
        analyses: List[dict],
        legacy_flags: List[bool],
    ) -> None:
        """ G1 (shadow): звірка GAP SCORE з рішеннями старого алгоритму.
        Нічого не змінює — лише лог + self.last_gap_score_summary для дебагу.

        MATCH        — обидва алгоритми згодні (межа/не межа)
        CONFLICT     — старий підтвердив межу, score < 60 (слабка межа?)
        NEW_CANDIDATE — score >= 60, старий НЕ підтвердив (пропущена межа?)
        """
        if not analyses:
            return

        counts = {"NORMAL": 0, "SUSPICIOUS": 0, "PROBABLE_BOUNDARY": 0, "STRONG_BOUNDARY": 0}
        for a in analyses:
            counts[a["decision"]] = counts.get(a["decision"], 0) + 1

        match_n = conflict_n = new_cand_n = 0
        new_candidates, conflicts = [], []
        for k, a in enumerate(analyses):
            legacy = bool(legacy_flags[k]) if k < len(legacy_flags) else False
            new_algo = a["decision"] in ("PROBABLE_BOUNDARY", "STRONG_BOUNDARY")
            if legacy and new_algo:
                match_n += 1
            elif legacy and not new_algo:
                conflict_n += 1
                conflicts.append(
                    f"G={a['gap_width']:.0f}px score={a['raw_score']:.0f} "
                    f"vac={a['vacuum_overlap']:.2f} r_loc={a['ratio_local']:.1f}"
                )
            elif not legacy and new_algo:
                new_cand_n += 1
                new_candidates.append(
                    f"G={a['gap_width']:.0f}px score={a['raw_score']:.0f} "
                    f"r_avg={a['ratio_avg']:.2f} r_min={a['ratio_min']:.2f} "
                    f"r_loc={a['ratio_local']:.1f}"
                )

        self.last_gap_score_summary = {
            "pairs": len(analyses),
            "counts": counts,
            "legacy_boundaries": int(sum(1 for f in legacy_flags if f)) if legacy_flags else 0,
            "match": match_n,
            "conflict": conflict_n,
            "new_candidate": new_cand_n,
            "new_candidates": new_candidates,
            "conflicts": conflicts,
        }

        logger.info(
            "📊 GAP SCORE SUMMARY: пар=%d | NORMAL=%d SUSPICIOUS=%d PROBABLE=%d "
            "STRONG=%d | legacy меж=%d | MATCH=%d CONFLICT=%d NEW_CANDIDATE=%d",
            len(analyses), counts["NORMAL"], counts["SUSPICIOUS"],
            counts["PROBABLE_BOUNDARY"], counts["STRONG_BOUNDARY"],
            self.last_gap_score_summary["legacy_boundaries"],
            match_n, conflict_n, new_cand_n,
        )
        for c in conflicts:
            logger.info("🟥 CONFLICT (legacy=так, score<60): %s", c)
        for c in new_candidates:
            logger.info("   🔎 NEW_CANDIDATE (score>=60, legacy=ні): %s", c)

          # 🆕 TGT: логуємо стабільні проміжки
        stable_gaps = [k for k, v in self._gap_history.items()
                       if v[0] >= self._gap_history_threshold]
        if stable_gaps:
            logger.info(
                "⏱️ TGT STABLE GAPS (%d): %s",
                len(stable_gaps),
                ", ".join(f"{k}(n={self._gap_history[k][0]})"
                          for k in stable_gaps[:5])
            )

        # 🆕v2: витіснення за ВІКОМ. Порівнюємо вік (v[1]), а не лічильник.
        stale = [k for k, v in self._gap_history.items()
                 if v[1] < self._gap_seq - self._gap_history_max_age]
        for k in stale:
            del self._gap_history[k]
        if stale:
            logger.debug("🧹 TGT: очищено %d застарілих ключів (за віком)", len(stale))

 

    def _extract_and_ocr_sentence(self, panorama: np.ndarray, timestamp: float) -> bool:
        """
        Розпізнає текст на панорамі з візуальною дебаг-візуалізацією та розумним сегментуванням за графіком.
        """
        if panorama is None or panorama.shape[1] < 20:
            return False
        self._dup_zone_px = 0.0          # 🆕 C-DROP: за замовчуванням зони немає
        self._dup_start_x = None
        try:
            # 0. Зберігаємо оригінальну панораму ДО очищення
            original_panorama = panorama.copy()

            # # 1. Видаляємо сірі смуги з копії оригіналу
            # clean_panorama = self._remove_left_gray_bands(
            #     original_panorama.copy()
            # )

            if self.fade_side == 'left':
                clean_panorama = self._remove_left_gray_bands(original_panorama.copy())
            else:
                clean_panorama = self._remove_right_gray_bands(original_panorama.copy())

            # 2. Зберігаємо панораму ПІСЛЯ очищення
            debug_dir = self.output_dir / "debug" / "panorama"
            debug_dir.mkdir(parents=True, exist_ok=True)

            # 2. Зберігаємо панораму ПІСЛЯ очищення
            cv2.imwrite(
                str(debug_dir / f"panorama_{self.frame_counter}.png"),
                clean_panorama,
            )

            # 3. Далі весь OCR працює з очищеною панорамою
            panorama = clean_panorama

            # Працює АДАПТИВНО: захищає темні фони (Al Jazeera) і чистить лише вузькі шви на світлих (Al Hadath).
            gray0 = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
            median_brightness = float(np.median(gray0))

            # КРОК 1: Якщо фон в цілому темний (медіана < 110), НЕ чіпаємо стовпці!
            # Це запобігає знищенню білого тексту на синьому/чорному фоні.
            if median_brightness < 110.0:
                logger.debug(f"🛡️ Очищення стовпців пропущено: темний фон (медіана яскравості={median_brightness:.1f})")
            else:
                # КРОК 2: Для світлих фонів шукаємо саме ВУЗЬКІ шви, а не широкі темні зони.
                dark_frac = np.mean(gray0 < 100, axis=0)
                dark_cols = np.where(dark_frac > 0.85)[0]

                if len(dark_cols) > 0:
                    # Групуємо суміжні темні стовпці, щоб перевірити їх ширину
                    groups = []
                    current_group = [dark_cols[0]]
                    for cx in dark_cols[1:]:
                        if cx - current_group[-1] <= 2:  # дозволяємо мікро-розриви в 1-2 пікселі
                            current_group.append(cx)
                        else:
                            groups.append(current_group)
                            current_group = [cx]
                    groups.append(current_group)

                    # Фільтруємо лише справжні шви стиснення (ширина <= 6 пікселів).
                    # Широкі темні зони (>6px) — це фон або текст, їх не чіпаємо!
                    #true_seams = [g for g in groups if len(g) <= 6]
                    true_seams = [
                        g for g in groups
                        if (g[-1] - g[0] + 1) <= 6
                    ]
                    flat_seams = [cx for g in true_seams for cx in g]

                    if len(flat_seams) > 0:
                        orig = panorama.copy()
                        clean_mask = np.ones(panorama.shape[1], dtype=bool)
                        clean_mask[flat_seams] = False
                        w0 = panorama.shape[1]

                        cleaned_count = 0
                        for cx in flat_seams:
                            l = cx - 1
                            while l >= 0 and not clean_mask[l]:
                                l -= 1
                            r = cx + 1
                            while r < w0 and not clean_mask[r]:
                                r += 1

                            if l >= 0 and r < w0:
                                panorama[:, cx] = ((orig[:, l].astype(np.int16) + orig[:, r].astype(np.int16)) // 2).astype(np.uint8)
                                cleaned_count += 1
                            elif l >= 0:
                                panorama[:, cx] = orig[:, l]
                                cleaned_count += 1
                            elif r < w0:
                                panorama[:, cx] = orig[:, r]
                                cleaned_count += 1

                        ignored_count = len(dark_cols) - cleaned_count
                        logger.debug(f"🧹 BGR-панорама: заліковано ВУЗЬКИХ швів: {cleaned_count} (ігноровано широких темних зон: {ignored_count})")


                        # 4.  М'ЯКЕ ОЧИЩЕННЯ ПЕРЕД OCR (прибирає смуги стиснення)
            gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
            
            # Крок 1: Дуже легке видалення вертикальних смуг (H.264 macroblocks)
            denoised = cv2.medianBlur(gray, 3)
            
            #  Крок 1.5: ЗАХИСТ для темних фондів (Al Jazeera). 
            # Якщо медіанна яскравість низька, НЕ чіпаємо стовпці, інакше текст буде знищено!
            #  Крок 1.5: АДАПТИВНЕ очищення для ТЕМНИХ фондів (Al Jazeera).
            # Замість пропуску очищення — адаптуємо поріг під яскравість фону.
            median_brightness = float(np.median(gray))
            
            # Визначаємо поріг для виявлення швів:
            # - Для світлих фонів (медіана > 110): шви мають яскравість < 100
            # - Для темних фонів (медіана ~60): шви мають яскравість ~20-30
            if median_brightness < 110.0:
                # 🆕 ДЛЯ ДУЖЕ ТЕМНИХ ФОНІВ (Al Ghad): ще м'якший поріг
                threshold = max(8.0, median_brightness * 0.5)  # 0.5 замість 0.4, мінімум 8 замість 15
                logger.debug(f"🛡️ Адаптивне очищення: ТЕМНИЙ фон (медіана={median_brightness:.1f}), поріг={threshold:.1f}")
            else:
                threshold = 100.0
                logger.debug(f"🛡️ Стандартне очищення: світлий фон (медіана={median_brightness:.1f}), поріг={threshold:.1f}")
            
            # ЗАВЖДИ виконуємо очищення (не пропускаємо!)
            gray_clean = denoised.copy()
            dark_frac = np.mean(gray_clean < threshold, axis=0)  # ЗМІНЕНО: 100 → threshold
            dark_cols = np.where(dark_frac > 0.8)[0]
            
            if len(dark_cols) > 0:
                clean_mask = dark_frac <= 0.8
                w1 = gray_clean.shape[1]
                for cx in dark_cols:
                    l = cx - 1
                    while l >= 0 and not clean_mask[l]:
                        l -= 1
                    r = cx + 1
                    while r < w1 and not clean_mask[r]:
                        r += 1
                    if l >= 0 and r < w1:
                        gray_clean[:, cx] = ((denoised[:, l].astype(np.int16) + denoised[:, r].astype(np.int16)) // 2).astype(np.uint8)
                    elif l >= 0:
                        gray_clean[:, cx] = denoised[:, l]
                    elif r < w1:
                        gray_clean[:, cx] = denoised[:, r]
            
            # Крок 2: CLAHE тепер працює на ОЧИЩЕНОМУ зображенні
            clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray_clean)
            
            # Крок 3: Конвертація в BGR для PaddleOCR
            enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

            # 1. Unsharp Mask (Підвищення різкості)
            # Критично важливо для арабської в'язі: робить краї літер "жорсткішими", 
            # допомагаючи OCR бачити розриви між злитими словами.
            blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.5)  # Радіус 1.5 краще для тонкого тексту (35px)
            enhanced = cv2.addWeighted(enhanced, 2.0, blurred, -1.0, 0)

            # 2. Збереження країв + видалення шуму (Bilateral Filter)
            # Працює швидше за fastNlMeansDenoisingColored і не "милить" літери.
            # d=3 (діаметр), sigmaColor=25, sigmaSpace=25
            enhanced = cv2.bilateralFilter(enhanced, 3, 25, 25)

            logger.debug(f"🎨 Покращення якості: Unsharp + Bilateral (розмір {enhanced.shape})")

            logger.warning(
                "🟩 OCR DIMENSIONS | panorama=%s | enhanced=%s",
                panorama.shape,
                enhanced.shape
            )

            # 5. Запуск OCR на очищеному зображенні
            result = self.ocr.predict(enhanced)
            self.ocr_calls += 1
            
            raw_blocks = []
            for page in result:
                texts = page.get('rec_texts', [])
                scores = page.get('rec_scores', [])
                
                # Безпечне отримання боксів (уникаємо помилок з NumPy arrays)
                # boxes = page.get('rec_polys', [])
                # if not boxes:
                #     boxes = page.get('dt_boxes', [])
                boxes = page.get('rec_polys', None)

                if boxes is None or len(boxes) == 0:
                    boxes = page.get('dt_boxes', [])

                for i, txt in enumerate(texts):
                    txt = txt.strip()
                    score = float(scores[i]) if i < len(scores) else 0.0

                    if score <= 0.4 or len(txt) <= 2:
                        continue
                    if self._is_excluded_block(txt):
                        continue
                    # блок без ЖОДНОЇ літери цільової мови і не число → сміття
                    if (not any(self._is_target_lang_char(c) for c in txt)
                            and not txt.strip().isdigit()):
                        continue
                    if i >= len(boxes):
                        continue

                    # 🔑 КРОК A: Зберігаємо обидва краї OCR-блока (x_min і x_max)
                    box = np.asarray(boxes[i], dtype=np.float32).reshape(-1, 2)
                    if len(box) < 4 or not np.isfinite(box).all():
                        continue
                    
                    xs = box[:, 0]
                    raw_blocks.append({
                        "x_min": float(xs.min()),
                        "x_max": float(xs.max()),
                        "text": txt,
                        "score": score,
                        "box": box
                    })

            if not raw_blocks:
                logger.warning("OCR: текст на панорамі не знайдено")
                return False

            # 3. Сортування СПРАВА НАЛІВО (від найбільшого x_min до найменшого)
            #raw_blocks.sort(key=lambda item: item["x_min"], reverse=True)
            # 🆕 КРОК 5.1: Універсальне сортування залежно від напрямку каналу
            raw_blocks = self._sort_blocks_by_direction(raw_blocks)
            logger.debug("📊 Блоки відсортовано для %s напрямку", "RTL" if self.is_rtl else "LTR")

              # 🆕 Статистика OCR (Діагностика якості розпізнавання)
            total_blocks_detected = len(raw_blocks)
            avg_confidence = np.mean([b['score'] for b in raw_blocks]) if raw_blocks else 0
            total_text_chars = sum(len(b['text']) for b in raw_blocks)
            
            logger.info(
                f"📊 OCR СТАТИСТИКА: блоків={total_blocks_detected}, "
                f"середня впевненість={avg_confidence:.2f}, "
                f"символів={total_text_chars}"
            )

            #  ДІАГНОСТИКА ПРОПУСКІВ
            if total_blocks_detected < 5:
                logger.warning(
                    f"🟥 МАЛО БЛОКІВ: {total_blocks_detected} (норма: 10-20). "
                    f"Можлива причина: агресивна фільтрація або погане очищення"
                )
                
            short_blocks = [b for b in raw_blocks if len(b['text']) < 10]
            if len(short_blocks) > len(raw_blocks) * 0.3:
                logger.warning(
                    f"🟥 БАГАТО КОРОТКИХ БЛОКІВ: {len(short_blocks)}/{len(raw_blocks)} "
                    f"(30%+). Можливо, текст розбито на занадто дribні частини через сепаратори"
                )

            #  E2 CROSS-PANORAMA MATCHING (діагностичний режим) — ОБОВ'ЯЗКОВО
            # ПЕРЕД _detect_edge_words. Дві причини (порядок критичний):
            #   1) буфер ще тримає класи B ПОПЕРЕДНЬОЇ панорами — детекція
            #      нижче перезапише їх класами B цієї (однопанорамна глибина);
            #   2) _edge_prev_meta ще «попередня» → _expected_prev_overlap_px
            #      рахує зону дублювання саме для пари «попередня ↔ ця».
            # Після матчу детекція класифікує краї цієї панорами й наповнить
            # буфер її власними кандидатами (ланцюжок продовжується).
            self._cross_panorama_match(raw_blocks, panorama, timestamp)

             # EDGE WORD DETECTION — лише спостереження
            self._detect_edge_words(raw_blocks, panorama, timestamp)

            # gaps, profile = self._detect_sentence_gaps(panorama) працює без математики aljazeera
            # if gaps:
            #     logger.info(f"✅ Знайдено {len(gaps)} пробілів між реченнями")

            # # УНІФІКАЦІЯ: використовуємо ТІ САМІ gaps для сегментації,
            # # що й для візуалізації ++. Більше немає окремого vacuum_runs!
            # vacuum_runs = gaps

            #  Використовуємо TSM, якщо він доступний
            if getattr(self, 'tsm_mapper', None):
                gaps, gap_kinds = self._run_tsm_analysis(panorama)
                profile = None  # TSM не повертає ink_profile, візуалізація це безпечно обробить (перевірка if profile is not None)
            else:
                # Фолбек на старий алгоритм, якщо TSM не ініціалізовано
                gaps, profile, gap_kinds = self._detect_sentence_gaps(panorama)

            if gaps:
                logger.info(f"✅ Знайдено {len(gaps)} пробілів між реченнями (dot={gap_kinds.count('dot')}, vacuum={gap_kinds.count('vacuum')})")
            
            # УНІФІКАЦІЯ: використовуємо ТІ САМІ gaps для сегментації
            vacuum_runs = gaps

            # gaps, profile, gap_kinds = self._detect_sentence_gaps(panorama)
            # if gaps:
            #     logger.info(f"✅ Знайдено {len(gaps)} пробілів між реченнями (dot={gap_kinds.count('dot')}, vacuum={gap_kinds.count('vacuum')})")
            #     # УНІФІКАЦІЯ: використовуємо ТІ САМІ gaps для сегментації,
            #     # що й для візуалізації ++. Більше немає окремого vacuum_runs!
            #     vacuum_runs = gaps

            # 🆕 PRE-SPLIT v2: ріжемо КОЖЕН блок, всередині якого є розділювач.
            # Виправляє випадки, коли OCR повертає 2+ блоки, але один із них —
            # гігантський злиток із кількома крапками • всередині (панорами #115, #153).
            if gaps:
                expanded = []
                split_hits = 0
                for blk in raw_blocks:
                    parts = self._split_block_by_gaps(blk, gaps)
                    flat = [pb for g in parts for pb in g]
                    if len(flat) > 1:
                        split_hits += 1
                    expanded.extend(flat)
                if len(expanded) > len(raw_blocks):
                    logger.info(
                        f"🟪 PRE-SPLIT v2: {len(raw_blocks)} блоків → {len(expanded)} "
                        f"псевдо-блоків (розрізано злитків: {split_hits}) "
                        f"за {len(gaps)} розділювачами"
                    )
                    raw_blocks = expanded
                    #raw_blocks.sort(key=lambda item: item["x_min"], reverse=True)
                    raw_blocks = self._sort_blocks_by_direction(raw_blocks)

            # ==========================================
            # ГЕОМЕТРИЧНА СЕГМЕНТАЦІЯ (використовує gaps з _detect_sentence_gaps)
            # ==========================================

            sentence_groups = []
            confirmed_boundaries = []
            # Межа між сусідніми групами.
            group_boundaries = []
            w_pan = panorama.shape[1]
            h_pan = panorama.shape[0]
            
            MIN_VACUUM_WIDTH = int(getattr(self, "min_gap_width", 50))
            MIN_GAP_OVERLAP_RATIO = 0.60
            
            if gaps and raw_blocks:
                current_group = [raw_blocks[0]]

                # (shadow): підготовка до аналізу геометрії проміжків
                gap_analyses = self._analyze_gap_scores(raw_blocks, gaps, panorama)
                legacy_boundary_flags = []
                
                # for i in range(1, len(raw_blocks)):
                #     # right_block = raw_blocks[i - 1]  # Фізично правий блок (більший x)
                #     # left_block = raw_blocks[i]       # Фізічно лівий блок (менший x)
                #     if self.is_rtl:
                #                         # 🆕 FIX P0-1: блок, який ПРИЄДНУЄМО до групи, — завжди
                #         # наступний у порядку ЧИТАННЯ (raw_blocks вже відсортовано
                #         # _sort_blocks_by_direction). left/right — лише ФІЗИЧНІ ролі
                #         # для геометрії проміжку, вони НЕ керують групацією.
                #         next_block = raw_blocks[i]
                #         right_block = raw_blocks[i - 1]  # Фізично правий блок
                #         left_block = raw_blocks[i]       # Фізично лівий блок
                #         gap_left = left_block["x_max"]
                #         gap_right = right_block["x_min"]
                #     else:
                #         left_block = raw_blocks[i - 1]   # Фізично лівий блок
                #         right_block = raw_blocks[i]      # Фізично правий блок
                #         gap_left = left_block["x_max"]
                #         gap_right = right_block["x_min"]
                #     gap_width = gap_right - gap_left
                for i in range(1, len(raw_blocks)):
                    # 🆕 FIX P0-1: блок, який ПРИЄДНУЄМО до групи, — завжди
                    # наступний у порядку ЧИТАННЯ (raw_blocks вже відсортовано
                    # _sort_blocks_by_direction). left/right — лише ФІЗИЧНІ ролі
                    # для геометрії проміжку, вони НЕ керують групацією.
                    next_block = raw_blocks[i]

                    if self.is_rtl:
                        right_block = raw_blocks[i - 1]   # фізично правий
                        left_block  = raw_blocks[i]       # фізично лівий
                    else:
                        left_block  = raw_blocks[i - 1]   # фізично лівий
                        right_block = raw_blocks[i]       # фізично правий

                    # спільна геометрія проміжку — однакова для RTL і LTR
                    gap_left  = left_block["x_max"]
                    gap_right = right_block["x_min"]
                    gap_width = gap_right - gap_left
                    
                    # Фізичний проміжок між краями цих двох конкретних блоків
                    gap_left = left_block["x_max"]
                    gap_right = right_block["x_min"]
                    gap_width = gap_right - gap_left
                    if gap_width < -50:
                        logger.warning(
                            "ДЕГЕНЕРАТИВНА ГЕОМЕТРІЯ ПАРІ: gap=%.0f (right=[%.0f..%.0f] left=[%.0f..%.0f]) "
                            "— ймовірно псевдо-блок без власних координат",
                            gap_width, right_block["x_min"], right_block["x_max"],
                            left_block["x_min"], left_block["x_max"],
    )
                    
                    is_sentence_boundary = False
                    pair_boundary = None  # 🆕 Зміна 2: ініціалізуємо для поточної пари

                    logger.info(
                        "OCR GAP: frame=%d right=[%.1f, %.1f] left=[%.1f, %.1f] "
                        "gap=[%.1f, %.1f] width=%.1f",
                        self.frame_counter,
                        right_block["x_min"], right_block["x_max"],
                        left_block["x_min"], left_block["x_max"],
                        gap_left, gap_right, gap_width
                    )

                    ga = gap_analyses[i - 1] if i - 1 < len(gap_analyses) else None

                    if ga and (ga["gap_width"] > 30 or ga["vacuum_overlap"] > 0 or ga["raw_score"] >= 30):
                        tgt_marker = ""
                        if ga.get("temporal_bonus", 0) > 0:
                            tgt_marker = (f"TGT+{ga['temporal_bonus']} "
                                          f"(stable={ga.get('stability_count', 0)}, "
                                          f"key={ga.get('gap_key', '?')})")
                        
                        logger.info(
                            "GAP ANALYSIS: frame=%d x_start=%.1f x_end=%.1f | G=%.0f | W_r=%.0f W_l=%.0f | r_min=%.2f r_avg=%.2f "
                            "r_loc=%.1f(med=%.0f) | vac=%.2f(run=%.0f) | words=%d/%d | "
                            "vac_s=%.0f loc_s=%d avg_s=%d min_s=%d txt_s=%d -> score=%.0f [%s]%s",
                            self.frame_counter, gap_left, gap_right,
                            ga["gap_width"], ga["w_right"], ga["w_left"],
                            ga["ratio_min"], ga["ratio_avg"],
                            ga["ratio_local"], ga["local_median_gap"],
                            ga["vacuum_overlap"], ga["vacuum_run_width"],
                            ga["words_before"], ga["words_after"],
                            ga["scores"]["vacuum"], ga["scores"]["local"],
                            ga["scores"]["avg"], ga["scores"]["min"], ga["scores"]["text"],
                            ga["raw_score"], ga["decision"],
                            tgt_marker,
                        )

                    if gap_width <= 0:
                        logger.debug(
                            "OCR BLOCK OVERLAP: right=[%.1f, %.1f] "
                            "left=[%.1f, %.1f] overlap=%.1f",
                            right_block["x_min"], right_block["x_max"],
                            left_block["x_min"], left_block["x_max"],
                            -gap_width
                        )

                    # 🆕 Зміна 1: Перевірка за фізичними краями блоків + тип межі
                    for gap_idx, (vac_start, vac_end) in enumerate(gaps):
                        kind = gap_kinds[gap_idx] if gap_idx < len(gap_kinds) else "vacuum"
                        sep_center = (vac_start + vac_end) / 2.0
                        
                        # Межа підтверджується якщо:
                        # 1. Це крапка (dot) → завжди межа
                        # 2. Це vacuum І ширина >= min_gap_width
                        is_valid_boundary = False
                        if kind == "dot":
                            is_valid_boundary = True
                        elif kind == "vacuum":
                            gap_width = vac_end - vac_start
                            if gap_width >= self.min_gap_width:
                                is_valid_boundary = True
                        
                        # Перевіряємо, що блоки не перекриваються і розділювач лежить саме між їхніми краями
                        if is_valid_boundary and gap_left <= gap_right and gap_left <= sep_center <= gap_right:
                            is_sentence_boundary = True
                            pair_boundary = (int(vac_start), int(vac_end))
                            confirmed_boundaries.append(pair_boundary)
                            break
                                        

                    if is_sentence_boundary:
                        sentence_groups.append(current_group)
                        group_boundaries.append(pair_boundary)
                        current_group = [next_block]       # було [left_block]
                    else:
                        current_group.append(next_block)   # було left_block
                    legacy_boundary_flags.append(bool(is_sentence_boundary))
                
                if current_group:
                    sentence_groups.append(current_group)

                self._log_gap_score_summary(gap_analyses, legacy_boundary_flags)
                logger.info(f"📝 Графік розрізав текст на {len(sentence_groups)} речень")
            else:
                sentence_groups = [raw_blocks]

            #SPLIT-•: якщо OCR злив усе в ОДИН блок, а розділювачі знайдені —
            # ріжемо текст на речення пропорційно до координат (RTL)  це для aljazeera - він не працює адже замінений на оптимальний і є як запас
            if len(sentence_groups) == 1 and len(sentence_groups[0]) == 1 and gaps:
                new_groups = self._split_block_by_gaps(sentence_groups[0][0], gaps)
                if len(new_groups) > 1:
                    blk = sentence_groups[0][0]
                    confirmed_boundaries = [
                        (int(gs), int(ge)) for gs, ge in gaps
                        if blk["x_min"] < gs < ge < blk["x_max"]
                    ]
                    sentence_groups = new_groups
                    group_boundaries = confirmed_boundaries[:len(new_groups) - 1]
            
                    logger.info(
                        f"🟪 SPLIT-•: єдиний блок розрізано на {len(sentence_groups)} речень "
                        f"за {len(confirmed_boundaries)} розділювачами"
                    )

                        # === ЛОГУВАННЯ КООРДИНАТ МЕЖ ===
            if confirmed_boundaries:
                logger.info(f"📍 КООРДИНАТИ МЕЖ (всього {len(confirmed_boundaries)}): {confirmed_boundaries}")
                for idx, (gs, ge) in enumerate(confirmed_boundaries):
                    logger.info(f"   🟨 Межа #{idx+1}: x={gs} до {ge} (ширина={ge-gs}px)")
            else:
                logger.info("📍 Меж речень не знайдено.")

            # ==========================================
            # 🎨 УНІВЕРСАЛЬНА ВІЗУАЛІЗАЦІЯ ДЛЯ ДЕБАГУ (Працює для ВСІХ каналів)
            # ==========================================
            debug_img = panorama.copy()
            h_pan, w_pan = debug_img.shape[:2]

            # 1. Лінія лівого краю (М'який червоний + згладжування)
            cv2.line(debug_img, (0, 0), (0, h_pan), (100, 100, 255), 1, lineType=cv2.LINE_AA)

            # 2. Рамки навколо OCR-блоків (М'які кольори + згладжування)
            for idx, block in enumerate(raw_blocks):
                box = block["box"]
                pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
                
                if idx == 0:
                    color = (100, 255, 100)  # М'який зелений
                elif idx == len(raw_blocks) - 1:
                    color = (100, 100, 255)  # М'який червоний
                else:
                    color = (255, 255, 150)  # М'який жовтий/циан
                    
                # Додаємо lineType=cv2.LINE_AA для згладжування країв лінії
                cv2.polylines(debug_img, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
                
                # Підпис блоку (Сірий колір тексту менше відволікає)
                label = f"{idx}:{block['text'][:12]}"
                cv2.putText(debug_img, label, (int(block['x_min']), int(pts[0][0][1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, lineType=cv2.LINE_AA)

                            # 3. Візуалізація підтверджених меж (Gaps)


                # 3. Візуалізація підтверджених меж (Gaps) — ПЕРСИКОВИЙ колір
                for gap_idx, (gap_start, gap_end) in enumerate(confirmed_boundaries):
                    mid_x = int((gap_start + gap_end) // 2)
                    gap_width = gap_end - gap_start
                    
                    # Яскравіші кольори для кращого контрасту
                    colors = [
                        (0, 140, 255),    # яскраво-помаранчевий
                        (255, 100, 100),  # кораловий
                        (0, 200, 150),    # бірюзовий
                        (180, 100, 255),  # фіолетовий
                        (255, 200, 0),    # жовтий
                    ]
                    gap_color = colors[gap_idx % len(colors)]
                    
                    # Товста лінія по центру межі (2px)
                    cv2.line(debug_img, (mid_x, 0), (mid_x, h_pan), gap_color, 2, lineType=cv2.LINE_AA)
                    
                    # Текст з інформацією про gap - ЗБІЛЬШЕНИЙ
                    label = f"G{gap_idx+1}:{gap_width}px"
                    
                    # Отримуємо розмір тексту для правильного позиціонування
                    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # Позиція тексту з відступом
                    text_x = mid_x - text_w // 2
                    text_y = 25 + (gap_idx * 22)  # збільшений крок між рядками
                    
                    # Перевірка щоб не вийшов за межі зліва/справа
                    text_x = max(5, min(text_x, w_pan - text_w - 5))
                    text_y = min(text_y, h_pan - 10)
                    
                    # Фон під текстом (прямокутник) для кращої читаємості
                    cv2.rectangle(debug_img, 
                                (text_x - 4, text_y - text_h - 4), 
                                (text_x + text_w + 4, text_y + baseline + 4), 
                                (0, 0, 0), -1)
                    
                    # Текст з товстою чорною обводкою (3px)
                    cv2.putText(debug_img, label, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, lineType=cv2.LINE_AA)
                    # Білий текст поверх
                    cv2.putText(debug_img, label, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                    
                    # Прямокутник навколо зони межі (2px)
                    cv2.rectangle(debug_img, (gap_start, 0), (gap_end, h_pan), 
                                gap_color, 2, lineType=cv2.LINE_AA)
                    
                    # Напівпрозоре заповнення зони
                    overlay = debug_img.copy()
                    cv2.rectangle(overlay, (gap_start, 0), (gap_end, h_pan), gap_color, -1)
                    cv2.addWeighted(overlay, 0.12, debug_img, 0.88, 0, debug_img)

                    
            # 4. Візуалізація згрупованих речень (Sentence Groups)
            # Використовуємо пастельні кольори, щоб не перекривати текст

            # 5. Профіль чорнила
            if profile is not None and len(profile) == w_pan:
                debug_img = self._draw_ink_overlay(debug_img, profile, gaps, 
                                                   label="PANORAMA", timestamp=timestamp)

            # Збереження зображення
            debug_dir = self.output_dir / "debug" / "ocr_visualization"
            debug_dir.mkdir(parents=True, exist_ok=True)
            time_str = datetime.fromtimestamp(timestamp).strftime('%H%M%S')
            vis_path = debug_dir / f"ocr_vis_frame_{time_str}_{self.frame_counter}.jpg"
            cv2.imwrite(str(vis_path), debug_img)


            # # ==========================================
            # # 🎨 ВІЗУАЛІЗАЦІЯ МЕЖ РЕЧЕНЬ (замінює секцію "сирі R/L блоки")
            # # ==========================================
            # debug_img2 = panorama.copy()
            # h_pan, w_pan = debug_img2.shape[:2]

            # # Створюємо один overlay для всіх меж (оптимізація)
            # overlay = debug_img2.copy()

            # for gap_idx, (gap_start, gap_end) in enumerate(confirmed_boundaries):
            #     gs_int, ge_int = int(gap_start), int(gap_end)
            #     gap_width = ge_int - gs_int
                
            #     # Кольори чергуються для кращої видимості
            #     colors = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 255)]
            #     gap_color = colors[gap_idx % len(colors)]
                
            #     # А) Тонкі вертикальні лінії по краях межі (1px)
            #     cv2.line(debug_img2, (gs_int, 0), (gs_int, h_pan), gap_color, 1)
            #     cv2.line(debug_img2, (ge_int, 0), (ge_int, h_pan), gap_color, 1)
                
            #     # Б) Напівпрозора заливка зони межі (додаємо в overlay)
            #     cv2.rectangle(overlay, (gs_int, 0), (ge_int, h_pan), gap_color, -1)
                
            #     # В) ТОЧНІ КООРДИНАТИ
            #     coord_label = f"{gs_int}-{ge_int} ({gap_width}px)"
                
            #     # Розміщення тексту з обмеженням по висоті панорами
            #     text_x = gs_int + 2
            #     # Чергуємо 2 рівні замість 3, щоб текст не виходив за межі (35px висота)
            #     text_y = 12 + (gap_idx % 2) * 15
            #     text_y = min(text_y, h_pan - 5)  # Обмеження
                
            #     # Текст з чорною обводкою для читабельності
            #     cv2.putText(debug_img2, coord_label, (text_x, text_y),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            #     cv2.putText(debug_img2, coord_label, (text_x, text_y),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, gap_color, 1)

            # # Застосовуємо всі заливки одночасно (20% прозорості)
            # cv2.addWeighted(overlay, 0.2, debug_img2, 0.8, 0, debug_img2)

            # # Збереження
            # debug_dir = self.output_dir / "debug" / "sentence_boundaries"
            # debug_dir.mkdir(parents=True, exist_ok=True)
            # time_str = datetime.fromtimestamp(timestamp).strftime('%H%M%S')
            # vis_path = debug_dir / f"boundaries_frame_{time_str}_{self.frame_counter}.jpg"
            # cv2.imwrite(str(vis_path), debug_img2)

            # ==========================================
            # КОНВЕЄР: позиційна обробка груп (справа → ліво)
            # ==========================================
            # def norm_words(t):
            #     return self.assembler.normalize_arabic_text(t).split()

            def norm_words(t):
               return self._normalize_text(t).split()  # Заміна

            def contains_seq(long_w, short_w):
                if not short_w or len(short_w) > len(long_w):
                    return False
                return any(
                    long_w[i:i + len(short_w)] == short_w
                    for i in range(len(long_w) - len(short_w) + 1)
                )


            def save_now():
                self.assembler.check_message_complete(force_save=True, reason="boundary", current_time=timestamp)
                #self._save_completed_messages()
                return self._save_completed_messages()

            cur = self.assembler.current_message
            cur_w = norm_words(cur.text) if cur else []
            w_pan = panorama.shape[1]

            for group_idx, group in enumerate(sentence_groups):
                group_text = ' '.join(b['text'] for b in group)
                # Межа, яка безпосередньо стосується цієї групи.
                group_boundary = None

                if group_idx < len(group_boundaries):
                    group_boundary = group_boundaries[group_idx]
                elif group_idx > 0:
                                    # Межа, яка розділила поточну групу від попередньої
                    group_boundary = group_boundaries[group_idx - 1] if group_idx > 0 else None
                
                # Очищення тексту (залишаємо твою логіку)
                for pattern in self.excluded_patterns:
                    group_text = group_text.replace(pattern, '')
                group_text = re.sub(r'\s*[-—]\s*', ' ', group_text)
                group_text = re.sub(r'\s+', ' ', group_text).strip()
                group_text = re.sub(r'[٠-٩]', lambda m: str(ord(m.group()) - ord('٠')), group_text)
                
                if len(group_text) < 5:
                    continue

                g_w = norm_words(group_text)
                is_last = (group_idx == len(sentence_groups) - 1)
                # C-DROP: група цілком у зоні дублювання = хвіст минулої панорами
                _ds = self._dup_start_x
                if not is_last and _ds is not None and self._dup_zone_px > 0:
                    gx0 = min(b['x_min'] for b in group)
                    gx1 = max(b['x_max'] for b in group)
                    inside = (gx0 >= _ds - 8) if self.is_rtl else (gx1 <= _ds + 8)
                    if inside:
                        logger.info("🗑️ C-DROP [%.0f..%.0f] dup_start=%.0f: %s",
                                    gx0, gx1, _ds, group_text[:40])
                        continue


                # 🛡️ ПРАВИЛО 1 (Виправлено B1): G доповнює буфер (G ⊇ current).
                # Додаємо 'not is_last', щоб не зберігати парціальну "голову", яка ще росте.
                # Замість прямої мутації тексту, ми скидаємо старий буфер і починаємо з кращої версії.
                if cur_w and not is_last and len(g_w) > len(cur_w) and contains_seq(g_w, cur_w):
                    save_now()  # Скидаємо неповний/гірший буфер
                    self.assembler.add_fragment(TickerFragment(
                        text=group_text, timestamp=timestamp, frame_index=self.frame_counter,
                        confidence=float(np.mean([b['score'] for b in group])),
                        boundaries=[], boundary_confirmed=False,
                        x_min=min(b['x_min'] for b in group), x_max=max(b['x_max'] for b in group)
                    ))
                    cur = self.assembler.current_message
                    cur_w = g_w
                    continue

                # ПРАВИЛО 2 (Виправлено + Захищено): G є хвостом буфера (G ⊆ current).
                # Додаємо мінімальну довжину, щоб уникнути випадкового збігу коротких слів (напр. "في", "من")
                MIN_TAIL_TOKENS = 4
                
                if (cur_w and 
                    len(g_w) >= MIN_TAIL_TOKENS and 
                    len(g_w) < len(cur_w) and 
                    contains_seq(cur_w, g_w)):
                    
                    save_now()
                    cur, cur_w = None, []
                    logger.debug("🗑️ КОНВЕЄР: Відкинуто T-хвіст (надійний збіг >= %d токенів)", MIN_TAIL_TOKENS)
                    continue

                # 🆕 Зміна 2 (продовження): Використовуємо pair_boundary замість confirmed_boundaries[0]
                boundary_position_ratio = None
                boundary_near_edge = False
                
                # if pair_boundary is not None:
                #     boundary_center_x = (pair_boundary[0] + pair_boundary[1]) / 2.0
                #     boundary_position_ratio = round(boundary_center_x / w_pan, 2)
                #     boundary_near_edge = (boundary_position_ratio < 0.15) or (boundary_position_ratio > 0.85)

                if group_boundary is not None:
                    boundary_center_x = (
                        group_boundary[0] + group_boundary[1]
                    ) / 2.0

                    boundary_position_ratio = round(
                        boundary_center_x / w_pan, 2
                    )

                    boundary_near_edge = (
                        boundary_position_ratio < 0.15
                        or boundary_position_ratio > 0.85
    )

                if boundary_near_edge and self.assembler.current_message:
                    self.assembler.current_message.completion_reason = "edge_truncation"
                    self.assembler.current_message.completeness = "incomplete"
                    logger.warning("⚠️ ОБРИВ НА КРАЮ: Межа новини на %.0f%% ширини панорами.", boundary_position_ratio * 100)

                # Створення фрагмента зі всіма твоїми полями для сумісності з dataclass
                fragment = TickerFragment(
                    text=group_text,
                    timestamp=timestamp,
                    frame_index=self.frame_counter,
                    confidence=float(np.mean([b['score'] for b in group])),
                    # boundaries=([list(b) for b in confirmed_boundaries] if group_idx == 0 else []),
                    # boundary_confirmed=(group_idx > 0 and bool(confirmed_boundaries)),
                    boundaries=(
                        [list(group_boundary)]
                        if group_boundary is not None
                        else []
                    ),

                    boundary_confirmed=(group_boundary is not None),
                    x_min=min(b['x_min'] for b in group),
                    x_max=max(b['x_max'] for b in group),
                    boundary_position_ratio=boundary_position_ratio,
                    boundary_near_edge=boundary_near_edge,
                    tail_of_saved=(group_idx == 0),
                    left_incomplete=is_last,
                )

                #ПРАВИЛО 5 (H): ліва група = голова новини → у буфер.
                # ВАЖЛИВО: відмова асемблера НЕ означає, що текст у буфері.
                if is_last:
                    success, reason = self.assembler.add_fragment(fragment)

                    DUP_REASONS = {
                        "duplicate", "duplicate_of_saved", "duplicate_history",
                        "no_novelty", "dropped_no_novelty"}

                    if not success and reason not in DUP_REASONS:
                        logger.warning(
                            "⚠️ H відхилено (%s) → flush старого + retry: %s...",
                            reason, group_text[:40]
                        )
                        save_now()                      # зберігаємо/закриваємо старий буфер
                        success, reason = self.assembler.add_fragment(fragment)
                        if not success:
                            logger.error("❌ H не прийнято навіть після flush: %s", reason)

                    cur = self.assembler.current_message
                    cur_w = norm_words(cur.text) if cur else []
                    logger.info("🚂 H-початок у буфер (статус: %s): %s...", reason, group_text[:40])
                    continue

                # ПРАВИЛО 6: середня група = повна новина (F).
                # Якщо boundary_confirmed=True, асемблер може відхилити фрагмент через наявний current.
                # У такому разі робимо flush + retry, щоб текст не загубився.
                # success, reason = self.assembler.add_fragment(fragment)
                # if not success:
                #     logger.debug("⚠️ КОНВЕЄР: F-група відхилена (%s), виконуємо flush + retry", reason)
                #     save_now()
                #     self.assembler.add_fragment(fragment)
                
                # saved_n = save_now()                #отримуємо кількість збережених
                # cur, cur_w = None, []

                success, reason = self.assembler.add_fragment(fragment)

                if not success:
                    logger.warning(
                        "🟥 КОНВЕЄР: F-група відхилена (%s), виконуємо flush + retry",
                        reason
                    )
                    save_now()
                    success, reason = self.assembler.add_fragment(fragment)
                    if not success:
                        logger.error(
                            "❌ F-група не прийнята навіть після flush: %s | текст: %s",
                            reason,
                            group_text[:80]
                        )
                        cur = self.assembler.current_message
                        cur_w = norm_words(cur.text) if cur else []
                        continue

                saved_n = save_now()
                cur, cur_w = None, []

                logger.info(
                            "КОНВЕЄР: F-новина %s: %s...",
                            "збережена" if saved_n else "відкинута фільтрами",
                            group_text[:40])

            # Фінальна перевірка за таймаутом
            if self.assembler.check_message_complete(timeout_seconds=9.0, current_time=timestamp, reason="timeout"):
                self._save_completed_messages()

            assembler_stats = self.assembler.get_statistics()
            # logger.info(
            #     f"⬜ АСЕМБЛЕР: збережено={assembler_stats.get('messages_completed', 0)}, "
            #     f"дублікатів={assembler_stats.get('dropped_no_novelty', 0)}, "
            #     f"поточна довжина={len(self.assembler.current_message.text) if self.assembler.current_message else 0}"
            # )

            logger.info(
                f"⬜ АСЕМБЛЕР: збережено={assembler_stats.get('messages_emitted', 0)}, "
                f"дублікатів={assembler_stats.get('dropped_no_novelty', 0)}, "
                f"поточна довжина={len(self.assembler.current_message.text) if self.assembler.current_message else 0}"
            )
                
            return True

        except Exception as e:
            logger.exception(f"Критична помилка OCR панорами: {e}")
            return False

    
    def process_frame(self, frame: np.ndarray, timestamp: float, frame_index: int):
        """
        Обробляє кадр, створює панораму і передає її в OCR для розумного склеювання.
        """
        self.frame_counter += 1
        self.frames_processed += 1

               #ОКРЕМИЙ ПАЙПЛАЙН ДЛЯ ВЕРТИКАЛЬНОГО ТІКЕРА
        if getattr(self, 'ticker_motion', None) == "vertical_flip" and getattr(self, 'flip_handler', None):
            return self.flip_handler.process_frame(frame, timestamp, frame_index, self.frame_counter)

        
        # 1. Витягуємо ROI
        ticker_roi = self.extract_ticker_roi(frame)
        if ticker_roi.size == 0:
            return
            
        # 2. Перевірка контрасту
        gray = cv2.cvtColor(ticker_roi, cv2.COLOR_BGR2GRAY)
        if np.std(gray) < 5: 
            return

        # 3. Дебаг-знімок ROI з АНАЛІЗОМ ЧОРНИЛА (ІДЕНТИЧНИЙ до панорами)
        if frame_index % 30 == 0:
            debug_dir = self.output_dir / "debug"
            debug_dir.mkdir(exist_ok=True)
            time_str = datetime.fromtimestamp(timestamp).strftime('%H%M%S')
            debug_path = debug_dir / f"ticker_roi_debug_{time_str}_{frame_index}.jpg"
            try:
                roi_profile, roi_gaps = self._get_ink_profile_and_gaps(ticker_roi)
                debug_roi_img = self._draw_ink_overlay(ticker_roi, roi_profile, roi_gaps, label="ROI FRAME", timestamp=timestamp)
                cv2.imwrite(str(debug_path), debug_roi_img)
            except Exception as e:
                logger.debug(f"Помилка збереження ROI дебагу: {e}")

        # ==========================================
        # 4. 🛡️ РОЗУМНЕ ДОДАВАННЯ В БУФЕР (Фільтр заморожених кадрів)
        # ==========================================
        #fade = self._detect_left_fade(ticker_roi)

        
        # 🆕 КРОК 3.1: Детекція fade з правильного боку залежно від напрямку каналу
        if self.is_rtl:
            fade = self._detect_left_fade(ticker_roi)
        else:
            fade = self._detect_right_fade(ticker_roi)

        # prev_gray = self.frame_buffer[-1].get("gray") if self.frame_buffer else None
        # if prev_gray is not None:
        #     # Порівнюємо ліві 85% ширини: правий край — годинник,
        #     # він тікає щосекунди і не має заважати детекції паузи.
        #     cmp_w = int(gray.shape[1] * 0.85)
        #     diff_mean = float(cv2.absdiff(gray[:, :cmp_w], prev_gray[:, :cmp_w]).mean())
        # else:
        #     diff_mean = 1e9   # буфер порожній → перший кадр приймаємо завжди


        prev_gray = self.frame_buffer[-1].get("gray") if self.frame_buffer else None
        if prev_gray is not None:
            # 🆕 КРОК 4: Універсальна зона порівняння (ігнорує обидва краї: fade та годинник)
            w = gray.shape[1]
            
            # Відступ: гарантовано покриває і fade_zone_px, і можливі годинники/логотипи
            # Мінімум fade_zone_px або 15% ширини кадру (що більше)
            margin = max(self.fade_zone_px, int(w * 0.15))
            
            # Захист від виродження (якщо fade занадто великий і з'їдає весь кадр)
            if margin * 2 < w:
                # Беремо тільки центральну частину кадру
                a = gray[:, margin:-margin]
                b = prev_gray[:, margin:-margin]
            else:
                # Фолбек: якщо краї надто широкі, беремо середні 50%
                a = gray[:, w//4 : 3*w//4]
                b = prev_gray[:, w//4 : 3*w//4]
                
            diff_mean = float(cv2.absdiff(a, b).mean())
        else:
            diff_mean = 1e9   # буфер порожній → перший кадр приймаємо завжди


        if diff_mean < 2.0:
            # ── ⏸️ ПАУЗА: кадр піксельно той самий → у буфер НЕ беремо.
            self._pause_frames = getattr(self, '_pause_frames', 0) + 1

            # Тримаємо таймаут асемблера живим і під час паузи
            if self.assembler.current_message and self._pause_frames % 30 == 0:
                if self.assembler.check_message_complete(timeout_seconds=9.0, current_time=timestamp, reason="timeout"):
                    self._save_completed_messages()

            if self._pause_frames == 3 and not getattr(self, '_pause_reported', False):
                self._pause_reported = True
                # механізм порівняння сусідніх кадрів (cv2.absdiff). Якщо різниця між пікселями поточного і попереднього кадру менша за 2.0, система вважає, що тікер не рухається (завис).
                logger.info("⏸️ ТІКЕР ЗУПИНИВСЯ: %d кадрів без руху (рядок на екрані, але нерухомий)", self._pause_frames)
                self._emit_system_event("ticker_paused", "рядок нерухомий ≥3 кадрів") # Розкоментуй, якщо є цей метод

            if self._pause_frames >= 600:      # ~10 хв повної тиші
                self.frame_buffer.clear()
                self._pause_frames = 0
            return   # ← ВИХІД: НЕ додаємо в буфер, не витрачаємо бюджет "20"

        # ── рух повернувся ──
        if getattr(self, '_pause_frames', 0) > 0:
            #Як тільки з'являється кадр, де текст зрушив хоча б на 1 піксель, система пише ▶️ ТІКЕР ВІДНОВИВ ПРОКРУТКУ і продовжує збирати панораму.
            logger.info("▶️ ТІКЕР ВІДНОВИВ ПРОКРУТКУ (після %d заморожених кадрів)", self._pause_frames)
            self._emit_system_event("ticker_resumed", "рух відновився") # Розкоментуй, якщо є цей метод
        
        self._pause_frames = 0
        self._pause_reported = False

        # Додаємо в буфер ТІЛЬКИ рухомі кадри + зберігаємо gray для наступної перевірки
        self.frame_buffer.append({
            'image': ticker_roi,
            'gray': gray,          # 🆕 потрібно для детекції паузи наступним кадром
            'timestamp': timestamp,
            'fade': fade,
        })

        
        # Діагностика знаку dx — лише раз на 30 кадрів
        if len(self.frame_buffer) >= 2 and self.frame_counter % 30 == 0:
            self._debug_check_dx_sign(
                self.frame_buffer[-2]["image"],
                self.frame_buffer[-1]["image"],
            )



                # 5. Кожні  кадрів будуємо панораму і розпізнаємо
        if len(self.frame_buffer) >= 10:
            self.panorama, shifts = self._create_panorama()

            # 🆕 Швидкість рахуємо на КОЖНІЙ панорамі (Варіант 4)
            roi_width = self.frame_buffer[0]["image"].shape[1]
            speed_stats = self._compute_speed_from_shifts(shifts, roi_width)
            if speed_stats:
                self.speed_stats = speed_stats
                self._log_speed_metric(timestamp, speed_stats)   # Варіант 1

                # 🆕 sanity-check як захист від хибного зшивання
                if not speed_stats.get("sane", True):
                    logger.warning(
                        "Панораму відкинуто: швидкість поза фізичними межами"
                    )
                    self.panorama = None

            # if self.panorama is not None:
            #     self._extract_and_ocr_sentence(self.panorama, timestamp)

            if self.panorama is not None:
                # 🆕 АНАЛІЗ ЧОРНИХ РЕГІОНІВ ПЕРЕД OCR
                black_location = self._analyze_black_region_location(self.panorama)
                
                if black_location != "none":
                    logger.warning(
                        "🟩 ЧОРНА ЗОНА НА ПАНОРАМІ: %s | Ширина панорами: %dpx", 
                        black_location, self.panorama.shape[1]
                    )
                    
                    # Якщо це стик, можемо навіть зберегти панораму окремо для дебагу
                    if black_location.startswith("seam_at_"):
                        debug_dir = self.output_dir / "debug" / "black_seams"
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        time_str = datetime.fromtimestamp(timestamp).strftime('%H%M%S')
                        cv2.imwrite(
                            str(debug_dir / f"seam_{time_str}_{self.frame_counter}.jpg"), 
                            self.panorama
                        )

                # Передаємо в OCR тільки якщо немає критичних чорних смуг посередині
                # (Алгоритм OCR все одно щось прочитає, але це засмітить лог)
                self._extract_and_ocr_sentence(self.panorama, timestamp)

            # # Очищаємо буфер, залишаючи останній кадр для плавного переходу
            # last_item = self.frame_buffer[-1]
            # self.frame_buffer.clear()
            # self.frame_buffer.append(last_item)

            # ═══ В3в: після розриву тримаємо кадри З МОМЕНТУ розриву —
            # початок нового контенту не губиться
            resume_idx = getattr(self, "_resume_from", None)
            self._resume_from = None
            
            # Якщо був розрив — зберігаємо кадри з індексу розриву до кінця
            # Якщо розриву не було — зберігаємо тільки останній кадр (як раніше)
            if resume_idx is not None and 0 < resume_idx < len(self.frame_buffer):
                keep = list(self.frame_buffer)[resume_idx:]
                logger.debug(f"🔄 RESUME: зберігаємо {len(keep)} кадрів з індексу {resume_idx}")
            else:
                keep = [self.frame_buffer[-1]]
            
            self.frame_buffer.clear()
            for fr in keep:
                self.frame_buffer.append(fr)



    @staticmethod
    def _lev1(a: str, b: str) -> bool:
        """Рівність слів з точністю до 1 редагування (OCR-шум:
        الحوثي/الحوتي, النار/لنار, اميرك/أميركا)."""
        if a == b:
            return True
        la, lb = len(a), len(b)
        if abs(la - lb) > 1:
            return False
        if la == lb:
            return sum(x != y for x, y in zip(a, b)) <= 1
        if la > lb:
            a, b = b, a
        # b довше на 1: перевірка однієї вставки
        i = j = diff = 0
        while i < len(a) and j < len(b):
            if a[i] == b[j]:
                i += 1
            else:
                diff += 1
                if diff > 1:
                    return False
            j += 1
        return True
        

    def _fuzzy_eq(self, a: str, b: str) -> bool:
        """✅ FIX: lev1 тільки для слів ≥4 літер. Короткі (ve/de/da, في/من)
        інакше збігаються між собою і дають хибні дублікати."""
        if a == b:
            return True
        if min(len(a), len(b)) < 4:
            return False
        return self._lev1(a, b)

    def _normalize_text(self, text: str) -> str:
        """
        🆕 КРОК 6.1: Універсальна нормалізація тексту залежно від мови.
        Замінює прямі виклики до normalize_arabic_text.
        """
        if self.ocr_lang == 'ar':
            return self.assembler.normalize_arabic_text(text)
        elif self.ocr_lang == 'tr':
            return self._normalize_turkish_text(text)
        else:
            return text.strip()

    @staticmethod
    def _tr_lower(s: str) -> str:
        """✅ FIX: турецький lower. str.lower() дає 'İ'→'i̇' (2 символи)."""
        return s.replace('I', 'ı').replace('İ', 'i').lower()

    def _normalize_turkish_text(self, text: str) -> str:
        """
        Нормалізація турецького тексту.
        Виправляє типові OCR-помилки PaddleOCR для турецької мови.
        """
        if not text:
            return ""
        
        # 1. Базова очистка
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.replace('_', ' ')
            # ✅ FIX: уніфікація апострофів (İstanbul'da / İstanbul'da)
        for ap in (''', ''', 'ʼ', '`', '´'):
            text = text.replace(ap, "'")
        
        # 2. 🆕 ВИПРАВЛЕННЯ PADDLEOCR ДІАКРИТИКИ (Турецька)
        # PaddleOCR часто замінює турецькі літери на латинські з акутами/макронами
        replacements = {
            # i з крапкою (İ) — найчастіша помилка
            'Í': 'İ', 'í': 'i',
            'Î': 'İ', 'î': 'i',
            # Ö
            'Ô': 'Ö', 'ô': 'ö',
            'Ó': 'Ö', 'ó': 'ö',
            # Ü
            'Û': 'U', 'û': 'u',
            'Ú': 'Ü', 'ú': 'ü',
            # Ş
            'Š': 'Ş', 'š': 'ş',
            'Ś': 'Ş', 'ś': 'ş',
            # Ç
            'Č': 'Ç', 'č': 'ç',
            'Ć': 'Ç', 'ć': 'ç',
            # Ğ (часто втрачається)
            'Ğ': 'Ğ', 'ğ': 'ğ',
            'Ĝ': 'Ğ', 'ĝ': 'ğ',
            # Інші артефакти
            'Â': 'A', 'â': 'a',
            'Ê': 'E', 'ê': 'e',
            'Ù': 'U', 'ù': 'u',
            'Ñ': 'N', 'ñ': 'n',
        }
        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)
        
        # 3.  Видалення специфічних OCR-символів (галюцинації PaddleOCR)
        # ⑨, ① — OCR часто плутає цифри в колах з літерами
        # π — OCR плутає з n або m
        # △ — OCR генерує замість пробілів або розділювачів
        # © — OCR плутає з O або 0
        # ✅ FIX: π/© окремо — сміття; всередині слова — спотворена літера
        text = re.sub(r'[①②③④⑤⑥⑦⑧⑨⑩®™]', '', text)
        text = text.replace('△', ' ')
        text = re.sub(r'(?<!\w)[π©](?!\w)', ' ', text)
        text = re.sub(r'(?<=\d)©|©(?=\d)', '0', text)
        text = text.replace('©', 'O').replace('π', 'n')
        
        # 4. 🆕 Видалення арабських/інді цифр (якщо раптом потрапили)
        text = re.sub(r'[٠-٩]', lambda m: str(ord(m.group()) - ord('٠')), text)
        # ✅ FIX: додано Extended Arabic-Indic (۰-۹)
        text = re.sub(r'[۰-۹]', lambda m: str(ord(m.group()) - ord('۰')), text)
        
        # 5. 🆕 Очищення від типових OCR-сміттєвих патернів

        # Стало (без IGNORECASE, без реальних слів):
        text = re.sub(r"\b(?:ww|wmu|ahter|aer|ese|en shpbor uns|thabét m|Ceniz|Coyarba|Eapa|Mer'k|Mer't|GEGL|SEGL|CECL|CECT|cres|CESr|cEer|crer)\b",
              ' ', text)
        
        # 6. Фінальна очистка
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _is_target_lang_char(self, ch: str) -> bool:
        """
        🆕 КРОК 6.4: Перевіряє, чи символ належить до цільової мови.
        Замінює жорсткі фільтри арабських літер в E2.
        """
        if self.ocr_lang == 'ar':
            return "\u0600" <= ch <= "\u06FF"  # Арабський блок
        elif self.ocr_lang == 'tr':
            # Латинські літери (включаючи турецькі діакритики: ğ, ş, ı, İ, ç, ö, ü)
            return ch.isalpha()
        else:
            return ch.isalpha()

    def _fuzzy_contains_seq(self, hay: List[str], needle: List[str]) -> bool:
        """needle — контигуальний run у hay; допускається ≤1 «фuzzy»-слово
        (інакше будь-які два речення збігаються через спільні прийменники)."""
        n = len(needle)
        if n == 0 or n > len(hay):
            return False
        for i in range(len(hay) - n + 1):
            fuzzy = 0
            ok = True
            for a, b in zip(hay[i:i + n], needle):
                if a != b:
                    if fuzzy == 1 or not self._fuzzy_eq(a, b):
                        ok = False
                        break
                    fuzzy = 1
            if ok:
                return True
        return False

    def _fuzzy_word_overlap(self, a: List[str], b: List[str]) -> float:
        """Частка спільних слів від КОРОТШОГО списку (fuzzy).
        Ловить редакційні варіанти однієї новини, які не є вкладеними."""
        if not a or not b:
            return 0.0
        short, long_ = (a, b) if len(a) <= len(b) else (b, a)
        used = [False] * len(long_)
        hit = 0
        for w in short:
            for j, v in enumerate(long_):
                if not used[j] and self._fuzzy_eq(w, v):
                    used[j] = True
                    hit += 1
                    break
        return hit / len(short)

    def _check_emission_duplicate(self, text: str) -> Optional[str]:
        """
        П5: порівнює кандидат з останніми емісіями (нормалізовані слова,
        fuzzy ≤1 на слово — щоб الحوثي/الحوتи не розбивало детекцію).
        Повертає причину відсіювання або None (пропустити).
        """
        #words = self.assembler.normalize_arabic_text(text).split()
        words = self._normalize_text(text).split()  # ✅
        if len(words) < 3:
            return None                      # надкоротке ловить Фільтр 1
        for saved in list(self.emitted_norm)[-self.EMITTED_KEEP:]:
            sw = saved.split()
            if not sw:
                continue
            # 1) кандидат повністю лежить у вже емітованому → хвіст/дубль
            if self._fuzzy_contains_seq(sw, words):
                return "subset_of_emitted"
            # 2) вже емітоване лежить у кандидаті:
            #    довший на ≥3 слова → це ПОВНІША версія, пропускаємо (емітимо);
            #    інакше → повтор → відсіяти

            if self._fuzzy_contains_seq(words, sw):
                if len(words) < len(sw) + 3:
                    return "near_repeat"
                # ✅ FIX: було return None — свіжіші емісії не перевірялись
                continue
            # 3) редакційний варіант тієї ж новини (не вкладений):
            #    ≥65% спільних слів від коротшого і кандидат не довший
            if (len(words) <= len(sw)
                    and self._fuzzy_word_overlap(words, sw) >= 0.65):
                return "word_overlap_duplicate"
        return None


    def _emit_system_event(self, event: str, note: str = ""):
        """Системна подія у ticker_news.jsonl — те саме місце, що й новини.

        Внизу розрізняй за полем "type": у подій "system_event", у звичайних
        новин цього поля немає — сумісність зі споживачами не ламається.
        """
        try:
            record = {
                "type": "system_event",
                "event": event,   # ticker_paused / ticker_resumed / content_changed
                "note": note,
                "channel": self.channel_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "frame": self.frame_counter,
            }
            output_file = self.output_dir / "ticker_news.jsonl"
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.debug(f"Не вдалося записати system_event: {e}")

    
    def _save_completed_messages(self):
        """Зберігає завершені повідомлення."""
        messages = self.assembler.get_completed_messages()

        if not messages:
            logger.debug("🟥 [_save_completed_messages] Немає повідомлень для збереження")
            return 0  # повертаємо кількість збережених

        output_file = self.output_dir / "ticker_news.jsonl"

        saved_n = 0                     # лічильник успішних збережень

        try:
            with open(output_file, 'a', encoding='utf-8') as f:
                for msg in messages:
                    raw_text = msg.text or ""          # 🔧 П5: ДО .strip()!
                    clean_text = raw_text.strip()
                    if not clean_text:
                        continue

                    # 🔧 П5: підтверджені межі — з урахуванням обох імен поля
                    cb = getattr(msg, 'confirmed_boundaries', None)
                    if cb is None:
                        cb = getattr(msg, 'boundaries', None) or []

                    # 🛡️ ФІЛЬТР 1 (🔧 П5): короткий уламок легітимний ЛИШЕ з
                    # підтвердженою межею. Поріг 20 знаків: чистих арабських
                    # новин коротше немає, а уламки прогону були 10–18 знаків.
                    if len(clean_text) < 20 and not cb:
                        logger.debug(
                            f"🗑️ Відкинуто короткий уламок "
                            f"({len(clean_text)} зн., меж нема): '{clean_text}'"
                        )
                        continue

                    # 🛡️ ФІЛЬТР 2: детекція обривів слів (початок або кінець)
                    # 🆕 КРОК 6.2: Мовно-залежний список коротких слів
                    is_truncated = False
                    truncation_reason = None
                    words = clean_text.split()
                    if len(words) > 0:
                        first_word = words[0]
                        last_word = words[-1]
                        
                        # 🆕 КРОК 6.2: Список коротких слів залежить від мови
                        if self.ocr_lang == 'tr':
                            # Турецькі сполучники/прийменники (дуже короткі слова — норма)
                            valid_short_words = {'ve', 'ile', 'için', 'ama', 'fakat', 'ki', 
                                                'de', 'da', 'mi', 'mı', 'mu', 'mü', 
                                                'su', 'at', 'et', 'ne', 'en', 'o'}
                        else:
                            # Арабські прийменники
                            valid_short_words = {'و', 'في', 'من', 'إلى', 'عن', 'على', 
                                                'لا', 'لم', 'لن', 'هو', 'هي', 'ال'}
                        def _is_num(w_: str) -> bool:
                            return bool(w_) and all(c in "0123456789٠١٢٣٤٥٦٧٨٩" for c in w_)
                        if len(first_word) < 3 and not _is_num(first_word) and first_word not in valid_short_words:
                            is_truncated = True
                            truncation_reason = "broken_start_word"
                        elif len(last_word) < 3 and not _is_num(last_word) and last_word not in valid_short_words:
                            is_truncated = True
                            truncation_reason = "broken_end_word"

                        # 🔧 П5: по СИРОМУ тексті — після .strip() хвостового
                        # пробілу немає і цей regex був мертвий
                        elif re.search(r'(و\s+$|في\s+$|من\s+$|إلى\s+$|ال\s+$|بـ\s+$)', raw_text):
                            is_truncated = True
                            truncation_reason = "ends_with_preposition"

                    if is_truncated:
                        logger.warning(
                            "🟥 ВІДКИДАЄМО ОБРИВОК: reason=%s | Текст: %r",
                            truncation_reason,
                            clean_text[-40:]
                        )
                        continue

                    # Узгодження прапорців обрізання (виправлення суперечності в даних)
                    # Якщо асемблер позначив це як обрізання краєм панорами, 
                    # ми примусово вмикаємо прапорці для коректного відображення у дашборді
                    if getattr(msg, "completion_reason", "") == "edge_truncation":
                        is_truncated = True
                        truncation_reason = truncation_reason or "edge"

                    # 🛡️ ФІЛЬТР 3: явні склейки двох новин
                    # if clean_text.count("مصادر الحدث") > 1 or clean_text.count("رويترز") > 1:
                    #     logger.debug(f"🗑️ Відкинуто дубльовану склейку: '{clean_text[:50]}...'")
                    #     continue

                    # 🛡️ ФІЛЬТР 3: явні склейки двох новин
                    # 🆕 КРОК 6.3: Мовно-залежні маркери каналу
                    is_glued = False
                    if self.ocr_lang == 'tr':
                        # Турецькі маркери каналу
                        tr_markers = ["Son dakika", "TRT Haber", "Habertürk", "Anadolu Ajansı", "AA"]
                        for marker in tr_markers:
                            if clean_text.count(marker) > 1:
                                is_glued = True
                                break
                    else:
                        # Арабські маркери каналу
                        ar_markers = ["مصادر الحدث", "رويترز", "العربية", "الجزيرة"]
                        for marker in ar_markers:
                            if clean_text.count(marker) > 1:
                                is_glued = True
                                break

                    if is_glued:
                        logger.debug(f"🗑️ Відкинуто дубльовану склейку: '{clean_text[:50]}...'")
                        continue

                    # 🛡️ ФІЛЬТР 4 (🔧 П5): нечіткі дублі проти історії емісій
                    drop_reason = self._check_emission_duplicate(clean_text)
                    if drop_reason:
                        logger.info(
                            "♻️ ВІДКИНЕНО як дубль (%s): %s",
                            drop_reason, clean_text[:60]
                        )
                        continue

                    # ==========================================
                    # ЯКЩО КОД ДОЙШОВ СЮДИ, ПОВІДОМЛЕННЯ ЯКІСНЕ
                    # ==========================================
                    self.ticker_count += 1
                    logger.info(f"\n🟢 TICKER #{self.ticker_count}:")
                    logger.info(f"   {clean_text}")

                    record = {
                        "message_id": msg.message_id,
                        "channel": msg.channel,
                        "text": msg.text,
                        "started_at": msg.started_at,
                        "completed_at": msg.completed_at,
                        "completeness": msg.completeness,
                        "ocr_confidence": msg.ocr_confidence,
                        "is_truncated": is_truncated,
                        "truncation_reason": truncation_reason,
                        "fragment_count": len(msg.fragments),
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "confirmed_boundaries": list(cb),
                        "boundary_frames": getattr(msg, "boundary_frames", []),
                        "completion_reason": getattr(msg, "completion_reason", "unknown"),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    saved_n += 1                    # інкрементуємо лічильник
                     
                    #TGT: скидаємо лічильники для ключів, пов'язаних зі
                    # збереженим текстом.
                    # 🆕v2: БЕЗ norm_words — вона вкладена в
                    # _extract_and_ocr_sentence і тут недоступна (уникнення NameError).
                    #saved_words = self.assembler.normalize_arabic_text(clean_text).split()
                    saved_words = self._normalize_text(clean_text).split()  # ✅
                    if saved_words:
                        probe = saved_words[:3] + saved_words[-3:]
                        keys_to_remove = [k for k in list(self._gap_history)
                                          if any(w and w in k for w in probe)]
                        for k in keys_to_remove:
                            del self._gap_history[k]


                    # 🔧 П5: пам'ятаємо емісію для Фільтра 4
                    try:
                        #norm = self.assembler.normalize_arabic_text(clean_text)
                        norm = self._normalize_text(clean_text)
                        self.emitted_norm.append(norm)
                        if len(self.emitted_norm) > self.EMITTED_KEEP * 3:
                            del self.emitted_norm[:-self.EMITTED_KEEP * 2]
                    except Exception:
                        pass

        except Exception as e:
            logger.error(f" [_save_completed_messages] Помилка збереження: {e}", exc_info=True)
        return saved_n 


        
    def get_statistics(self) -> Dict:
        """Повертає статистику сервісу."""
        assembler_stats = self.assembler.get_statistics()
        
        return {
            "channel": self.channel_id,
            "frames_processed": self.frames_processed,
            "ocr_calls": self.ocr_calls,
            "messages_detected": self.messages_detected,
            "ocr_calls_per_frame": self.ocr_calls / self.frames_processed if self.frames_processed > 0 else 0,
            "ticker_speed": self.speed_stats or None,
            **assembler_stats
        }
    

def main():
    """Тестовий запуск сервісу."""
    logging.basicConfig(
        level=logging.INFO, # Можна змінити на DEBUG, якщо треба, але з фільтрами нижче
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )

       # ВИМИКАЄМО СПАМ ВІД СТОРОННІХ БІБЛІОТЕК
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("paddle").setLevel(logging.WARNING) # PaddleOCR теж іноді любить спамити

    
    # Ініціалізація
    profile_manager = ChannelProfileManager("channel_profiles.json")
    service = TickerService(channel_id="alhadath", profile_manager=profile_manager, output_dir="/tmp/ticker_output")
    service.init_ocr()
    logger.info(" Ticker Service готовий до роботи")

    
    try:
        while True:
            # Імітація обробки кадрів
            time.sleep(1.0)
            
            stats = service.get_statistics()
            if stats['frames_processed'] % 60 == 0:
                logger.info(f"📊 Статистика: {stats}")
    

    except KeyboardInterrupt:
        logger.info("Зупинка сервісу")

        # 1. Перевіряємо, чи залишилася незавершена новина
        current = service.assembler.current_message

        if current is not None:
            logger.info(
                "Незавершене повідомлення перед зупинкою: %s",
                current.text
            )

        # 2. Примусово завершуємо поточне повідомлення
        service.assembler.check_message_complete(force_save=True)

        # 3. Зберігаємо всі завершені повідомлення
        service._save_completed_messages()


if __name__ == "__main__":
    import argparse
    import sys

    # Налаштування базового логування для консолі
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser(description="Ticker Service — витягування біжучого рядка")
    
    # 🆕 Обов'язковий аргумент каналу
    parser.add_argument(
        '--channel', 
        type=str, 
        required=True, 
        help="Назва каналу (наприклад: al_hadath, al_arabiya, cnn_arabic)"
    )
    
    # Додаткові аргументи (можете налаштувати під свої шляхи)
    parser.add_argument(
        '--config', 
        type=str, 
        default="channels_config.json", 
        help="Шлях до JSON-файлу з профілями каналів"
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default="./news_output", 
        help="Базова директорія для збереження JSON-логів"
    )
    
    args = parser.parse_args()

    try:
        # 1. Ініціалізуємо менеджер профілів
        profile_manager = ChannelProfileManager(args.config)
        
        # 2. Створюємо екземпляр сервісу для конкретного каналу
        logger.info(f"🚀 Запуск Ticker Service для каналу: {args.channel}")
        service = TickerService(
            channel_id=args.channel,
            profile_manager=profile_manager,
            output_dir=args.output
        )
        
        # 3. Ініціалізуємо OCR (якщо потрібно)
        service.init_ocr()
        
        # 4. Запускаємо основний цикл обробки (замініть на ваш реальний метод запуску)
        # service.run() 
        
    except ValueError as e:
        logger.error(f"❌ Помилка запуску: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("❌ Критична помилка в Ticker Service")
        sys.exit(1)