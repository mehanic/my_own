"""
ticker_flip.py
Обробник вертикального дискретного тікера (FLIP-режим).
Використовується для каналів, де текст виїжджає знизу, зупиняється для читання, і потім їде вгору.
"""

from datetime import datetime
import json
import os

import cv2
import numpy as np
import re
import logging
from typing import List, Dict, Optional, Callable, Tuple

logger = logging.getLogger(__name__)

try:
    from ticker_assembler import TickerFragment
except ImportError:
    TickerFragment = None
    logger.warning("⚠️ ticker_flip: TickerFragment не знайдено, емісія буде неможливою")


class FlipTickerHandler:
    def __init__(self, parent_service, profile, assembler, excluded_patterns, save_callback):
        # Зберігаємо посилання на батьківський сервіс для доступу до спільних ресурсів
        self.parent_service = parent_service 
        
        self.profile = profile
        self.assembler = assembler
        self.excluded_patterns = excluded_patterns
        self.save_callback = save_callback

        # 🆕 Ініціалізація всіх параметрів з профілю
        self.diff_threshold = float(profile.get('flip_diff_threshold', 0.03))
        self.confirm_frames = int(profile.get('flip_confirm_frames', 15))
        self.min_dwell_sec = float(profile.get('flip_min_dwell_sec', 0.8))
        self.max_episode_frames = int(profile.get('flip_max_episode_frames', 80))
        self.ocr_conf_min = float(profile.get('flip_ocr_conf_min', 0.6))
        self.clock_mask_px = int(profile.get('flip_clock_mask_px', 0))
        
        # Стан машини
        self.state = "MOTION"
        self.static_run = 0
        self.consecutive_motion_frames = 0
        self.prev_roi = None    # 🆕 Додано
        self.prev_gray = None
        self.episode = []
        self.emitted = False
        self.last_norm = ""
        
        # Статистика
        self.ocr_calls = 0
        self._last_brightness = 0.0
        self._last_check_timestamp = 0.0  #  ДОДАНО
        self._last_text_hash = ""  #  ДОДАНО (для хеша тексту)

        logger.info("🔄 FLIP-РЕЖИМ ініціалізовано: вертикальний дискретний тікер (вхід знизу → пауза → виїзд вгору)")

    @staticmethod
    def _sharpness(gray: np.ndarray) -> float:
        """Різкість кадру (варіація Лапласіана) — для вибору найкращого кадру в епізоді."""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())


    def _draw_ocr_visualization(self, bgr_img: np.ndarray, blocks: List[dict], timestamp: float, frame_index: int):
        """Малює рамки OCR-блоків та зберігає зображення для дебагу (аналог горизонтального пайплайну)."""
        debug_img = bgr_img.copy()
        h, w = debug_img.shape[:2]
        
        # 1. Малюємо рамки навколо блоків
        for idx, block in enumerate(blocks):
            box = block["box"]
            pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
            
            # Кольори: перший - зелений, останній - червоний, інші - жовтий
            if idx == 0:
                color = (100, 255, 100)
            elif idx == len(blocks) - 1:
                color = (100, 100, 255)
            else:
                color = (255, 255, 150)
                
            cv2.polylines(debug_img, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
            
            # Підпис блоку
            label = f"{idx}:{block['text'][:12]}"
            cv2.putText(debug_img, label, (int(block['x_min']), int(pts[0][0][1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, lineType=cv2.LINE_AA)

        # 2. Збереження зображення
        debug_dir = self.parent_service.output_dir / "debug" / "ocr_visualization"
        debug_dir.mkdir(parents=True, exist_ok=True)
        time_str = datetime.fromtimestamp(timestamp).strftime('%H%M%S')
        vis_path = debug_dir / f"ocr_vis_f{frame_index}_{time_str}.jpg"
        
        cv2.imwrite(str(vis_path), debug_img)
        logger.debug(f"💾 Збережено візуалізацію OCR: {vis_path.name}")


    def _log_flip_coordinates(self, blocks: List[dict], frame_index: int, timestamp: float):
        """Логує координати кожного OCR-блоку вертикального тікера у окремий JSONL файл."""
        # 1. Якщо блоків немає, виходимо одразу
        if not blocks:
            return
            
        # 2. Формуємо ім'я файлу (таке саме, як у coordinates_tap, щоб дані були в одному місці)
        channel_safe_name = str(self.parent_service.channel_id).replace(" ", "").replace("_", "").lower()
        coords_file = self.parent_service.output_dir / f"coordinates_{channel_safe_name}.jsonl"
        
        try:
            with open(coords_file, 'a', encoding='utf-8') as f:
                for idx, b in enumerate(blocks):
                    # 3. Безпечний розрахунок координат (захист від від'ємних або нульових значень)
                    y_min = float(b["box"][:, 1].min())
                    y_max = float(b["box"][:, 1].max())
                    x_min = float(b["x_min"])
                    x_max = float(b["x_max"])
                    
                    w = max(1.0, x_max - x_min)
                    h = max(1.0, y_max - y_min)
                    
                    record = {
                        "seq": idx + 1,
                        "ts": datetime.fromtimestamp(timestamp).isoformat(),
                        "ts_epoch": float(timestamp),
                        "pid": os.getpid(),
                        "panorama_seq": 0,
                        "type": "block",
                        "text": b["text"],
                        "x_start": round(x_min, 1),
                        "w": round(w, 1),
                        "y_start": round(y_min, 1),
                        "h": round(h, 1),
                        "score": round(b["score"], 2),
                        "decision": "STRONG",
                        "frame_index": int(frame_index)  # Явне перетворення в int
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            # 4. Лог успіху (ви побачите це в консолі, якщо увімкнете DEBUG, або зможете додати INFO)
            logger.info(f"💾 Записано {len(blocks)} блоків координат у {coords_file.name}")
            
        except Exception as e:
            # 5. КРИТИЧНО: logger.error, щоб ви точно побачили помилку, якщо щось піде не так
            logger.error(f"Помилка запису координат FLIP у {coords_file}: {e}", exc_info=True)

    def _detect_text_color(self, bgr_roi: np.ndarray) -> str:
        """
        Визначає колір тексту в ROI.
        Повертає: 'yellow' (жовтий), 'white' (білий), або 'unknown'
        """
        hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
        
        #  ВИПРАВЛЕНО: Маска для ВСІХ яскравих пікселів (Hue 0-179, Saturation 0-255, Value 200-255)
        lower_brightness = np.array([0, 0, 200])
        upper_brightness = np.array([179, 255, 255]) 
        mask = cv2.inRange(hsv, lower_brightness, upper_brightness)
        
        if np.count_nonzero(mask) < 100:
            return 'unknown'
        
        h_values = hsv[mask > 0, 0]
        s_values = hsv[mask > 0, 1]
        
        if len(h_values) == 0:
            return 'unknown'
        
        median_hue = np.median(h_values)
        median_saturation = np.median(s_values)
        
        # Логіка визначення:
        if median_saturation < 30:
            return 'white'
        elif 20 <= median_hue <= 45 and median_saturation > 40: #  Розширили діапазон для надійності
            return 'yellow'
        else:
            return 'white'

    def _vertical_shift(self, g1: np.ndarray, g2: np.ndarray) -> float:
        """Обчислює вертикальний зсув (dy) між кадрами через phaseCorrelate."""
        b1 = cv2.GaussianBlur(g1, (3, 3), 0)
        b2 = cv2.GaussianBlur(g2, (3, 3), 0)
        win = cv2.createHanningWindow(b1.shape[::-1], cv2.CV_32F)
        try:
            (dx, dy), resp = cv2.phaseCorrelate(np.float32(b1), np.float32(b2), win)
        except cv2.error:
            return 0.0
        return float(dy) if resp > 0.05 else 0.0

    def _calculate_frame_diff(self, gray1: np.ndarray, gray2: np.ndarray) -> float:
        """Повертає відсоток змінених пікселів між двома сірими кадрами (0.0 до 1.0)."""
        if gray1 is None or gray2 is None or gray1.shape != gray2.shape:
            return 1.0
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        return float(np.count_nonzero(thresh)) / float(thresh.size)

    def process_frame(self, frame: np.ndarray, timestamp: float, frame_index: int, frame_counter: int) -> bool:
        """Основний метод обробки кадру для вертикального тікера."""
        # 1. Отримуємо ROI від батьківського сервісу
        roi = self.parent_service.extract_ticker_roi(frame)
        if roi.size == 0:
            return False

        current_brightness = float(np.median(roi))
        logger.debug(f"📊 ROI STATS: shape={roi.shape}, brightness={current_brightness:.1f}")
        
        # Маскування годинника (робимо це рано, щоб не заважав аналізу)
        if self.clock_mask_px > 0:
            roi = roi[:, :-self.clock_mask_px]
            
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Зберігаємо попередній стан
        prev_roi, prev_gray = self.prev_roi, self.prev_gray
        self.prev_roi = roi.copy()
        self.prev_gray = gray.copy()
        
        if prev_roi is None or prev_roi.shape != roi.shape:
            self.state = "MOTION"
            return False

        # 2. Обчислюємо метрики руху (ТЕПЕР diff доступний для всіх наступних перевірок!)
        diff = self._calculate_frame_diff(prev_gray, gray)  
        dy = self._vertical_shift(prev_gray, gray)

        # 🆕 ЛОГІКА ВИЯВЛЕННЯ НОВОГО ПОВІДОМЛЕННЯ
        if self.emitted:
            brightness_change = abs(current_brightness - self._last_brightness)

            # 3. ВИЯВЛЕННЯ ЗМІНИ ТЕКСТУ ЧЕРЕЗ РУХ (diff тепер визначений!)
            if not hasattr(self, '_last_diff_check'):
                self._last_diff_check = timestamp
            
            time_since_check = timestamp - self._last_diff_check
            if diff > 0.15 and time_since_check > 1.0:
                self._last_diff_check = timestamp
                logger.info(f"🔄 ВИЯВЛЕНО ЗМІНУ ТЕКСТУ: diff={diff:.3f} (текст рухається/змінився)")
                self.emitted = False
                self.static_run = 0
                self.episode = []
                self.state = "MOTION"
            
            # 1. Перевірка за яскравістю
            if brightness_change > 8.0:
                logger.info(f"🆕 ВИЯВЛЕНО НОВУ СМУГУ: яскравість змінилася з {self._last_brightness:.1f} на {current_brightness:.1f}")
                self.emitted = False
                self.static_run = 0
                self.episode = []
                self.state = "MOTION"
            
            # 2. ПРИМУСОВА ПЕРЕВІРКА КОЖНІ 2 СЕКУНДИ
            elif (timestamp - self._last_check_timestamp) > 2.0:
                self._last_check_timestamp = timestamp
                logger.info(f"⏰ ПРИМУСОВА ПЕРЕВІРКА: прошло 2 секунди, emitted={self.emitted}")
                self.emitted = False
                self.static_run = 0
                self.episode = []
                self.state = "STATIC"



        self._last_brightness = current_brightness
        
        #  Умова руху: вертикальний зсув ВАЖЛИВІШИЙ за загальну зміну
        # dy > 3.5 — це реальний рух тексту вгору/вниз
        # diff > threshold — це може бути зміна фону/кольору
        moving = (abs(dy) > 3.5) or (diff > self.diff_threshold and abs(dy) > 1.0)

        # 3. Машина станів з DEBOUNCE (захист від одиночних шумових кадрів)
        if moving:
            self.consecutive_motion_frames += 1
            
            # 🛡️ ЗАХИСТ: Тільки якщо 3 кадри ПОСПІЛЬ показують рух, вважаємо це реальним рухом тексту
            if self.consecutive_motion_frames >= 3:
                logger.warning(f"🔄 FLIP RESET: Виявлено реальний рух. diff={diff:.3f}, dy={dy:.2f}, consecutive={self.consecutive_motion_frames}")
                
                # Зберігаємо дебаг-кадр при реальному скиданні
                debug_dir = self.parent_service.output_dir / "debug"
                debug_dir.mkdir(exist_ok=True)
                filename = f"flip_RESET_f{frame_index}_b{int(current_brightness)}_d{diff:.2f}.jpg"
                cv2.imwrite(str(debug_dir / filename), roi)

                self.state = "MOTION"
                self.static_run = 0
                self.episode = []
                self.emitted = False
                self.consecutive_motion_frames = 0  # Скидаємо лічильник після справжнього ресету
            else:
                # Це одиночний/подвійний шумовий кадр. Не скидаємо static_run у 0!
                # Лише трохи "відкочуємо" його назад, щоб дати системі шанс відновитися.
                self.static_run = max(0, self.static_run - 1)
                logger.debug(f"⚠️ FLIP NOISE IGNORED: diff={diff:.3f}, consecutive={self.consecutive_motion_frames}. static_run зменшено до {self.static_run}")
        else:
            # Кадр стабільний: скидаємо лічильник шуму і нарощуємо static_run
            self.consecutive_motion_frames = 0
            self.static_run += 1
            
            if self.state != "STATIC":
                self.state = "STATIC"
                self.episode = []
                self.emitted = False
                
            # Накопичуємо кадри епізоду
            if len(self.episode) < self.max_episode_frames:
                self.episode.append({
                    "ts": timestamp, 
                    "bgr": roi.copy(),
                    "gray": gray.copy(), 
                    "sharp": self._sharpness(gray),
                })
                if len(self.episode) == 1:
                    logger.info(f"🎬 FLIP EPISODE START: frame={frame_index}, ts={timestamp:.2f}")

                    # Зберігаємо дебаг-кадр початку епізоду
                    debug_dir = self.parent_service.output_dir / "debug"
                    debug_dir.mkdir(exist_ok=True)
                    filename = f"start_f{frame_index}_b{int(current_brightness)}_d{diff:.2f}.jpg"
                    cv2.imwrite(str(debug_dir / filename), roi)
                    logger.debug(f"💾 Збережено дебаг-кадр: {filename}")
                            
            # Якщо пауза підтверджена — запускаємо OCR
            if not self.emitted and self.static_run >= self.confirm_frames:
                logger.info(f"✅ FLIP PAUSE CONFIRMED: static_run={self.static_run}, frames={len(self.episode)}")
                #  Не ставимо emitted=True тут, а тільки після успішного OCR
                if self._finalize_episode(timestamp, frame_index):
                    self.emitted = True  #  Тільки якщо OCR успішний
                else:
                    # OCR не вдалося — скидаємо, щоб спробувати знову
                    self.static_run = 0
                    self.episode = []

        # 4. Періодичний статус кожні 30 кадрів
        if frame_counter % 30 == 0:
            logger.info(f"🔄 FLIP STATUS: state={self.state}, diff={diff:.3f}, dy={dy:.2f}, "
                f"static={self.static_run}, episode={len(self.episode)}, emitted={self.emitted}")
            
        return True
    
    def _finalize_episode(self, timestamp: float, frame_index: int) -> bool:
        """Завершення епізоду: вибір найрізкішого кадру, OCR, очищення, емісія.
        Повертає True якщо OCR успішний, False інакше."""
        if not self.episode:
            return False
            
        dwell = self.episode[-1]["ts"] - self.episode[0]["ts"]
        if dwell < self.min_dwell_sec:
            logger.info("⏭️ FLIP: епізод надто короткий (%.2fs < %.2fs) — пропущено",
                        dwell, self.min_dwell_sec)
            return False
        logger.info(f"📸 FINALIZING EPISODE: dwell={dwell:.2f}s, frames={len(self.episode)}, "
                f"best_sharp={max(e['sharp'] for e in self.episode):.0f}")

            
        # 1. Вибираємо кадр з максимальною різкістю
        best = max(self.episode, key=lambda e: e["sharp"])
        
        # 🆕 2. ЗБЕРЕЖЕННЯ "ПАНОРАМИ" (найкращого кадру епізоду) для дебагу
        debug_dir = self.parent_service.output_dir / "debug" / "panorama"
        debug_dir.mkdir(parents=True, exist_ok=True)
        time_str = datetime.fromtimestamp(best["ts"]).strftime('%H%M%S')
        pano_path = debug_dir / f"panorama_f{frame_index}_{time_str}.jpg"
        cv2.imwrite(str(pano_path), best["bgr"])
        logger.debug(f"💾 Збережено панораму (найкращий кадр): {pano_path.name}")

        # 3. Запуск OCR (тепер отримуємо і blocks)
        text, score, blocks = self._ocr_frame(best["bgr"])

          # ФІЛЬТРАЦІЯ СМІТТЯ З ПРАВОГО КРАЮ (годинник та артефакти)  неперевірено для арабських каналів
        # if blocks:
        #     filtered_blocks = []
        #     for b in blocks:
        #         x_min = b["x_min"]
        #         w = b["x_max"] - b["x_min"]
        #         score = b["score"]
                
        #         # Пропускаємо годинник (правий край > 850px) та дрібні артефакти
        #         if x_min > 850 or (w < 30 and score < 0.75):
        #             continue
        #         filtered_blocks.append(b)
            
        #     blocks = filtered_blocks

        # 4. ЧІТКЕ ЛОГУВАННЯ КООРДИНАТ У ТЕРМІНАЛ
        if blocks:
            logger.info(f"📐 КООРДИНАТИ OCR БЛОКІВ (всього {len(blocks)}):")
            for i, b in enumerate(blocks):
                x_min = int(b["x_min"])
                x_max = int(b["x_max"])
                width = x_max - x_min
                y_min = int(b["box"][:, 1].min())
                y_max = int(b["box"][:, 1].max())
                conf = b["score"]
                
                # Безпечне обрізання тексту для терміналу (макс 35 символів)
                # Арабський текст винесено в самий кінець рядка, щоб не ламати LTR вирівнювання цифр
                txt_preview = b["text"][:35] + ("..." if len(b["text"]) > 35 else "")
                
                logger.info(
                    f"  [{i+1:2d}] X: {x_min:4d} - {x_max:4d} (шир: {width:3d}px) | "
                    f"Y: {y_min:3d} - {y_max:3d} | Впевн: {conf:.2f} | Текст: '{txt_preview}'"
                )
            
            # Виклик візуалізації та запису у файл (як і було)
            self._draw_ocr_visualization(best["bgr"], blocks, best["ts"], frame_index)
            self._log_flip_coordinates(blocks, frame_index, best["ts"])

        # 🆕 5. ВИЗНАЧАЄМО КОЛІР ТЕКСТУ
         # 🆕 ВИЗНАЧАЄМО КОЛІР ТЕКСТУ
        text_color = self._detect_text_color(best["bgr"])
        logger.info(f"🎨 DETECTED TEXT COLOR: {text_color.upper()}")

        logger.info(f"👁️ RAW OCR TEXT (до очищення): '{text}' (len={len(text)})")
        
        if not text or score < self.ocr_conf_min:
            logger.info("⏭️ FLIP: OCR порожній/слабкий (conf=%.2f, dwell=%.2fs) — пропущено",
                        score, dwell)
            return False
            
        # Очищення тексту (ідентичне горизонтальному конвеєру)
        original_text = text
        for p in self.excluded_patterns:
            text = text.replace(p, '')
        text = re.sub(r'\s*[-—]\s*', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[٠-٩]', lambda m: str(ord(m.group()) - ord('٠')), text)

        if len(original_text) > len(text) + 15:
            logger.warning(f"⚠️ ОЧИЩЕННЯ ВІДРІЗАЛО ТЕКСТ! Було: '{original_text}' -> Стало: '{text}'")
        
        
        if len(text) < 12:
            logger.info(f"⏭️ FLIP SKIP: text too short ({len(text)} chars < 12)")
            return False
            
        # 🆕 Використовуємо нормалізатор батьківського асемблера
        norm = self.parent_service.assembler.normalize_arabic_text(text)
        if norm == self.last_norm:
            logger.info("♻️ FLIP-REPEAT: те саме повідомлення поспіль — не дублюємо: %s...", text[:40])
            return False  
            
        self.last_norm = norm
        logger.info("🔄 FLIP MESSAGE: dwell=%.2fs frames=%d conf=%.2f sharp=%.0f | %s",
                    dwell, len(self.episode), score, best["sharp"], text[:80])
                    
        # Емісія в асемблер
        if TickerFragment is None:
            logger.error("FLIP: TickerFragment не імпортовано, емісія неможлива")
            return False
            
        fragment = TickerFragment(
            text=text,
            timestamp=best["ts"],
            frame_index=frame_index,
            confidence=score,
            boundaries=[], 
            boundary_confirmed=True,  # вертикальний тікер дає ціле повідомлення
            x_min=0.0, 
            x_max=1.0,
            color=text_color,
        )
        
        ok, reason = self.assembler.add_fragment(fragment)
        if ok:
            self.assembler.check_message_complete(force_save=True, reason="flip_complete")
            if self.save_callback:
                self.save_callback()
            #  Зберігаємо хеш тексту для майбутніх перевірок
            self._last_text_hash = hash(norm)
        else:
            logger.info("♻️ FLIP: відхилено асемблером (%s): %s...", reason, text[:40])
        
        return True  #  Успішне завершення

    def _ocr_frame(self, bgr: np.ndarray) -> Tuple[str, float, List[dict]]:
        # 1. Grayscale та базове очищення
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 3)
        
        # 2. CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(denoised)
        
        # 3. Unsharp Mask та Bilateral (швидше працюють на 1-канальному зображенні!)
        blurred = cv2.GaussianBlur(enhanced_gray, (0, 0), 1.5)
        enhanced_gray = cv2.addWeighted(enhanced_gray, 2.0, blurred, -1.0, 0)
        enhanced_gray = cv2.bilateralFilter(enhanced_gray, 3, 25, 25)
        
        # 4. Конвертація в BGR безпосередньо перед OCR (якщо модель цього вимагає)
        enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        # 🆕 Використовуємо OCR екземпляр батьківського сервісу
        result = self.parent_service.ocr.predict(enhanced_bgr)
        self.ocr_calls += 1
        
        blocks = []
        for page in result:
            texts = page.get('rec_texts', [])
            scores = page.get('rec_scores', [])
            boxes = page.get('rec_polys', None)
            if boxes is None or len(boxes) == 0:
                boxes = page.get('dt_boxes', [])
                
            for i, t in enumerate(texts):
                t = (t or "").strip()
                s = float(scores[i]) if i < len(scores) else 0.0
                if s <= 0.4 or len(t) <= 2 or i >= len(boxes):
                    continue
                    
                box = np.asarray(boxes[i], dtype=np.float32).reshape(-1, 2)
                if len(box) < 4 or not np.isfinite(box).all():
                    continue
                    
                blocks.append({
                        "x_min": float(box[:, 0].min()),
                        "x_max": float(box[:, 0].max()),
                        "text": t,
                        "score": s,
                        "box": box
                    })
                
        if not blocks:
            return "", 0.0, []

        total_chars = sum(len(b["text"]) for b in blocks)
        avg_conf = np.mean([b["score"] for b in blocks])
        logger.info(f"🔤 OCR RESULT: blocks={len(blocks)}, chars={total_chars}, "
                    f"avg_conf={avg_conf:.2f}, top_conf={max(b['score'] for b in blocks):.2f}")

        for i, b in enumerate(blocks[:3]):
            logger.debug(f"  Block {i}: '{b['text'][:40]}...' conf={b['score']:.2f}, y={b['box'][:, 1].min():.0f}, x={b['box'][:, 0].min():.0f}")
                            
        # Сортування: рядки зверху вниз (Y), всередині рядка справа наліво (-X) для арабської
        blocks.sort(key=lambda b: (round(b["box"][:, 1].min() / 20), -b["box"][:, 0].min()))
        text = " ".join(b["text"] for b in blocks)
        score = float(np.mean([b["score"] for b in blocks]))
        return text, score, blocks