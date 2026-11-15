#!/usr/bin/env python3
"""
Ticker Assembler — збирає фрагменти OCR у повні повідомлення.
"""
import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import difflib
import uuid
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class TickerFragment:
    """Один фрагмент тексту з OCR."""
    text: str
    timestamp: float
    frame_index: int
    confidence: float
    overlap_start: Optional[float] = None
    overlap_end: Optional[float] = None
    boundary_confirmed: bool = False

    # Координати підтверджених меж речень у цьому фрагменті
    # (у пікселях панорами; для логування та аналізу)
    boundaries: List[List[int]] = field(default_factory=list)
    # Горизонтальні координати OCR-фрагмента
    x_min: Optional[float] = None
    x_max: Optional[float] = None

    #НОВІ ПОЛЯ ДЛЯ АНАЛІЗУ КРАЇВ
    boundary_position_ratio: Optional[float] = None  # 0.0 до 1.0
    boundary_near_edge: bool = False

    tail_of_saved: bool = False      # ЗОНА 1: хвіст вже збереженої новини
    left_incomplete: bool = False    # ЗОНА 3: неповний початок → чекає продовження

    # 🆕 ДОДАНО: Колір тексту (для розрізнення термінових/звичайних новин)
    color: str = "unknown"           # Можливі значення: 'yellow', 'white', 'unknown'

@dataclass
class TickerMessage:
    """Повне повідомлення з біжучого рядка."""
    message_id: str
    channel: str
    text: str
    started_at: float
    completed_at: Optional[float] = None
    fragments: List[TickerFragment] = field(default_factory=list)
    completeness: str = "capturing"  # capturing, waiting_complete, complete, incomplete
    ocr_confidence: float = 0.0
    duplicate_check_hash: str = ""
    #Усі підтверджені межі з усіх фрагментів (агрегація для JSON)
    boundaries: List[List[int]] = field(default_factory=list)
    #З яких frame_index ці межі (для прив'язки до часу)
    boundary_frames: List[int] = field(default_factory=list)
    x_min: float | None = None
    x_max: float | None = None
    completion_reason: str = "unknown"  # timeout, boundary, forced, edge_truncation,
    awaiting_continuation: bool = False
    
class TickerAssembler:
    """
    Збирає фрагменти OCR у повні повідомлення.
    Ключові завдання:
    1. Виявлення перекриттів між фрагментами
    2. Нормалізація арабського тексту
    3. Виправлення помилок OCR
    4. Визначення початку/кінця повідомлення
    """
    

    def __init__(self, channel: str):
        self.channel = channel
        self.current_message: Optional[TickerMessage] = None
        self.completed_messages: List[TickerMessage] = []
        self.recent_messages: deque = deque(maxlen=6)

        self.stats = {
            'total_fragments_received': 0,
            'merges_performed': 0,
            'dropped_low_overlap': 0,
            'dropped_no_novelty': 0,
            'messages_emitted': 0,
            'messages_completed': 0,
            'dropped_prefix_buffer': 0,
        }
        

    def normalize_arabic_text(self, text: str) -> str:
        if not text:
            return ""
        
        # 1. Видаляємо діакритичні знаки (ташкіл), які OCR часто плутає
        text = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', text)
        
        # 2. Уніфікація літер (критично для арабського збігу слів)
        text = text.replace('ى', 'ي')      # Alef Maksura -> Ya
        text = text.replace('ة', 'ه')      # Ta Marbuta -> Ha
        text = text.replace('أ', 'ا')      # Alef with Hamza -> Plain Alef
        text = text.replace('إ', 'ا')      # арабська алеф U+0627  (опціонально, залежить від стилю каналу)
        text = text.replace('ؤ', 'و')      # Waw with Hamza -> Waw
        
        # 3. Видаляємо подвоєння літер, які часто виникають через артефакти OCR (напр. "اللل")
        text = re.sub(r'([ا-ي])\1{2,}', r'\1', text)
        
        # 4. Нормалізація пробілів
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


    def _char_tail_head_ratio(
        self, norm_old: List[str], norm_new: List[str]
    ) -> Tuple[float, int]:
        """А8: символьне зіставлення хвіст→голова з М'ЯКИМИ якорями.
        Ловить важко мангляні слова всередині перекриття (обрізане навпіл
        'المدفعية'->'المد'), коли словний і шовний рівні мовчать.
        Якір: ХОЧА БИ ОДИН блок (size>=3) торкається кінця хвоста, І
        ХОЧА БИ ОДИН — початку голови (не обов'язково той самий блок).
        Повертає (ratio, b_end) — крайня права позиція збігу в голові
        (символи; сумісна з drop-циклом merge 4.5)."""
        tail_s = ' '.join(norm_old[-8:])
        head_s = ' '.join(norm_new[:8])
        if min(len(tail_s), len(head_s)) < 16:
            return 0.0, 0

        sm = difflib.SequenceMatcher(None, tail_s, head_s, autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size >= 3]
        if not blocks:
            return 0.0, 0

        if not (any(b.a + b.size >= len(tail_s) - 3 for b in blocks)
                and any(b.b <= 3 for b in blocks)):
            return 0.0, 0

        b_end = max(b.b + b.size for b in blocks)
        return sm.ratio(), b_end


    def _seam_word_overlap(
        self, norm_old: List[str], norm_new: List[str]
    ) -> Tuple[int, int]:
        """А6v3: k слів на шві, СКЛЕЄНИХ без пробілів, порівняні символьно.
        Ловить розщеплені/склеєні слова OCR (تراجعوا عن ↔ ترا اجعواعن,
        في المخيمات ↔ فيالمخيمات). Повертає (k, covered_b) або (0, 0)."""
        for k in (3, 2):
            if len(norm_old) >= k and len(norm_new) >= k:
                a = ''.join(norm_old[-k:])
                b = ''.join(norm_new[:k])
                if min(len(a), len(b)) < 9:   # було 10: пропускало канонічний 9-символьний шів تراجعواعن
                    continue
                sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
                blocks = list(sm.get_matching_blocks())
                if not any(bl.size >= 2 for bl in blocks):
                    continue
                # якір: хоча б один збіг (навіть 1 символ — транспозиція дає такі хвости)
                # торкається кінця a І початку b
                if (any(bl.a + bl.size >= len(a) - 1 for bl in blocks)
                        and any(bl.b <= 1 for bl in blocks)
                        and sm.ratio() >= 0.78):
                    covered_b = max(
                        (bl.b + bl.size for bl in blocks if bl.size >= 2),
                        default=0,
                    )
                    logger.debug("🔗 А6v3: шовні %d слів (ratio=%.2f, covered=%d)",
                                 k, sm.ratio(), covered_b)
                    return k, covered_b
        return 0, 0
    

    def calculate_overlap(
        self, text1: str, text2: str
    ) -> Tuple[float, str, int]:
        """
        Знаходить перекриття між двома OCR-фрагментами на рівні слів.

        Повертає:
            (overlap_score, overlap_type, overlap_words)

        overlap_type:
            suffix_prefix     — кінець text1 збігається з початком text2
            prefix_suffix     — кінець text2 збігається з початком text1
            text1_in_text2    — text1 повністю міститься в text2
            text2_in_text1    — text2 повністю міститься в text1
            none              — надійного збігу не знайдено
        """
        norm1 = self.normalize_arabic_text(text1).split()
        norm2 = self.normalize_arabic_text(text2).split()

        if not norm1 or not norm2:
            return 0.0, "none", 0

        def find_subsequence(short, long):
            """Повертає позицію входження або -1."""
            if len(short) > len(long):
                return -1

            for i in range(len(long) - len(short) + 1):
                if long[i:i + len(short)] == short:
                    return i

            return -1

        # Повне входження одного фрагмента в інший.
        if len(norm1) <= len(norm2):
            if find_subsequence(norm1, norm2) >= 0:
                return len(norm1) / len(norm2), "text1_in_text2", len(norm1)

        if len(norm2) <= len(norm1):
            if find_subsequence(norm2, norm1) >= 0:
                return len(norm2) / len(norm1), "text2_in_text1", len(norm2)

        # def word_similarity(a, b):
        #     if a == b:
        #         return 1.0
        #     if min(len(a), len(b)) < 3:
        #         return 0.0
        #     if abs(len(a) - len(b)) > 2:
        #         if min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a)):
        #             return 0.85            # 🆕 А6 у merge
        #         return 0.0
        #     return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()

        def word_similarity(a, b):
            if a == b:
                return 1.0

            if min(len(a), len(b)) < 3:
                return 0.0

            if abs(len(a) - len(b)) > 2:
                # 🆕 A6: обрізане слово = префікс повного (الف → الفدرالية)
                if min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a)):
                    return 0.85
                return 0.0

            return difflib.SequenceMatcher(
                None, a, b, autojunk=False
            ).ratio()

        def find_boundary_overlap(left, right):
            """
            Порівнює суфікс left із префіксом right.
            Повертає кількість спільних слів або 0.
            """
            max_size = min(len(left), len(right), 15)

            for size in range(max_size, 0, -1):
                a_words = left[-size:]
                b_words = right[:size]

                scores = [
                    word_similarity(a, b)
                    for a, b in zip(a_words, b_words)
                ]

                exact = sum(a == b for a, b in zip(a_words, b_words))

                if exact == size:
                    return size

                fuzzy_match = (
                    size >= 2
                    and exact >= 1
                    and sum(scores) / size >= 0.65
                    and sum(1 for s in scores if s >= 0.60) >= (size - 1)
                    and min(scores) >= 0.50  # 🆕 ЗАКРИТТЯ ЛАЗІВКИ
                )

                if fuzzy_match:
                    return size

            return 0

        # Основний напрямок: кінець text1 -> початок text2.
        overlap = find_boundary_overlap(norm1, norm2)
        overlap_type = "suffix_prefix"

        # Перевіряємо протилежний напрямок, якщо збігу немає.
        if overlap == 0:
            overlap = find_boundary_overlap(norm2, norm1)
            overlap_type = "prefix_suffix"

        # ==========================================================
        # 🆕 ОНОВЛЕНИЙ БЛОК (А6v3): Останній шанс перед відмовою
        # ==========================================================
        if overlap == 0:
            # А6v2: ПЕРШИЙ пріоритет — шовні слова (найкраще ловить розщеплені/склеєні слова)
            k, _cov = self._seam_word_overlap(norm1, norm2)
            if k:
                return 0.20, "suffix_prefix", k

            k, _cov = self._seam_word_overlap(norm2, norm1)
            if k:
                return 0.20, "prefix_suffix", k

            # 🆕 А8: символьний рівень — другий останній шанс
            r, b_end = self._char_tail_head_ratio(norm1, norm2)
            if r >= 0.75:
                return 0.20, "suffix_prefix", max(1, b_end // 6)

            r, b_end = self._char_tail_head_ratio(norm2, norm1)
            if r >= 0.75:
                return 0.20, "prefix_suffix", max(1, b_end // 6)

            return 0.0, "none", 0



        # Нормалізація до ДОВЖИНИ КОРОТШОГО тексту. 
        score = overlap / min(len(norm1), len(norm2)) if min(len(norm1), len(norm2)) > 0 else 0.0
        return score, overlap_type, overlap


    def merge_fragments(
        self,
        accumulated_text: str,
        new_frag: TickerFragment,
    ) -> str:
        """
        Об'єднує OCR-фрагменти на рівні слів.

        - Зберігає оригінальне написання слів.
        - Прибирає підтверджене перекриття.
        - Підтримує точні та обережні нечіткі збіги.
        - Якщо збігу немає, зберігає обидва фрагменти.
        - Не визначає межі новин.
        """
        old_text = accumulated_text.strip()
        new_text = new_frag.text.strip()

        if not old_text:
            return new_text
        if not new_text:
            return old_text

        old_words = old_text.split()
        new_words = new_text.split()

        norm_old = self.normalize_arabic_text(old_text).split()
        norm_new = self.normalize_arabic_text(new_text).split()

        if not norm_old or not norm_new:
            return f"{old_text} {new_text}".strip()

        def contains_sequence(short, long):
            if len(short) > len(long):
                return False

            return any(
                long[i:i + len(short)] == short
                for i in range(len(long) - len(short) + 1)
            )

        # 1. Новий фрагмент уже повністю є в накопиченому.
        if len(norm_new) <= len(norm_old):
            if contains_sequence(norm_new, norm_old):
                return old_text

        # 2. Новий фрагмент містить увесь накопичений текст.
        if len(norm_old) <= len(norm_new):
            if contains_sequence(norm_old, norm_new):
                return new_text

        def word_similarity(a, b):
            if a == b:
                return 1.0

            #ЗМІНЕНО з < 4 на < 3, щоб дозволити 3-літерні арабські корені
            if min(len(a), len(b)) < 3:
                return 0.0


            if abs(len(a) - len(b)) > 2:
                # 🆕 A6: обрізане слово = префікс повного (السو → السوداني)
                if min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a)):
                    return 0.85
                return 0.0

            return difflib.SequenceMatcher(
                None, a, b, autojunk=False
            ).ratio()

        def find_overlap(left, right):
            """
            Зіставляє суфікс left із префіксом right.
            Повертає кількість підтверджених спільних слів.
            """
            max_size = min(len(left), len(right), 15)

            for size in range(max_size, 0, -1):
                left_part = left[-size:]
                right_part = right[:size]

                scores = [
                    word_similarity(a, b)
                    for a, b in zip(left_part, right_part)
                ]

                exact = sum(
                    a == b for a, b in zip(left_part, right_part)
                )

                # Усі слова збігаються точно.
                if exact == size:
                    return size

                fuzzy_match = (
                    size >= 2
                    and exact >= 1
                    and sum(scores) / size >= 0.65  # 🆕 Синхронізовано з calculate_overlap
                    and sum(1 for s in scores if s >= 0.60) >= (size - 1)
                    and min(scores) >= 0.50  # 🆕 ЗАКРИТТЯ ЛАЗІВКИ
                )

                if fuzzy_match:
                    return size

            return 0
        # 3. Основний напрямок: кінець старого → початок нового (+A5 дорощування)
        overlap = find_overlap(norm_old, norm_new)

        if overlap:                                          
            _ot, _nt = old_words[-1], new_words[overlap - 1]   
            if len(_nt) > len(_ot) and _nt.startswith(_ot):    
                old_text = ' '.join(old_words[:-1] + [_nt])    
                
            remaining_words = new_words[overlap:]

            if not remaining_words:
                return old_text

            return f"{old_text} {' '.join(remaining_words)}".strip()

        # 4. Зворотний напрямок:
        # кінець нового фрагмента -> початок старого.
        overlap = find_overlap(norm_new, norm_old)

        if overlap:
            # Новий фрагмент містить текст, що йде перед старим.
            preceding_words = new_words[:-overlap]

            if not preceding_words:
                return old_text

            return f"{' '.join(preceding_words)} {old_text}".strip()

        # ==========================================================
        # 4.4 А6v3: шовні слова. Ріжемо по покриттю solid-блоків (size>=2),
        # інакше асиметрія (2 старих слова ↔ 1 нове) викидає зайве слово.
        # ==========================================================
        k, covered_b = self._seam_word_overlap(norm_old, norm_new)
        if k:
            drop_words, pos = 0, 0
            for w in new_words:
                if pos >= covered_b:
                    break
                pos += len(w) + 1
                drop_words += 1

            remaining_words = new_words[drop_words:]
            if not remaining_words:
                return old_text
            return f"{old_text} {' '.join(remaining_words)}".strip()


        # ==========================================================
        # 4.5 А8: Символьний рівень — ріж голову нового по символьному покриттю
        # ==========================================================
        r, covered = self._char_tail_head_ratio(norm_old, norm_new)
        if r >= 0.75:  # 🆕 ЗМІНЕНО: поріг знижено з 0.80 до 0.75 для кращого збігу
            drop_words, used = 0, 0
            for w in new_words:
                if used >= covered:
                    break
                used += len(w) + 1
                drop_words += 1

            remaining_words = new_words[drop_words:]
            if not remaining_words:
                return old_text
            return f"{old_text} {' '.join(remaining_words)}".strip()

        # ==========================================================
        # 5. Немає підтвердженого перекриття (overlap == 0).
        # КРИТИЧНО: не роздуваємо текст, якщо старого вже достатньо багато.
        # ==========================================================
        if len(old_text) > 50:
            # Якщо старий текст вже довгий, а новий фрагмент не перекривається,
            # найімовірніше, це або нова думка, або OCR читає те саме з помилками.
            # Безпечніше повернути НОВИЙ текст, ніж створювати ланцюжок дублікатів.
            logger.debug(
                f"⚠️ Розрив ланцюжка: overlap=0, old_len={len(old_text)}. "
                f"Повертаємо новий текст замість додавання."
            )
            return new_text
        return f"{old_text} {new_text}".strip()


    

 
    def add_fragment(self, fragment: TickerFragment) -> tuple[bool, str]:
        
        self.stats["total_fragments_received"] += 1

        # Очищаємо текст фрагмента.
        new_text = fragment.text.strip()

        # Порожній фрагмент не додаємо.
        if not new_text:
            return True, "empty"

        # Перший фрагмент: створюємо нове повідомлення.
        if self.current_message is None:
            self.current_message = TickerMessage(
                message_id=(
                    f"ticker_{self.channel}_{uuid.uuid4().hex[:8]}"
                ),
                channel=self.channel,
                text=new_text,
                started_at=fragment.timestamp,
                completed_at=fragment.timestamp,
                fragments=[fragment],
                ocr_confidence=fragment.confidence,
                boundaries=list(fragment.boundaries),
                boundary_frames=(
                    [fragment.frame_index]
                    if fragment.boundaries
                    else []
                ),
                 # Ініціалізуємо геометрію з першого фрагмента
                x_min=fragment.x_min,
                x_max=fragment.x_max,
            )

            self.stats["merges_performed"] += 1

            logger.debug(
                "Створено нове повідомлення: frame=%s chars=%d",
                fragment.frame_index,
                len(new_text),
            )

            return True, "started"

        current_text = self.current_message.text.strip()

        # Розділяємо лише за явно підтвердженою межею.
        if getattr(fragment, "boundary_confirmed", False):
            _ov, _typ, _w = self.calculate_overlap(current_text, new_text)
            _short = min(len(self.normalize_arabic_text(current_text).split()),
                         len(self.normalize_arabic_text(new_text).split()))
            _same = (_typ in ("text1_in_text2", "text2_in_text1")
                     or _w >= max(3, int(0.6 * _short)))
            if _same:
                logger.info("🔗 A1: межу проігноровано — це продовження тієї ж новини "
                            "(ov=%.2f type=%s words=%d)", _ov, _typ, _w)
            else:
                edge_warning = ""
                if getattr(fragment, "boundary_near_edge", False):
                    ratio = getattr(fragment, "boundary_position_ratio", 0.5)
                    side = "початок" if ratio < 0.15 else "кінець"
                    edge_warning = f" ⚠️ УВАГА: обрив на {side} панорами (можлива втрата частини тексту)"

                logger.info(
                    "🛑 Підтверджена межа повідомлення: frame=%s%s",
                    fragment.frame_index, edge_warning
                )

                if getattr(fragment, "boundary_near_edge", False) and getattr(fragment, "boundary_position_ratio", 1.0) > 0.85:
                    if self.current_message:
                        self.current_message.completeness = "incomplete"

                return False, "boundary"
        
        # Нормалізація потрібна лише для порівняння.
        prev_words = self.normalize_arabic_text(
            current_text
        ).split()

        new_words = self.normalize_arabic_text(
            new_text
        ).split()

        # Обчислюємо перекриття між поточним і новим фрагментом.
        overlap, overlap_type, overlap_words = (
            self.calculate_overlap(
                current_text,
                new_text,
            )
        )

        # 🆕 ГЕОМЕТРИЧНА ПЕРЕВІРКА: Якщо текст не збігається, але координати логічні
        is_geometrically_adjacent = False
        if getattr(fragment, 'x_min', None) is not None and getattr(fragment, 'x_max', None) is not None:
            current_x_min = getattr(self.current_message, 'x_min', None)
            current_x_max = getattr(self.current_message, 'x_max', None)
            
            if current_x_min is not None and current_x_max is not None:
                # Для RTL: новий текст має бути праворуч (більший X) або поруч з попереднім
                distance = abs(fragment.x_min - current_x_max)
                if distance < 200.0: 
                    is_geometrically_adjacent = True

        # 🆕 ЛОГІКА ОБРОБКИ НИЗЬКОГО OVERLAP З УРАХУВАННЯМ ГЕОМЕТРІЇ
        if overlap < 0.10 and len(prev_words) >= 3 and len(new_words) >= 3:
            prev_tail = " ".join(prev_words[-3:])
            new_head = " ".join(new_words[:3])
            
            # 🛡️ А5б: Захист від гібридів. Якщо це хвіст вже збереженої новини,
            # ми категорично відмовляємося клеїти його до нової голови.
            if (getattr(fragment, "tail_of_saved", False)
                    and not getattr(fragment, "left_incomplete", False)):
                logger.debug("🛡️ А5б: Відхилено T-хвіст, щоб уникнути гібридного речення.")
                return True, "duplicate_history"

            # if is_geometrically_adjacent:
            #     logger.info(
            #         f"🔗 ГЕОМЕТРИЧНЕ ЗЛИТТЯ: overlap={overlap:.2f}, але координати суміжні (дистанція < 200px). "
            #         f"Додаємо текст: '{new_text[:30]}...'"
            #     )
            #     # Ми дозволяємо виконанню йти далі до merge_fragments

            if is_geometrically_adjacent:
                # 🆕 H4-geo: геометрія САМА ПО СОБІ не доводить продовження.
                # Нуль текстових зачіпок (type=none) + низька символьна схожість
                # = РІЗНІ новини, що просто стоять поруч на стрічці.
                _char_ratio = difflib.SequenceMatcher(
                    None,
                    self.normalize_arabic_text(current_text),
                    self.normalize_arabic_text(new_text),
                ).ratio()
                if overlap_type == "none" and _char_ratio <= 0.45:
                    logger.info(
                        "🗑️ H4-geo: відмова (geometry only, ratio=%.2f) — різні новини: "
                        "'%s...' + '%s...'", _char_ratio, current_text[:30], new_text[:30]
                    )
                    self.stats["dropped_low_overlap"] += 1
                    return False, "no_logical_connection"
                logger.info(
                    f"🔗 ГЕОМЕТРИЧНЕ ЗЛИТТЯ: overlap={overlap:.2f}, але координати суміжні (дистанція < 200px). "
                    f"Додаємо текст: '{new_text[:30]}...'"
                )
            elif getattr(self.current_message, "awaiting_continuation", False):
                # КОНВЕЄР: поточне повідомлення — неповний початок з лівого краю
                # попередньої панорами. Новий фрагмент ГАРАНТОВАНО його продовження
                # (зона 3 панорами N = зона 1-2 панорами N+1). Мерджимо попри overlap=0.
                ratio = difflib.SequenceMatcher(
                    None,
                    self.normalize_arabic_text(current_text),
                    self.normalize_arabic_text(new_text),
                ).ratio()

                # 🆕 A7: обрізане останнє слово буфера (≤3 знаки) — частина
                # першого слова нового. Дорощуємо ЗАМІСТЬ відмови/склейки.
                _ow, _nw = current_text.split(), new_text.split()
                _o, _n = _ow[-1], _nw[0]
                # _a7 = (len(_o) <= 3 and len(_n) > len(_o)
                #        and (_n.startswith(_o) or _n.endswith(_o)))
                _a7 = (1 <= len(_o) <= 3 
                    and len(_n) - len(_o) >= 2  # 🆕 мінімум 2 знаки різниці
                    and (_n.startswith(_o) or _n.endswith(_o)))

                if _a7:
                    merged = ' '.join(_ow[:-1] + _nw)
                    logger.info("🔗 A7: дорощено обрізане '%s' → '%s'", _o, _n)
                elif ratio <= 0.45:
                    logger.info("🗑️ H1: відмова склейки (ratio=%.2f) — різна новина", ratio)
                    logger.info(
                                f"🚂 КОНВЕЄР: ВІДХИЛЕНО (fuzzy ratio={ratio:.2f}): "
                                f"'{current_text[:30]}...' vs '{new_text[:30]}...'"
                            )
                    self.stats["dropped_low_overlap"] += 1
                    return False, "no_logical_connection"
                else:
                    merged = new_text if len(new_words) >= len(prev_words) else current_text

                logger.info(
                    f"🚂 КОНВЕЄР: примусове продовження (fuzzy ratio={ratio:.2f}): "
                    f"'{current_text[:30]}...' + '{new_text[:30]}...'"
                )
                self.current_message.text = merged
                self.current_message.fragments.append(fragment)
                self.current_message.completed_at = fragment.timestamp
                self.current_message.awaiting_continuation = getattr(fragment, "left_incomplete", False)
                return True, "merged"
            else:
                logger.warning(
                    f"ВІДХИЛЕНО: overlap={overlap:.2f} і геометрія не суміжна. "
                    f"Кінець старого: '[{prev_tail}]' | Початок нового: '[{new_head}]'"
                )
                self.stats["dropped_low_overlap"] += 1 
                return False, "no_logical_connection"

        # Далі йде ваш існуючий код:
        if overlap < 0.05 and len(prev_words) >= 5 and len(new_words) >= 5:
            logger.debug(
                f"🔍 DEBUG OVERLAP FAIL: "
                f"prev_tail={prev_words[-5:]} | new_head={new_words[:5]}"
            )

        logger.debug(
            "ASSEMBLY: overlap=%.2f type=%s words=%d "
            "current_words=%d new_words=%d frame=%s",
            overlap,
            overlap_type,
            overlap_words,
            len(prev_words),
            len(new_words),
            fragment.frame_index,
        )

                # 🆕 ПЕРЕВІРКА ПРОТИ ІСТОРІЇ: чи не є фрагмент хвостом ВЖЕ ЗБЕРЕЖЕНОЇ новини?
        # Це захищає від "склейок" типу [початок нової] + [хвіст старої з JSONL].
        # Перевіряємо проти останніх 6 збережених повідомлень (recent_messages).
        if new_words:
            for saved_text in reversed(self.recent_messages):
                saved_norm = self.normalize_arabic_text(saved_text).split()
                if not saved_norm or len(new_words) > len(saved_norm):
                    continue
                # Шукаємо, чи new_words є безперервною підпослідовністю saved_norm
                if any(
                    saved_norm[i:i + len(new_words)] == new_words
                    for i in range(len(saved_norm) - len(new_words) + 1)
                ):
                    self.stats["dropped_no_novelty"] += 1
                    logger.debug(
                        "🗑️ Фрагмент = частина ВЖЕ ЗБЕРЕЖЕНОЇ новини, відкинуто: %s...",
                        new_text[:40],
                    )
                    return True, "duplicate_history"


        # Перевіряємо, чи весь новий фрагмент уже є
        # безперервною послідовністю в поточному повідомленні.
        duplicate = False

        if new_words and len(new_words) <= len(prev_words):
            for i in range(
                len(prev_words) - len(new_words) + 1
            ):
                if (
                    prev_words[i:i + len(new_words)]
                    == new_words
                ):
                    duplicate = True
                    break

        if duplicate:
            self.stats["dropped_no_novelty"] += 1

            logger.debug(
                "Дублікат пропущено: frame=%s words=%d",
                fragment.frame_index,
                len(new_words),
            )

            return True, "duplicate"

        # Об'єднуємо фрагменти.
        # Якщо перекриття відсутнє, merge_fragments()
        # має зберегти обидва фрагменти.
        merged_text = self.merge_fragments(
            current_text,
            fragment,
        )
        # Якщо текст виріс понад 150 символів, а збігу так і не знайдено (overlap < 0.05),
        # це означає, що ми склеюємо різні новини. Примусово розриваємо ланцюжок.
        if len(merged_text) > 150 and overlap < 0.05:
            logger.warning(
                f"⚠️ ПЕРЕПОВНЕННЯ БЕЗ ЗБІГУ: довжина={len(merged_text)}, overlap={overlap:.2f}. "
                f"Примусовий розрив, щоб уникнути злипання новин."
            )
            return False, "safety_valve"

        self.current_message.text = merged_text
        self.current_message.fragments.append(fragment)
        self.current_message.completed_at = fragment.timestamp
        self.current_message.awaiting_continuation = getattr(fragment, "left_incomplete", False)

        #Оновлюємо геометрію повідомлення координатами найсвіжішого фрагмента
        # (Раніше x_min/x_max лишались None → геометричне злиття не працювало)
        if fragment.x_min is not None and fragment.x_max is not None:
            self.current_message.x_min = fragment.x_min
            self.current_message.x_max = fragment.x_max

        # Додаємо нові межі без повторів.
        if fragment.boundaries:
            for boundary in fragment.boundaries:
                if boundary not in self.current_message.boundaries:
                    self.current_message.boundaries.append(
                        boundary
                    )
                    self.current_message.boundary_frames.append(
                        fragment.frame_index
                    )

        # Перераховуємо середню впевненість OCR.
        fragments = self.current_message.fragments

        if fragments:
            self.current_message.ocr_confidence = (
                sum(f.confidence for f in fragments)
                / len(fragments)
            )

        self.stats["merges_performed"] += 1

        logger.debug(
            "Фрагмент оброблено: overlap=%.2f type=%s "
            "overlap_words=%d result_chars=%d",
            overlap,
            overlap_type,
            overlap_words,
            len(merged_text),
        )

        return True, "merged"


    
    def check_message_complete(self, timeout_seconds: float = 6.0, 
                               current_time: Optional[float] = None, 
                               force_save: bool = False,
                               reason: Optional[str] = None) -> bool:
        if not self.current_message:
            return False
        
        # 🆕 ВИЗНАЧЕННЯ ПРИЧИНИ
        if reason is None:
            if force_save:
                reason = "forced"
            else:
                reason = "timeout"
        
        # А4: не затираємо edge_truncation, виставлений upstream (сервісом)
        if self.current_message.completion_reason != "edge_truncation":
            self.current_message.completion_reason = reason
        
        # 🆕 Логіка примусового збереження
        if force_save:
            if current_time and current_time > (self.current_message.completed_at or 0):
                self.current_message.completed_at = current_time
            if self.current_message.completeness != "incomplete":
                self.current_message.completeness = "complete"
                
            self.completed_messages.append(self.current_message)
            if self.current_message.text:
                self.recent_messages.append(self.current_message.text)
            self.stats['messages_emitted'] += 1
            logger.info(f"✅ Примусово завершено (reason={reason}): {self.current_message.text[:80]}...")
            self.current_message = None
            return True
        
        if current_time is None:
            current_time = datetime.now().timestamp()
        
        time_since_last = current_time - self.current_message.completed_at
        effective_timeout = timeout_seconds
        
        if time_since_last > effective_timeout:
            # А4: не затираємо incomplete/edge_truncation, виставлені upstream
            if self.current_message.completeness != "incomplete":
                self.current_message.completeness = "complete"
            if self.current_message.completion_reason != "edge_truncation":
                self.current_message.completion_reason = "timeout"
            self.completed_messages.append(self.current_message)
            self.recent_messages.append(self.current_message.text)  # пам'ятаємо для перевірки дублікатів
            self.stats['messages_completed'] += 1
            self.stats['messages_emitted'] += 1
            logger.info(f"✅ Завершено (тиша {time_since_last:.1f}s, reason=timeout): {self.current_message.text[:80]}...")
            self.current_message = None
            return True
        
        if self.current_message.completeness != "incomplete":
            self.current_message.completeness = "capturing"
        return False
    
    
    def get_completed_messages(self) -> List[TickerMessage]:
        """Повертає завершені повідомлення та очищає список."""
        messages = self.completed_messages.copy()
        self.completed_messages.clear()
        return messages
    
    # def get_statistics(self) -> Dict:
    #     """Статистика роботи збирача."""
    #     total_fragments = sum(len(m.fragments) for m in self.completed_messages)
    #     avg_confidence = sum(m.ocr_confidence for m in self.completed_messages) / len(self.completed_messages) if self.completed_messages else 0
        
    #     # 🆕 ДОДАНО: Об'єднуємо базову статистику з нашою новою детальною діагностикою
    #     return {
    #         **self.stats,  # <-- Цей рядок додає всі нові лічильники (dropped, merges тощо)
    #         "completed_messages": len(self.completed_messages),
    #         "total_fragments_in_completed": total_fragments,
    #         "avg_fragments_per_message": total_fragments / len(self.completed_messages) if self.completed_messages else 0,
    #         "avg_ocr_confidence": round(avg_confidence, 3),
    #         "current_message_fragments": len(self.current_message.fragments) if self.current_message else 0
    #     }

    def get_statistics(self) -> Dict:
        """Статистика роботи збирача."""
        total_fragments = sum(len(m.fragments) for m in self.completed_messages)
        avg_confidence = (
            sum(m.ocr_confidence for m in self.completed_messages) 
            / len(self.completed_messages)
            if self.completed_messages else 0
        )

        return {
            **self.stats,   # вже містить 'messages_completed' (історія), merges_performed тощо
            "queued_messages": len(self.completed_messages),          # 🆕 в черзі, ще не забрані
            "total_fragments_in_queue": total_fragments,              # 🆕 фрагменти в черзі
            "avg_fragments_per_queued": (
                total_fragments / len(self.completed_messages) 
                if self.completed_messages else 0
            ),
            "avg_ocr_confidence": round(avg_confidence, 3),
            "current_message_fragments": (
                len(self.current_message.fragments) 
                if self.current_message else 0
            ),
        }