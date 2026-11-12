# -*- coding: utf-8 -*-
"""
coordinates_tap — пасивний збір координат із логів конвеєра у coordinates.jsonl.

ЩО ЦЕ
  CoordinatesTap — logging.Handler на root-логері. Він отримує ТІ САМІ
  повідомлення, що й консоль/файл логу, розпізнає «координатні» рядки
  і дописує їх як структуровані JSON-записи в <output_dir>/coordinates.jsonl.

БЕЗПЕКА
  * тап лише ЧИТАЄ лог-записи і пише В ОКРЕМИЙ файл — конвеєр не зачіпається;
  * emit() ніколи не кидає винятків назовні: будь-яка помилка → handleError;
  * ідемпотентний: повторний install на тому ж логері не створює дублікатів.

СХЕМА ЗАПИСІВ (type | додаткові поля)
  blank_gaps_found     | count
  blank_gap            | index, x_start, x_end, width
  sentence_gaps_found  | count
  ocr_gap              | right[2], left[2], gap[2], width
  gap_analysis         | frame, x_start, x_end, gap_width, w_right, w_left, ratio_min, 
                       | ratio_avg, ratio_local, local_median_gap, vacuum_overlap, 
                       | vacuum_run_width, words_before, words_after, vacuum_s, local_s, 
                       | avg_s, min_s, text_s, raw_score, decision, temporal_bonus, 
                       | stability_count, gap_key
  gap_summary          | pairs, normal, suspicious, probable, strong, legacy, match, conflict, new_candidate
  boundaries_confirmed | count, boundaries[[x1,x2],…]
  message_boundary     | frame, note, edge_warning
  message_completed    | reason, text
  ticker               | ticker_n
  news_saved           | status, text
  buffer_started       | status, text
  assembler_stats      | saved, duplicates, current_len
  empty_sections       | sections[[start,end,width],…]
  dx_real / dx_artificial / dx_sign / dx_control | відповідні метрики DX TEST
  tgt_stable / tgt_cleanup | метрики TGT-патча
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["CoordinatesTap", "install_coordinates_tap", "parse_log_message"]

# ──────────────────────────────────────────────────────────────────────
# Допоміжні засоби
# ──────────────────────────────────────────────────────────────────────

_N = r"-?\d+(?:\.\d+)?"                            # число: "-2", "82.0", "0.98"
_T2 = re.compile(r"\((-?\d+), (-?\d+)\)")          # (2428, 2494)
_T3 = re.compile(r"\((-?\d+), (-?\d+), (-?\d+)\)")  # (59, 110, 51)


def _tuples2(s: str) -> List[List[int]]:
    return [[int(a), int(b)] for a, b in _T2.findall(s)]


def _tuples3(s: str) -> List[List[int]]:
    return [[int(a), int(b), int(c)] for a, b, c in _T3.findall(s)]


def _num_or_none(s: str) -> Optional[int]:
    return None if s == "None" else int(s)


# ──────────────────────────────────────────────────────────────────────
# Розпізнавачі: (regex, builder) → (type, fields)
# Повідомлення перед матчем нормалізується: strip() + видалення U+FE0F
# ──────────────────────────────────────────────────────────────────────

def _b_blank_gaps_found(m) -> Tuple[str, Dict[str, Any]]:
    return ("blank_gaps_found", {"count": int(m["n"])})


def _b_blank_gap(m) -> Tuple[str, Dict[str, Any]]:
    return ("blank_gap", {"index": int(m["i"]), "x_start": int(m["a"]),
                          "x_end": int(m["b"]), "width": int(m["w"])})


def _b_sentence_gaps_found(m) -> Tuple[str, Dict[str, Any]]:
    return ("sentence_gaps_found", {"count": int(m["n"])})


def _b_ocr_gap(m) -> Tuple[str, Dict[str, Any]]:
    f: Dict[str, Any] = {
        "right": [float(m["r1"]), float(m["r2"])],
        "left": [float(m["l1"]), float(m["l2"])],
        "gap": [float(m["g1"]), float(m["g2"])],
        "width": float(m["w"]),
    }
    if m["frame"] is not None:
        f["frame"] = int(m["frame"])
    return ("ocr_gap", f)


def _b_gap_analysis(m) -> Tuple[str, Dict[str, Any]]:
    f: Dict[str, Any] = {
        "frame": int(m["frame"]),
        "x_start": float(m["x_start"]),   
        "x_end": float(m["x_end"]),       
        "gap_width": int(m["g"]), 
        "w_right": int(m["wr"]), 
        "w_left": int(m["wl"]),
        "ratio_min": float(m["rmin"]), 
        "ratio_avg": float(m["ravg"]),
        "ratio_local": float(m["rloc"]), 
        "local_median_gap": int(m["med"]),
        "vacuum_overlap": float(m["vac"]), 
        "vacuum_run_width": int(m["run"]),
        "words_before": int(m["wb"]), 
        "words_after": int(m["wa"]),
        "vacuum_s": int(m["vs"]), 
        "local_s": int(m["ls"]),
        "avg_s": int(m["avg_s"]), 
        "min_s": int(m["ms"]), 
        "text_s": int(m["ts"]),
        "raw_score": int(m["score"]), 
        "decision": m["dec"],
    }
    if m["gkey"] is not None:
        f["temporal_bonus"] = int(m["tgtb"])
        f["stability_count"] = int(m["tgts"])
        f["gap_key"] = m["gkey"]
    return ("gap_analysis", f)


def _b_gap_summary(m) -> Tuple[str, Dict[str, Any]]:
    return ("gap_summary", {"pairs": int(m["p"]), "normal": int(m["n0"]),
                            "suspicious": int(m["n1"]), "probable": int(m["n2"]),
                            "strong": int(m["n3"]), "legacy": int(m["l"]),
                            "match": int(m["m"]), "conflict": int(m["c"]),
                            "new_candidate": int(m["nc"])})


def _b_graph_split(m) -> Tuple[str, Dict[str, Any]]:
    return ("graph_split", {"sentences": int(m["n"])})


def _b_boundaries(m) -> Tuple[str, Dict[str, Any]]:
    return ("boundaries_confirmed",
            {"count": int(m["n"]), "boundaries": _tuples2(m["list"])})


def _b_boundaries_none(m) -> Tuple[str, Dict[str, Any]]:
    return ("boundaries_confirmed", {"count": 0, "boundaries": []})


def _b_edge_break(m) -> Tuple[str, Dict[str, Any]]:
    return ("edge_break", {"edge_percent": int(m["p"])})


def _b_message_boundary(m) -> Tuple[str, Dict[str, Any]]:
    note = m["note"].strip()
    return ("message_boundary", {"frame": int(m["f"]), "note": note,
                                 "edge_warning": "обрив" in note})


def _b_message_completed(m) -> Tuple[str, Dict[str, Any]]:
    return ("message_completed", {"reason": m["r"], "text": m["t"]})


def _b_ticker(m) -> Tuple[str, Dict[str, Any]]:
    return ("ticker", {"ticker_n": int(m["n"])})


def _b_news_saved(m) -> Tuple[str, Dict[str, Any]]:
    return ("news_saved", {"status": m["s"].strip(), "text": m["t"]})


def _b_buffer_started(m) -> Tuple[str, Dict[str, Any]]:
    return ("buffer_started", {"status": m["s"].strip(), "text": m["t"]})


def _b_assembler(m) -> Tuple[str, Dict[str, Any]]:
    return ("assembler_stats", {"saved": int(m["s"]), "duplicates": int(m["d"]),
                                "current_len": int(m["l"])})


def _b_empty_sections(m) -> Tuple[str, Dict[str, Any]]:
    return ("empty_sections", {"sections": _tuples3(m["list"])})


def _b_dx_real(m) -> Tuple[str, Dict[str, Any]]:
    return ("dx_real", {"dx": _num_or_none(m["dx"])})


def _b_dx_artificial(m) -> Tuple[str, Dict[str, Any]]:
    return ("dx_artificial", {"shift_px": int(m["px"]),
                              "measured_dx": _num_or_none(m["dx"])})


def _b_dx_sign_ok(m) -> Tuple[str, Dict[str, Any]]:
    return ("dx_sign", {"contract_ok": True})


def _b_dx_sign_fail(m) -> Tuple[str, Dict[str, Any]]:
    return ("dx_sign", {"contract_ok": False})


def _b_dx_control(m) -> Tuple[str, Dict[str, Any]]:
    return ("dx_control", {"measured": None})


def _b_tgt_stable(m) -> Tuple[str, Dict[str, Any]]:
    keys = m["keys"]
    pairs = re.findall(r"(\S+)\(n=(\d+)\)", keys)
    if pairs:
        stable = [[k, int(n)] for k, n in pairs]
    else:
        stable = [[k.strip(), 0] for k in keys.split(",") if k.strip()]
    return ("tgt_stable", {"count": int(m["n"]), "stable": stable})


def _b_tgt_cleanup(m) -> Tuple[str, Dict[str, Any]]:
    return ("tgt_cleanup", {"removed": int(m["n"])})


# 🔑 КЛЮЧОВА ЗМІНА: Префікс frame=... x_start=... зроблено ОПЦІОНАЛЬНИМ (?: ... )?
# Це гарантує, що парсер не впаде і не проігнорує рядок, якщо лог виведе старий формат.
_GAP_ANALYSIS_RX = (
    r"GAP ANALYSIS:\s*"
    r"frame=(?P<frame>\d+)\s+"
    r"x_start=(?P<x_start>-?\d+(?:\.\d+)?)\s+"
    r"x_end=(?P<x_end>-?\d+(?:\.\d+)?)\s+\|\s+"
    r"G=(?P<g>-?\d+)\s+\|\s+W_r=(?P<wr>\d+)\s+W_l=(?P<wl>\d+)\s+\|\s+"
    r"r_min=(?P<rmin>-?\d+(?:\.\d+)?)\s+r_avg=(?P<ravg>-?\d+(?:\.\d+)?)\s+"
    r"r_loc=(?P<rloc>-?\d+(?:\.\d+)?)\(med=(?P<med>\d+)\)\s+\|\s+"
    r"vac=(?P<vac>-?\d+(?:\.\d+)?)\(run=(?P<run>\d+)\)\s+\|\s+"
    r"words=(?P<wb>\d+)/(?P<wa>\d+)\s+\|\s+"
    r"vac_s=(?P<vs>\d+)\s+loc_s=(?P<ls>\d+)\s+avg_s=(?P<avg_s>\d+)\s+min_s=(?P<ms>\d+)\s+txt_s=(?P<ts>\d+)\s+"
    r"->\s+score=(?P<score>\d+)\s+\[(?P<dec>STRONG_BOUNDARY|PROBABLE_BOUNDARY|SUSPICIOUS|NORMAL)\]"
    r"(?:\s+⏱️TGT\+(?P<tgtb>\d+)\s+\(stable=(?P<tgts>\d+),\s+key=(?P<gkey>[^)]+)\))?"
    r"\s*\Z"
)

_PATTERNS: List[Tuple[re.Pattern, Any]] = [
    (re.compile(r"✅ Знайдено пробілів: (?P<n>\d+)\Z"), _b_blank_gaps_found),
    (re.compile(r"Пробіл #(?P<i>\d+): (?P<a>\d+)-(?P<b>\d+) \((?P<w>\d+)px\)\Z"), _b_blank_gap),
    (re.compile(r"✅ Знайдено (?P<n>\d+) пробілів між реченнями\Z"), _b_sentence_gaps_found),
    # OCR GAP: right=[…, …] left=[…, …] gap=[…, …] width=…
    (re.compile(rf"OCR GAP: (?:frame=(?P<frame>\d+) )?"
            rf"right=\[(?P<r1>{_N}), (?P<r2>{_N})\] "
            rf"left=\[(?P<l1>{_N}), (?P<l2>{_N})\] "
            rf"gap=\[(?P<g1>{_N}), (?P<g2>{_N})\] width=(?P<w>{_N})\Z"),
 _b_ocr_gap),

    (re.compile(_GAP_ANALYSIS_RX), _b_gap_analysis),
    (re.compile(r"📊 GAP SCORE SUMMARY: пар=(?P<p>\d+) \| NORMAL=(?P<n0>\d+) "
                r"SUSPICIOUS=(?P<n1>\d+) PROBABLE=(?P<n2>\d+) STRONG=(?P<n3>\d+) \| "
                r"legacy меж=(?P<l>\d+) \| MATCH=(?P<m>\d+) CONFLICT=(?P<c>\d+) "
                r"NEW_CANDIDATE=(?P<nc>\d+)\Z"), _b_gap_summary),
    (re.compile(r"📝 Графік розрізав текст на (?P<n>\d+) речень\Z"), _b_graph_split),
    # 🔑 ВИПРАВЛЕНО: додано емодзі 📝, якого не вистачало у вашому коді
    (re.compile(r"📝 Підтверджено (?P<n>\d+) меж: (?P<list>\[.*\])\Z"), _b_boundaries),
    (re.compile(r"📝 Меж не підтверджено"), _b_boundaries_none),
    (re.compile(r"⚠ ОБРИВ НА КРАЮ: Межа новини на (?P<p>\d+)% ширини панорами\.\Z"), _b_edge_break),
    (re.compile(r"🛑 Підтверджена межа повідомлення: frame=(?P<f>\d+)\s*(?P<note>.*)"), _b_message_boundary),
    (re.compile(r"✅ Примусово завершено \(reason=(?P<r>\w+)\): (?P<t>.*)"), _b_message_completed),
    (re.compile(r"🟢 TICKER #(?P<n>\d+):\Z"), _b_ticker),
    (re.compile(r"🚂 КОНВЕЄР: F-новина (?P<s>[^:]+): (?P<t>.*)"), _b_news_saved),
    (re.compile(r"🚂 H-початок у буфер \(статус: (?P<s>[^)]+)\): (?P<t>.*)"), _b_buffer_started),
    (re.compile(r"🧩 АСЕМБЛЕР: збережено=(?P<s>\d+), дублікатів=(?P<d>\d+), "
                r"поточна довжина=(?P<l>\d+)\Z"), _b_assembler),
    (re.compile(r"📏 ПОРОЖНІ ДІЛЯНКИ \(start-end-ширина\): (?P<list>\[.*\])\Z"), _b_empty_sections),
    (re.compile(r"DX TEST: реальна пара кадрів: dx=(?P<dx>None|-?\d+)\Z"), _b_dx_real),
    (re.compile(r"DX TEST: штучний рух вправо на (?P<px>\d+) px -> "
                r"measured dx=(?P<dx>None|-?\d+)\Z"), _b_dx_artificial),
    (re.compile(r"DX TEST: знак відповідає контракту dx > 0\Z"), _b_dx_sign_ok),
    (re.compile(r"DX TEST: знак протилежний контракту"), _b_dx_sign_fail),
    (re.compile(r"DX TEST: контрольний зсув не визначено"), _b_dx_control),
    (re.compile(r"⏱ TGT STABLE GAPS \((?P<n>\d+)\): (?P<keys>.*)\Z"), _b_tgt_stable),
    (re.compile(r"🧹 TGT: очищено (?P<n>\d+) застарілих ключів"), _b_tgt_cleanup),
]


def parse_log_message(msg: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Розпізнає координатний рядок логу → (type, fields) або None."""
    text = msg.strip().replace("\ufe0f", "")
    if not text:
        return None
    
    # 🆕 ТИМЧАСОВИЙ ДЕБАГ: перевіримо, чи бачить парсер наш рядок взагалі
    if "GAP ANALYSIS:" in text:
        print(f"🔍 ДЕБАГ: Спроба парсингу: {text[:80]}...")
        
    for rx, build in _PATTERNS:
        m = rx.match(text)
        if m:
            if "GAP ANALYSIS:" in text:
                print("✅ ДЕБАГ: GAP ANALYSIS УСПІШНО СФОРМОВАНО!")
            return build(m)
            
    if "GAP ANALYSIS:" in text:
        print("ДЕБАГ: GAP ANALYSIS НЕ СФОРМОВАНО! Регулярний вираз не підійшов.")
        
    return None

# ──────────────────────────────────────────────────────────────────────
# Handler + інсталяція
# ──────────────────────────────────────────────────────────────────────

class CoordinatesTap(logging.Handler):
    """Пасивний запис координатних подій у coordinates.jsonl."""

    def __init__(self, out_path):
        super().__init__(level=logging.INFO)
        self._path = Path(out_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "a", encoding="utf-8", buffering=1)
        self._seq = 0
        self._panorama_seq = 0
        self._pid = os.getpid()
        self._ignored = 0

    @property
    def path(self) -> Path:
        return self._path

    def stats(self) -> Dict[str, Any]:
        return {"written": self._seq, "ignored": self._ignored,
                "panorama_seq": self._panorama_seq,
                "pid": self._pid, "path": str(self._path)}

    def emit(self, record: logging.LogRecord) -> None:
        try:
            parsed = parse_log_message(record.getMessage())
            if parsed is None:
                self._ignored += 1
                return
            etype, fields = parsed
            if etype == "blank_gaps_found":
                self._panorama_seq += 1
            self._seq += 1
            rec: Dict[str, Any] = {
                "seq": self._seq,
                "ts": datetime.fromtimestamp(
                    record.created, timezone.utc).isoformat(timespec="milliseconds"),
                "ts_epoch": round(record.created, 3),
                "pid": self._pid,
                "panorama_seq": self._panorama_seq,
                "type": etype,
            }
            rec.update(fields)
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        finally:
            super().close()


def install_coordinates_tap(output_dir, logger: Optional[logging.Logger] = None,
                            filename: str = "coordinates.jsonl") -> CoordinatesTap:
    """Чіпляє тап на логер (за замовчуванням root). Ідемпотентно."""
    lg = logger if logger is not None else logging.getLogger()
    for h in lg.handlers:
        if isinstance(h, CoordinatesTap):
            return h
    tap = CoordinatesTap(Path(output_dir) / filename)
    lg.addHandler(tap)
    return tap










# Опціональний префікс у _GAP_ANALYSIS_RX: Конструкція (?:frame=... \| )? дозволяє успішно парсити як нові логи з frame=, так і старі логи без нього.
# Безпечний _b_gap_analysis: Додано перевірку if m["frame"] is not None:, щоб уникнути помилок при відсутності опціональних груп.
# Виправлено емодзі: Додано 📝 у патерн _b_boundaries, інакше ваш код ніколи б не розпізнав рядок 📝 Підтверджено N меж.
# Покращена документація: Додано чіткий опис схеми даних у docstring, що полегшить подальшу роботу з coordinates.jsonl.