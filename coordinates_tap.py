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
    f = {"count": int(m["n"])}
    if m.groupdict().get("dot") is not None:
        f["dot"] = int(m["dot"]); f["vacuum"] = int(m["vac"])
    return ("sentence_gaps_found", f)


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
    g = m.groupdict()
    from_dot = g.get("dot") is not None      # 4161 має dot=, 4664 — ні

    # 4161 друкує words=after/before, 4664 — before/after
    a, b = int(g["w1"]), int(g["w2"])
    w_after, w_before = (a, b) if from_dot else (b, a)

    f: Dict[str, Any] = {
        "frame": int(g["frame"]),
        "x_start": float(g["x_start"]),
        "x_end": float(g["x_end"]),
        "gap_width": int(float(g["g"])),
        "w_right": int(float(g["wr"])),
        "w_left": int(float(g["wl"])),
        "ratio_min": float(g["rmin"]),
        "ratio_avg": float(g["ravg"]),
        "ratio_local": float(g["rloc"]),
        "local_median_gap": float(g["med"]),
        "vacuum_overlap": float(g["vac"]),
        "vacuum_run_width": int(float(g["run"])),
        "words_before": w_before,
        "words_after": w_after,
        "vacuum_s": float(g["vs"]),
        "local_s": int(float(g["ls"])),
        "avg_s": int(float(g["avg_s"])),
        "min_s": int(float(g["ms"])),
        "text_s": int(float(g["ts"])),
        "raw_score": float(g["score"]),
        "decision": g["dec"],
        "source": "analyze" if from_dot else "segment",   # який логер
    }
    if from_dot:
        f["has_dot"] = (g["dot"] == "YES")
        f["dot_bonus"] = int(g["dotb"])

    raw = (g.get("tgt_raw") or "").strip()
    if raw:
        mm = re.search(r"TGT\+(\d+)\s*\(stable=(\d+),\s*key=(.+)\)\s*$", raw)
        if mm:
            f["temporal_bonus"] = int(mm.group(1))
            f["stability_count"] = int(mm.group(2))
            f["gap_key"] = mm.group(3)
        else:
            f["tgt_raw"] = raw          # формат маркера змінився — не втрачаємо запис
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


def _b_news_saved(m) -> Tuple[str, Dict[str, Any]]:
    return ("news_saved", {"status": m["s"].strip(), "text": m["t"]})


def _b_buffer_started(m) -> Tuple[str, Dict[str, Any]]:
    return ("buffer_started", {"status": m["s"].strip(), "text": m["t"]})


def _b_assembler(m) -> Tuple[str, Dict[str, Any]]:
    return ("assembler_stats", {"saved": int(m["s"]), "duplicates": int(m["d"]),
                                "current_len": int(m["l"])})


def _b_empty_sections(m) -> Tuple[str, Dict[str, Any]]:
    return ("empty_sections", {"sections": _tuples3(m["list"])})

def _b_panorama_built(m) -> Tuple[str, Dict[str, Any]]:
    return ("panorama_built", {"width": int(m["w"]), "height": int(m["h"]),
                               "shifts": int(m["s"]), "cum_dy": float(m["dy"])})

def _b_dx_real(m) -> Tuple[str, Dict[str, Any]]:
    return ("dx_real", {"dx": _num_or_none(m["dx"])})


def _b_dx_artificial(m) -> Tuple[str, Dict[str, Any]]:
    return ("dx_artificial", {"shift_px": int(m["px"]),
                              "measured_dx": _num_or_none(m["dx"])})

def _b_message_timeout(m) -> Tuple[str, Dict[str, Any]]:
    return ("message_completed", {"reason": m["r"], "silence_s": float(m["sec"]),
                                  "text": m["t"], "via": "timeout"})

def _b_dx_sign_ok(m) -> Tuple[str, Dict[str, Any]]:
    return ("dx_sign", {"contract_ok": True})


def _b_dx_sign_fail(m) -> Tuple[str, Dict[str, Any]]:
    return ("dx_sign", {"contract_ok": False})


def _b_dx_control(m) -> Tuple[str, Dict[str, Any]]:
    return ("dx_control", {"measured": None})

def _b_ticker(m) -> Tuple[str, Dict[str, Any]]:
    f = {"ticker_n": int(m["n"])}
    if m.groupdict().get("t"):
        f["text"] = m["t"]
    return ("ticker", f)

def _b_cdrop(m):
    return ("c_drop", {"x_start": int(m["a"]), "x_end": int(m["b"]),
                       "dup_start": int(m["d"]), "text": m["t"]})

def _b_a1(m):
    return ("a1_boundary_ignored", {"overlap": float(m["ov"]),
                                    "overlap_type": m["typ"], "words": int(m["w"])})
def _b_pause(m):
    return ("ticker_paused", {"frozen_frames": int(m["n"])})

def _b_resume(m):
    return ("ticker_resumed", {"frozen_frames": int(m["n"])})

def _b_cross_pano(m):
    return ("cross_pano_summary", {"panorama": int(m["pan"]),
                                   "hits": int(m["hit"]), "total": int(m["tot"])})

def _b_boundary_item(m):
    return ("boundary_item", {"index": int(m["i"]), "x_start": int(m["a"]),
                              "x_end": int(m["b"]), "width": int(m["w"])})


def _b_tgt_stable(m) -> Tuple[str, Dict[str, Any]]:
    keys = m["keys"]
    pairs = re.findall(r"(\S+)\(n=(\d+)\)", keys)
    if pairs:
        stable = [[k, int(n)] for k, n in pairs]
    else:
        stable = [[k.strip(), 0] for k in keys.split(",") if k.strip()]
    return ("tgt_stable", {"count": int(m["n"]), "stable": stable})

def _b_d1(m):
    return ("prefix_buffer_dropped", {"patch": "D1", "text": m["t"]})

def _b_d2(m):
    return ("prefix_buffer_dropped", {"patch": "D2", "branch": m["branch"], "text": m["t"]})

def _b_content_break(m):
    return ("content_break", {"frame": int(m["frame"]), "dx": int(m["dx"]),
                              "response": float(m["resp"])})

def _b_edge_summary(m):
    return ("edge_summary", {"panorama": int(m["pan"]), "a_seam": int(m["a"]),
                             "b_entry": int(m["b"]), "b_break": int(m["brk"]),
                             "b_pending": int(m["pend"]), "b_stream": int(m["strm"]),
                             "c_overlap": int(m["c"]), "width": int(m["w"]),
                             "fade_crop": int(m["fc"]), "seams": int(m["s"]),
                             "dup_zone": [int(m["dz0"]), int(m["dz1"])],
                             "dt_prev": float(m["dt"]), "speed_px_s": int(m["v"])})

def _b_cross_full(m):
    return ("cross_pano_full", {"tail": m["tail"], "sentence": m["sent"],
                                "verified": int(m["ver"]), "need": int(m["need"]),
                                "exact": int(m["ex"]), "lev1": int(m["l1"])})

def _b_fade_instab(m):
    return ("fade_instability", {"std_px": float(m["std"])})

def _b_crop_width(m):
    return ("crop_width", {"min": int(m["mn"]), "max": int(m["mx"]),
                           "mean": float(m["mean"]), "expected": float(m["exp"]),
                           "anomalies": int(m["anom"])})

def _b_vdrift(m):
    return ("vertical_drift", {"raw_sum": float(m["raw"]),
                               "applied_sum": float(m["app"]), "max_cum": float(m["cum"])})


def _b_tgt_cleanup(m) -> Tuple[str, Dict[str, Any]]:
    return ("tgt_cleanup", {"removed": int(m["n"])})


_F = r"-?\d+(?:\.\d+)?"      # int або float
_P = r"\s*\|\s*"             # "|" з пробілами або без

_GAP_ANALYSIS_RX = (
    rf"GAP ANALYSIS:\s*"
    rf"frame=(?P<frame>\d+)\s+"
    rf"x_start=(?P<x_start>{_F})\s+"
    rf"x_end=(?P<x_end>{_F}){_P}"
    rf"G=(?P<g>{_F}){_P}"
    rf"W_r=(?P<wr>{_F})\s+W_l=(?P<wl>{_F}){_P}"
    rf"r_min=(?P<rmin>{_F})\s+r_avg=(?P<ravg>{_F})\s+"
    rf"r_loc=(?P<rloc>{_F})\(med=(?P<med>{_F})\){_P}"
    rf"vac=(?P<vac>{_F})\(run=(?P<run>{_F})\){_P}"
    rf"words=(?P<w1>\d+)/(?P<w2>\d+){_P}"
    rf"vac_s=(?P<vs>{_F})\s+loc_s=(?P<ls>{_F})\s+avg_s=(?P<avg_s>{_F})\s+"
    rf"min_s=(?P<ms>{_F})\s+txt_s=(?P<ts>{_F})\s+"
    rf"(?:dot=(?P<dot>YES|NO)\(\+(?P<dotb>\d+)\)\s+)?"     # тільки 4161
    rf"->\s+score=(?P<score>{_F})\s+"
    rf"\[(?P<dec>STRONG_BOUNDARY|PROBABLE_BOUNDARY|SUSPICIOUS|NORMAL)\]"
    rf"\s*(?P<tgt_raw>.*?)\s*\Z"                            # хвіст (TGT або порожньо)
)

_PATTERNS: List[Tuple[re.Pattern, Any]] = [
    # 🆕 D3: рівно одна подія на побудовану панораму, ДО OCR-подій
    (re.compile(r"Панорама: width=(?P<w>\d+) height=(?P<h>\d+) "
                r"shifts=(?P<s>\d+) cum_dy=(?P<dy>-?\d+(?:\.\d+)?)px\Z"),
     _b_panorama_built),
    (re.compile(r"✅ Знайдено пробілів: (?P<n>\d+)\Z"), _b_blank_gaps_found),
    (re.compile(r"Пробіл #(?P<i>\d+): (?P<a>\d+)-(?P<b>\d+) \((?P<w>\d+)px\)\Z"), _b_blank_gap),
    (re.compile(r"✅ Знайдено (?P<n>\d+) пробілів між реченнями"
                r"(?: \(dot=(?P<dot>\d+), vacuum=(?P<vac>\d+)\))?\Z"),
     _b_sentence_gaps_found),
    # OCR GAP: right=[…, …] left=[…, …] gap=[…, …] width=…
    (re.compile(rf"OCR GAP: (?:frame=(?P<frame>\d+) )?"
            rf"right=\[(?P<r1>{_N}), (?P<r2>{_N})\] "
            rf"left=\[(?P<l1>{_N}), (?P<l2>{_N})\] "
            rf"gap=\[(?P<g1>{_N}), (?P<g2>{_N})\] width=(?P<w>{_N})\Z"),
    _b_ocr_gap),

    (re.compile(_GAP_ANALYSIS_RX), _b_gap_analysis),
    (re.compile(r"📊 GAP SCORE SUMMARY: пар=(?P<p>\d+)\s*\|\s*NORMAL=(?P<n0>\d+)\s+"
                r"SUSPICIOUS=(?P<n1>\d+)\s+PROBABLE=(?P<n2>\d+)\s+STRONG=(?P<n3>\d+)\s*\|\s*"
                r"legacy меж=(?P<l>\d+)\s*\|\s*MATCH=(?P<m>\d+)\s+CONFLICT=(?P<c>\d+)\s+"
                r"NEW_CANDIDATE=(?P<nc>\d+)\Z"), _b_gap_summary),
    (re.compile(r"📝 Графік розрізав текст на (?P<n>\d+) речень\Z"), _b_graph_split),
    # 🔑 ВИПРАВЛЕНО: додано емодзі 📝, якого не вистачало у вашому коді
    (re.compile(r"📍 КООРДИНАТИ МЕЖ \(всього (?P<n>\d+)\): (?P<list>\[.*\])\Z"), _b_boundaries),
    (re.compile(r"📝 Меж не підтверджено"), _b_boundaries_none),
    (re.compile(r"⚠ ОБРИВ НА КРАЮ: Межа новини на (?P<p>\d+)% ширини панорами\.\Z"), _b_edge_break),
    (re.compile(r"🛑 Підтверджена межа повідомлення: frame=(?P<f>\d+)\s*(?P<note>.*)"), _b_message_boundary),
    (re.compile(r"✅ Примусово завершено \(reason=(?P<r>\w+)\): (?P<t>.*)"), _b_message_completed),
    (re.compile(r"✅ Завершено \(тиша (?P<sec>\d+(?:\.\d+)?)s, reason=(?P<r>\w+)\): (?P<t>.*)"),
     _b_message_timeout),
    (re.compile(r"🟢 TICKER #(?P<n>\d+):(?: (?P<t>.*))?\Z"), _b_ticker),
    (re.compile(r"🚂 КОНВЕЄР: F-новина (?P<s>[^:]+): (?P<t>.*)"), _b_news_saved),
    (re.compile(r"🚂 H-початок у буфер \(статус: (?P<s>[^)]+)\): (?P<t>.*)"), _b_buffer_started),
    (re.compile(r"[⬜🧩] АСЕМБЛЕР: збережено=(?P<s>\d+), дублікатів=(?P<d>\d+), "
                r"поточна довжина=(?P<l>\d+)\Z"), _b_assembler),
    (re.compile(r"📏 ПОРОЖНІ ДІЛЯНКИ \(start-end-ширина\): (?P<list>\[.*\])\Z"), _b_empty_sections),
        (re.compile(r"[🟧🗑] C-DROP \[(?P<a>\d+)\.\.(?P<b>\d+)\] dup_start=(?P<d>\d+): (?P<t>.*)"), _b_cdrop),
    (re.compile(r"🔗 A1: межу проігноровано.*\(ov=(?P<ov>[\d.]+) type=(?P<typ>\w+) words=(?P<w>\d+)\)"), _b_a1),
    (re.compile(r"⏸ ТІКЕР ЗУПИНИВСЯ: (?P<n>\d+) кадрів"), _b_pause),
    (re.compile(r"▶ ТІКЕР ВІДНОВИВ ПРОКРУТКУ \(після (?P<n>\d+) заморожених кадрів\)"), _b_resume),
    (re.compile(r"⬜ CROSS-PANO SUMMARY: панорама (?P<pan>\d+): збігів=(?P<hit>\d+) з (?P<tot>\d+)"), _b_cross_pano),
    (re.compile(r"🟨 Межа #(?P<i>\d+): x=(?P<a>\d+) до (?P<b>\d+) \(ширина=(?P<w>\d+)px\)"), _b_boundary_item),
    (re.compile(r"DX TEST: реальна пара кадрів: dx=(?P<dx>None|-?\d+)\Z"), _b_dx_real),
    (re.compile(r"DX TEST: штучний рух вправо на (?P<px>\d+) px -> "
                r"measured dx=(?P<dx>None|-?\d+)\Z"), _b_dx_artificial),
    (re.compile(r"DX TEST: знак відповідає контракту dx > 0\Z"), _b_dx_sign_ok),
    (re.compile(r"DX TEST: знак протилежний контракту"), _b_dx_sign_fail),
    (re.compile(r"DX TEST: контрольний зсув не визначено"), _b_dx_control),
    (re.compile(r"⏱ TGT STABLE GAPS \((?P<n>\d+)\): (?P<keys>.*)\Z"), _b_tgt_stable),
    (re.compile(r"🧹 TGT: очищено (?P<n>\d+) застарілих ключів"), _b_tgt_cleanup),
        (re.compile(r"🗑 D1: буфер-префікс відкинуто без збереження: (?P<t>.*)"), _b_d1),
    (re.compile(r"🗑 D2\((?P<branch>[HF])\): буфер-префікс відкинуто перед retry: (?P<t>.*)"), _b_d2),
    (re.compile(r"🟪 РОЗРИВ КОНТЕНТУ: кадр (?P<frame>\d+) \(dx=(?P<dx>-?\d+), resp=(?P<resp>[\d.]+)\)"), _b_content_break),
    (re.compile(r"🧭 EDGE SUMMARY: pan=(?P<pan>\d+) A_seam=(?P<a>\d+) B_entry=(?P<b>\d+)"
                r"\(break=(?P<brk>\d+),pending=(?P<pend>\d+),stream=(?P<strm>\d+)\) C_overlap=(?P<c>\d+)"
                r"\| w=(?P<w>\d+) fade_crop=(?P<fc>\d+) seams=(?P<s>\d+)"
                r"\| dup_zone=(?P<dz0>\d+)\.\.(?P<dz1>\d+) dt_prev=(?P<dt>[\d.]+)с v=(?P<v>\d+)px/с"), _b_edge_summary),
    (re.compile(r"🔗 CROSS-PANO FULL: '(?P<tail>[^']+)' → речення підтверджене ЦІЛИКОМ: '(?P<sent>[^']+)'"
                r"\| verified=(?P<ver>\d+)/(?P<need>\d+) \(exact=(?P<ex>\d+), lev1=(?P<l1>\d+)\)"), _b_cross_full),
    (re.compile(r"🟥 FADE INSTABILITY: std=(?P<std>[\d.]+)px"), _b_fade_instab),
    (re.compile(r"🟪 CROP WIDTH: min=(?P<mn>\d+)px, max=(?P<mx>\d+)px, mean=(?P<mean>[\d.]+)px "
                r"\(намічено: (?P<exp>[\d.]+)px, аномалій: (?P<anom>\d+)\)"), _b_crop_width),
    (re.compile(r"↕ VERTICAL DRIFT: raw_sum=(?P<raw>[\d.]+)px, applied_sum=(?P<app>[\d.]+)px, "
                r"max_cum=(?P<cum>[\d.]+)px"), _b_vdrift),

        # D9-hold: діагностика затриманих речень
    (re.compile(r"⏳ D9-hold: утримано (?P<t>.{10,80})"), lambda m: ("d9_hold", {"text": m["t"]})),
    (re.compile(r"✅ D9-release: склеєно до gap: \+(?P<n>\d+) слів"), lambda m: ("d9_release", {"added_words": int(m["n"])})),
]


def parse_log_message(msg: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Розпізнає координатний рядок логу → (type, fields) або None."""
    text = msg.strip().replace("\ufe0f", "")
    if not text:
        return None

    for rx, build in _PATTERNS:
        m = rx.match(text)
        if m:
            return build(m)
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
            if etype == "panorama_built":
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
    target = Path(output_dir) / filename
    for h in lg.handlers:
        if isinstance(h, CoordinatesTap) and h.path == target:
            return h
    tap = CoordinatesTap(Path(output_dir) / filename)
    lg.addHandler(tap)
    return tap

