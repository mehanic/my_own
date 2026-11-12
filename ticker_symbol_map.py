#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ticker_symbol_map.py — карта позицій символів і межі речень у рядку-тікері
(арабські новинні канали, письмо справа наліво).

Ідея
----
Ширина пробілу, яку повертає OCR, ненадійна: OCR губить роздільник («•»,
логотип каналу), склеює сусідні слова або рве рядок там, де роздільника немає.
Тому межі шукаємо у пікселях панорами, а не в OCR-боксах:

  1. чорнило  — темні безколірні пікселі відносно локального фону;
  2. кластери — безперервні по X відрізки чорнила (зв'язні шматки слів);
  3. пробіли  — три природні класи ширин, пороги рахуються з самого кадру:
        intra  (всередині слова, між незв'язними літерами)
        word   (між словами)
        sep    (поля довкола роздільника, ≥ ~1.8 × типового пробілу)
  4. роздільник речень — вузький ізольований гліф із sep-полями з обох боків
     (крапка-буліт або іконка-логотип). Усі роздільники одного тікера однакові,
     тому форма перевіряється шаблоном; шаблон можна зберегти у профіль каналу
     й знаходити знак навіть біля самого краю кадру;
  5. пунктуація всередині речення — за формою компонент у «власних» колонках
     (крапки арабських літер завжди лежать над/під тілом літери, пунктуація — ні):
        ":"  дві крапки одна над одною      "."  крапка на базовій лінії
        ".." кілька крапок поспіль           "-"  коротка горизонтальна риска

Вихід — JSON:
  boundaries  — межі речень (зона від слова до слова + позиція самого знака);
  sentences   — речення: x-межі, чим обмежені, чи повні, пунктуація всередині;
  symbol_map  — усі токени у порядку читання (справа наліво):
                word / space / punct / separator з координатами.

Використання
------------
  python ticker_symbol_map.py panorama_134.png -o out.json
  python ticker_symbol_map.py *.png -o out.json --debug-dir dbg --ocr
  python ticker_symbol_map.py pan.png --save-profile alarab.profile.json
  python ticker_symbol_map.py new.png --profile alarab.profile.json -o out.json

Як бібліотека:
  import ticker_symbol_map as tsm
  res, profile = tsm.analyze("panorama.png")
  tsm.classify_gap(res, 1650, 1688, scale=...)   # перевірити «підозрілий» ocr_gap
  tsm.assign_boxes(ocr_boxes, res, scale=...)    # розкласти свої OCR-бокси по реченнях

Залежності: opencv-python, numpy. Для --ocr потрібен tesseract з мовою ara.
Координати x0/x1 — пікселі вхідного зображення, x1 не включно.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Налаштування (розміри відносні — до висоти кадру або висоти літер «em»)
# --------------------------------------------------------------------------
DEFAULT_CFG = dict(
            ink_ratio=0.72,        # піксель = чорнило, якщо яскравість < ink_ratio * фон
            bg_min=150,            # фон тікера має бути світлим
            ink_resid=16,          # чорнило не має власного кольору (відхилення від t * фон)
            overlay_resid=60,      # яскраво-кольорові пікселі = сторонні оверлеї, маскуємо з околом
            close_k=0.30,          # ядро оцінки фону, частка висоти кадру
            rule_row_frac=0.85,    # рядок пікселів, зайнятий на >85 % = лінія-рамка
            min_comp_area=3,       # дрібніший шум викидаємо
            sep_gap_ratio=1.8,     # sep-поле ≥ 1.8 × медіани міжслівного пробілу
            sep_jump_ratio=1.6,    # або природний розрив у розподілі ширин
            sep_max_w_em=1.6,      # роздільник не ширший за 1.6 em
            sep_min_h_em=0.30,
            tpl_size=16,           # шаблон роздільника 16×16
            tpl_same=0.60,         # мінімальна схожість роздільників між собою
            tpl_match=0.75,        # схожість із шаблоном, коли одне з полів невідоме (край кадру)
            dot_max_em=0.45,       # «крапка» ≤ 0.45 em у будь-якому вимірі

            mark_sat_min=60,
            mark_max_h_em=1.2,
        )

class TickerSymbolMapper:
    """
    Піксельний аналізатор меж речень у тікері.
    Зберігає стан (вивчений профіль роздільника) між викликами.
    """
    def __init__(self, config: dict = None):
        self.cfg = config or DEFAULT_CFG
        self.profile = None  # Тут зберігатиметься вивчений шаблон роздільника (template)
        
    def set_profile(self, profile: dict):
        """Дозволяє завантажити профіль з channel_profiles.json"""
        self.profile = profile

    @staticmethod
    def _runs(mask) -> list[tuple[int, int]]:
        """Відрізки True у 1-D масці: [(start, end_exclusive), ...]."""
        m = np.asarray(mask, dtype=bool)
        d = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
        return [(int(s), int(e)) for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0])]

    @staticmethod
    def _otsu_1d(values) -> float | None:
        """Поріг Оцу між двома модами списку ширин; серед майже рівних — найширша «долина»."""
        v = np.sort(np.asarray(values, dtype=float))
        uniq = np.unique(v)
        if len(uniq) < 2:
            return None
        cands = []
        for a, b in zip(uniq[:-1], uniq[1:]):
            lo, hi = v[v <= a], v[v > a]
            cands.append((len(lo) * len(hi) * (lo.mean() - hi.mean()) ** 2, b - a, (a + b) / 2.0))
        top = max(c[0] for c in cands)
        return max((c for c in cands if c[0] >= 0.97 * top), key=lambda c: c[1])[2]

    @staticmethod
    def _tpl(dark: np.ndarray, c: dict, size: int) -> np.ndarray:
        """Нормалізований шаблон форми гліфа: півтонова «темність» у квадраті довкола центру."""
        side = max(c["w"], c["h"]) + 2
        cx, cy = (c["x0"] + c["x1"]) / 2.0, (c["y0"] + c["y1"]) / 2.0
        M = np.float32([[1, 0, side / 2.0 - cx], [0, 1, side / 2.0 - cy]])
        crop = cv2.warpAffine(dark, M, (side, side), flags=cv2.INTER_LINEAR, borderValue=0)
        t = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        t -= t.mean()
        n = float(np.linalg.norm(t))
        return t / n if n > 1e-6 else t

    @staticmethod
    def _ncc(a: np.ndarray, b: np.ndarray) -> float:
        return float((a * b).sum())


    # --------------------------------------------------------------------------
    # 1. Чорнило та смуга тексту
    # --------------------------------------------------------------------------
    def extract_ink(self, img: np.ndarray) -> dict:
        """ink  — булева маска чорнила; dark — півтонова карта чорнила (0..1);
        norm — нормалізоване зображення (білий фон, темний текст) для OCR;
        xr — робочі колонки; band — рядки смуги тексту."""
        H, W = img.shape[:2]
        V = img.max(axis=2)

        MN = img.min(axis=2)
        sat = (V.astype(np.int16) - MN.astype(np.int16)).clip(0, 255).astype(np.uint8)  # 🆕 насиченість

        k = max(9, int(round(H * self.cfg["close_k"])) | 1)
        ker = np.ones((k, k), np.uint8)
        # фон по кожному каналу: морфологічне закриття «заливає» тонкі темні штрихи
        bg3 = cv2.merge([cv2.morphologyEx(img[:, :, c], cv2.MORPH_CLOSE, ker) for c in range(3)]).astype(np.float32)
        bg = bg3.max(axis=2)
        # модель «чорне чорнило на фоні»: піксель = t * колір_фону, t — частка фону
        t = V.astype(np.float32) / np.maximum(bg, 1.0)
        resid = np.abs(img.astype(np.float32) - t[:, :, None] * bg3).max(axis=2)

                # яскраво-кольорові пікселі на світлому фоні = кандидати у знаки-роздільники
        colored = (sat > self.cfg["mark_sat_min"]) & (bg >= self.cfg["bg_min"])
        overlay = cv2.dilate((resid > self.cfg["overlay_resid"]).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        ink = (t < self.cfg["ink_ratio"]) & (bg >= self.cfg["bg_min"]) & (resid < self.cfg["ink_resid"]) & ~overlay

        # робочі колонки: найдовший відрізок зі світлим фоном (відсікає службові панелі)
        cr = self._runs(np.median(bg, axis=0) >= self.cfg["bg_min"])
        if not cr:
            raise ValueError("не знайдено світлої смуги тікера")
        x_lo, x_hi = max(cr, key=lambda r: r[1] - r[0])
        ink[:, :x_lo] = False
        ink[:, x_hi:] = False

        # суцільні горизонтальні лінії (рамка тікера) — не текст
        ink[ink[:, x_lo:x_hi].mean(axis=1) > self.cfg["rule_row_frac"], :] = False
        row_frac = ink[:, x_lo:x_hi].mean(axis=1)

        # смуга тексту: рядки, де фон світлий у ≥90 % колонок і є чорнило
        bright = np.percentile(bg[:, x_lo:x_hi], 10, axis=1) >= self.cfg["bg_min"] - 10
        rr = self._runs(bright & (row_frac > 0.003))
        if not rr:
            raise ValueError("не знайдено рядка тексту")
        # текст = багато переходів чорнило/фон у рядку; суцільні темні плями дають мало
        trans = (ink[:, x_lo + 1:x_hi] != ink[:, x_lo:x_hi - 1]).sum(axis=1)
        y0, y1 = max(rr, key=lambda r: trans[r[0]:r[1]].sum())
        ink[:y0, :] = False
        ink[y1:, :] = False

        colored[:y0, :] = False
        colored[y1:, :] = False
        colored[:, :x_lo] = False
        colored[:, x_hi:] = False

        dark = np.where(ink, 1.0 - t, 0.0).astype(np.float32)
        norm = np.full((H, W), 255, np.uint8)
        norm[y0:y1, x_lo:x_hi] = np.where(overlay, 255, np.clip(t * 255, 0, 255))[y0:y1, x_lo:x_hi].astype(np.uint8)
        #return dict(ink=ink, dark=dark, norm=norm, xr=(int(x_lo), int(x_hi)), band=(int(y0), int(y1)))

        return dict(ink=ink, dark=dark, norm=norm, xr=(int(x_lo), int(x_hi)), band=(int(y0), int(y1)),
                    sat=sat, colored=colored)   # sat/colored для кольорових знаків


    # --------------------------------------------------------------------------
    # 2. Кластери (колонкові відрізки чорнила) та зв'язні компоненти
    # --------------------------------------------------------------------------
    def _mk_cluster(self, cc: list[dict]) -> dict:
        x0, x1 = min(c["x0"] for c in cc), max(c["x1"] for c in cc)
        y0, y1 = min(c["y0"] for c in cc), max(c["y1"] for c in cc)
        return dict(x0=x0, x1=x1, w=x1 - x0, y0=y0, y1=y1, h=y1 - y0, area=sum(c["area"] for c in cc),
                    comps=cc, kind="glyph")


    def build_clusters(self, ink: np.ndarray) -> list[dict]:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
        comps = []
        for i in range(1, n):
            x, y, w, h, a = (int(v) for v in stats[i])
            if a < self.cfg["min_comp_area"]:
                ink[lab == i] = False
                continue
            comps.append(dict(x0=x, x1=x + w, y0=y, y1=y + h, w=w, h=h, area=a))
        comps.sort(key=lambda c: c["x0"])
        clusters = []
        for s, e in self._runs(ink.any(axis=0)):
            clusters.append(self._mk_cluster([c for c in comps if c["x0"] >= s and c["x1"] <= e]))
        return clusters

    # def add_colored_marks(self, px: dict, clusters: list, em: float) -> list:
    #     """🆕 Кольорові знаки-роздільники (червона крапка • на білому фоні).
    #     Компактні насичені компоненти додаються як звичайні кластери;
    #     маски ink/dark/norm оновлюються, щоб шаблон NCC бачив форму знака."""
    #     colored = px.get("colored")
    #     if colored is None or not colored.any():
    #         return clusters
    #     sat = px["sat"]
    #     n, lab, stats, _ = cv2.connectedComponentsWithStats(colored.astype(np.uint8), 8)
    #     added, ids = [], []
    #     for i in range(1, n):
    #         x, y, w, h, a = (int(v) for v in stats[i])
    #         if a < self.cfg["min_comp_area"]:
    #             continue
    #         if w > self.cfg["sep_max_w_em"] * em or h > self.cfg["mark_max_h_em"] * em:
    #             continue                      # велика кольорова пляма = оверлей/логотип, не знак
    #         if h < self.cfg["sep_min_h_em"] * em or w < 2:
    #             continue
    #         ids.append(i)
    #         added.append(self._mk_cluster(
    #             [dict(x0=x, x1=x + w, y0=y, y1=y + h, w=w, h=h, area=int(a))]))
    #     if not added:
    #         return clusters
    #     mark = np.zeros(colored.shape, dtype=bool)
    #     for i in ids:
    #         mark |= (lab == i)
    #     # знак стає чорнилом; темність беремо з насиченості (для шаблону форми)
    #     px["ink"] = px["ink"] | mark
    #     px["dark"] = np.maximum(px["dark"],
    #                             np.where(mark, sat.astype(np.float32) / 255.0, 0.0)).astype(np.float32)
    #     px["norm"] = np.where(mark,
    #                           np.clip(255 - sat.astype(np.int16), 0, 255).astype(np.uint8),
    #                           px["norm"])
    #     return sorted(clusters + added, key=lambda c: c["x0"])
    
    #поміняв коли тестували iraqianews  для розпізнавання вертикальних
    def add_colored_marks(self, px: dict, clusters: list, em: float) -> list:
        """🆕 Кольорові знаки-роздільники (червона крапка • на білому фоні)."""
        colored = px.get("colored")
        if colored is None or not colored.any():
            import logging
            logging.getLogger(__name__).warning("🔍 TSM add_colored_marks: colored маска порожня!")
            return clusters
        
        sat = px["sat"]
        n, lab, stats, _ = cv2.connectedComponentsWithStats(colored.astype(np.uint8), 8)
        added, ids = [], []
        
        # 🆕 ДІАГНОСТИКА
        candidates = 0
        rejected_size = 0
        rejected_small = 0
        
        for i in range(1, n):
            x, y, w, h, a = (int(v) for v in stats[i])
            if a < self.cfg["min_comp_area"]:
                continue
            
            candidates += 1
            
            # 🆕 АБСОЛЮТНИЙ ФІЛЬТР для великих прямокутників (Iraqia News)
            is_large_separator = (10 <= w <= 35) and (15 <= h <= 50)
            
            if not is_large_separator:
                # Стандартний фільтр для дрібних крапок/іконок
                if w > self.cfg["sep_max_w_em"] * em or h > self.cfg["mark_max_h_em"] * em:
                    rejected_size += 1
                    if candidates <= 5:  # Логуємо перші 5 відхилених
                        import logging
                        logging.getLogger(__name__).warning(
                            f"🔍 TSM add_colored_marks: відхилено компонент #{i}: "
                            f"w={w}px, h={h}px, em={em:.1f}px, "
                            f"max_w={self.cfg['sep_max_w_em']*em:.0f}px, "
                            f"max_h={self.cfg['mark_max_h_em']*em:.0f}px"
                        )
                    continue
                if h < self.cfg["sep_min_h_em"] * em or w < 2:
                    rejected_small += 1
                    continue
            
            ids.append(i)
            added.append(self._mk_cluster(
                [dict(x0=x, x1=x + w, y0=y, y1=y + h, w=w, h=h, area=int(a))]))
        
        # 🆕 ФІНАЛЬНА ДІАГНОСТИКА
        import logging
        logging.getLogger(__name__).info(
            f"🔍 TSM add_colored_marks: candidates={candidates}, "
            f"added={len(added)}, rejected_size={rejected_size}, rejected_small={rejected_small}"
        )
        
        if not added:
            return clusters
        
        mark = np.zeros(colored.shape, dtype=bool)
        for i in ids:
            mark |= (lab == i)
        
        px["ink"] = px["ink"] | mark
        px["dark"] = np.maximum(px["dark"],
                                np.where(mark, sat.astype(np.float32) / 255.0, 0.0)).astype(np.float32)
        px["norm"] = np.where(mark,
                            np.clip(255 - sat.astype(np.int16), 0, 255).astype(np.uint8),
                            px["norm"])
        return sorted(clusters + added, key=lambda c: c["x0"])


    def _is_dot(self, c: dict, dmax: float) -> bool:
        return (max(c["w"], c["h"]) <= dmax and 0.5 <= c["w"] / c["h"] <= 2.2
                and c["area"] >= 0.5 * c["w"] * c["h"])


    def split_free_dots(self, clusters: list[dict], em: float) -> list[dict]:
        """Двокрапка часто стоїть впритул до літери (0 px між колонками) і зливається з нею
        в один кластер. Відокремлюємо крапки, у чиїх колонках немає іншого чорнила."""
        dmax = self.cfg["dot_max_em"] * em
        out = []
        for cl in clusters:
            cc = cl["comps"]
            free = [c for c in cc if self._is_dot(c, dmax)]
            if len(cc) < 2 or not free or len(free) == len(cc):
                out.append(cl)
                continue
            changed = True
            while changed and free:
                changed = False
                rest = [c for c in cc if not any(c is f for f in free)]
                for d in list(free):
                    if any(d["x0"] < r["x1"] and r["x0"] < d["x1"] for r in rest):
                        free = [f for f in free if f is not d]
                        changed = True
            if not free:
                out.append(cl)
                continue
            free.sort(key=lambda c: c["x0"])
            groups = [[free[0]]]
            for d in free[1:]:
                if d["x0"] < max(c["x1"] for c in groups[-1]):
                    groups[-1].append(d)
                else:
                    groups.append([d])
            rest = [c for c in cc if not any(c is f for f in free)]
            cuts = [min(c["x0"] for c in g) for g in groups]
            buckets: dict[int, list] = {}
            for r in rest:
                buckets.setdefault(sum(r["x0"] >= x for x in cuts), []).append(r)
            parts = [self._mk_cluster(g) for g in groups] + [self._mk_cluster(b) for b in buckets.values()]
            out.extend(sorted(parts, key=lambda c: c["x0"]))
        return out


    def text_metrics(self, ink: np.ndarray, clusters: list[dict], xr) -> dict:
        """em — типова висота літери; базова лінія з урахуванням нахилу панорами."""
        hs = [c["h"] for c in clusters if c["w"] >= 4]
        em = float(np.median(hs)) if hs else float(ink.shape[0]) / 3
        x_lo, x_hi = xr
        xs, yc = [], []
        step = max(32, int(em * 4))
        for a in range(x_lo, x_hi - step + 1, step // 2):   # центр мас чорнила у вікнах
            ys = np.where(ink[:, a:a + step])[0]
            if len(ys) > 30:
                xs.append(a + step / 2)
                yc.append(float(ys.mean()))
        slope, icpt = 0.0, float(np.mean(yc)) if yc else ink.shape[0] / 2
        if len(xs) >= 6:                                     # робастна пряма
            xs_a, yc_a = np.array(xs), np.array(yc)
            for _ in range(3):
                slope, icpt = np.polyfit(xs_a, yc_a, 1)
                res = np.abs(yc_a - (slope * xs_a + icpt))
                keep = res <= max(1.0, np.percentile(res, 80))
                if keep.sum() < 4:
                    break
                xs_a, yc_a = xs_a[keep], yc_a[keep]
        # базова лінія = найщільніший рядок після вирівнювання нахилу
        H = ink.shape[0]
        hist = np.zeros(3 * H)
        yy, xx = np.where(ink)
        rel = np.round(yy - (slope * xx + icpt)).astype(int) + H
        np.add.at(hist, np.clip(rel, 0, 3 * H - 1), 1)
        return dict(em=em, slope=float(slope), icpt=float(icpt), base_rel=int(np.argmax(hist)) - H)


    def baseline_at(self, m: dict, x: float) -> float:
        return m["slope"] * x + m["icpt"] + m["base_rel"]


    # --------------------------------------------------------------------------
    # 3. Класи пробілів
    # --------------------------------------------------------------------------
    def gap_thresholds(self, widths: list[int]):
        """(t_word, t_sep, info). Пороги динамічні — з розподілу ширин пробілів у кадрі."""
        ws = sorted(w for w in widths if w >= 1)
        info: dict = dict(n_gaps=len(ws))
        if len(ws) < 4:
            return None, None, info
        # intra / word: Оцу на нижніх 90 % (щоб sep-клас не перетягував поріг)
        t_word = self._otsu_1d([w for w in ws if w <= np.percentile(ws, 90)]) or self._otsu_1d(ws)
        if t_word is None:
            return None, None, info
        word_gaps = [w for w in ws if w > t_word]
        word_med = float(np.median(word_gaps)) if word_gaps else None
        info.update(word_space_median=word_med)
        if not word_med:
            return float(t_word), None, info

        # (а) природний розрив у верхній частині розподілу: рідкісні й значно ширші
        t_jump, best = None, 0.0
        uniq = sorted(set(ws))
        for a, b in zip(uniq[:-1], uniq[1:]):
            if a <= t_word:
                continue
            ratio = b / a
            if ratio >= self.cfg["sep_jump_ratio"] and sum(w >= b for w in ws) <= 0.25 * len(ws) and ratio > best:
                best, t_jump = ratio, (a + b) / 2.0
        # (б) кратність до типового міжслівного пробілу
        t_ratio = self.cfg["sep_gap_ratio"] * word_med
        t_sep = t_jump if (t_jump and t_jump >= 1.4 * word_med) else t_ratio
        info.update(t_jump=t_jump, t_ratio=t_ratio)
        return float(t_word), float(t_sep), info


    # --------------------------------------------------------------------------
    # 4. Роздільники речень
    # --------------------------------------------------------------------------
    def detect_separators(self, dark, clusters, t_sep, m, profile=None):
        """Позначає clusters[i]['kind'] = 'separator'. Повертає (шаблон, (w, h)) або (None, None)."""
        em = m["em"]
        S = self.cfg["tpl_size"]
        n = len(clusters)
        for i, c in enumerate(clusters):
            c["gap_l"] = c["x0"] - clusters[i - 1]["x1"] if i > 0 else None       # None = край кадру
            c["gap_r"] = clusters[i + 1]["x0"] - c["x1"] if i < n - 1 else None

        def shape_ok(c):
            return (c["kind"] == "glyph" and c["w"] <= self.cfg["sep_max_w_em"] * em
                    and c["h"] >= self.cfg["sep_min_h_em"] * em)

        tpl, tpl_wh = None, None
        if profile and profile.get("template"):
            tpl = np.array(profile["template"], dtype=np.float32)
            tpl_wh = (profile["w_em"] * em, profile["h_em"] * em)

        # (а) ізольовані гліфи: sep-поля з обох боків
        iso = []
        if t_sep:
            iso = [c for c in clusters if shape_ok(c) and c["gap_l"] is not None and c["gap_r"] is not None
                and c["gap_l"] >= t_sep and c["gap_r"] >= t_sep]
        if iso and tpl is None:
            ts = [self._tpl(dark, c, S) for c in iso]
            # медоїд: гліф, найбільш схожий на решту (роздільник повторюється);
            # за рівності — той, у кого ширші поля
            score = [sum(self._ncc(a, b) for b in ts) + 0.01 * min(c["gap_l"], c["gap_r"]) / t_sep
                    for a, c in zip(ts, iso)]
            k = int(np.argmax(score))
            group = [c for c, t in zip(iso, ts) if self._ncc(ts[k], t) >= self.cfg["tpl_same"]]
            tpl = np.mean([self._tpl(dark, c, S) for c in group], axis=0)
            tpl /= max(float(np.linalg.norm(tpl)), 1e-6)
            tpl_wh = (float(np.median([c["w"] for c in group])), float(np.median([c["h"] for c in group])))
        for c in iso:
            sc = self._ncc(tpl, self._tpl(dark, c, S))
            if sc >= self.cfg["tpl_same"]:          # сторонній уламок не схожий на решту — пропускаємо
                c.update(kind="separator", tpl_score=round(sc, 3), how="isolated")

        # (б) збіг із шаблоном там, де одне з полів невідоме (край кадру) або трохи вужче
        if tpl is not None:
            lim = 0.75 * t_sep if t_sep else 0.75 * em * (profile or {}).get("side_gap_em", 0.8)
            for c in clusters:
                if not shape_ok(c):
                    continue
                if not (0.7 * tpl_wh[0] <= c["w"] <= 1.35 * tpl_wh[0] and 0.7 * tpl_wh[1] <= c["h"] <= 1.35 * tpl_wh[1]):
                    continue
                sc = self._ncc(tpl, self._tpl(dark, c, S))
                sides = [g for g in (c["gap_l"], c["gap_r"]) if g is not None]
                if sc >= self.cfg["tpl_match"] and sides and all(g >= lim for g in sides):
                    c.update(kind="separator", tpl_score=round(sc, 3), how="template")

        for c in clusters:                     # тип знака
            if c["kind"] != "separator":
                continue
            fill = c["area"] / float(c["w"] * c["h"])
            if len(c["comps"]) == 1 and fill >= 0.68 and 0.7 <= c["w"] / c["h"] <= 1.4 and c["h"] <= 0.8 * em:
                c.update(char="•", name="bullet")
            else:
                c.update(char="◉", name="icon")   # графічний знак (логотип); символ умовний
        return tpl, tpl_wh


    # --------------------------------------------------------------------------
    # 5. Пунктуація за формою
    # --------------------------------------------------------------------------
    def classify_punct(self, clusters, m):
        em = m["em"]
        dmax = self.cfg["dot_max_em"] * em
        for idx, cl in enumerate(clusters):
            if cl["kind"] != "glyph" or idx in (0, len(clusters) - 1):   # крайні кластери — обрізки
                continue
            cc = cl["comps"]
            base = self.baseline_at(m, (cl["x0"] + cl["x1"]) / 2)
            if len(cc) == 2 and all(self._is_dot(c, dmax) for c in cc):
                a, b = sorted(cc, key=lambda c: c["y0"])
                ov = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
                if ov >= 0.5 * min(a["w"], b["w"]) and b["y0"] - a["y1"] >= 1 and cl["h"] <= 1.1 * em:
                    cl.update(kind="punct", char=":", name="colon")
            elif len(cc) == 1:
                c = cc[0]
                dot = self._is_dot(c, dmax)
                mid = base - 0.35 * em
                if dot and abs(c["y1"] - 1 - base) <= max(2.0, 0.2 * em):
                    cl.update(kind="punct", char=".", name="dot")
                elif (c["w"] >= 1.8 * c["h"] and 2 <= c["h"] <= 0.3 * em and c["w"] <= em
                        and c["area"] >= 0.7 * c["w"] * c["h"]
                        and abs((c["y0"] + c["y1"] - 1) / 2 - mid) <= 0.25 * em):
                    cl.update(kind="punct", char="-", name="dash")
                elif dot:
                    cl.update(kind="punct", char="·", name="mid_dot")


    # --------------------------------------------------------------------------
    # 6. Карта символів і речення
    # --------------------------------------------------------------------------
    def build_map(self, clusters, t_word, t_sep, m):
        """Токени у порядку читання (справа наліво) + речення між роздільниками."""
        em = m["em"]
        tokens: list[dict] = []
        word: list[dict] = []

        def flush_word():
            if word:
                x0, x1 = min(c["x0"] for c in word), max(c["x1"] for c in word)
                tokens.append(dict(type="word", x0=x0, x1=x1, w=x1 - x0,
                                glyphs=[[c["x0"], c["x1"]] for c in word]))
                word.clear()

        prev = None
        for c in sorted(clusters, key=lambda c: -c["x0"]):
            if prev is not None:
                gap = prev["x0"] - c["x1"]
                same_word = c["kind"] == "glyph" and prev["kind"] == "glyph" and gap <= t_word
                if not same_word:
                    flush_word()
                    if gap > 0:
                        cls = "intra" if gap <= t_word else ("wide" if (t_sep and gap >= t_sep) else "word")
                        tokens.append(dict(type="space", x0=c["x1"], x1=prev["x0"], w=gap, cls=cls))
            if c["kind"] == "glyph":
                word.append(c)
            else:
                tok = dict(type=c["kind"], char=c["char"], name=c["name"], x0=c["x0"], x1=c["x1"], w=c["w"],
                        y0=c["y0"], y1=c["y1"])
                if c["kind"] == "separator":
                    g = [v for v in (c["gap_l"], c["gap_r"]) if v is not None]
                    conf = 0.5 * min(1.5, min(g) / (t_sep or em)) / 1.5 + 0.5 * max(0.0, c["tpl_score"])
                    tok.update(gap_left=c["gap_l"], gap_right=c["gap_r"], tpl_score=c["tpl_score"],
                            how=c["how"], confidence=round(min(1.0, conf), 2))
                tokens.append(tok)
            prev = c
        flush_word()

        # кілька крапок поспіль → ".." / "..."
        merged: list[dict] = []
        for t in tokens:
            if (t["type"] == "punct" and t["name"] == "dot" and len(merged) >= 2
                    and merged[-1]["type"] == "space" and merged[-1]["cls"] == "intra"
                    and merged[-2]["type"] == "punct" and merged[-2]["name"] in ("dot", "dots")):
                merged.pop()
                p = merged[-1]
                p.update(x0=t["x0"], w=p["x1"] - t["x0"], char=p["char"] + ".", name="dots",
                        y0=min(p["y0"], t["y0"]), y1=max(p["y1"], t["y1"]))
                continue
            merged.append(t)
        tokens = merged
        for i, t in enumerate(tokens):
            t["i"] = i
            # поле роздільника — «sep»; широкий пробіл без знака лишається «wide»
            if t["type"] == "space" and t["cls"] == "wide":
                nb = [tokens[j]["type"] for j in (i - 1, i + 1) if 0 <= j < len(tokens)]
                if "separator" in nb:
                    t["cls"] = "sep"

        # речення: шматки між роздільниками; без роздільника з боку — обрізане краєм кадру
        sentences: list[dict] = []
        cur: list[dict] = []
        start_sep = None

        def close(end_sep):
            body = [t for t in cur if t["type"] != "space"]
            if not body:
                return
            x1, x0 = max(t["x1"] for t in body), min(t["x0"] for t in body)
            s = dict(
                idx=len(sentences), x_start=x1, x_end=x0, width=x1 - x0,
                start=dict(by="separator", token=start_sep["i"]) if start_sep else dict(by="frame_edge"),
                end=dict(by="separator", token=end_sep["i"]) if end_sep else dict(by="frame_edge"),
                complete=bool(start_sep and end_sep),
                tokens=[body[0]["i"], body[-1]["i"]],
                n_words=sum(t["type"] == "word" for t in body),
                punct=[dict(char=t["char"], name=t["name"], x0=t["x0"], x1=t["x1"], token=t["i"])
                    for t in body if t["type"] == "punct"],
            )
            sentences.append(s)

        for t in tokens:
            if t["type"] == "separator":
                close(t)
                cur, start_sep = [], t
            else:
                cur.append(t)
        close(None)
        return tokens, sentences


    # --------------------------------------------------------------------------
    # 7. Необов'язковий OCR кожного речення окремо (tesseract)
    # --------------------------------------------------------------------------
    def ocr_sentences(self, norm, sentences, band, slope, lang="ara"):
        """Кожне речення розпізнається окремим шматком — склеїтись через роздільник нічому."""
        exe = shutil.which("tesseract")
        if not exe:
            print("  [ocr] tesseract не знайдено — пропускаю", file=sys.stderr)
            return
        H, W = norm.shape
        with tempfile.TemporaryDirectory() as td:
            for s in sentences:
                a, b = max(0, s["x_end"] - 6), min(W, s["x_start"] + 6)
                crop = norm[:, a:b]
                M = np.float32([[1, 0, 0], [-slope, 1, slope * (b - a) / 2.0]])    # вирівняти нахил
                crop = cv2.warpAffine(crop, M, (b - a, H), flags=cv2.INTER_LINEAR, borderValue=255)
                rows = np.where((crop < 160).any(axis=1))[0]
                if len(rows) == 0:
                    continue
                crop = crop[max(0, rows.min() - 3):rows.max() + 4]
                crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                crop = cv2.copyMakeBorder(crop, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255)
                p = os.path.join(td, f"s{s['idx']}.png")
                cv2.imwrite(p, crop)
                r = subprocess.run([exe, p, "stdout", "-l", lang, "--psm", "7"],
                                capture_output=True, text=True, encoding="utf-8")
                s["text_ocr"] = " ".join(r.stdout.split())


    # --------------------------------------------------------------------------
    # Інтеграція у власний конвеєр
    # --------------------------------------------------------------------------
    def classify_gap(self, result: dict, x0: float, x1: float, scale: float = 1.0) -> dict:
        """Перевірка «підозрілого» проміжку між OCR-боксами (ваша подія ocr_gap).
        scale = px_цього_зображення / px_ваших_координат.
        boundary  — у проміжку стоїть роздільник: справжня межа речень;
        lost_text — у проміжку є чорнило без роздільника: OCR загубив слово/число, межі немає;
        blank     — чорнила немає і роздільника немає: порожнє місце, межу не підтверджено."""
        a, b = sorted((x0 * scale, x1 * scale))
        inside = [t for t in result["symbol_map"] if t["type"] != "space" and t["x1"] > a and t["x0"] < b]
        seps = [t for t in inside if t["type"] == "separator"]
        if seps:
            return dict(verdict="boundary", mark=seps[0]["char"], mark_x=[seps[0]["x0"], seps[0]["x1"]],
                        confidence=seps[0]["confidence"])
        lost = [t for t in inside if min(t["x1"], b) - max(t["x0"], a) >= 0.5 * t["w"]]
        if lost:
            return dict(verdict="lost_text", tokens=[t["i"] for t in lost])
        return dict(verdict="blank")


    def assign_boxes(self, boxes: list[dict], result: dict, scale: float = 1.0) -> list[str]:
        """boxes: [{'x0','x1','text'}, ...] — слова/фрагменти вашого OCR.
        Повертає тексти речень, зібрані за X-центрами боксів (справа наліво)."""
        out: list[list] = [[] for _ in result["sentences"]]
        for bx in boxes:
            xc = (bx["x0"] + bx["x1"]) / 2 * scale
            for s in result["sentences"]:
                if s["x_end"] - 2 <= xc <= s["x_start"] + 2:
                    out[s["idx"]].append((xc, bx["text"]))
                    break
        return [" ".join(t for _, t in sorted(v, key=lambda p: -p[0])) for v in out]


# --------------------------------------------------------------------------
# Головна функція
# --------------------------------------------------------------------------
    def analyze_image(self, img: np.ndarray, ocr: bool = False, debug_dir: str | None = None,
                      source_name: str | None = None) -> tuple[dict, dict | None]:
        """Повертає (result, new_profile).
        Профіль береться з self.profile; якщо вивчено новий шаблон —
        self.profile оновлюється автоматично (пам'ять класу)."""
        if img is None or img.size == 0:
            raise ValueError("порожнє зображення")
        H, W = img.shape[:2]
        px = self.extract_ink(img)
        ink, xr, band = px["ink"], px["xr"], px["band"]
        clusters = self.build_clusters(ink)
        if len(clusters) < 3:
            raise ValueError("замало тексту в кадрі")
        m = self.text_metrics(ink, clusters, xr)
        clusters = self.split_free_dots(clusters, m["em"])
        clusters = self.add_colored_marks(px, clusters, m["em"])   # червоні/кольорові роздільники

        gaps = [g for g in (clusters[i + 1]["x0"] - clusters[i]["x1"] for i in range(len(clusters) - 1)) if g > 0]
        t_word, t_sep, ginfo = self.gap_thresholds(gaps)
        if t_word is None:
            t_word = 0.3 * m["em"]
        tpl, tpl_wh = self.detect_separators(px["dark"], clusters, t_sep, m, profile=self.profile)
        self.classify_punct(clusters, m)          # після роздільників: буліт не сплутати з крапкою
        tokens, sentences = self.build_map(clusters, t_word, t_sep, m)
        if ocr:
            self.ocr_sentences(px["norm"], sentences, band, m["slope"])

        seps = [t for t in tokens if t["type"] == "separator"]
        side = [g for t in seps for g in (t["gap_left"], t["gap_right"]) if g is not None]
        hist: dict[int, int] = {}
        for g in gaps:
            hist[g] = hist.get(g, 0) + 1
        result = dict(
            file=source_name,
            image=dict(width=W, height=H, work_x=list(xr), text_band_y=list(band)),
            direction="rtl",
            metrics=dict(em=round(m["em"], 1), skew_px_per_1000=round(m["slope"] * 1000, 2),
                         baseline_y_left=round(self.baseline_at(m, xr[0]), 1),
                         baseline_y_right=round(self.baseline_at(m, xr[1]), 1)),
            gaps=dict(histogram={str(k): hist[k] for k in sorted(hist)},
                      t_word=t_word, t_sep=t_sep, word_space_median=ginfo.get("word_space_median"),
                      classes="intra ≤ t_word < word < t_sep ≤ sep|wide"),
            separator=dict(count=len(seps), char=seps[0]["char"] if seps else None,
                           name=seps[0]["name"] if seps else None,
                           glyph_w=tpl_wh[0] if (seps and tpl_wh) else None,
                           glyph_h=tpl_wh[1] if (seps and tpl_wh) else None,
                           side_gap_median=float(np.median(side)) if side else None),
            boundaries=[dict(x0=t["x0"] - (t["gap_left"] or 0), x1=t["x1"] + (t["gap_right"] or 0),
                             mark_x0=t["x0"], mark_x1=t["x1"], char=t["char"], confidence=t["confidence"])
                        for t in seps],
            wide_gaps=[dict(x0=t["x0"], x1=t["x1"], w=t["w"]) for t in tokens
                       if t["type"] == "space" and t["cls"] == "wide"],
            sentences=sentences,
            symbol_map=tokens,
        )
        new_profile = None
        if tpl is not None and seps and side:
            new_profile = dict(name=seps[0]["name"], char=seps[0]["char"],
                               w_em=round(tpl_wh[0] / m["em"], 3), h_em=round(tpl_wh[1] / m["em"], 3),
                               side_gap_em=round(float(np.median(side)) / m["em"], 3),
                               template=[[round(float(v), 4) for v in row] for row in tpl])
            self.profile = new_profile   # 🔑 єдина нова дія: клас запам'ятовує шаблон
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            base = os.path.splitext(source_name or "image")[0]
            self.draw_debug(img, result, os.path.join(debug_dir, base + "_map.png"))
        return result, new_profile

    def analyze(self, path: str, ocr: bool = False, debug_dir: str | None = None) -> tuple[dict, dict | None]:
        """Повертає (result, profile). profile — вивчений шаблон роздільника або None.
        Ваша відтестована функція: читає файл і делегує всю логіку в analyze_image."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return self.analyze_image(img, ocr=ocr, debug_dir=debug_dir, source_name=os.path.basename(path))


    @staticmethod
    def draw_debug(img, res, out_path, zoom=2):
        """PNG з розміткою: червоне — роздільники та їхні поля, синє — пунктуація,
        кольорові лінії знизу — речення (зірочка = обрізане краєм кадру)."""
        H, W = img.shape[:2]
        pad, z = 14, zoom
        canvas = np.full((H + 2 * pad, W, 3), 255, np.uint8)
        canvas[pad:pad + H] = img
        canvas = cv2.resize(canvas, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
        palette = [(200, 120, 0), (0, 150, 0), (160, 0, 160), (0, 120, 200)]
        for s in res["sentences"]:
            col = palette[s["idx"] % len(palette)]
            y = (pad + H + 5) * z
            cv2.line(canvas, (s["x_end"] * z, y), (s["x_start"] * z, y), col, 3)
            tag = f'S{s["idx"]}' + ("" if s["complete"] else "*")
            cv2.putText(canvas, tag, (max(0, s["x_start"] * z - 34), (pad + H + 13) * z), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, col, 1, cv2.LINE_AA)
        for t in res["symbol_map"]:
            if t["type"] == "separator":
                cv2.rectangle(canvas, (t["x0"] * z - 3, (pad + t["y0"]) * z - 3),
                            (t["x1"] * z + 2, (pad + t["y1"]) * z + 2), (0, 0, 255), 2)
                xa, xb = (t["x0"] - (t["gap_left"] or 0)) * z, (t["x1"] + (t["gap_right"] or 0)) * z
                cv2.line(canvas, (xa, 6), (xb, 6), (0, 0, 255), 2)
                cv2.line(canvas, (xa, 2), (xa, 12), (0, 0, 255), 1)
                cv2.line(canvas, (xb, 2), (xb, 12), (0, 0, 255), 1)
            elif t["type"] == "punct":
                cv2.rectangle(canvas, (t["x0"] * z - 2, (pad + t["y0"]) * z - 2),
                            (t["x1"] * z + 1, (pad + t["y1"]) * z + 1), (255, 100, 0), 1)
                cv2.putText(canvas, t["char"] if t["char"].isascii() else ".", (t["x0"] * z - 2, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1, cv2.LINE_AA)
            elif t["type"] == "space" and t["cls"] == "word":
                xm = (t["x0"] + t["x1"]) * z // 2
                cv2.line(canvas, (xm, (pad + H) * z), (xm, (pad + H + 2) * z), (150, 150, 150), 1)
            elif t["type"] == "space" and t["cls"] == "wide":
                cv2.line(canvas, (t["x0"] * z, 6), (t["x1"] * z, 6), (0, 160, 255), 2)
        cv2.imwrite(out_path, canvas)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Карта позицій символів і межі речень у тікері (RTL).")
    ap.add_argument("images", nargs="+", help="панорами / кадри тікера (PNG, JPG)")
    ap.add_argument("-o", "--out", default="ticker_symbol_map.json", help="вихідний JSON")
    ap.add_argument("--profile", help="профіль каналу (шаблон роздільника) з --save-profile")
    ap.add_argument("--save-profile", help="зберегти вивчений шаблон роздільника у файл")
    ap.add_argument("--ocr", action="store_true", help="розпізнати текст кожного речення окремо (tesseract -l ara)")
    ap.add_argument("--debug-dir", help="куди покласти PNG з розміткою")
    ap.add_argument("--compact", action="store_true", help="без symbol_map (лише межі й речення)")
    a = ap.parse_args(argv)

    # 🆕 Створюємо екземпляр класу
    mapper = TickerSymbolMapper()
    
    # Завантажуємо профіль, якщо вказано
    if a.profile:
        with open(a.profile, encoding="utf-8") as f:
            mapper.set_profile(json.load(f))
    
    results = []
    for p in a.images:
        try:
            res, new_profile = mapper.analyze(p, ocr=a.ocr, debug_dir=a.debug_dir)
        except (ValueError, FileNotFoundError) as e:      # кадр без тексту тощо
            results.append(dict(file=os.path.basename(p), error=str(e)))
            print(f"{os.path.basename(p)}: ПОМИЛКА {e}", file=sys.stderr)
            continue
        if a.compact:
            res.pop("symbol_map")
        results.append(res)
        print(f'{res["file"]}: речень {len(res["sentences"])}, роздільників {res["separator"]["count"]} '
              f'({res["separator"]["name"]}), t_word={res["gaps"]["t_word"]}, t_sep={res["gaps"]["t_sep"]}')
    
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(dict(tool="ticker_symbol_map", version=1, results=results), f, ensure_ascii=False, indent=1)
    
    # Зберігаємо вивчений профіль, якщо потрібно
    if a.save_profile and mapper.profile:
        with open(a.save_profile, "w", encoding="utf-8") as f:
            json.dump(mapper.profile, f, ensure_ascii=False)
    elif a.save_profile:
        print("профіль не збережено: у кадрах немає роздільника з полями з обох боків", file=sys.stderr)
    
    return 0