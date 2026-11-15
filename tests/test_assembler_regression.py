#!/usr/bin/env python3
"""Регресійні тести асемблера (без зовнішніх залежностей).

Запуск:  python3 tests/test_assembler_regression.py
         (або pytest tests/)
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from ticker_assembler import TickerAssembler, TickerFragment  # noqa: E402

HEADLINE = ("الجيش الإسرائيلي يعلن استهداف مواقع لحزب الله "
            "في جنوب لبنان خلال الساعات الماضية")
SHORT_JUNK = "قصف مدفعي"          # 2 слова: проходить повз гард overlap


def test_short_fragment_does_not_wipe_buffer():
    """Фрагмент із <3 слів без перекриття НЕ має затирати накопичену новину.

    Гард `overlap < 0.10 and len(prev_words) >= 3 and len(new_words) >= 3`
    пропускає короткі фрагменти далі, а merge_fragments раніше повертав
    для них new_text — накопичений заголовок зникав назавжди.
    """
    a = TickerAssembler("test")
    a.add_fragment(TickerFragment(text=HEADLINE, timestamp=1.0,
                                  frame_index=1, confidence=0.9))
    assert len(a.current_message.text) > 50

    ok, reason = a.add_fragment(TickerFragment(text=SHORT_JUNK, timestamp=2.0,
                                               frame_index=2, confidence=0.9))

    assert ok is False, "асемблер має відмовити, а не мовчки злити буфер"
    assert reason == "no_logical_connection"
    assert a.current_message.text == HEADLINE, "буфер затерто — новина втрачена"


def test_merge_fragments_signals_instead_of_replacing():
    """merge_fragments повертає None (сигнал flush+retry), а не новий текст."""
    a = TickerAssembler("test")
    frag = TickerFragment(text=SHORT_JUNK, timestamp=2.0,
                          frame_index=2, confidence=0.9)
    assert a.merge_fragments(HEADLINE, frag) is None


def test_awaiting_continuation_set_on_creation():
    """H-група (left_incomplete=True) створює буфер → прапорець має бути True.

    Інакше гілка «КОНВЕЄР» у add_fragment недосяжна і продовження з
    наступної панорами відхиляється як no_logical_connection.
    """
    a = TickerAssembler("test")
    a.add_fragment(TickerFragment(text=HEADLINE, timestamp=1.0, frame_index=1,
                                  confidence=0.9, left_incomplete=True))
    assert a.current_message.awaiting_continuation is True


def test_awaiting_continuation_false_for_complete_fragment():
    a = TickerAssembler("test")
    a.add_fragment(TickerFragment(text=HEADLINE, timestamp=1.0, frame_index=1,
                                  confidence=0.9, left_incomplete=False))
    assert a.current_message.awaiting_continuation is False


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"✅ {name}")
            except AssertionError as exc:
                failed += 1
                print(f"❌ {name}: {exc}")
    sys.exit(1 if failed else 0)
