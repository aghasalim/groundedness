# -*- coding: utf-8 -*-
"""Script group of a text, for per-script decision thresholds."""
import unicodedata

def script_of(text):
    counts = {}
    for ch in text:
        if not ch.isalpha(): continue
        try: name = unicodedata.name(ch)
        except ValueError: continue
        g = ('cyrillic' if name.startswith('CYRILLIC') else 'arabic' if name.startswith('ARABIC') else 'cjk' if name.startswith(('CJK', 'HIRAGANA', 'KATAKANA', 'HANGUL')) else
             'indic' if name.startswith(('DEVANAGARI', 'BENGALI', 'THAI')) else 'other' if name.startswith(('GEORGIAN', 'HEBREW', 'GREEK', 'ARMENIAN')) else 'latin')
        counts[g] = counts.get(g, 0) + 1
    return max(counts, key=counts.get) if counts else 'latin'

if __name__ == '__main__':
    assert script_of('Salam! Konsultasiya 25 AZN') == 'latin' and script_of('Здравствуйте') == 'cyrillic' and script_of('مرحباً') == 'arabic' and script_of('नमस्ते') == 'indic' and script_of('こんにちは') == 'cjk' and script_of('გამარჯობა') == 'other'
    print('ok')
