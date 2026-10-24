"""Риск ошибки в полях, которых нет в MRZ: место рождения и «кем выдан».

Риск слова — наибольший из двух:
- само слово. Частое слово (встречалось в train хотя бы MIN_SEEN раз) — сглаженная
  доля его неверных прочтений. Редкое (название села) — доля ошибок редких слов того
  же типа и той же уверенности символов. Тип: слово есть в словаре (разметка train и
  справочник подразделений); нет в словаре; нет, но в словаре есть частое слово на
  одну правку от него — «ОБЛАОТИ», «ОПРОСАМ»: такие почти всегда ошибки, даже при
  уверенности 0.999;
- пара «слово + следующее слово или конец строки», если она частая: «ОБЛ» в конце
  строки обычно «ОБЛ.», «ПС» перед регионом — «ПО».
Риск поля — вероятность хотя бы одной ошибки среди его слов.
"""
from __future__ import annotations

import csv
import difflib
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import fields as F

TEXT_FIELDS = ["Birth_place_ru", "Issue_organization_ru"]
END = "<END>"
CONF_BINS = [0.0, 0.9, 0.97, 0.99, 0.995, 0.998, 0.999, 1.01]
MIN_SEEN = 5             # с какого числа прочтений слово или пара считаются частыми
PRIOR_STRENGTH = 20      # сглаживание доли ошибок частых слов к средней
NEAR_MIN_COUNT = 3       # словарное слово для сравнения встречалось хотя бы столько раз
KNOWN, UNKNOWN, NEAR = 0, 1, 2
SUBDIVISIONS = Path(__file__).parent / "reference" / "subdivisions.csv"


def tokens_with_conf(raw: dict | None) -> list[tuple[str, float]]:
    """Слова поля и минимальная уверенность символов в каждом."""
    if not raw or not raw.get("lines"):
        return []
    out: list[tuple[str, float]] = []
    for line in raw["lines"]:
        text, conf = line["text"].upper(), list(line.get("char_conf") or [])
        if len(conf) != len(text):
            conf = [min(conf) if conf else 0.0] * len(text)
        word: list[str] = []
        wconf: list[float] = []
        for ch, c in zip(text + " ", conf + [1.0]):
            if ch == " ":
                if word:
                    out.append(("".join(word), min(wconf)))
                word, wconf = [], []
            else:
                word.append(ch)
                wconf.append(c)
    return out


def pairs(words: list[str]) -> list[str]:
    return [f"{w}|{words[i + 1] if i + 1 < len(words) else END}" for i, w in enumerate(words)]


def token_truth(viz: list[str], gold: list[str]) -> tuple[list[bool], list[bool]]:
    """Верно ли слово само по себе и вместе со следующим словом (или концом строки)."""
    ok = [False] * len(viz)
    for tag, a, b, _, _ in difflib.SequenceMatcher(None, viz, gold, autojunk=False).get_opcodes():
        if tag == "equal":
            for i in range(a, b):
                ok[i] = True
    pair = [ok[i] and (ok[i + 1] if i + 1 < len(viz) else viz[-1:] == gold[-1:])
            for i in range(len(viz))]
    return ok, pair


def _bin(conf: float) -> int:
    for i in range(len(CONF_BINS) - 1):
        if CONF_BINS[i] <= conf < CONF_BINS[i + 1]:
            return i
    return len(CONF_BINS) - 2


def _deletes(w: str) -> set[str]:
    return {w[:i] + w[i + 1:] for i in range(len(w))} | {w}


class Lexicon:
    """Словарь слов с поиском соседей на расстоянии одной правки (symmetric delete)."""

    def __init__(self, counts: dict[str, int]) -> None:
        self.counts = counts
        self.index: dict[str, set[str]] = defaultdict(set)
        for w, c in counts.items():
            if c >= NEAR_MIN_COUNT:
                for d in _deletes(w):
                    self.index[d].add(w)

    def kind(self, w: str, own: int = 0) -> int:
        """own — сколько раз слово попало в словарь из разметки этого же документа."""
        if self.counts.get(w, 0) - own > 0:
            return KNOWN
        near = set().union(*(self.index.get(d, set()) for d in _deletes(w)))
        return NEAR if near - {w} else UNKNOWN


def reference_words(path: Path = SUBDIVISIONS) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                for w in F.norm(row["name"]).split():
                    counts[w] += 1
    return counts


@dataclass
class RiskModel:
    words: dict[str, tuple[int, int]]    # слово → (прочтений, неверных)
    pairs: dict[str, tuple[int, int]]    # «слово|следующее» → (прочтений, неверных)
    rare_rate: list[list[float]]         # доля ошибок редких слов: [тип][корзина уверенности]
    word_base: float
    pair_base: float
    lexicon: Lexicon

    def word_risk(self, word: str, conf: float) -> float:
        n, bad = self.words.get(word, (0, 0))
        if n >= MIN_SEEN:
            return (bad + PRIOR_STRENGTH * self.word_base) / (n + PRIOR_STRENGTH)
        return self.rare_rate[self.lexicon.kind(word)][_bin(conf)]

    def pair_risk(self, key: str) -> float:
        n, bad = self.pairs.get(key, (0, 0))
        if n >= MIN_SEEN:
            return (bad + PRIOR_STRENGTH * self.pair_base) / (n + PRIOR_STRENGTH)
        return 0.0

    def field_risk(self, raw: dict | None) -> float:
        toks = tokens_with_conf(raw)
        if not toks:
            return 1.0
        log_ok = 0.0
        for (w, c), key in zip(toks, pairs([w for w, _ in toks])):
            r = max(self.word_risk(w, c), self.pair_risk(key))
            log_ok += math.log1p(-min(r, 0.999999))
        return 1.0 - math.exp(log_ok)


def fit(preds: list[dict], gold: dict[str, dict]) -> RiskModel:
    words: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    prs: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    lex_counts = reference_words()
    occurrences = []                     # (слово, уверенность, верно, сколько раз в разметке документа)
    for pred in preds:
        g = gold.get(pred["doc_id"])
        if g is None:
            continue
        own: dict[str, int] = defaultdict(int)
        for f in TEXT_FIELDS:
            for w in F.norm(g[f]).split():
                lex_counts[w] += 1
                own[w] += 1
        for f in TEXT_FIELDS:
            toks = tokens_with_conf(pred["viz"].get(f))
            if not toks:
                continue
            ws = [w for w, _ in toks]
            ok, pair_ok = token_truth(ws, F.norm(g[f]).split())
            for (w, c), key, o, po in zip(toks, pairs(ws), ok, pair_ok):
                words[w][0] += 1
                words[w][1] += not o
                prs[key][0] += 1
                prs[key][1] += not po
                occurrences.append((w, c, o, own.get(w, 0)))
    lexicon = Lexicon(dict(lex_counts))
    nb = len(CONF_BINS) - 1
    tot = [[0] * nb for _ in range(3)]
    bad = [[0] * nb for _ in range(3)]
    for w, c, o, own_count in occurrences:
        if words[w][0] < MIN_SEEN:
            # Тип без разметки этого же документа: иначе верное слово всегда «есть в словаре».
            k = lexicon.kind(w, own_count)
            tot[k][_bin(c)] += 1
            bad[k][_bin(c)] += not o
    rate = [[(bad[k][i] + 1) / (tot[k][i] + 2) for i in range(nb)] for k in range(3)]

    def base(d: dict) -> float:
        freq = [v for v in d.values() if v[0] >= MIN_SEEN]
        return sum(b for _, b in freq) / max(1, sum(n for n, _ in freq))

    return RiskModel({k: (v[0], v[1]) for k, v in words.items()},
                     {k: (v[0], v[1]) for k, v in prs.items()},
                     rate, base(words), base(prs), lexicon)
