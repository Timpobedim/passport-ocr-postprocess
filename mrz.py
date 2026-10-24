"""Машиночитаемая зона внутреннего паспорта РФ: разбор, контрольные цифры, транслитерация.

Строка 1: PNRUS + ФАМИЛИЯ<<ИМЯ<ОТЧЕСТВО, кириллица записана латиницей и цифрами
(Ч → 3, Ш → 4, Я → 8 …). Строка 2 (44 символа):

    0–8   серия (3 первые цифры) и номер        9  контрольная цифра
    10–12 RUS                                   13–18 дата рождения ГГММДД, 19 контрольная
    20    пол M/F                               21–27 срок действия (пусто) и контрольная
    28–41 доп. данные: 4-я цифра серии, дата выдачи ГГММДД, код подразделения, «<»
    42    контрольная доп. данных               43 общая контрольная цифра
"""
from __future__ import annotations

from dataclasses import dataclass

CYR_TO_MRZ = {
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ё": "2", "Ж": "J", "З": "Z",
    "И": "I", "Й": "Q", "К": "K", "Л": "L", "М": "M", "Н": "N", "О": "O", "П": "P", "Р": "R",
    "С": "S", "Т": "T", "У": "U", "Ф": "F", "Х": "H", "Ц": "C", "Ч": "3", "Ш": "4", "Щ": "W",
    "Ъ": "X", "Ы": "Y", "Ь": "9", "Э": "6", "Ю": "7", "Я": "8", "-": "<", " ": "<",
}
MRZ_TO_CYR = {v: k for k, v in CYR_TO_MRZ.items() if k not in "- "}

_WEIGHTS = (7, 3, 1)


def _value(ch: str) -> int:
    if ch.isdigit():
        return int(ch)
    if "A" <= ch <= "Z":
        return ord(ch) - ord("A") + 10
    if ch == "<":
        return 0
    raise ValueError(ch)


def check_digit(s: str) -> str | None:
    try:
        return str(sum(_value(c) * _WEIGHTS[i % 3] for i, c in enumerate(s)) % 10)
    except ValueError:
        return None


def to_mrz(name: str) -> str | None:
    """Кириллица визуальной зоны → запись MRZ; None, если есть символ вне таблицы."""
    out = []
    for ch in name.upper():
        if ch not in CYR_TO_MRZ:
            return None
        out.append(CYR_TO_MRZ[ch])
    return "".join(out)


@dataclass(frozen=True)
class Line2:
    series3: str
    number: str
    birth: str          # ГГММДД
    sex: str            # M / F
    series4: str
    issue: str          # ГГММДД
    code: str           # 6 цифр
    checks_ok: bool     # все четыре контрольные цифры сошлись


def _parse2(s: str) -> Line2 | None:
    if len(s) != 44:
        return None
    doc, birth, opt = s[0:9], s[13:19], s[28:42]
    ok = (check_digit(doc) == s[9] and check_digit(birth) == s[19]
          and check_digit(opt) == s[42]
          and check_digit(s[0:10] + s[13:20] + s[21:43]) == s[43])
    return Line2(doc[:3], doc[3:9], birth, s[20], opt[0], opt[1:7], opt[7:13], ok)


def parse_line2(s: str) -> Line2 | None:
    """Строка 2. Если длина сбита на один символ, пробуем удалить или вставить «<» так,
    чтобы сошлись все контрольные цифры; иначе — что получилось или None."""
    s = s.strip()
    direct = _parse2(s)
    if direct and direct.checks_ok:
        return direct
    variants = []
    if len(s) == 45:
        variants = [s[:i] + s[i + 1:] for i in range(45)]
    elif len(s) == 43:
        variants = [s[:i] + "<" + s[i:] for i in range(44)]
    # Пол контрольными цифрами не защищён — без M/F на своём месте вариант не годится.
    fixed = {v for v in variants if (p := _parse2(v)) and p.checks_ok and p.sex in "MF"}
    if len(fixed) == 1:
        return _parse2(fixed.pop())
    return direct


def parse_names(line1: str) -> tuple[str, str, str] | None:
    """Фамилия, имя, отчество в записи MRZ (без перевода в кириллицу)."""
    s = line1.strip()
    if not s.startswith("PN") or len(s) < 6:
        return None
    body = s[5:].rstrip("<")
    if "<<" not in body:
        return None
    surname, rest = body.split("<<", 1)
    given = rest.split("<")
    given = [g for g in given if g]
    if len(given) == 2:
        return surname, given[0], given[1]
    if len(given) == 1:
        return surname, given[0], ""
    return None


def from_mrz(s: str) -> str | None:
    out = []
    for ch in s:
        if ch == "<":
            out.append("-")
        elif ch in MRZ_TO_CYR:
            out.append(MRZ_TO_CYR[ch])
        else:
            return None
    return "".join(out)
