"""Поля документа: значение из визуальной зоны и его сверка с MRZ."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import mrz

FIELDS = ["Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru", "Birth_date",
          "Birth_place_ru", "Issue_date", "Issue_organisation_code", "Licence_number",
          "Issue_organization_ru"]
NAME_FIELDS = FIELDS[:3]
# Поля, которые MRZ подтверждает; место рождения и «кем выдан» в ней не записаны.
MRZ_FIELDS = ["Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru", "Birth_date",
              "Issue_date", "Issue_organisation_code", "Licence_number"]

DATE = re.compile(r"^(\d{2})\.(\d{2})\.(\d{4})$")
CODE = re.compile(r"^(\d{3})-(\d{3})$")
NUMBER = re.compile(r"^(\d{2}) (\d{2}) (\d{6})$")


def norm(s: str | None) -> str:
    return " ".join(str(s or "").split()).upper()


@dataclass
class Field:
    value: str | None           # что выдадим
    conf: float = 0.0           # средняя уверенность детектора по полю
    min_char: float = 0.0       # минимальная уверенность по символу
    n_lines: int = 0
    mrz_ok: bool = False        # значение подтверждено MRZ
    source: str = "viz"         # viz | mrz — откуда взято значение


@dataclass
class Doc:
    doc_id: str
    source: str
    quality: dict
    fields: dict[str, Field] = field(default_factory=dict)
    mrz_lines: int = 0
    mrz_checks: bool = False    # строка 2 прошла все контрольные цифры
    mrz_min_char: float = 0.0


def _viz(raw: dict | None) -> Field:
    if raw is None:
        return Field(None)
    lines = raw.get("lines") or []
    chars = [c for ln in lines for c in (ln.get("char_conf") or [])]
    text = " ".join(ln["text"] for ln in lines)
    return Field(norm(text) or None, float(raw.get("conf") or 0.0),
                 min(chars) if chars else 0.0, len(lines))


def _yymmdd(date: str | None) -> str | None:
    m = DATE.match(date or "")
    return f"{m.group(3)[2:]}{m.group(2)}{m.group(1)}" if m else None


def _full_date(yymmdd: str, century_hint: str | None, *, birth: bool) -> str:
    yy, mm, dd = yymmdd[:2], yymmdd[2:4], yymmdd[4:]
    if century_hint and century_hint[2:] == yy:
        year = century_hint
    elif birth:
        year = ("20" if int(yy) <= 12 else "19") + yy
    else:
        year = ("20" if int(yy) <= 30 else "19") + yy
    return f"{dd}.{mm}.{year}"


def build(pred: dict) -> Doc:
    doc = Doc(pred["doc_id"], pred.get("source", ""), pred.get("quality") or {})
    for f in FIELDS:
        doc.fields[f] = _viz(pred["viz"].get(f))
    raw = pred.get("mrz") or {}
    lines = raw.get("lines") or []
    doc.mrz_lines = len(lines)
    chars = [c for ln in lines for c in (ln.get("char_conf") or [])]
    doc.mrz_min_char = min(chars) if chars else 0.0
    if len(lines) != 2:
        return doc
    line1, line2 = lines[0]["text"], lines[1]["text"]
    _check_names(doc, line1)
    l2 = mrz.parse_line2(line2)
    if l2 is None:
        return doc
    doc.mrz_checks = l2.checks_ok
    _check_line2(doc, l2)
    return doc


def _check_names(doc: Doc, line1: str) -> None:
    body = line1.strip()[5:].rstrip("<")
    names = [doc.fields[f].value for f in NAME_FIELDS]
    if any(n is None for n in names[:2]):
        return
    parts = [mrz.to_mrz(n or "") for n in names]
    if any(p is None for p in parts):
        return
    expected = parts[0] + "<<" + parts[1] + ("<" + parts[2] if parts[2] else "")
    if expected == body:
        for f in NAME_FIELDS:
            doc.fields[f].mrz_ok = True


def _check_line2(doc: Doc, l2: mrz.Line2) -> None:
    f = doc.fields
    sex = {"M": "МУЖ.", "F": "ЖЕН."}.get(l2.sex)
    if sex and f["Sex_ru"].value == sex:
        f["Sex_ru"].mrz_ok = True
    if not l2.checks_ok:
        # Без контрольных цифр MRZ лишь второй независимый прочтённый вариант.
        pass
    bd = f["Birth_date"]
    if bd.value and _yymmdd(bd.value) == l2.birth:
        bd.mrz_ok = True
    iss = f["Issue_date"]
    if iss.value and _yymmdd(iss.value) == l2.issue:
        iss.mrz_ok = True
    code = f["Issue_organisation_code"]
    if code.value and code.value.replace("-", "") == l2.code and CODE.match(code.value):
        code.mrz_ok = True
    num = f["Licence_number"]
    if num.value and (m := NUMBER.match(num.value)):
        if m.group(1) + m.group(2)[0] == l2.series3 and m.group(2)[1] == l2.series4 \
                and m.group(3) == l2.number:
            num.mrz_ok = True
