"""Постобработка OCR паспорта РФ: принять документ с полями или попросить переснять.

    python postprocess.py data/test_predictions.jsonl answers.jsonl

Уверенность модели сама по себе ненадёжна (у неверных символов бывает 0.999),
поэтому каждое поле подтверждается независимо:
- ФИО, пол, даты, серия и номер, код подразделения — совпадают с MRZ, у строки 2
  которой сошлись все контрольные цифры;
- место рождения и «кем выдан» в MRZ нет: строки покрывают рамку поля (нет
  потерянной строки), а риск ошибки по словам (risk.py) не выше порога;
- паспорт действителен на 01.10.2026.
Работает на CPU без сети; модель риска обучается на train при запуске (~2 с),
файлы train ищутся рядом с входным файлом.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import fields as F
import mrz
import risk

ROOT = Path(__file__).parent
CHECK_DATE = dt.date(2026, 10, 1)

# Подобраны на train кросс-валидацией (experiment.py).
MIN_COVERAGE = 0.75      # сумма высот строк / высота рамки поля
MAX_GAP = 0.25           # пустая доля рамки сверху или снизу
MAX_TEXT_RISK = 0.03     # суммарный риск ошибки в месте рождения и «кем выдан»


def coverage(raw: dict | None) -> tuple[float, float]:
    """Доля рамки, покрытая строками, и наибольший пустой край."""
    if not raw or not raw.get("lines") or not raw.get("box"):
        return 0.0, 1.0
    _, by, _, bh = raw["box"]
    if bh <= 0:
        return 0.0, 1.0
    lines = raw["lines"]
    covered = sum(ln["box"][3] for ln in lines) / bh
    top = min(ln["box"][1] for ln in lines)
    bottom = max(ln["box"][1] + ln["box"][3] for ln in lines)
    return covered, max((top - by) / bh, (by + bh - bottom) / bh)


def parse_date(s: str | None) -> dt.date | None:
    m = F.DATE.match(s or "")
    if not m:
        return None
    try:
        return dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None


def _birthday(born: dt.date, age: int) -> dt.date:
    try:
        return born.replace(year=born.year + age)
    except ValueError:           # 29 февраля
        return dt.date(born.year + age, 3, 1)


def valid_on(born: dt.date, issued: dt.date, when: dt.date = CHECK_DATE) -> tuple[bool, str]:
    """Действителен ли паспорт: до 20 лет, до 45 лет, после 45 — бессрочно.
    90 дней на замену после 20 и 45 лет (Положение о паспорте) не используем:
    такие документы отклоняем — ошибка дороже отказа."""
    if issued > when:
        return False, "выдан позже даты проверки"
    if issued < _birthday(born, 14):
        return False, "выдан до 14 лет"
    for age in (20, 45):
        limit = _birthday(born, age)
        if issued < limit <= when:
            return False, f"не заменён в {age} лет"
    return True, ""


def fix_dates(doc: F.Doc, pred: dict) -> None:
    """Век в годе визуальной зоны бывает искажён (1088 вместо 1988), а в MRZ год
    двузначный. День, месяц и две последние цифры года совпали с MRZ, век невозможен —
    берём единственный правдоподобный век."""
    lines = (pred.get("mrz") or {}).get("lines") or []
    if len(lines) != 2:
        return
    l2 = mrz.parse_line2(lines[1]["text"])
    if not l2 or not l2.checks_ok:
        return
    for name, yymmdd in (("Birth_date", l2.birth), ("Issue_date", l2.issue)):
        fld = doc.fields[name]
        m = F.DATE.match(fld.value or "")
        if not m or m.group(3)[:2] in ("19", "20"):
            continue
        if f"{m.group(3)[2:]}{m.group(2)}{m.group(1)}" != yymmdd:
            continue
        options = [f"{yymmdd[4:]}.{yymmdd[2:4]}.{c}{yymmdd[:2]}" for c in ("19", "20")]
        plausible = [o for o in options if (d := parse_date(o)) and 1920 <= d.year and d <= CHECK_DATE]
        if len(plausible) == 1:
            fld.value, fld.mrz_ok, fld.source = plausible[0], True, "mrz"
    # Дата рождения должна давать возраст при выдаче от 14 лет.
    born, issued = parse_date(doc.fields["Birth_date"].value), parse_date(doc.fields["Issue_date"].value)
    if born and issued and issued < _birthday(born, 14):
        doc.fields["Birth_date"].mrz_ok = False


def decide(pred: dict, model: risk.RiskModel, max_risk: float = MAX_TEXT_RISK
           ) -> tuple[bool, dict, list[str], float]:
    """Решение, значения полей, причины отказа и оценка риска текстовых полей."""
    doc = F.build(pred)
    fix_dates(doc, pred)
    reasons = []
    if not doc.mrz_checks:
        reasons.append("MRZ не прочитана или не сошлись контрольные цифры")
    for f in F.MRZ_FIELDS:
        if not doc.fields[f].mrz_ok:
            reasons.append(f"{f}: не совпадает с MRZ")
    text_risk = 0.0
    for f in risk.TEXT_FIELDS:
        raw = pred["viz"].get(f)
        if doc.fields[f].value is None:
            reasons.append(f"{f}: не найдено")
            text_risk = 1.0
            continue
        cov, gap = coverage(raw)
        if cov < MIN_COVERAGE or gap > MAX_GAP:
            reasons.append(f"{f}: строки не покрывают рамку")
        text_risk = 1 - (1 - text_risk) * (1 - model.field_risk(raw))
    if text_risk > max_risk:
        reasons.append("риск ошибки в месте рождения или «кем выдан»")
    born, issued = parse_date(doc.fields["Birth_date"].value), parse_date(doc.fields["Issue_date"].value)
    if born and issued:
        ok, why = valid_on(born, issued)
        if not ok:
            reasons.append(f"недействителен: {why}")
    else:
        reasons.append("даты не разобраны")
    values = {f: doc.fields[f].value for f in F.FIELDS}
    return not reasons, values, reasons, text_risk


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def train_dir(pred_path: str | Path) -> Path:
    """Разметка train — рядом с входным файлом, иначе в data/ проекта."""
    for d in (Path(pred_path).parent, ROOT / "data"):
        if (d / "train_labels.jsonl").exists() and (d / "train_predictions.jsonl").exists():
            return d
    sys.exit("нужны train_predictions.jsonl и train_labels.jsonl рядом с входным файлом или в data/")


def train_model(data_dir: Path = ROOT / "data") -> risk.RiskModel:
    preds = read_jsonl(data_dir / "train_predictions.jsonl")
    gold = {d["doc_id"]: d["fields"] for d in read_jsonl(data_dir / "train_labels.jsonl")}
    return risk.fit(preds, gold)


def main(pred_path: str, out_path: str) -> None:
    model = train_model(train_dir(pred_path))
    accepted = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for pred in read_jsonl(pred_path):
            ok, values, _, _ = decide(pred, model)
            accepted += ok
            row = {"doc_id": pred["doc_id"], "decision": "accept" if ok else "reject"}
            if ok:
                row["fields"] = values
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"принято {accepted}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("python postprocess.py <predictions.jsonl> <answers.jsonl>")
    main(sys.argv[1], sys.argv[2])
