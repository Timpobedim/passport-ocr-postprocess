import datetime as dt
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fields as F  # noqa: E402
import mrz  # noqa: E402
import postprocess as pp  # noqa: E402
import risk  # noqa: E402

LINE1 = "PNRUSZAINTINOV<<ZAKERIQ<NAZIFULOVI3<<<<<<<<<"
LINE2 = "6122839687RUS1006102M<<<<<<<4250911620004<28"
GOLD = {
    "Last_name_ru": "ЗАИНТИНОВ", "First_name_ru": "ЗАКЕРИЙ", "Middle_name_ru": "НАЗИФУЛОВИЧ",
    "Sex_ru": "МУЖ.", "Birth_date": "10.06.2010", "Birth_place_ru": "С. ТУРУНОВО БАТЫРЕВСКОГО Р-НА",
    "Issue_date": "11.09.2025", "Issue_organisation_code": "620-004", "Licence_number": "61 24 283968",
    "Issue_organization_ru": "ОТДЕЛ ПО ВОПРОСАМ МИГРАЦИИ ОМВД РОССИИ ПО РЯЗАНСКОЙ ОБЛАСТИ В КАСИМОВСКОМ РАЙОНЕ",
}
SPLIT = {"Birth_place_ru": ["С. ТУРУНОВО", "БАТЫРЕВСКОГО Р-НА"],
         "Issue_organization_ru": ["ОТДЕЛ ПО ВОПРОСАМ МИГРАЦИИ ОМВД", "РОССИИ ПО РЯЗАНСКОЙ ОБЛАСТИ В",
                                   "КАСИМОВСКОМ РАЙОНЕ"]}


def make_doc(doc_id="d1", **override):
    viz = {}
    for f, value in GOLD.items():
        texts = SPLIT.get(f, [value])
        lines = [{"text": t, "box": [0, 30 * i, 400, 30], "char_conf": [0.9995] * len(t)}
                 for i, t in enumerate(texts)]
        viz[f] = {"box": [0, 0, 400, 30 * len(texts)], "conf": 0.9995, "lines": lines}
    doc = {"doc_id": doc_id, "source": "web", "quality": {"sharpness": 200, "doc_px": 1700, "glare": 0.1},
           "viz": viz, "mrz": {"lines": [{"text": LINE1, "char_conf": [0.999] * 44},
                                         {"text": LINE2, "char_conf": [0.999] * 44}]}}
    for f, text in override.items():
        doc["viz"][f]["lines"][0]["text"] = text
        doc["viz"][f]["lines"][0]["char_conf"] = [0.9995] * len(text)
    return doc


def test_check_digits_of_real_line():
    l2 = mrz.parse_line2(LINE2)
    assert l2.checks_ok
    assert (l2.series3, l2.series4, l2.number) == ("612", "4", "283968")
    assert (l2.birth, l2.issue, l2.code, l2.sex) == ("100610", "250911", "620004", "M")


def test_broken_digit_fails_checks():
    assert not mrz.parse_line2(LINE2.replace("283968", "283969")).checks_ok


def test_extra_char_is_repaired_only_when_checks_agree():
    l2 = mrz.parse_line2(LINE2[:20] + "<" + LINE2[20:])
    assert l2 is not None and l2.checks_ok and l2.number == "283968"


def test_transliteration_table():
    assert mrz.to_mrz("НАЗИФУЛОВИЧ") == "NAZIFULOVI3"
    assert mrz.to_mrz("ЩЁЛКОВ-ЯШИН") == "W2LKOV<84IN"
    assert mrz.to_mrz("SMITH") is None


def test_clean_document_is_confirmed_by_mrz():
    doc = F.build(make_doc())
    assert doc.mrz_checks
    assert all(doc.fields[f].mrz_ok for f in F.MRZ_FIELDS)


def test_mismatch_with_mrz_is_not_confirmed():
    doc = F.build(make_doc(Last_name_ru="ЗАИНТИНОБ", Licence_number="61 24 283969"))
    assert not doc.fields["Last_name_ru"].mrz_ok
    assert not doc.fields["Licence_number"].mrz_ok
    # ФИО сверяются со строкой 1 целиком: расхождение в фамилии снимает подтверждение со всех трёх.
    assert not doc.fields["First_name_ru"].mrz_ok
    assert doc.fields["Birth_date"].mrz_ok


def test_broken_century_is_restored_from_mrz():
    pred = make_doc(Birth_date="10.06.7010")
    doc = F.build(pred)
    pp.fix_dates(doc, pred)
    assert doc.fields["Birth_date"].value == "10.06.2010"
    assert doc.fields["Birth_date"].mrz_ok


@pytest.mark.parametrize("born, issued, ok", [
    ("2010-06-10", "2025-09-11", True),     # 16 лет, до 20 далеко
    ("2006-01-01", "2020-03-01", False),    # 20 лет исполнилось до 01.10.2026, не заменён
    ("2006-12-01", "2020-12-10", True),     # 20 исполнится после даты проверки
    ("1981-04-21", "2001-05-01", False),    # 45 лет исполнилось, паспорт с 20 лет
    ("1960-01-01", "2005-02-01", True),     # выдан после 45 — бессрочный
    ("2013-01-01", "2026-01-01", False),    # выдан до 14 лет
    ("1990-01-01", "2026-11-01", False),    # выдан позже даты проверки
])
def test_validity(born, issued, ok):
    assert pp.valid_on(dt.date.fromisoformat(born), dt.date.fromisoformat(issued))[0] is ok


def test_missing_line_is_detected_by_box():
    raw = make_doc()["viz"]["Issue_organization_ru"]
    assert pp.coverage(raw)[0] == pytest.approx(1.0)
    raw["lines"] = raw["lines"][:2]
    cov, gap = pp.coverage(raw)
    assert cov < pp.MIN_COVERAGE and gap > pp.MAX_GAP


def test_word_near_frequent_word_is_flagged():
    lex = risk.Lexicon({"ОБЛАСТИ": 50, "ОТДЕЛ": 50})
    assert lex.kind("ОБЛАСТИ") == risk.KNOWN
    assert lex.kind("ОБЛАОТИ") == risk.NEAR
    assert lex.kind("ТУРУНОВО") == risk.UNKNOWN


def _train_set(n=30):
    preds, gold = [], {}
    for i in range(n):
        preds.append(make_doc(f"t{i}"))
        gold[f"t{i}"] = dict(GOLD)
    # Пара прочтений с типичной ошибкой «С» вместо «О».
    for i in range(3):
        bad = make_doc(f"b{i}")
        bad["viz"]["Issue_organization_ru"]["lines"][1]["text"] = "РОССИИ ПО РЯЗАНСКОЙ СБЛАСТИ В"
        preds.append(bad)
        gold[f"b{i}"] = dict(GOLD)
    return preds, gold


def test_decide_accepts_clean_and_rejects_typo():
    preds, gold = _train_set()
    model = risk.fit(preds, gold)
    ok, values, why, _ = pp.decide(make_doc(), model)
    assert ok, why
    assert values == {f: GOLD[f] for f in F.FIELDS}
    typo = make_doc()
    typo["viz"]["Issue_organization_ru"]["lines"][1]["text"] = "РОССИИ ПО РЯЗАНСКОЙ СБЛАСТИ В"
    assert not pp.decide(typo, model)[0]


def test_cli_end_to_end(tmp_path):
    preds, gold = _train_set()
    with open(tmp_path / "train_predictions.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in preds)
    with open(tmp_path / "train_labels.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps({"doc_id": k, "fields": v}, ensure_ascii=False) + "\n" for k, v in gold.items())
    test = [make_doc("x1"), deepcopy(make_doc("x2"))]
    test[1]["mrz"] = None
    with open(tmp_path / "test_predictions.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in test)
    out = tmp_path / "answers.jsonl"
    subprocess.run([sys.executable, str(ROOT / "postprocess.py"), str(tmp_path / "test_predictions.jsonl"),
                    str(out)], check=True, capture_output=True)
    rows = {r["doc_id"]: r for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())}
    assert rows["x1"]["decision"] == "accept" and rows["x1"]["fields"]["Licence_number"] == "61 24 283968"
    assert rows["x2"] == {"doc_id": "x2", "decision": "reject"}
