"""Честная оценка на train: модель риска для каждой пятой части обучается без неё.

    python experiment.py            # перебор порога риска
    python experiment.py --errors   # ошибки принятых и причины отказов при выбранном пороге
"""
from __future__ import annotations

import sys
from collections import Counter

import fields as F
import postprocess as pp
import risk

K = 5
THRESHOLDS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]


def cross_validate() -> list[tuple[dict, bool, dict, list[str], float]]:
    preds = pp.read_jsonl(pp.ROOT / "data" / "train_predictions.jsonl")
    gold = {d["doc_id"]: d["fields"] for d in pp.read_jsonl(pp.ROOT / "data" / "train_labels.jsonl")}
    fold = {doc_id: i % K for i, doc_id in enumerate(sorted(gold))}
    models = [risk.fit([p for p in preds if fold[p["doc_id"]] != k], gold) for k in range(K)]
    rows = []
    for pred in preds:
        ok, values, why, r = pp.decide(pred, models[fold[pred["doc_id"]]], max_risk=1.0)
        wrong = {f: (values[f], gold[pred["doc_id"]][f]) for f in F.FIELDS
                 if F.norm(values[f]) != F.norm(gold[pred["doc_id"]][f])}
        rows.append((pred, ok, wrong, why, r))
    return rows


def main(show_errors: bool) -> None:
    rows = cross_validate()
    n = len(rows)
    print(f"{'порог':>7} {'принято':>8} {'верно':>6} {'ошибок':>7} {'сумма риска':>12}")
    for t in THRESHOLDS:
        acc = [(w, r) for _, ok, w, _, r in rows if ok and r <= t]
        bad = sum(bool(w) for w, _ in acc)
        print(f"{t:7.3f} {len(acc):8} {len(acc) - bad:6} {bad:7} {sum(r for _, r in acc):12.2f}")
    if not show_errors:
        return
    t = pp.MAX_TEXT_RISK
    print(f"\nошибки при пороге {t}:")
    for pred, ok, wrong, _, r in rows:
        if ok and r <= t and wrong:
            print(" ", pred["doc_id"], pred["source"], f"риск {r:.4f}", wrong)
    reasons: Counter = Counter()
    for _, ok, _, why, r in rows:
        if r > t:
            why = why + ["риск"]
        for x in why:
            reasons[x.split(":")[0]] += 1
    print("\nпричины отказов (документ может иметь несколько):")
    for x, c in reasons.most_common():
        print(f"  {c:5}  {x}")
    acc = Counter(p["source"] for p, ok, w, _, r in rows if ok and r <= t)
    tot = Counter(p["source"] for p, *_ in rows)
    print("\nпринято по источникам:", {s: f"{acc[s]}/{tot[s]}" for s in tot})


if __name__ == "__main__":
    main("--errors" in sys.argv)
