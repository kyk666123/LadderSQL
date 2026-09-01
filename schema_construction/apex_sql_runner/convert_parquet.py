import pandas as pd, json, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

df = pd.read_parquet(a.parquet)
rows = []
for i, r in df.iterrows():
    d = r.to_dict()
    qid = d.get("question_id", None)
    if qid is None or (isinstance(qid, float) and pd.isna(qid)) or str(qid) == "nan":
        qid = i
    rows.append({
        "question_id": str(qid),
        "db_id": d["db_id"],
        "question": d["question"],
        "evidence": (d.get("evidence", "") or ""),
    })
json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"wrote {len(rows)} -> {a.out}")
