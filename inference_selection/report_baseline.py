"""单文件基线+作用域报告器 (读 env: BIRD_FILE/EXEC_CACHE/BON_DB_DIR)。
报告: N / BASE固定正确 / contested / 多数投票EX / pass@32 / 单结果组 vs 多结果组。"""
import os, sys
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import bird_common as B, gated_common as G

samples = B.load_samples()
info = G.classify(samples)
per = info["per_sample"]
N = len(samples)

single = multi = 0
for sid in range(N):
    cls = per[sid]["clusters"]
    if len(cls) <= 1:
        single += 1
    else:
        multi += 1

contested_ids = set(info["contested_ids"])

maj = 0
passk = 0
base_fixed = 0
for sid in range(N):
    cls = per[sid]["clusters"]
    if not cls:
        continue
    top = max(cls, key=lambda c: c["n"])
    if top.get("correct"):
        maj += 1
    if any(c.get("correct") for c in cls):
        passk += 1
    if sid not in contested_ids and any(c.get("correct") for c in cls):
        base_fixed += 1

print(f"FILE={os.path.basename(B.BIRD_FILE)}")
print(f"N={N}")
print(f"单结果组={single}  多结果组={multi}")
print(f"contested(有对有错,需判别)={len(contested_ids)}")
print(f"BASE固定正确={base_fixed}")
print(f"多数投票EX = {maj}/{N} = {100*maj/N:.2f}%")
print(f"pass@32天花板 = {passk}/{N} = {100*passk/N:.2f}%")
print(f"[理论]contested全判对上限 = (BASE {base_fixed}+contested {len(contested_ids)})/{N} = {100*(base_fixed+len(contested_ids))/N:.2f}%")
