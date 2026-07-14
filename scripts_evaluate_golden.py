import json
from pathlib import Path
from src.security import scan_input

path = Path(__file__).parent / "golden_dataset" / "golden_dataset.jsonl"
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
passed = 0
for row in rows:
    result = scan_input(row["input"])
    expected_block = row["category"] == "attack"
    ok = (not result.allowed) if expected_block else result.allowed
    passed += int(ok)
    print(f"{row['id']}: {'PASS' if ok else 'FAIL'} | allowed={result.allowed} | reasons={result.reasons}")
print(f"\nResultado: {passed}/{len(rows)} ({passed/len(rows):.1%})")
