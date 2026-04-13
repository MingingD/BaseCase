# script to pull cases from huggingface and save to cases.json
# run with: python src/fetch_cases.py

import json
import os
from collections import Counter
from datasets import load_dataset

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cases.json')

print("loading dataset...")
ds = load_dataset("mkezhng/courtlistenercases4", split="train")
print(f"got {len(ds)} records")

cases = []
for i, row in enumerate(ds):
    text = row.get("text") or ""
    text = text[:3000]

    absolute_url = row.get("absolute_url") or ""
    slug = [s for s in absolute_url.split("/") if s][-1] if absolute_url else ""
    case_name = slug.replace("-", " ").title() if slug else f"Case {i}"

    category = row.get("_category") or row.get("category") or ""
    url = ("https://www.courtlistener.com" + absolute_url) if absolute_url else ""

    cases.append({
        "id": str(row.get("opinion_id", i)),
        "case_name": case_name,
        "category": category,
        "text": text,
        "url": url,
    })

with open(output_path, "w") as f:
    json.dump(cases, f, indent=2)

print(f"saved {len(cases)} cases to {output_path}")
print("categories:", dict(Counter(c["category"] for c in cases)))
