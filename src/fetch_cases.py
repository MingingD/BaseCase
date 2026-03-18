"""
Run once locally to download CourtListener cases from HuggingFace and save to cases.json.
Usage: python src/fetch_cases.py
"""
import json
import os
from datasets import load_dataset

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cases.json')

def main():
    print("Loading dataset from HuggingFace...")
    ds = load_dataset("mkezhng/courtlistenercases", split="train")
    print(f"Loaded {len(ds)} records")

    cases = []
    for i, row in enumerate(ds):
        text = row.get("text") or row.get("opinion_text") or ""
        text = text[:3000]

        case_name = row.get("case_name") or row.get("name") or f"Case {i}"
        category = row.get("category") or row.get("practice_area") or ""
        opinion_id = row.get("opinion_id")
        # Always use CourtListener canonical URL when we have an opinion_id
        if opinion_id:
            url = f"https://www.courtlistener.com/opinion/{opinion_id}/"
        else:
            url = row.get("viewable_url") or ""

        cases.append({
            "id": str(row.get("opinion_id", i)),
            "case_name": case_name,
            "category": category,
            "text": text,
            "url": url,
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(cases, f, indent=2)

    print(f"Saved {len(cases)} cases to {OUTPUT_PATH}")

    # Show category distribution
    from collections import Counter
    cats = Counter(c["category"] for c in cases)
    print("Category distribution:", dict(cats))

if __name__ == "__main__":
    main()
