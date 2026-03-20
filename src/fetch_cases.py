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

        # Derive case name from the URL slug (new schema has no case_name field)
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

    with open(OUTPUT_PATH, "w") as f:
        json.dump(cases, f, indent=2)

    print(f"Saved {len(cases)} cases to {OUTPUT_PATH}")

    # Show category distribution
    from collections import Counter
    cats = Counter(c["category"] for c in cases)
    print("Category distribution:", dict(cats))

if __name__ == "__main__":
    main()
