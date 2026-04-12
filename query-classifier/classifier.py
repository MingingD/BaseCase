import re
import math
from typing import Any, Dict, List
import pprint
from keywords import category_keywords

category_keywords = category_keywords


class RuleBasedLegalClassifier:
    def __init__(
        self,
        category_keywords,
        low_confidence_threshold=0.02,
        ambiguity_ratio=0.82,
        secondary_ambiguity_min_score=0.022,
        secondary_ambiguity_fraction_of_top=0.24,
    ):
        """
        category_keywords:
            dictionary that has the keywords for each category

        low_confidence_threshold:
            if best normalized score is below this, treat as low confidence

        ambiguity_ratio:
            categories with score >= top_score * ratio count as tied with the leader.

        secondary_ambiguity_* (hybrid):
            If only one category sits inside the ratio band but the runner-up still has
            a meaningful normalized score and is at least a fraction of the winner,
            treat as ambiguous (two-category stories like FMLA + surgery).
        """
        self.category_keywords = category_keywords
        self.low_confidence_threshold = low_confidence_threshold
        self.ambiguity_ratio = ambiguity_ratio
        self.secondary_ambiguity_min_score = secondary_ambiguity_min_score
        self.secondary_ambiguity_fraction_of_top = secondary_ambiguity_fraction_of_top

        self.category_scales = {}
        for category, keywords in self.category_keywords.items():
            total_weight = sum(keywords.values())
            self.category_scales[category] = math.sqrt(total_weight) if total_weight > 0 else 1.0

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text):
        return text.split()

    def _non_overlapping_keyword_hits(self, processed_query: str, keyword_weights: dict):
        """
        Longest phrases first; each character span matched at most once.
        Reduces double-counting from e.g. both 'pain' and 'pain and suffering'.
        """
        items = sorted(keyword_weights.items(), key=lambda kv: len(kv[0]), reverse=True)
        n = len(processed_query)
        covered = [False] * n
        matched_keywords = []
        raw_score = 0.0

        for keyword, weight in items:
            pattern = re.compile(r"\b" + re.escape(keyword.lower()) + r"\b")
            for m in pattern.finditer(processed_query):
                start, end = m.start(), m.end()
                if start >= n or end > n:
                    continue
                if any(covered[start:end]):
                    continue
                raw_score += weight
                matched_keywords.append({"keyword": keyword, "weight": weight})
                for i in range(start, end):
                    covered[i] = True

        return raw_score, matched_keywords

    def compute_scores(self, query):
        processed_query = self.preprocess(query)
        tokens = self.tokenize(processed_query)
        query_length = max(len(tokens), 1)

        results = []

        for category, keyword_weights in self.category_keywords.items():
            raw_score, matched_keywords = self._non_overlapping_keyword_hits(
                processed_query, keyword_weights
            )

            length_factor = math.sqrt(query_length)
            category_factor = self.category_scales[category]

            normalized_score = raw_score / (length_factor * category_factor)

            results.append({
                "category": category,
                "raw_score": raw_score,
                "normalized_score": normalized_score,
                "matched_keywords": matched_keywords,
            })

        results.sort(key=lambda x: x["normalized_score"], reverse=True)
        return results

    def classify_from_scores(self, scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Same decision logic as classify(), but uses a precomputed score list
        (e.g. after blending with semantic similarity in routes).
        """
        scores = sorted(scores, key=lambda x: x["normalized_score"], reverse=True)

        if not scores or scores[0]["normalized_score"] == 0.0:
            return {
                "status": "no_match",
                "reason": "No matches found, all scores are 0.",
                "category": None,
                "score": None,
                "candidates": [],
                "scores": scores,
            }

        top_score = scores[0]["normalized_score"]

        if top_score < self.low_confidence_threshold:
            candidates = [
                {"category": s["category"], "score": s["normalized_score"]}
                for s in scores[:3]
                if s["normalized_score"] > 0
            ]
            return {
                "status": "low_confidence",
                "reason": (
                    f"Top score ({top_score:.3f}) is below confidence threshold; "
                    "pick one or more legal areas to narrow results."
                ),
                "category": None,
                "score": None,
                "candidates": candidates,
                "scores": scores,
            }

        floor = top_score * self.ambiguity_ratio
        selected_categories = []
        for score_data in scores:
            current_score = score_data["normalized_score"]
            current_cat = score_data["category"]
            if current_score >= floor:
                selected_categories.append((current_cat, current_score))
            else:
                break

        if len(selected_categories) > 1:
            return {
                "status": "ambiguous",
                "reason": (
                    f"{len(selected_categories)} areas within {self.ambiguity_ratio:.0%} "
                    "of the top score."
                ),
                "category": None,
                "score": None,
                "candidates": [
                    {"category": c, "score": s} for c, s in selected_categories
                ],
                "scores": scores,
            }

        # Hybrid ambiguity: one clear ratio-band winner, but #2 still has real keyword mass
        if len(scores) >= 2:
            s0 = float(scores[0]["normalized_score"])
            s1 = float(scores[1]["normalized_score"])
            if s1 > 0 and s1 >= self.secondary_ambiguity_min_score:
                if s1 >= self.secondary_ambiguity_fraction_of_top * s0:
                    return {
                        "status": "ambiguous",
                        "reason": (
                            "Top area is ahead, but a second area also has clear keyword support."
                        ),
                        "category": None,
                        "score": None,
                        "candidates": [
                            {"category": scores[0]["category"], "score": s0},
                            {"category": scores[1]["category"], "score": s1},
                        ],
                        "scores": scores,
                    }

        return {
            "status": "ok",
            "reason": None,
            "category": selected_categories[0][0],
            "score": top_score,
            "candidates": [],
            "scores": scores,
        }

    def classify(self, query) -> Dict[str, Any]:
        return self.classify_from_scores(self.compute_scores(query))


# expected_label: category id for status ok, or status name ambiguous | low_confidence | no_match
CLASSIFIER_TEST_CASES: List[Dict[str, str]] = [
    {"query": "I slipped on a wet floor at the grocery store and hurt my back badly.", "expected_label": "personal_injury"},
    {"query": "Rear-ended at a red light, whiplash and ER visit, other driver was texting.", "expected_label": "personal_injury"},
    {"query": "Dog bite at the park, needed stitches and rabies shots.", "expected_label": "personal_injury"},
    {"query": "pain and suffering after slip and fall on ice in the parking lot.", "expected_label": "personal_injury"},
    {"query": "My employer fired me the day after I filed an OSHA complaint; feels like retaliation.", "expected_label": "employment_labor"},
    {"query": "Not paid overtime for 60-hour weeks; they misclassified me as exempt.", "expected_label": "employment_labor"},
    {"query": "Hostile work environment and sexual harassment by my supervisor; thinking about EEOC.", "expected_label": "employment_labor"},
    # Hybrid ambiguity: strong employment + meaningful PI (surgery) but outside ratio band.
    {"query": "I was fired the week I came back from FMLA after my knee surgery; is that wrongful termination?", "expected_label": "ambiguous"},
    {"query": "Can I use MIT-licensed code in a commercial SaaS if I keep the license notice?", "expected_label": "copyright"},
    {"query": "Received a DMCA takedown for my app; I think it is fair use.", "expected_label": "copyright"},
    {"query": "Forked a GPL project and sold a hosted version without sharing source—what is the risk?", "expected_label": "copyright"},
    {"query": "My boss fired me right after I got injured in a bad car accident.", "expected_label": "ambiguous"},
    # Employment terms dominate; copyright "photo" is a weaker single hit.
    {"query": "Wrongful termination and they also used my photo in ads without my permission.", "expected_label": "employment_labor"},
    {"query": "My boss threatened wrongful termination over my copyright software, and then I was injured at the store, leaving me with medical bills.", "expected_label": "ambiguous"},
    {"query": "I am looking for a new job because my current daily commute is getting way too long.", "expected_label": "low_confidence"},
    # No keyword hits → no_match (not low_confidence with zero top score).
    {"query": "Something bad happened and I want to know my rights.", "expected_label": "no_match"},
    {"query": "I am having a serious dispute with my neighbor over the property line fence and I need to know if I can take them to civil court.", "expected_label": "no_match"},
    {"query": "How do I incorporate an LLC in Delaware?", "expected_label": "no_match"},
    {"query": "Immigration visa denied; can I appeal?", "expected_label": "no_match"},
    {"query": "Can I use an open source MIT license for my software if I plan on commercial use?", "expected_label": "copyright"},
]


def _matches_expected(result: Dict[str, Any], expected_label: str) -> bool:
    st = result["status"]
    if expected_label in ("ambiguous", "low_confidence", "no_match"):
        return st == expected_label
    if st != "ok":
        return False
    return result.get("category") == expected_label


if __name__ == "__main__":
    classifier = RuleBasedLegalClassifier(
        category_keywords=category_keywords,
        low_confidence_threshold=0.02,
        ambiguity_ratio=0.82,
    )

    for case in CLASSIFIER_TEST_CASES:
        q = case["query"]
        expected = case["expected_label"]
        result = classifier.classify(q)
        ok = _matches_expected(result, expected)

        if result["status"] == "ok":
            actual_display = result["category"]
        elif result["status"] == "ambiguous":
            actual_display = [c["category"] for c in result["candidates"]]
        elif result["status"] == "low_confidence":
            actual_display = {"status": "low_confidence", "top_candidates": result["candidates"]}
        else:
            actual_display = "no_match"

        print("-" * 77)
        print("query:   ", q)
        print("expected:", expected)
        print("actual:  ", result["status"], "|", actual_display)
        print("PASS" if ok else "FAIL — see above")
        if not ok:
            pprint.pprint(result)
