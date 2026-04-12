import json
import os
import sys
import numpy as np
from flask import send_from_directory, request, jsonify
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'query-classifier'))
from classifier import RuleBasedLegalClassifier
from keywords import category_keywords

cases_path = os.path.join(os.path.dirname(__file__), 'cases.json')
with open(cases_path) as f:
    CASES = json.load(f)

# build TF-IDF + LSA (SVD) index over all cases
corpus = [f"{c['case_name']} {c['text']}" for c in CASES]
VECTORIZER = TfidfVectorizer(stop_words='english', max_features=10000)
TFIDF_MATRIX = VECTORIZER.fit_transform(corpus)

n_samples, n_features = TFIDF_MATRIX.shape
# TruncatedSVD requires n_components < min(n_samples, n_features) for stable fits
_max_k = min(100, n_samples, n_features)
if _max_k >= n_samples:
    _max_k = n_samples - 1
SVD_K = max(2, _max_k) if n_samples > 2 else max(1, min(n_samples, n_features))

SVD_MODEL = TruncatedSVD(n_components=SVD_K, random_state=42)
SVD_MATRIX = SVD_MODEL.fit_transform(TFIDF_MATRIX)
SVD_MATRIX = normalize(SVD_MATRIX)

FEATURE_NAMES = VECTORIZER.get_feature_names_out()
# Name latent dimensions by top loading terms (explainability)
DIMENSION_LABELS = {}
_n_label_dims = min(10, SVD_K)
for i in range(_n_label_dims):
    comp = SVD_MODEL.components_[i]
    top_idx = comp.argsort()[-5:][::-1]
    top_terms = [FEATURE_NAMES[j] for j in top_idx]
    DIMENSION_LABELS[i] = f"Dimension {i}: {', '.join(top_terms)}"

CLASSIFIER = RuleBasedLegalClassifier(category_keywords)

CATEGORY_MAP = {
    'personal_injury': 'personal injury',
    'employment_labor': 'employment',
    'copyright': 'copyright',
}

CATEGORY_LABELS = {
    'personal_injury': 'Personal Injury',
    'employment_labor': 'Employment Law',
    'copyright': 'Copyright',
}


def _indices_for_category_keys(cat_keys):
    """Case row indices whose category is in cat_keys; empty cat_keys means all."""
    if not cat_keys:
        return list(range(len(CASES)))
    indices = [
        i for i, c in enumerate(CASES)
        if c.get("category", "") in cat_keys
    ]
    return indices if indices else list(range(len(CASES)))


def _category_keys_from_request():
    """Repeatable ?category= — order preserved, deduped, only known keys."""
    seen = set()
    keys = []
    for raw in request.args.getlist("category"):
        k = (raw or "").strip()
        if k in CATEGORY_MAP and k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


def _human_labels_for_keys(keys):
    if not keys:
        return None
    return ", ".join(CATEGORY_LABELS.get(k, k) for k in keys)


def _serialize_classification(
    *,
    status: str,
    reason,
    candidates_raw,
    needs_user_category: bool,
):
    candidates = []
    for c in candidates_raw or []:
        key = c.get("category")
        if key not in CATEGORY_MAP:
            continue
        score = c.get("score")
        try:
            score_f = round(float(score), 4)
        except (TypeError, ValueError):
            score_f = 0.0
        candidates.append({
            "key": key,
            "label": CATEGORY_LABELS.get(key, key),
            "score": score_f,
        })
    return {
        "status": status,
        "needs_user_category": bool(needs_user_category),
        "reason": reason,
        "candidates": candidates,
    }


def _query_svd_vector(q: str):
    """TF-IDF -> SVD latent space, L2-normalized for cosine similarity."""
    query_tfidf = VECTORIZER.transform([q])
    q_svd = SVD_MODEL.transform(query_tfidf)
    return normalize(q_svd)


def _activated_dimension_labels(query_svd: np.ndarray, top_n: int = 3):
    """Top latent dimensions by |activation| for 'Why this result?' explainability."""
    vec = query_svd[0]
    order = np.argsort(np.abs(vec))[::-1]
    labels = []
    for d in order:
        if d >= _n_label_dims:
            continue
        if len(labels) >= top_n:
            break
        labels.append(DIMENSION_LABELS.get(d, f"Dimension {d}"))
    return labels


def register_routes(app):
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        else:
            return send_from_directory(app.static_folder, 'index.html')

    @app.route("/api/search")
    def search():
        q = request.args.get("q", "").strip()
        user_cat_keys = _category_keys_from_request()

        # Browse mode: pill(s) with no search query
        if not q:
            if user_cat_keys:
                allow = set(user_cat_keys)
                category_cases = [
                    c for c in CASES
                    if c.get("category", "") in allow
                ]
                hits = [{
                    "case_name": c["case_name"],
                    "category": c.get("category", ""),
                    "similarity": 1.0,
                    "snippet": c["text"][:300],
                    "url": c.get("url", ""),
                    "why": [],
                } for c in category_cases]
                return jsonify({
                    "results": hits,
                    "detected_category": _human_labels_for_keys(user_cat_keys),
                    "confidence": None,
                    "activated_dimensions": [],
                    "classification": _serialize_classification(
                        status="browse",
                        reason=None,
                        candidates_raw=[],
                        needs_user_category=False,
                    ),
                })
            # No query, no category — return all cases as default browse
            hits = [{
                "case_name": c["case_name"],
                "category": c.get("category", ""),
                "similarity": 1.0,
                "snippet": c["text"][:300],
                "url": c.get("url", ""),
                "why": [],
            } for c in CASES]
            return jsonify({
                "results": hits,
                "detected_category": None,
                "confidence": None,
                "activated_dimensions": [],
                "classification": _serialize_classification(
                    status="browse",
                    reason=None,
                    candidates_raw=[],
                    needs_user_category=False,
                ),
            })

        # User chose one or more categories — skip classifier, union those buckets
        if user_cat_keys:
            confidence = None
            classification = _serialize_classification(
                status="user_selected",
                reason=None,
                candidates_raw=[],
                needs_user_category=False,
            )
            indices = _indices_for_category_keys(user_cat_keys)
            detected_category = None
            human_category = _human_labels_for_keys(user_cat_keys)
        else:
            result = CLASSIFIER.classify(q)
            status = result.get("status", "ok")
            detected_category = result.get("category")
            confidence = result.get("score")
            human_category = None

            if status == "ok" and detected_category in CATEGORY_MAP:
                indices = _indices_for_category_keys([detected_category])
                classification = _serialize_classification(
                    status="ok",
                    reason=result.get("reason"),
                    candidates_raw=result.get("candidates"),
                    needs_user_category=False,
                )
                human_category = CATEGORY_LABELS.get(detected_category)
            elif status == "ambiguous":
                detected_category = None
                confidence = None
                keys = [c["category"] for c in (result.get("candidates") or [])]
                indices = _indices_for_category_keys(keys)
                human_category = _human_labels_for_keys(keys)
                classification = _serialize_classification(
                    status="ambiguous",
                    reason=(
                        "Several legal areas matched; results mix cases from each."
                    ),
                    candidates_raw=result.get("candidates"),
                    needs_user_category=False,
                )
            elif status == "low_confidence":
                detected_category = None
                confidence = None
                keys = [c["category"] for c in (result.get("candidates") or [])]
                indices = _indices_for_category_keys(keys)
                classification = _serialize_classification(
                    status="low_confidence",
                    reason=(
                        "Keyword signal is weak — select one or more categories "
                        "above to mix cases from those areas."
                    ),
                    candidates_raw=result.get("candidates"),
                    needs_user_category=True,
                )
            else:
                # no_match or unknown fallback
                detected_category = None
                confidence = None
                indices = list(range(len(CASES)))
                classification = _serialize_classification(
                    status="no_match",
                    reason=result.get("reason"),
                    candidates_raw=result.get("candidates"),
                    needs_user_category=False,
                )

        query_svd = _query_svd_vector(q)
        activated_dimensions = _activated_dimension_labels(query_svd, top_n=3)

        sub_matrix = SVD_MATRIX[indices]
        sims = cosine_similarity(query_svd, sub_matrix).flatten()

        # normalize so scores add up to 1 (Daming's idea)
        total = sims.sum()
        if total > 0:
            sims = sims / total

        top_k = min(5, len(indices))
        top_local = np.argsort(sims)[::-1][:top_k]

        hits = []
        for local_idx in top_local:
            global_idx = indices[local_idx]
            case = CASES[global_idx]
            hits.append({
                "case_name": case["case_name"],
                "category": case.get("category", ""),
                "similarity": round(float(sims[local_idx]), 4),
                "snippet": case["text"][:300],
                "url": case.get("url", ""),
                "why": list(activated_dimensions),
            })

        return jsonify({
            "results": hits,
            "detected_category": human_category,
            "confidence": round(float(confidence), 4) if confidence is not None else None,
            "activated_dimensions": activated_dimensions,
            "classification": classification,
        })

    @app.route("/api/episodes")
    def episodes_search():
        q = request.args.get("title", request.args.get("q", ""))
        from flask import redirect
        return redirect(f"/api/search?q={q}")

    @app.route("/api/config")
    def config():
        return jsonify({"use_llm": False})
