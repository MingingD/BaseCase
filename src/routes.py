import json
import os
import sys
import numpy as np
from flask import send_from_directory, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'query-classifier'))
from classifier import RuleBasedLegalClassifier
from keywords import category_keywords

cases_path = os.path.join(os.path.dirname(__file__), 'cases.json')
with open(cases_path) as f:
    CASES = json.load(f)

# build tfidf index over all cases
corpus = [f"{c['case_name']} {c['text']}" for c in CASES]
VECTORIZER = TfidfVectorizer(stop_words='english', max_features=10000)
TFIDF_MATRIX = VECTORIZER.fit_transform(corpus)

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
        category_param = request.args.get("category", "").strip()

        # Browse mode: pill clicked with no search query
        if not q:
            if category_param and category_param in CATEGORY_MAP:
                category_cases = [
                    c for c in CASES
                    if c.get("category", "") == category_param
                ]
                hits = [{
                    "case_name": c["case_name"],
                    "category": c.get("category", ""),
                    "similarity": 1.0,
                    "snippet": c["text"][:300],
                    "url": c.get("url", ""),
                } for c in category_cases]
                return jsonify({
                    "results": hits,
                    "detected_category": CATEGORY_LABELS.get(category_param),
                    "confidence": None,
                })
            # No query, no category — return all cases as default browse
            hits = [{
                "case_name": c["case_name"],
                "category": c.get("category", ""),
                "similarity": 1.0,
                "snippet": c["text"][:300],
                "url": c.get("url", ""),
            } for c in CASES]
            return jsonify({"results": hits, "detected_category": None, "confidence": None})

        # If a pill is active, force that category; otherwise classify the query
        if category_param and category_param in CATEGORY_MAP:
            detected_category = category_param
            confidence = None
        else:
            result = CLASSIFIER.classify(q)
            detected_category = result.get("category")
            confidence = result.get("score")

        # filter by category
        if detected_category and detected_category in CATEGORY_MAP:
            indices = [
                i for i, c in enumerate(CASES)
                if c.get("category", "") == detected_category
            ]
            if not indices:
                indices = list(range(len(CASES)))
        else:
            detected_category = None
            indices = list(range(len(CASES)))

        query_vec = VECTORIZER.transform([q])
        sub_matrix = TFIDF_MATRIX[indices]
        sims = cosine_similarity(query_vec, sub_matrix).flatten()

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
            })

        human_category = CATEGORY_LABELS.get(detected_category) if detected_category else None
        return jsonify({
            "results": hits,
            "detected_category": human_category,
            "confidence": round(float(confidence), 4) if confidence is not None else None,
        })

    @app.route("/api/episodes")
    def episodes_search():
        q = request.args.get("title", request.args.get("q", ""))
        from flask import redirect
        return redirect(f"/api/search?q={q}")

    @app.route("/api/config")
    def config():
        return jsonify({"use_llm": False})
