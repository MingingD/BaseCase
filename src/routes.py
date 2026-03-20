"""
Routes: React app serving and legal case search API.
"""
import json
import os
import sys
import numpy as np
from flask import send_from_directory, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import query classifier
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'query-classifier'))
from classifier import RuleBasedLegalClassifier
from keywords import category_keywords

# ── Load cases ────────────────────────────────────────────────────────────────
_cases_path = os.path.join(os.path.dirname(__file__), 'cases.json')
with open(_cases_path) as f:
    CASES = json.load(f)

# ── Build TF-IDF index (Lectures 4–5) ────────────────────────────────────────
_corpus = [f"{c['case_name']} {c['text']}" for c in CASES]
VECTORIZER = TfidfVectorizer(stop_words='english', max_features=10000)
TFIDF_MATRIX = VECTORIZER.fit_transform(_corpus)

# ── Classifier ────────────────────────────────────────────────────────────────
CLASSIFIER = RuleBasedLegalClassifier(category_keywords)

# Maps classifier category keys → substring found in cases.json category field
CATEGORY_MAP = {
    'personal_injury': 'personal injury',
    'employment_labor': 'employment',
    'copyright':        'copyright',
}

CATEGORY_LABELS = {
    'personal_injury': 'Personal Injury',
    'employment_labor': 'Employment Law',
    'copyright':        'Copyright',
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
        if not q:
            return jsonify({"results": [], "detected_category": None, "confidence": None})

        # Classify query
        result = CLASSIFIER.classify(q)
        detected_category = result.get("category")
        confidence = result.get("score")

        # Filter indices by category if confident match
        if detected_category and detected_category in CATEGORY_MAP:
            cat_substr = CATEGORY_MAP[detected_category]
            indices = [
                i for i, c in enumerate(CASES)
                if cat_substr.lower() in c.get("category", "").lower()
            ]
            # Fall back to full corpus if filter yields nothing
            if not indices:
                indices = list(range(len(CASES)))
        else:
            detected_category = None
            indices = list(range(len(CASES)))

        # TF-IDF cosine similarity (Lectures 6–7)
        query_vec = VECTORIZER.transform([q])
        sub_matrix = TFIDF_MATRIX[indices]
        sims = cosine_similarity(query_vec, sub_matrix).flatten()

        # Normalize scores to relative probabilities (Daming's suggestion)
        total = sims.sum()
        if total > 0:
            sims = sims / total

        # Top-5 by similarity
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

    # Keep /api/episodes as alias so existing wiring doesn't break
    @app.route("/api/episodes")
    def episodes_search():
        q = request.args.get("title", request.args.get("q", ""))
        from flask import redirect
        return redirect(f"/api/search?q={q}")

    @app.route("/api/config")
    def config():
        return jsonify({"use_llm": False})
