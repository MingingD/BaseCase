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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'llmRAG'))
from rag import run_case_rag, run_case_rag_chat, rewrite_query_for_retrieval

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


def _normalize_title(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _resolve_case_by_name(case_name: str):
    """
    Find authoritative case payload from CASES by title.
    Tries exact match first, then normalized equality, then substring fallback.
    """
    raw = (case_name or "").strip()
    if not raw:
        return None

    for case in CASES:
        if (case.get("case_name") or "") == raw:
            return case

    wanted = _normalize_title(raw)
    for case in CASES:
        if _normalize_title(case.get("case_name") or "") == wanted:
            return case

    for case in CASES:
        c_name = (case.get("case_name") or "")
        c_norm = _normalize_title(c_name)
        if wanted and (wanted in c_norm or c_norm in wanted):
            return case
    return None


def _resolve_case_by_idx(case_idx):
    try:
        i = int(case_idx)
    except (TypeError, ValueError):
        return None
    if i < 0 or i >= len(CASES):
        return None
    return CASES[i]


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


def _wants_query_rewrite() -> bool:
    raw = (request.args.get("rewrite", "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _activated_dimension_labels(query_svd: np.ndarray, top_n: int = 3):
    """Top latent dimensions by |activation|, with +/- sign, for query-level explainability."""
    vec = query_svd[0]
    order = np.argsort(np.abs(vec))[::-1]
    labels = []
    for d in order:
        if d >= _n_label_dims:
            continue
        if len(labels) >= top_n:
            break
        sign = '+' if vec[d] >= 0 else '-'
        labels.append(f"({sign}) {DIMENSION_LABELS.get(d, f'Dimension {d}')}")
    return labels


def _per_hit_latent_overlap_labels(
    query_vec: np.ndarray,
    doc_vec: np.ndarray,
    top_n: int = 3,
):
    """
    Latent dimensions ranked by contribution q_i * d_i to cosine(query, doc).
    Sign prefix (+/-) shows whether query and document activate the dimension
    in the same direction (+) or opposite directions (-).
    """
    q = np.asarray(query_vec, dtype=float).reshape(-1)
    d = np.asarray(doc_vec, dtype=float).reshape(-1)
    if q.size != d.size or q.size == 0:
        return []
    contrib = q * d
    cap = min(int(contrib.shape[0]), _n_label_dims)
    if cap <= 0:
        return []
    order = np.argsort(-np.abs(contrib[:cap]))
    labels = []
    for dim in order:
        if len(labels) >= top_n:
            break
        val = float(contrib[dim])
        if abs(val) < 1e-10:
            continue
        sign = '+' if val >= 0 else '-'
        labels.append(f"({sign}) {DIMENSION_LABELS.get(int(dim), f'Dimension {int(dim)}')}")
    return labels


def _sentence_aware_prefix(text: str, max_len: int = 320) -> str:
    """Browse / fallback: prefer breaking at sentence or word, not mid-token."""
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    window = t[:max_len]
    for sep in (". ", "? ", "! ", "\n"):
        idx = window.rfind(sep)
        if idx > max_len // 4:
            return (t[: idx + len(sep)].strip() + "…")
    sp = window.rfind(" ")
    if sp > 40:
        return t[:sp].rstrip() + "…"
    return window.rstrip() + "…"


def _iter_text_windows(text: str, window_chars: int = 420, stride: int = 140):
    """Overlapping spans for query-aligned excerpt selection."""
    t = (text or "").strip()
    n = len(t)
    if n == 0:
        return
    if n <= window_chars:
        yield 0, t
        return
    start = 0
    while start < n:
        end = min(start + window_chars, n)
        yield start, t[start:end]
        if end >= n:
            break
        next_start = start + stride
        if next_start < n:
            sp = t.find(" ", next_start)
            if sp != -1 and sp < next_start + 50:
                next_start = sp + 1
        start = next_start


def _trim_chunk_at_word(chunk: str, max_out: int = 380) -> str:
    excerpt = chunk.strip()
    if len(excerpt) <= max_out:
        return excerpt
    excerpt = excerpt[:max_out]
    sp = excerpt.rfind(" ")
    if sp > max_out // 2:
        excerpt = excerpt[:sp]
    return excerpt.rstrip() + "…"


def _best_snippet_for_query(
    text: str,
    query_tfidf,
    *,
    window_chars: int = 420,
    stride: int = 140,
    max_out: int = 380,
    min_cos: float = 0.001,
):
    """
    Pick the text window whose TF-IDF vector is most similar to the query.
    Returns (snippet, snippet_is_excerpt). Falls back to sentence-aware prefix
    when the document is short or similarity is negligible.
    """
    t = (text or "").strip()
    if not t:
        return "", False

    windows = list(_iter_text_windows(t, window_chars, stride))
    if not windows:
        return _sentence_aware_prefix(t, max_out), False

    chunk_texts = [w[1] for w in windows]
    if not chunk_texts:
        return _sentence_aware_prefix(t, max_out), False

    W = VECTORIZER.transform(chunk_texts)
    sims = cosine_similarity(query_tfidf, W).flatten()
    best_i = int(np.argmax(sims))
    best_cos = float(sims[best_i])

    if best_cos < min_cos or not np.isfinite(best_cos):
        return _sentence_aware_prefix(t, max_out), False

    start, chunk = windows[best_i]
    body = _trim_chunk_at_word(chunk, max_out)
    prefix = "… " if start > 0 else ""
    return prefix + body, True


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
        use_rewrite = _wants_query_rewrite()
        effective_q = rewrite_query_for_retrieval(q) if q and use_rewrite else q
        rewrite_applied = bool(q and use_rewrite and effective_q and effective_q != q)
        user_cat_keys = _category_keys_from_request()

        # Browse mode: pill(s) with no search query
        if not q:
            if user_cat_keys:
                allow = set(user_cat_keys)
                category_cases = [
                    (i, c) for i, c in enumerate(CASES)
                    if c.get("category", "") in allow
                ]
                hits = [{
                    "case_idx": i,
                    "case_name": c["case_name"],
                    "category": c.get("category", ""),
                    "similarity": 1.0,
                    "snippet": _sentence_aware_prefix(c.get("text") or "", 320),
                    "snippet_is_excerpt": False,
                    "url": c.get("url", ""),
                    "why": [],
                } for i, c in category_cases]
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
                "case_idx": i,
                "case_name": c["case_name"],
                "category": c.get("category", ""),
                "similarity": 1.0,
                "snippet": _sentence_aware_prefix(c.get("text") or "", 320),
                "snippet_is_excerpt": False,
                "url": c.get("url", ""),
                "why": [],
            } for i, c in enumerate(CASES)]
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
            result = CLASSIFIER.classify(effective_q)
            status = result.get("status", "ok")
            detected_category = result.get("category")
            raw_conf = result.get("score")
            confidence = raw_conf if isinstance(raw_conf, (int, float)) else None
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
                    reason=(
                        "Enter more detail, or select the legal areas you want above."
                    ),
                    candidates_raw=result.get("candidates"),
                    needs_user_category=True,
                )

        query_svd = _query_svd_vector(effective_q)
        q_tfidf_snippet = VECTORIZER.transform([effective_q])
        activated_dimensions = _activated_dimension_labels(query_svd, top_n=3)

        sub_matrix = SVD_MATRIX[indices]
        sims = cosine_similarity(query_svd, sub_matrix).flatten()

        # normalize so scores add up to 1
        total = sims.sum()
        if total > 0:
            sims = sims / total

        top_k = min(10, len(indices))
        top_local = np.argsort(sims)[::-1][:top_k]

        hits = []
        for local_idx in top_local:
            global_idx = indices[local_idx]
            case = CASES[global_idx]
            snip, is_excerpt = _best_snippet_for_query(
                case.get("text") or "",
                q_tfidf_snippet,
            )
            doc_svd = np.asarray(SVD_MATRIX[global_idx]).reshape(-1)
            why_hit = _per_hit_latent_overlap_labels(query_svd[0], doc_svd, top_n=3)
            hits.append({
                "case_idx": global_idx,
                "case_name": case["case_name"],
                "category": case.get("category", ""),
                "similarity": round(float(sims[local_idx]), 4),
                "snippet": snip,
                "snippet_is_excerpt": is_excerpt,
                "url": case.get("url", ""),
                "why": why_hit,
            })

        return jsonify({
            "results": hits,
            "detected_category": human_category,
            "confidence": round(float(confidence), 4) if confidence is not None else None,
            "activated_dimensions": activated_dimensions,
            "classification": classification,
            "query_used_for_retrieval": effective_q,
            "query_rewrite_applied": rewrite_applied,
        })

    @app.route("/api/episodes")
    def episodes_search():
        q = request.args.get("title", request.args.get("q", ""))
        from flask import redirect
        return redirect(f"/api/search?q={q}")

    @app.route("/api/case-rag", methods=["POST"])
    def case_rag():
        payload = request.get_json(silent=True) or {}
        user_query = (payload.get("user_query") or "").strip()
        case_name = (payload.get("case_name") or "").strip()
        case_idx = payload.get("case_idx")

        if not user_query:
            return jsonify({"error": "Missing required field: user_query"}), 400
        if case_idx is None and not case_name:
            return jsonify({"error": "Missing required field: case_idx or case_name"}), 400

        case = _resolve_case_by_idx(case_idx)
        if not case and case_name:
            case = _resolve_case_by_name(case_name)
        if not case:
            return jsonify({"error": "Case not found for provided identifier"}), 404

        case_context = (case.get("text") or "").strip()
        if not case_context:
            return jsonify({"error": "Selected case has no opinion text"}), 422

        answer = run_case_rag(user_query, case_context, case.get("case_name") or case_name)
        return jsonify({
            "answer": answer,
            "case_name": case.get("case_name") or case_name,
        })

    @app.route("/api/case-rag-chat", methods=["POST"])
    def case_rag_chat():
        payload = request.get_json(silent=True) or {}
        case_idx = payload.get("case_idx")
        case_name = (payload.get("case_name") or "").strip()
        user_query = (payload.get("user_query") or "").strip()
        snippet = (payload.get("snippet") or "").strip()
        messages = payload.get("messages") or []

        if case_idx is None and not case_name:
            return jsonify({"error": "Missing required field: case_idx or case_name"}), 400
        if not user_query:
            return jsonify({"error": "Missing required field: user_query"}), 400
        if not isinstance(messages, list) or len(messages) == 0:
            return jsonify({"error": "Missing required field: messages"}), 400

        case = _resolve_case_by_idx(case_idx)
        if not case and case_name:
            case = _resolve_case_by_name(case_name)
        if not case:
            return jsonify({"error": "Case not found for provided identifier"}), 404

        answer = run_case_rag_chat(
            user_query=user_query,
            case_name=case.get("case_name") or case_name,
            snippet=snippet,
            messages=messages,
        )
        return jsonify({
            "answer": answer,
            "case_name": case.get("case_name") or case_name,
        })

    @app.route("/api/config")
    def config():
        return jsonify({"use_llm": False})
