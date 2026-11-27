"""
AuroraStay RAG API (Flask).

Endpoints
  GET  /                               welcome text (template-compatible)
  GET  /health                         liveness + index / LLM status
  POST /v1/relevant_chunks             retrieval only
  POST /v1/answer_with_relevant_chunks answer + sources + confidence
  POST /v1/answer                      answer + confidence
  POST /v1/answer/stream               same, streamed as Server-Sent Events
  POST /v1/feedback                    thumbs up / down from the UI
  GET  /v1/history                     saved questions, newest first
  GET  /v1/history/<id>                one saved question with its answer
  DELETE /v1/history/<id>              remove one saved question

Reliability features
  - every request is validated; bad input returns 400 with a JSON error,
    never a stack trace
  - LLM calls have a timeout and the answer call is tried three times;
    one question never takes longer than about 100 s (see rag_utils)
  - if the LLM is unavailable the API still returns the retrieved sources
    with a 503, so staff can read the policy text themselves
  - identical deterministic questions (temperature 0) are served from a
    1-hour cache
  - every response carries a request id (also in the X-Request-ID header)
  - a question is saved to the history only when the caller asks for it
    ("save_history": true), and a history failure never blocks an answer
"""

import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid

from flask import Flask, Response, g, jsonify, request, stream_with_context
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from rag_utils import (DEFAULT_K, DEFAULT_MODEL, TTLCache, assess, build_prompt, find_coverage,
                       format_chunk, invoke_with_retry, make_llm, retrieve_with_scores,
                       split_coverage)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

EMBED_MODEL = "HarishMaths/Hotel-Policy-Embedding"
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "faiss_index")
FEEDBACK_PATH = os.getenv("FEEDBACK_PATH", "feedback.jsonl")
HISTORY_PATH = os.getenv("HISTORY_PATH", "history.db")
HISTORY_MAX_ROWS = 500  # older questions are dropped
VERSION = "2.0.0"
ALLOWED_MODELS = [m.strip() for m in os.getenv(
    "ALLOWED_MODELS", "gpt-4o-mini,gpt-4o,gpt-4.1-mini,gpt-4.1").split(",") if m.strip()]
LIMITS = {"k": (1, 10), "temperature": (0.0, 1.0), "top_p": (0.0, 1.0), "max_tokens": (64, 2048)}
DEFAULTS = {"k": DEFAULT_K, "model_name": DEFAULT_MODEL, "temperature": 0.0, "top_p": 1.0, "max_tokens": 512}
MAX_QUERY_CHARS = 1000
LLM_ATTEMPTS = 3        # tries for the answer call (the proxy sometimes fails for a moment)
RETRY_PAUSE_S = 2.0     # pause before the second try; doubled before the third

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("aurora")

# ---------------------------------------------------------
# Initialize the Flask application
# ---------------------------------------------------------

rag_api = Flask("Aurora Stay API")
rag_api.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # 64 KB is far more than any real question needs
rag_api.json.sort_keys = False

# ---------------------------------------------------------
# Load the embedding model and the saved FAISS vector store (once)
# ---------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

for _name in ("index.faiss", "index.pkl"):
    if not os.path.exists(os.path.join(VECTORSTORE_PATH, _name)):
        raise SystemExit(f"FAISS index not found: {os.path.join(VECTORSTORE_PATH, _name)} is missing. "
                         "Build the index first (see README), then start the API again.")

vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True,  # our own index, built in the notebook
)
print(f"Vectors stored: {vectorstore.index.ntotal}")

# Warm-up: build the BM25 index and run the embedding model once so the
# first real request is as fast as the rest.
retrieve_with_scores("check-in time", k=1, vectorstore=vectorstore, rewrite=False)

answer_cache = TTLCache(maxsize=512, ttl=3600)
feedback_lock = threading.Lock()


# ---------------------------------------------------------
# Question history (one small SQLite file)
# ---------------------------------------------------------
# The front desk shares one screen per shift, so the history is shared
# too. Times are stored in UTC seconds; the UI shows them in local time.

history_lock = threading.Lock()


def _db():
    con = sqlite3.connect(HISTORY_PATH, timeout=5)
    con.row_factory = sqlite3.Row
    return con


def history_init():
    with history_lock, _db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS history (
            id TEXT PRIMARY KEY, ts REAL NOT NULL, query TEXT NOT NULL, answer TEXT NOT NULL,
            route TEXT, score REAL, feedback TEXT, payload TEXT NOT NULL)""")
        con.execute("CREATE INDEX IF NOT EXISTS history_ts ON history (ts DESC)")


def history_add(rid, query, payload):
    """Save one answered question. Never raises: history is a convenience."""
    try:
        conf = payload.get("confidence") or {}
        keep = {k: payload.get(k) for k in ("sources", "confidence", "latency_ms")}
        with history_lock, _db() as con:
            # OR IGNORE: a caller that repeats a request id cannot overwrite a saved row.
            con.execute("INSERT OR IGNORE INTO history VALUES (?,?,?,?,?,?,?,?)",
                        (rid, time.time(), query, payload.get("answer", ""), conf.get("route"),
                         conf.get("score"), None, json.dumps(keep)))
            con.execute("DELETE FROM history WHERE id NOT IN "
                        "(SELECT id FROM history ORDER BY ts DESC LIMIT ?)", (HISTORY_MAX_ROWS,))
    except Exception as exc:
        log.warning("rid=%s history not saved: %s", rid, type(exc).__name__)


def history_feedback(rid, rating):
    try:
        with history_lock, _db() as con:
            con.execute("UPDATE history SET feedback=? WHERE id=?", (rating, rid))
    except Exception as exc:
        log.warning("rid=%s history feedback not saved: %s", rid, type(exc).__name__)


try:
    history_init()
except Exception as exc:
    # The API still answers questions without the history file.
    log.warning("history is off, could not open %s: %s", HISTORY_PATH, type(exc).__name__)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

class LLMUnavailable(Exception):
    """The answer call failed. Carries the passages already retrieved."""

    def __init__(self, scored):
        super().__init__("LLM unavailable")
        self.scored = scored


def describe(exc) -> str:
    """Exception type and HTTP status only. The message of an upstream error
    can hold request details, so it is never logged."""
    status = getattr(exc, "status_code", None)
    return f"{type(exc).__name__} status={status}" if status else type(exc).__name__


class BadRequest(Exception):
    def __init__(self, message, code="invalid_request", status=400):
        super().__init__(message)
        self.message, self.code, self.status = message, code, status


def error(message, code, status, **extra):
    body = {"error": {"code": code, "message": message}, "request_id": g.get("rid")}
    body.update(extra)
    return jsonify(body), status


def llm_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY")) and bool(os.getenv("OPENAI_API_BASE"))


def parse(require_llm: bool) -> dict:
    """Validate the JSON body and return clean, checked parameters."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise BadRequest("Request body must be a JSON object.")
    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        raise BadRequest("'query' is required and must be a non-empty string.")
    query = " ".join(query.split())
    if len(query) > MAX_QUERY_CHARS:
        raise BadRequest(f"'query' must be at most {MAX_QUERY_CHARS} characters.")
    p = {"query": query, "save_history": data.get("save_history") is True}
    for name, (lo, hi) in LIMITS.items():
        raw = data.get(name, DEFAULTS[name])
        if isinstance(raw, bool):
            raise BadRequest(f"'{name}' must be a number between {lo} and {hi}.")
        try:
            val = float(raw)
            if name in ("k", "max_tokens"):
                # int() of infinity raises OverflowError; 3.9 is not a whole number.
                if int(val) != val:
                    raise ValueError(name)
                val = int(val)
        except (TypeError, ValueError, OverflowError):
            whole = "whole number" if name in ("k", "max_tokens") else "number"
            raise BadRequest(f"'{name}' must be a {whole} between {lo} and {hi}.")
        if not lo <= val <= hi:
            raise BadRequest(f"'{name}' must be between {lo} and {hi}.")
        p[name] = val
    if require_llm:
        model = data.get("model_name") or DEFAULTS["model_name"]
        if model not in ALLOWED_MODELS:
            raise BadRequest(f"'model_name' must be one of {ALLOWED_MODELS}.")
        p["model_name"] = model
        if not llm_configured():
            raise BadRequest("The LLM is not configured: set OPENAI_API_KEY and OPENAI_API_BASE.",
                             code="llm_not_configured", status=503)
    return p


def sources_json(scored_docs, cited=()):
    cited = set(cited)
    return [{
        "id": i,
        "chunk_id": d.metadata.get("chunk_id"),
        "document": d.metadata.get("document"),
        "policy_code": d.metadata.get("policy_code"),
        "section": d.metadata.get("section"),
        "text": d.page_content.split("\n", 1)[-1],
        "score": round(float(s), 3),
        "cited": i in cited,
    } for i, (d, s) in enumerate(scored_docs, start=1)]


def cache_key(p):
    return (p["query"].lower(), p["k"], p["model_name"], p["temperature"], p["top_p"], p["max_tokens"])


def answer_payload(p) -> dict:
    """Run (or fetch from cache) the full pipeline for validated params.

    Same steps as rag_utils.rag_detailed. They are written out here so that
    only a failed answer call raises LLMUnavailable, and the passages found
    before the failure can still be returned.
    """
    key = cache_key(p)
    if p["temperature"] == 0:
        hit = answer_cache.get(key)
        if hit is not None:
            return {**hit, "cached": True}
    t0 = time.perf_counter()
    scored = retrieve_with_scores(p["query"], p["k"], vectorstore, filtered=True)
    t1 = time.perf_counter()
    docs = [d for d, _ in scored]
    llm = make_llm(p["model_name"], p["temperature"], p["top_p"], p["max_tokens"])
    try:
        raw = invoke_with_retry(llm, build_prompt(p["query"], docs), LLM_ATTEMPTS, RETRY_PAUSE_S)
    except Exception as exc:  # upstream LLM errors, timeouts, auth
        log.warning("rid=%s LLM failure: %s", g.get("rid"), describe(exc))
        raise LLMUnavailable(scored) from None
    t2 = time.perf_counter()
    answer_text, coverage = split_coverage(raw)
    conf = assess(answer_text, coverage, scored)
    payload = {
        "answer": answer_text,
        "relevant_chunks": [format_chunk(d) for d in docs],
        "sources": sources_json(scored, conf["cited"]),
        "confidence": conf,
        "latency_ms": {"retrieval": round((t1 - t0) * 1000),
                       "generation": round((t2 - t1) * 1000),
                       "total": round((t2 - t0) * 1000)},
        "cached": False,
    }
    if p["temperature"] == 0:
        answer_cache.set(key, payload)
    return payload


# ---------------------------------------------------------
# Request lifecycle, error handlers
# ---------------------------------------------------------

@rag_api.before_request
def _start():
    supplied = request.headers.get("X-Request-ID", "")
    # Accept a caller's id only if it is short and plain (it is echoed into
    # headers and logs).
    g.rid = supplied if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", supplied) else uuid.uuid4().hex[:12]
    g.t0 = time.perf_counter()


@rag_api.after_request
def _finish(resp):
    resp.headers["X-Request-ID"] = g.get("rid", "")
    if request.path != "/health":
        log.info("rid=%s %s %s status=%s %.0fms", g.get("rid"), request.method, request.path,
                 resp.status_code, (time.perf_counter() - g.get("t0", time.perf_counter())) * 1000)
    return resp


@rag_api.errorhandler(BadRequest)
def _bad_request(e):
    return error(e.message, e.code, e.status)


@rag_api.errorhandler(404)
def _not_found(e):
    return error("Endpoint not found.", "not_found", 404)


@rag_api.errorhandler(405)
def _not_allowed(e):
    return error("Method not allowed for this endpoint.", "method_not_allowed", 405)


@rag_api.errorhandler(413)
def _too_large(e):
    return error("Request body too large.", "payload_too_large", 413)


@rag_api.errorhandler(Exception)
def _unexpected(e):
    log.exception("rid=%s unhandled error", g.get("rid"))
    return error("Unexpected server error. Please retry; if it persists, contact support "
                 "with the request id.", "internal_error", 500)


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

@rag_api.get("/")
def home():
    """Plain-text welcome message, kept exactly as in the course template."""
    return "Welcome to the Aurora Stay RAG API!"


@rag_api.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "version": VERSION,
        "vectors": int(vectorstore.index.ntotal),
        "embedding_model": EMBED_MODEL,
        "llm_configured": llm_configured(),
        "default_model": DEFAULT_MODEL,
        "allowed_models": ALLOWED_MODELS,
    })


@rag_api.post("/v1/relevant_chunks")
def relevant_chunks():
    """Return the top-k relevant chunks for a query (no LLM call).

    Example request: {"query": "What is the cancellation policy?", "k": 3}
    """
    p = parse(require_llm=False)
    t0 = time.perf_counter()
    scored = retrieve_with_scores(p["query"], p["k"], vectorstore)
    return jsonify({
        "query": p["query"],
        "k": p["k"],
        "relevant_chunks": [format_chunk(d) for d, _ in scored],
        "sources": sources_json(scored),
        "latency_ms": {"retrieval": round((time.perf_counter() - t0) * 1000)},
        "request_id": g.rid,
    })


def _answer_or_degrade(p):
    """Answer; if the answer call fails return the sources with a 503.

    Any other error is a bug on our side and goes to the 500 handler.
    """
    try:
        payload = answer_payload(p)
    except LLMUnavailable as exc:
        # Reuse the passages found before the failure: no second search and
        # no further LLM call.
        scored = exc.scored
        return None, error(
            "The answer service is temporarily unavailable. The most relevant policy "
            "sources are included so you can check them directly.",
            "llm_unavailable", 503,
            relevant_chunks=[format_chunk(d) for d, _ in scored],
            sources=sources_json(scored))
    if p["save_history"]:
        history_add(g.rid, p["query"], payload)
    return payload, None


@rag_api.post("/v1/answer_with_relevant_chunks")
def answer_with_relevant_chunks():
    """Answer + supporting chunks + confidence routing.

    Example request:
    {"query": "What is the cancellation policy?", "k": 6,
     "model_name": "gpt-4o-mini", "temperature": 0.0, "top_p": 1.0, "max_tokens": 512}
    """
    p = parse(require_llm=True)
    payload, err = _answer_or_degrade(p)
    if err:
        return err
    return jsonify({**{k: p[k] for k in ("query", "k", "model_name", "temperature", "top_p", "max_tokens")},
                    **payload, "request_id": g.rid})


@rag_api.post("/v1/answer")
def answer():
    """Answer only (plus confidence). Same request body as above."""
    p = parse(require_llm=True)
    payload, err = _answer_or_degrade(p)
    if err:
        return err
    return jsonify({"answer": payload["answer"], "confidence": payload["confidence"],
                    "latency_ms": payload["latency_ms"], "cached": payload["cached"],
                    "request_id": g.rid})


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@rag_api.post("/v1/answer/stream")
def answer_stream():
    """Stream the answer token by token as Server-Sent Events.

    Events: sources -> token* -> done   (or error)
    """
    p = parse(require_llm=True)
    rid = g.rid

    def failed(message="The answer service is temporarily unavailable. "
                       "Please check the sources above or retry.", code="llm_unavailable"):
        return _sse("error", {"error": {"code": code, "message": message}, "request_id": rid})

    def run():
        # Everything is inside one try so the stream always ends with a
        # "done" or an "error" event.
        try:
            yield from answer_events()
        except Exception:
            log.exception("rid=%s stream failed", rid)
            yield failed("Unexpected server error. Please retry; if it persists, contact "
                         "support with the request id.", "internal_error")

    def answer_events():
        t0 = time.perf_counter()
        key = cache_key(p)
        hit = answer_cache.get(key) if p["temperature"] == 0 else None
        if hit is not None:
            yield _sse("sources", {"sources": hit["sources"], "request_id": rid})
            yield _sse("token", {"t": hit["answer"]})
            if p["save_history"]:
                history_add(rid, p["query"], hit)
            yield _sse("done", {"answer": hit["answer"], "confidence": hit["confidence"],
                                "sources": hit["sources"], "latency_ms": hit["latency_ms"],
                                "cached": True, "request_id": rid})
            return
        scored = retrieve_with_scores(p["query"], p["k"], vectorstore, filtered=True)
        t1 = time.perf_counter()
        yield _sse("sources", {"sources": sources_json(scored), "request_id": rid})
        llm = make_llm(p["model_name"], p["temperature"], p["top_p"], p["max_tokens"], streaming=True)
        prompt = build_prompt(p["query"], [d for d, _ in scored])
        # text is the reply so far without leading blanks; sent counts the
        # characters of text already streamed.
        text, sent = "", 0
        hold = 24  # hold back enough characters to hide the COVERAGE line
        for attempt in range(LLM_ATTEMPTS):
            try:
                for part in llm.stream(prompt):
                    text = (text + (part.content or "")).lstrip()
                    cut, _ = find_coverage(text)
                    safe = len(text) - hold if cut is None else cut
                    if safe > sent:
                        yield _sse("token", {"t": text[sent:safe]})
                        sent = safe
                break
            except Exception as exc:
                log.warning("rid=%s stream failure (try %d): %s", rid, attempt + 1, describe(exc))
                # A failure before the first token can be retried without the
                # reader noticing. After that the stream has to end.
                if text or attempt == LLM_ATTEMPTS - 1:
                    yield failed()
                    return
                time.sleep(RETRY_PAUSE_S * (attempt + 1))
        answer_text, coverage = split_coverage(text)
        shown = text[:sent]
        if answer_text.startswith(shown) and len(answer_text) > len(shown):
            yield _sse("token", {"t": answer_text[len(shown):]})
        t2 = time.perf_counter()
        conf = assess(answer_text, coverage, scored)
        latency = {"retrieval": round((t1 - t0) * 1000), "generation": round((t2 - t1) * 1000),
                   "total": round((t2 - t0) * 1000)}
        sources = sources_json(scored, conf["cited"])
        payload = {"answer": answer_text, "relevant_chunks": [format_chunk(d) for d, _ in scored],
                   "sources": sources, "confidence": conf, "latency_ms": latency, "cached": False}
        if p["temperature"] == 0:
            answer_cache.set(key, payload)
        if p["save_history"]:
            history_add(rid, p["query"], payload)
        yield _sse("done", {"answer": answer_text, "confidence": conf, "sources": sources,
                            "latency_ms": latency, "cached": False, "request_id": rid})

    return Response(stream_with_context(run()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@rag_api.post("/v1/feedback")
def feedback():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise BadRequest("Request body must be a JSON object.")
    rating = data.get("rating")
    if rating not in ("up", "down"):
        raise BadRequest("'rating' must be 'up' or 'down'.")
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "request_id": str(data.get("request_id", ""))[:64],
        "rating": rating,
        "query": str(data.get("query", ""))[:MAX_QUERY_CHARS],
        "answer": str(data.get("answer", ""))[:4000],
        "comment": str(data.get("comment", ""))[:1000],
    }
    with feedback_lock, open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    if record["request_id"]:
        history_feedback(record["request_id"], rating)
    return jsonify({"ok": True, "request_id": g.rid})


@rag_api.get("/v1/history")
def history_list():
    """Saved questions, newest first. Optional: ?q=text to search, ?limit=1..500."""
    try:
        limit = max(1, min(HISTORY_MAX_ROWS, int(request.args.get("limit", 200))))
    except ValueError:
        raise BadRequest("'limit' must be a whole number.")
    q = " ".join(request.args.get("q", "").split())[:200]
    sql, args = "SELECT id, ts, query, route, score, feedback FROM history", []
    if q:
        # Escape the LIKE wildcards so a search for "50%" means the text "50%".
        like = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        sql += " WHERE query LIKE ? ESCAPE '\\'"
        args.append(f"%{like}%")
    with history_lock, _db() as con:
        rows = con.execute(sql + " ORDER BY ts DESC LIMIT ?", (*args, limit)).fetchall()
    return jsonify({"items": [dict(r) for r in rows], "request_id": g.rid})


def _history_id(item_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", item_id):
        raise BadRequest("Unknown history id.", code="not_found", status=404)
    return item_id


@rag_api.get("/v1/history/<item_id>")
def history_get(item_id):
    with history_lock, _db() as con:
        row = con.execute("SELECT * FROM history WHERE id=?", (_history_id(item_id),)).fetchone()
    if row is None:
        raise BadRequest("Unknown history id.", code="not_found", status=404)
    item = dict(row)
    item.update(json.loads(item.pop("payload")))
    return jsonify({**item, "request_id": g.rid})


@rag_api.delete("/v1/history/<item_id>")
def history_delete(item_id):
    with history_lock, _db() as con:
        gone = con.execute("DELETE FROM history WHERE id=?", (_history_id(item_id),)).rowcount
    if not gone:
        raise BadRequest("Unknown history id.", code="not_found", status=404)
    return jsonify({"ok": True, "request_id": g.rid})


# ---------------------------------------------------------
# Run the Flask application (local development only;
# in Docker the app is served by gunicorn)
# ---------------------------------------------------------

if __name__ == "__main__":
    rag_api.run(host="0.0.0.0", port=7860, debug=False, threaded=True)
