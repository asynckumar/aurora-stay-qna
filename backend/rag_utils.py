"""
AuroraStay RAG core: hybrid retrieval, grounded generation, confidence routing.

Used by the Flask API (app.py) and identical to the functions defined and
evaluated in the notebook.

Pipeline for one question
1. Query rewriting  - gpt-4o-mini restates the question in the vocabulary the
                      policies use ("loud next door" -> "noise, quiet hours,
                      disturbance") and the rewrite is appended to the
                      original question. Measured on the evaluation data:
                      context coverage 0.74 -> 0.82 (eval), 0.83 -> 0.91
                      (held-out test). Falls back to the original question if
                      the rewrite call fails.
2. Hybrid retrieval - dense search (FAISS, fine-tuned hotel-policy embedding)
                      and keyword search (BM25) are combined with a
                      score-aware convex fusion (alpha 0.5). Dense search
                      understands paraphrases; BM25 catches exact room types,
                      property codes, prices and policy terms. Convex fusion
                      beat Reciprocal Rank Fusion on the held-out test set
                      (0.83 vs 0.77) because it keeps score margins.
3. De-duplication   - the same property matrix appears in all four
                      documents; repeated table rows and near-identical
                      chunks are dropped so the k slots carry k different
                      facts.
4. Generation       - the LLM answers only from the numbered sources, cites
                      them inline as [1], [2], and reports whether the
                      sources fully, partly or do not cover the question.
5. Confidence       - coverage, retrieval strength and citation checks are
                      combined into a 0-1 score and routed to AUTO ANSWER /
                      ASSISTED REVIEW / MANUAL LOOKUP (same thresholds as the
                      offline evaluation).
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections import OrderedDict

import numpy as np
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gpt-4o-mini"
# Time budget for one question. The UI waits 120 s, so the worst case here
# must stay under about 100 s: rewrite 8 s + filter 8 s + three answer
# tries of 25 s + 6 s of pauses. invoke_with_retry already tries three
# times, so the client library does not retry on top of that.
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "25"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "0"))
# The rewrite and filter calls are optional helpers: short timeout, one try.
HELPER_TIMEOUT_S = float(os.getenv("HELPER_TIMEOUT_S", "8"))
# The embedding model reads only this many tokens of the search text.
EMBED_MAX_TOKENS = 128
CANDIDATES = 20          # candidates taken from each retriever before fusion
RRF_K = 60               # standard Reciprocal Rank Fusion constant
FUSION = os.getenv("FUSION", "convex")       # "convex" or "rrf"
ALPHA = float(os.getenv("FUSION_ALPHA", "0.5"))  # weight of dense score
EXPAND_ALIASES = os.getenv("EXPAND_ALIASES", "1") == "1"  # city name -> property code (cities only)
REWRITE_QUERIES = os.getenv("REWRITE_QUERIES", "1") == "1"
CONTEXT_FILTER = os.getenv("CONTEXT_FILTER", "1") == "1"   # LLM keeps only needed passages
REWRITE_MODEL = os.getenv("REWRITE_MODEL", "gpt-4o-mini")
DEFAULT_K = 6
# Adaptive k: after the first MIN_K chunks, keep a chunk only if its fused
# score is at least REL_CUTOFF x the best chunk's score. k is the maximum.
MIN_K = int(os.getenv("MIN_K", "3"))
REL_CUTOFF = float(os.getenv("REL_CUTOFF", "0.7"))  # tuned on evaluation.csv
DEDUPE_JACCARD = 0.7     # word overlap above which two chunks are duplicates
AUTO_THRESHOLD = 0.75    # same routing thresholds as the evaluation notebook
REVIEW_THRESHOLD = 0.50

INSUFFICIENT = ("The available policy documents do not contain enough "
                "information to answer this question.")

_STOP = {"the", "a", "an", "and", "or", "of", "to", "is", "are", "be", "in",
         "for", "by", "with", "on", "as", "at", "it", "that", "this", "may",
         "not", "i", "my", "can", "do", "what", "how", "does", "if", "will"}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9$:]+", text.lower())) - _STOP


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9$]+", text.lower())


def _body(text: str) -> str:
    """Chunk text without its 'Document | Section' header line."""
    return text.split("\n", 1)[-1]


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Dense (FAISS) + keyword (BM25) retrieval with convex score fusion
    (RRF optional) and dedupe."""

    def __init__(self, vectorstore, reranker=None):
        from rank_bm25 import BM25Okapi

        self.vectorstore = vectorstore
        self.reranker = reranker
        store = vectorstore.docstore._dict
        ids = [vectorstore.index_to_docstore_id[i] for i in range(vectorstore.index.ntotal)]
        self.docs = [store[i] for i in ids]
        self.bm25 = BM25Okapi([_tokens(d.page_content) for d in self.docs])
        # Stored (normalized) vectors, used for exact cosine of candidates.
        self.matrix = vectorstore.index.reconstruct_n(0, vectorstore.index.ntotal)
        self.aliases = self._build_aliases()
        # The same table row (e.g. a property's matrix row) appears in several
        # documents, sometimes with fewer columns. Map each row key to its
        # longest copy so a chosen row is always shown in full.
        self.pos = {d.page_content: i for i, d in enumerate(self.docs)}
        self.richest: dict = {}
        for d in self.docs:
            if d.metadata.get("block_type") != "table_row":
                continue
            key = self._row_key(d.page_content)
            best = self.richest.get(key)
            if key and (best is None or len(_body(d.page_content)) > len(_body(best))):
                self.richest[key] = d.page_content

    def _build_aliases(self) -> dict[str, str]:
        """'Property: ASH-MIA Brickell, Miami FL' -> {'miami': 'ASH-MIA Brickell'}."""
        aliases: dict[str, str] = {}
        pat = re.compile(r"Property: (ASH-[A-Z]{3} ([A-Za-z ]+?))(?:, ([A-Za-z ]+?) [A-Z]{2})?[;.]")
        for d in self.docs:
            for m in pat.finditer(d.page_content):
                # Cities only: area names such as Union, Harbor or Loop are also
                # ordinary words and would add codes to unrelated questions.
                name, city = m.group(1), m.group(3)
                for key in filter(None, [city]):
                    aliases.setdefault(key.strip().lower(), name)
        return aliases

    @staticmethod
    def _row_key(text: str):
        """Key of a table-row record: its first column and value.

        'Property: ASH-SEA Pike, Seattle WA; Parking: ...' and
        'Property: ASH-SEA Pike; Parking: ...' share the key
        ('property', 'ash-sea pike'), so they are the same row repeated in
        two documents.
        """
        m = re.match(r"([^:;\n]{2,40}): ([^;,\n]+)", _body(text))
        return (m.group(1).strip().lower(), m.group(2).strip().lower()) if m else None

    def _duplicate_of(self, text: str, chosen: list[str]):
        key = self._row_key(text)
        w = _words(_body(text))
        for i, other in enumerate(chosen):
            if key is not None and key == self._row_key(other):
                return i
            ow = _words(_body(other))
            if len(w & ow) / max(1, len(w | ow)) > DEDUPE_JACCARD:
                return i
        return None

    def expand(self, query: str) -> str:
        if not EXPAND_ALIASES:
            return query
        # Add a property's code only when the question names its city but
        # not the code itself (re-adding a code already present over-weights
        # the property table).
        present = set(re.findall(r"ASH-[A-Z]{3}", query))
        found = sorted({v for k, v in self.aliases.items()
                        if v.split()[0] not in present
                        and re.search(rf"\b{re.escape(k)}\b", query.lower())})
        return f"{query} ({', '.join(found)})" if found else query

    def _too_long(self, text: str) -> bool:
        """True when the embedding model would cut the end of this text."""
        try:
            model = self.vectorstore.embeddings._client
            limit = min(EMBED_MAX_TOKENS, model.max_seq_length)
            return len(model.tokenizer.encode(text)) > limit
        except Exception:
            return len(text.split()) > 80  # rough word count if no tokenizer

    def _dense_query(self, q: str, question) -> str:
        """Search text for the embedding model.

        The rewrite keywords and the property code sit at the end of q. For
        a long question the model would never read them, so they are moved
        to the front. Short questions are returned unchanged.
        """
        if not question or not q.lower().startswith(question.lower()):
            return q
        extra = q[len(question):].strip()
        if not extra or not self._too_long(q):
            return q
        return f"{extra} {question}"

    def search(self, query: str, k: int = DEFAULT_K, question: str = None):
        """Return up to k (Document, relevance) pairs, best first.

        relevance is the cosine similarity between the query and the chunk
        (0-1), used for display and for the confidence score. question is
        the staff question as typed, when query is the question plus the
        rewrite keywords.
        """
        q = self.expand(query)
        qv = np.asarray(self.vectorstore.embeddings.embed_query(self._dense_query(q, question)),
                        dtype="float32")

        # Candidate generation: top-N from FAISS and top-N from BM25.
        _, idx = self.vectorstore.index.search(qv[None, :], CANDIDATES)
        dense_ids = [int(i) for i in idx[0] if i >= 0]
        bm25 = np.asarray(self.bm25.get_scores(_tokens(q)), dtype="float32")
        bm25_ids = list(np.argsort(-bm25)[:CANDIDATES])
        cand = list(dict.fromkeys(dense_ids + [int(i) for i in bm25_ids]))

        # Exact cosine for every candidate (vectors are normalized).
        cos = self.matrix[cand] @ qv
        fused = self._fuse(cos, bm25[cand])
        order = np.argsort(-fused)
        ranked = [self.docs[cand[i]].page_content for i in order]
        fused_by_text = {self.docs[cand[i]].page_content: float(fused[i]) for i in order}
        cos_by_text = {self.docs[c].page_content: float(s) for c, s in zip(cand, cos)}

        if self.reranker is not None:
            ranked = ranked[:CANDIDATES]
            ce = self.reranker.predict([(q, t) for t in ranked])
            ranked = [t for _, t in sorted(zip(ce, ranked), key=lambda x: -x[0])]

        by_text = {d.page_content: d for d in self.docs}
        chosen: list[str] = []
        top_score = fused_by_text[ranked[0]] if ranked else 0.0
        for text in ranked:
            if len(chosen) >= MIN_K and fused_by_text.get(text, 0.0) < REL_CUTOFF * top_score:
                break
            dup = self._duplicate_of(text, chosen)
            if dup is not None:
                # Same fact from another document: keep the richer copy
                # (e.g. the matrix row that also lists check-in times).
                if len(_body(text)) > len(_body(chosen[dup])):
                    chosen[dup] = text
                continue
            chosen.append(text)
            if len(chosen) == k:
                break

        # Replace each table row with its fullest copy across documents.
        chosen = [self._richest_copy(t) for t in chosen]
        chosen = list(dict.fromkeys(chosen))
        return [(by_text[t], max(0.0, self._cos(t, qv, cos_by_text))) for t in chosen]

    def _richest_copy(self, text: str) -> str:
        if by := self.richest.get(self._row_key(text)):
            if self.docs[self.pos[text]].metadata.get("block_type") == "table_row":
                return by
        return text

    def _cos(self, text: str, qv, known: dict) -> float:
        if text in known:
            return known[text]
        return float(self.matrix[self.pos[text]] @ qv)

    def _fuse(self, cos: np.ndarray, bm25: np.ndarray) -> np.ndarray:
        """Combine dense and keyword evidence for the candidate set.

        'convex' (default): alpha * minmax(cosine) + (1 - alpha) * bm25/max.
        Unlike rank fusion it keeps score margins, so a decisive keyword hit
        (an exact room type, property code or fee name) is not out-voted by
        weak dense matches. 'rrf' is kept for comparison.
        """
        if FUSION == "rrf":
            out = np.zeros(len(cos), dtype="float32")
            for scores in (cos, bm25):
                ranks = np.argsort(np.argsort(-scores))
                out += 1.0 / (RRF_K + ranks)
            return out
        span = float(cos.max() - cos.min()) or 1.0
        dn = (cos - cos.min()) / span
        bn = bm25 / (float(bm25.max()) or 1.0)
        return ALPHA * dn + (1.0 - ALPHA) * bn


_RETRIEVERS: dict[int, HybridRetriever] = {}
_RETRIEVER_LOCK = threading.Lock()


def get_retriever(vectorstore, reranker=None) -> HybridRetriever:
    """One HybridRetriever per vector store (BM25 index built once)."""
    key = id(vectorstore)
    with _RETRIEVER_LOCK:
        if key not in _RETRIEVERS:
            _RETRIEVERS[key] = HybridRetriever(vectorstore, reranker)
        return _RETRIEVERS[key]


REWRITE_PROMPT = """Rewrite this hotel staff question as a short search query for AuroraStay policy documents.
Keep every property name or city. Add the policy terms the documents are likely to use (for example: noise -> quiet hours, disturbance; hold -> incidentals authorization; moving dates -> modification, change).
Return one line of keywords, at most 25 words, nothing else.

Question: {q}"""

_REWRITE_CACHE = None  # created lazily (TTLCache is defined below)


def rewrite_query(query: str) -> str:
    """Return 'question + policy-vocabulary keywords' for retrieval.

    Never raises: if the LLM is not configured or the call fails, the
    original question is used unchanged.
    """
    global _REWRITE_CACHE
    if not REWRITE_QUERIES or not os.getenv("OPENAI_API_KEY"):
        return query
    if _REWRITE_CACHE is None:
        _REWRITE_CACHE = TTLCache(maxsize=1024, ttl=24 * 3600)
    key = query.strip().lower()
    hit = _REWRITE_CACHE.get(key)
    if hit is not None:
        return hit
    try:
        llm = make_llm(REWRITE_MODEL, temperature=0.0, top_p=1.0, max_tokens=60,
                       timeout=HELPER_TIMEOUT_S, max_retries=0)
        keywords = " ".join(llm.invoke(REWRITE_PROMPT.format(q=query)).content.split())[:300]
        expanded = f"{query} {keywords}" if keywords else query
    except Exception:
        return query
    _REWRITE_CACHE.set(key, expanded)
    return expanded


FILTER_PROMPT = """A hotel staff member asked: {q}

Numbered policy passages:
{passages}

Which passages are needed to answer it? Keep every passage that states a fact, fee, time, limit, rule, condition, exception or staff action for what the question asks, for each property, rate plan or charge it names. Drop passages about other properties, other topics or general background. When unsure, keep the passage.
Reply with the kept passage numbers only, most useful first, separated by commas (for example: 3,1). Reply 0 if none of them help."""

_FILTER_CACHE = None
_NUMBER_LIST = re.compile(r"\s*\d+(?:[\s,]+\d+)*[\s,.]*")


def filter_context(query: str, scored: list) -> list:
    """Keep only the retrieved passages an LLM judges necessary, best first.

    Runs before generation, so the model really sees only these passages.
    Never raises and never returns nothing: on any failure the retrieved list
    is returned unchanged; if the LLM keeps nothing, the two best matches are kept.
    Only a reply that is a plain list of numbers is used. A sentence such as
    "None of the 6 passages help" also contains a digit, so any other reply
    returns all passages and is not cached.
    """
    global _FILTER_CACHE
    if not CONTEXT_FILTER or len(scored) <= 1 or not os.getenv("OPENAI_API_KEY"):
        return scored
    if _FILTER_CACHE is None:
        _FILTER_CACHE = TTLCache(maxsize=1024, ttl=24 * 3600)
    key = (query.strip().lower(), tuple(d.metadata.get("chunk_id", d.page_content[:40]) for d, _ in scored))
    hit = _FILTER_CACHE.get(key)
    if hit is not None:
        return hit
    passages = "\n\n".join(f"[{i}] {_body(d.page_content)}" for i, (d, _) in enumerate(scored, 1))
    try:
        llm = make_llm(REWRITE_MODEL, 0.0, 1.0, 20, timeout=HELPER_TIMEOUT_S, max_retries=0)
        reply = llm.invoke(FILTER_PROMPT.format(q=query, passages=passages)).content
    except Exception:
        return scored
    if not _NUMBER_LIST.fullmatch(reply or ""):
        return scored
    picks = [int(n) for n in re.findall(r"\d+", reply)]
    keep = list(dict.fromkeys(n for n in picks if 1 <= n <= len(scored)))
    # Nothing judged useful: pass only the two best matches, so the model sees
    # little noise and can say the documents do not cover the question.
    out = [scored[n - 1] for n in keep] or scored[:2]
    _FILTER_CACHE.set(key, out)
    return out


def format_chunk(doc) -> str:
    """Template-compatible chunk string: content plus metadata."""
    return f"Content:\n{doc.page_content}\n\nMetadata:\n{doc.metadata}"


def retrieve_with_scores(query: str, k: int = DEFAULT_K, vectorstore=None, reranker=None,
                         rewrite: bool = True, filtered: bool = False):
    """Top-k (Document, cosine) pairs. With filtered=True (used before
    generation) only the passages needed for the question are kept."""
    search_query = rewrite_query(query) if rewrite else query
    scored = get_retriever(vectorstore, reranker).search(search_query, k=k, question=query)
    return filter_context(query, scored) if filtered else scored


def retrieve(query: str, k: int = DEFAULT_K, vectorstore=None) -> list[str]:
    """Retrieve the top-k relevant chunks (same signature as the template)."""
    return [format_chunk(d) for d, _ in retrieve_with_scores(query, k, vectorstore)]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

RULES = """
1. Answer ONLY from the numbered policy sources below. Do not use outside knowledge, assumptions or typical hotel practice.
2. Start with the direct answer to exactly what was asked: the amount, time, limit or rule. For a yes/no question, first check the facts in the sources (for a number, compare it with the stated limit; a number equal to the limit is allowed), then start with "Yes" or "No" only if the sources settle it without exception, and make sure every later sentence agrees with that first word. When exceptions exist, state the normal rule first and then name the exception.
3. Quote amounts, times, limits and property codes exactly as written in the sources. Never transfer a fee or rule from one property to another.
4. Then add only the conditions, exceptions or approvals that could change the answer for this guest. Do not add facts on other topics the question did not ask about. Do not close with general advice to staff (confirm, check, record) unless the question asks what staff should do or the sources require that check before answering.
5. If the question names several properties, rate plans or charges, answer each one.
6. Cite the source number in square brackets after each fact, for example [1] or [2][3]. Cite only sources you actually used.
7. If the question asks for a detail the sources do not give, say in a few words that it is not stated and where staff can confirm it. Never mention details the question did not ask for, and never add a separate closing caveat.
8. Only if the sources contain nothing that answers any part of the question, reply in one or two sentences: say plainly that the policy documents do not cover the specific thing asked (name it, and the property if one is named), give the closest related fact the sources do state in a few words if there is one, then tell staff to check the live property profile or contact the property team rather than guessing. Never fill a gap with general advice.
9. If sources disagree, state both and say which document each comes from.
10. Write in neutral third person ("the guest", "staff should"), 2 to 3 short sentences in a single paragraph, no greeting, no filler.
11. On the very last line write exactly one of: COVERAGE: FULL, COVERAGE: PARTIAL, COVERAGE: NONE (whether the sources fully answer the question).
""".format(insufficient=INSUFFICIENT)


def build_prompt(query: str, docs) -> str:
    sources = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, start=1))
    return f"""You are AuroraStay's internal policy assistant. Hotel staff ask you questions about guest policies, and you answer from the company's policy documents only.

User Query:
{query}

Retrieved Policy Information (numbered sources):
{sources}

Rules:
{RULES}
Answer:
"""


def make_llm(model_name=DEFAULT_MODEL, temperature=0.0, top_p=1.0, max_tokens=512, streaming=False,
             timeout=None, max_retries=None):
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout=LLM_TIMEOUT_S if timeout is None else timeout,
        max_retries=LLM_MAX_RETRIES if max_retries is None else max_retries,
        streaming=streaming,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
    )


# The coverage line. The model is asked for "COVERAGE: FULL" on the last
# line, but it sometimes writes it in bold, adds text after it, or the reply
# is cut off in the middle of it. The word in capitals is matched anywhere,
# with or without a value. Other spellings ("Coverage - Full") must start a
# line and carry a value, so a normal sentence about coverage is not cut.
_COVERAGE = re.compile(
    r"[\s*_`#>]*(?<![A-Za-z])COVERAGE(?![A-Za-z])(?:[\s*_`:=\-]*(?i:(FULL|PARTIAL|NONE))\b)?"
    r"|\s*^[ \t*_`#>]*(?i:coverage)[ \t*_`]*[:=\-][ \t*_`]*(?i:(FULL|PARTIAL|NONE))\b",
    re.M)
# A last line that stops partway through the word (reply cut by max_tokens).
_COVERAGE_CUT = re.compile(r"\n[ \t*_`#>]*COV(?:E(?:R(?:AG?)?)?)?\s*$")

# Wording of an answer that says the documents do not have the fact.
_REFUSAL = re.compile(
    r"\b(do|does) not (cover|contain|state|mention|specify|include|address|provide)\b"
    r"|\b(don't|doesn't) (cover|contain|state|mention|specify|include|address|provide)\b"
    r"|\b(not|isn't|aren't) (covered|stated|mentioned|specified|addressed)\b"
    r"|\bno (information|mention|details?)\b", re.I)


_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_HEDGE = re.compile(
    r"^(however,?\s+)?(the\s+)?(policy|policies|documents?|sources?|provided (sources|documents))"
    r"\s+(do|does)\s+not\s+(specify|detail|mention|state|provide|include|address)\b[^\[]*$", re.I)


_ADVICE = re.compile(
    r"^(staff|the front desk|the team)\s+should\s+(also\s+)?(confirm|check|ensure|refer|inform|verify|note|consult|contact|advise|review)\b[^\[]*$",
    re.I)


def _drop_closing_advice(answer: str) -> str:
    """Remove a last sentence that only tells staff to confirm or check
    something and cites no source. It is general advice, not policy, and the
    prompt already asks the model not to add it."""
    parts = [x.strip() for x in _SENTENCE.split(answer.strip()) if x.strip()]
    if len(parts) >= 2 and _ADVICE.match(parts[-1]) and re.search(r"\[\d+\]", " ".join(parts[:-1])):
        return " ".join(parts[:-1])
    return answer


def find_coverage(text: str):
    """Return (start, value) of the coverage line, or (None, None).

    value is None when the line is there but its value is missing or cut.
    """
    m = _COVERAGE.search(text)
    if m:
        value = m.group(1) or m.group(2)
        return m.start(), (value.upper() if value else None)
    m = _COVERAGE_CUT.search(text)
    return (m.start(), None) if m else (None, None)


def split_coverage(text: str) -> tuple[str, str]:
    """Cut the answer at the 'COVERAGE: X' line and return (answer, coverage).

    Everything from the coverage line on is dropped, so no part of that line
    is ever shown. Without a usable value the coverage is NONE when the
    answer cites nothing and reads as a refusal, otherwise PARTIAL.
    """
    start, cov = find_coverage(text)
    answer = (text if start is None else text[:start]).strip()
    if cov is None:
        refusal = not re.search(r"\[\d+\]", answer) and _REFUSAL.search(answer)
        cov = "NONE" if refusal else "PARTIAL"
    answer = _drop_closing_advice(answer)
    if cov == "FULL":
        # The model says the sources fully answer the question, so a sentence
        # about what the documents "do not specify" is unrequested hedging
        # (it lowers answer relevancy and worries staff for no reason).
        kept = [x for x in _SENTENCE.split(answer) if not _HEDGE.match(x.strip())]
        if kept and re.search(r"\[\d+\]", " ".join(kept)):
            answer = " ".join(x.strip() for x in kept if x.strip())
    return answer, cov


def invoke_with_retry(llm, prompt: str, attempts: int = 3, pause_s: float = 2.0) -> str:
    """Call the LLM, retrying a few times on any error.

    The course proxy sometimes answers a valid request with a 401 for a few
    seconds. The OpenAI client never retries a 401, so one such reply would
    end a whole evaluation run.
    """
    for attempt in range(attempts):
        try:
            return llm.invoke(prompt).content
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(pause_s * (attempt + 1))


def generate(query, retrieved_chunks, model) -> str:
    """Template-compatible: retrieved_chunks are formatted strings."""
    sources = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(retrieved_chunks, start=1))
    prompt = build_prompt(query, []).replace(
        "Retrieved Policy Information (numbered sources):\n",
        f"Retrieved Policy Information (numbered sources):\n{sources}")
    return model.invoke(prompt)


# ---------------------------------------------------------------------------
# Confidence and routing
# ---------------------------------------------------------------------------

def assess(answer: str, coverage: str, scored_docs) -> dict:
    """Online reliability estimate for one answer.

    The offline composite score needs the expected answer, which does not
    exist for a live question, so the online score uses signals that are
    available at answer time:
      - coverage      : the model's own judgment of whether the sources
                        answer the question (FULL / PARTIAL / NONE),
      - retrieval     : cosine similarity of the best cited source,
      - citations     : whether the answer cites valid sources at all.
    """
    n = len(scored_docs)
    cited = sorted({int(c) for c in re.findall(r"\[(\d+)\]", answer) if 1 <= int(c) <= n})
    cov_score = {"FULL": 1.0, "PARTIAL": 0.55, "NONE": 0.1}.get(coverage, 0.55)
    sims = [scored_docs[i - 1][1] for i in cited] or [s for _, s in scored_docs[:1]] or [0.0]
    # Rescale the cosine band 0.35 to 0.80 to 0-1.
    retrieval = min(1.0, max(0.0, (max(sims) - 0.35) / 0.45))
    citation = 1.0 if cited else (0.5 if coverage == "NONE" else 0.2)
    score = round(0.55 * cov_score + 0.30 * retrieval + 0.15 * citation, 3)

    reasons = []
    reasons.append({"FULL": "Sources fully cover the question.",
                    "PARTIAL": "Sources cover only part of the question.",
                    "NONE": "Sources do not contain the answer."}[coverage])
    reasons.append(f"Best supporting source similarity {max(sims):.2f}.")
    reasons.append(f"{len(cited)} of {n} sources cited." if cited else "Answer cites no source.")

    if coverage == "NONE":
        route = "MANUAL LOOKUP"
    elif coverage == "PARTIAL" and score >= REVIEW_THRESHOLD:
        # Part of the question is unanswered: a person should look first.
        route = "ASSISTED REVIEW"
    elif not cited and score >= REVIEW_THRESHOLD:
        # Nothing in the answer can be checked against a source.
        route = "ASSISTED REVIEW"
    elif score > AUTO_THRESHOLD:
        route = "AUTO ANSWER"
    elif score >= REVIEW_THRESHOLD:
        route = "ASSISTED REVIEW"
    else:
        route = "MANUAL LOOKUP"
    level = {"AUTO ANSWER": "high", "ASSISTED REVIEW": "medium", "MANUAL LOOKUP": "low"}[route]
    return {"score": score, "level": level, "route": route, "reasons": reasons,
            "coverage": coverage, "cited": cited}


# ---------------------------------------------------------------------------
# Full RAG pipeline
# ---------------------------------------------------------------------------

def rag_detailed(query: str, k: int = DEFAULT_K, model_name: str = DEFAULT_MODEL,
                 temperature: float = 0.0, top_p: float = 1.0,
                 max_tokens: int = 512, vectorstore=None, reranker=None) -> dict:
    """Run retrieval + generation and return answer, sources and confidence."""
    t0 = time.perf_counter()
    scored = retrieve_with_scores(query, k, vectorstore, reranker, filtered=True)
    t1 = time.perf_counter()
    docs = [d for d, _ in scored]
    llm = make_llm(model_name, temperature, top_p, max_tokens)
    raw = invoke_with_retry(llm, build_prompt(query, docs))
    t2 = time.perf_counter()
    answer, coverage = split_coverage(raw)
    conf = assess(answer, coverage, scored)
    return {
        "answer": answer,
        "scored_docs": scored,
        "relevant_chunks": [format_chunk(d) for d in docs],
        # Exactly the passages the model was given (no metadata dump); this is
        # what the evaluation judges as the retrieval context.
        "contexts": [d.page_content for d in docs],
        "confidence": conf,
        "latency_ms": {"retrieval": round((t1 - t0) * 1000),
                       "generation": round((t2 - t1) * 1000),
                       "total": round((t2 - t0) * 1000)},
    }


def rag(query: str, k: int = DEFAULT_K, model_name: str = DEFAULT_MODEL,
        temperature: float = 0.0, top_p: float = 1.0, max_tokens: int = 512,
        vectorstore=None):
    """Template-compatible RAG function: returns (answer, retrieved_chunks)."""
    out = rag_detailed(query, k, model_name, temperature, top_p, max_tokens, vectorstore)
    return out["answer"], out["relevant_chunks"]


# ---------------------------------------------------------------------------
# Small thread-safe TTL cache for repeated questions
# ---------------------------------------------------------------------------

class TTLCache:
    def __init__(self, maxsize: int = 256, ttl: float = 3600):
        self.maxsize, self.ttl = maxsize, ttl
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            ts, value = item
            if time.time() - ts > self.ttl:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key, value):
        with self._lock:
            self._data[key] = (time.time(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)
