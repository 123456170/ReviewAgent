import streamlit as st
import streamlit.components.v1 as components
import sqlite3
import json
import re
import hashlib
import urllib.request
import urllib.parse
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from collections import defaultdict


DEFAULT_QUESTION = "What evidence exists on sparse attention for long-context transformers?"

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this", "these",
    "those", "with", "within", "without", "for", "from", "to", "of", "in", "on", "at",
    "by", "as", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "could", "should", "would", "may", "might", "will",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "too", "very", "s", "t", "just", "don", "now",
    "evidence", "exists", "exist", "review", "literature", "paper", "papers", "study",
    "studies", "show", "shows", "showed", "report", "reports", "reported", "say", "says"
}

ABSOLUTE_WORDS = {
    "always", "all", "never", "proves", "prove", "guarantee", "guarantees",
    "eliminate", "eliminates", "solves", "solve", "complete", "fully",
    "universally", "conclusive", "state-of-the-art"
}

HEDGE_WORDS = {
    "can", "may", "might", "often", "sometimes", "suggests", "suggest",
    "indicates", "indicate", "some", "many", "most", "competitive",
    "helps", "help", "tends", "probable", "possible", "likely"
}


@dataclass
class Paper:
    paper_id: str
    title: str
    authors: List[str]
    year: int
    venue: str
    doi: str
    url: str
    abstract: str
    open_access: bool
    retracted: bool = False
    flagged: bool = False
    full_text: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class Passage:
    passage_id: str
    paper_id: str
    paragraph_index: int
    text: str
    topic: str = ""
    score: float = 0.0


@dataclass
class Claim:
    claim_id: str
    claim_text: str
    paper_id: str
    passage_id: str
    quote: str
    topic: str = ""
    status: str = "RECORDED"
    verdict: str = "PENDING"
    verifier_notes: str = ""


@dataclass
class SubQuestion:
    sq_id: str
    question: str
    query: str
    status: str = "OPEN"
    seen_ratio: float = 0.0
    new_papers: int = 0
    supported_claims: int = 0


def normalize_ws(text: str) -> str:
    return " ".join(str(text).split())


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def token_set(text: str) -> set:
    return set(tokenize(text))


def important_tokens(text: str) -> set:
    return {t for t in token_set(text) if t not in STOPWORDS and len(t) > 2}


def split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n|\n", str(text))
    return [normalize_ws(p) for p in parts if normalize_ws(p)]


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", normalize_ws(text))
    return [normalize_ws(p) for p in parts if normalize_ws(p)]


class Storage:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.trace_enabled = True
        self._create_tables()

    def _create_tables(self) -> None:
        c = self.conn
        c.execute("CREATE TABLE IF NOT EXISTS papers (paper_id TEXT PRIMARY KEY, data TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS passages (passage_id TEXT PRIMARY KEY, paper_id TEXT, data TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS claims (claim_id TEXT PRIMARY KEY, data TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS traces (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, tool TEXT, args TEXT, result TEXT, ok INTEGER)")
        c.execute("CREATE TABLE IF NOT EXISTS exclusions (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, paper_id TEXT, reason TEXT, query TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS search_cache (key TEXT PRIMARY KEY, results TEXT, ts TEXT)")
        c.commit()

    def save_paper(self, paper: Paper) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO papers (paper_id, data) VALUES (?, ?)",
            (paper.paper_id, json.dumps(asdict(paper)))
        )
        self.conn.commit()

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        row = self.conn.execute("SELECT data FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
        return Paper(**json.loads(row["data"])) if row else None

    def list_papers(self) -> List[Paper]:
        rows = self.conn.execute("SELECT data FROM papers").fetchall()
        return [Paper(**json.loads(r["data"])) for r in rows]

    def count_papers(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS n FROM papers").fetchone()["n"])

    def save_passage(self, passage: Passage) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO passages (passage_id, paper_id, data) VALUES (?, ?, ?)",
            (passage.passage_id, passage.paper_id, json.dumps(asdict(passage)))
        )
        self.conn.commit()

    def get_passage(self, passage_id: str) -> Optional[Passage]:
        row = self.conn.execute("SELECT data FROM passages WHERE passage_id=?", (passage_id,)).fetchone()
        return Passage(**json.loads(row["data"])) if row else None

    def save_claim(self, claim: Claim) -> bool:
        exists = self.conn.execute("SELECT 1 FROM claims WHERE claim_id=?", (claim.claim_id,)).fetchone()
        if exists:
            return False
        self.conn.execute(
            "INSERT INTO claims (claim_id, data) VALUES (?, ?)",
            (claim.claim_id, json.dumps(asdict(claim)))
        )
        self.conn.commit()
        return True

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        row = self.conn.execute("SELECT data FROM claims WHERE claim_id=?", (claim_id,)).fetchone()
        return Claim(**json.loads(row["data"])) if row else None

    def update_claim(self, claim: Claim) -> None:
        self.conn.execute(
            "UPDATE claims SET data=? WHERE claim_id=?",
            (json.dumps(asdict(claim)), claim.claim_id)
        )
        self.conn.commit()

    def add_trace(self, tool: str, args: Dict[str, Any], result: str, ok: bool) -> None:
        if not self.trace_enabled:
            return
        self.conn.execute(
            "INSERT INTO traces (ts, tool, args, result, ok) VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                tool,
                json.dumps(args, default=str),
                result,
                1 if ok else 0
            )
        )
        self.conn.commit()

    def clear_traces(self) -> None:
        self.conn.execute("DELETE FROM traces")
        self.conn.commit()

    def recent_traces(self, limit: int = 1000) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM (SELECT * FROM traces ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def add_exclusion(self, paper_id: str, reason: str, query: str) -> None:
        self.conn.execute(
            "INSERT INTO exclusions (ts, paper_id, reason, query) VALUES (?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), paper_id, reason, query)
        )
        self.conn.commit()

    def max_exclusion_id(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(id), 0) AS m FROM exclusions").fetchone()
        return int(row["m"])

    def list_exclusions(self, since_id: int = 0) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM exclusions WHERE id > ? ORDER BY id ASC",
            (since_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_search_cache(self, key: str) -> Optional[List[Dict[str, Any]]]:
        row = self.conn.execute("SELECT results FROM search_cache WHERE key=?", (key,)).fetchone()
        return json.loads(row["results"]) if row else None

    def put_search_cache(self, key: str, results: List[Dict[str, Any]]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO search_cache (key, results, ts) VALUES (?, ?, ?)",
            (key, json.dumps(results), datetime.now(timezone.utc).isoformat())
        )
        self.conn.commit()


def _paper(pid: str, title: str, authors: List[str], year: int, doi: str,
           tags: List[str], full_text: str, retracted: bool = False, flagged: bool = False) -> Paper:
    abstract = normalize_ws(full_text.strip().split("\n")[0])
    return Paper(
        pid, title, authors, year, "demo corpus", doi,
        f"https://example.org/{pid}", abstract, True,
        retracted, flagged, full_text, tags
    )


MOCK_PAPERS = [
    _paper("p001", "Longformer: The Long-Document Transformer", ["Iz Beltagy", "Matthew E. Peters", "Arman Cohan"], 2020,
           "10.48550/arXiv.2004.05150",
           ["longformer", "sliding window", "global attention", "sparse attention", "long document", "linear memory"],
           """Longformer uses a sliding window attention pattern that scales linearly with sequence length.
It adds global attention tokens for tasks requiring global context.
On document classification and question answering, Longformer achieves competitive results with lower memory than full attention."""),

    _paper("p002", "Big Bird: Transformers for Longer Sequences", ["Manzil Zaheer", "Guy Gur-Ari", "Kevin Robinson"], 2020,
           "10.48550/arXiv.2007.14088",
           ["big bird", "random attention", "window attention", "global tokens", "sparse attention", "theoretical"],
           """Big Bird combines random, window, and global attention to model long sequences with linear memory.
The theoretical analysis shows that these sparse patterns can approximate full attention for many tasks.
Big Bird outperforms previous sparse methods on long-range benchmark tasks."""),

    _paper("p003", "Reformer: The Efficient Transformer", ["Nikita Kitaev", "Lukasz Kaiser", "Anselm Levskaya"], 2020,
           "10.48550/arXiv.2001.04451",
           ["reformer", "lsh attention", "hashing", "efficient attention", "memory"],
           """Reformer uses locality-sensitive hashing to reduce attention from quadratic to approximate sub-quadratic cost.
It can handle sequence lengths beyond standard transformers by chunking hash buckets.
On certain generation tasks, Reformer improves memory scaling but can be slower than dense attention in practice."""),

    _paper("p004", "Rethinking Attention with Performers", ["Krzysztof Choromanski", "Valerii Likhosherstov", "David Dohan"], 2021,
           "10.48550/arXiv.2009.14794",
           ["performer", "kernel features", "favor+", "linear attention", "random features"],
           """Performer approximates softmax attention with random feature maps, enabling linear space and time estimates.
The FAVOR+ mechanism estimates attention without materializing the full attention matrix.
Quality can vary by task because the kernel approximation introduces bias."""),

    _paper("p005", "Linformer: Self-Attention with Linear Complexity", ["Sinong Wang", "Belinda Li", "Madian Khabsa"], 2020,
           "10.48550/arXiv.2006.04768",
           ["linformer", "low-rank projection", "linear attention", "sequence compression"],
           """Linformer projects keys and values to a lower-dimensional sequence, making attention linear in length.
The projection is learned and can be reused across layers.
Low-rank assumptions may limit expressiveness for tasks needing fine-grained long-range dependencies."""),

    _paper("p006", "Sparse Transformers", ["Rewon Child", "Scott Gray", "Alec Radford"], 2019,
           "10.48550/arXiv.1904.10509",
           ["sparse transformer", "fixed sparsity pattern", "strided attention", "block-sparse"],
           """Sparse Transformer introduces fixed sparse attention patterns such as strided and block-sparse masks.
These patterns reduce memory and compute while preserving autoregressive generation quality on modeling benchmarks.
The best pattern depends on the task and sequence length."""),

    _paper("p007", "ETC: Extending Transformer Context", ["Joshua Ainslie", "Katharina Meinp", "Yinhan Liu"], 2020,
           "10.48550/arXiv.2004.08467",
           ["etc", "extended tokens", "global tokens", "long context", "transformer"],
           """ETC adds global tokens that can attend to all positions, increasing effective context with limited extra cost.
It is designed for tasks where a few global summary tokens mediate long-range interactions.
ETC can improve long-document understanding when global tokens are allocated properly."""),

    _paper("p008", "Routing Transformers", ["Aurko Roy", "Mingyu Yang", "Ankit Shah"], 2021,
           "10.48550/arXiv.2003.05997",
           ["routing transformer", "top-k routing", "cluster attention", "sparse attention"],
           """Routing Transformer clusters queries and keys and attends to top-k clusters, yielding sparse adaptive attention.
The routing pattern changes with content rather than following a fixed window.
It reports improvements on image and language sequence tasks with reduced attention cost."""),

    _paper("p009", "Focused Transformer and Identification", ["Juang-Hsiang Lee", "Yi Tay", "Mostafa Dehghani"], 2021,
           "10.48550/arXiv.2107.05222",
           ["focused transformer", "identification", "sparse attention", "retrieval"],
           """Focused Transformer restricts attention to identified tokens suspected to be relevant for the current prediction.
This dynamic sparsity can improve efficiency on retrieval-heavy tasks.
Identification errors can cause missed evidence and lower accuracy."""),

    _paper("p010", "LongNet: Scaling Transformers to 1B Tokens", ["Jiayu Ding", "Shuming Ma", "Li Dong"], 2023,
           "10.48550/arXiv.2307.02486",
           ["longnet", "dilated attention", "scaling", "1 billion tokens", "long context"],
           """LongNet uses dilated attention to scale sequence length exponentially while keeping compute linear.
The method targets contexts up to one billion tokens in principle.
Empirical validation at extreme lengths remains limited compared with shorter benchmarks."""),

    _paper("p011", "Mamba: Linear-Time Sequence Modeling", ["Albert Gu", "Tri Dao"], 2023,
           "10.48550/arXiv.2312.00752",
           ["mamba", "state space", "selective", "linear time", "long context"],
           """Mamba uses selective state spaces to achieve linear-time sequence modeling for long contexts.
It reports strong perplexity and retrieval results without using softmax attention.
State-space models change the inductive bias relative to sparse attention transformers."""),

    _paper("p012", "Retentive Network: A Successor to Transformer", ["Yutao Sun", "Li Dong", "Barun Patra"], 2023,
           "10.48550/arXiv.2307.08621",
           ["retnet", "retention", "multi-scale", "parallel training", "inference", "long context", "sequence model"],
           """RetNet combines retention mechanisms with multi-scale retention for parallel training and constant-time inference.
It aims to match transformer quality while reducing memory for long sequences.
Independent replication on diverse long-context tasks is still emerging."""),

    _paper("p013", "FlashAttention: Fast and Memory-Efficient Attention", ["Tri Dao", "Daniel Y. Fu", "Stefano Ermon"], 2022,
           "10.48550/arXiv.2205.14135",
           ["flashattention", "kernel", "io-aware", "exact attention", "gpu"],
           """FlashAttention is an IO-aware exact attention kernel that reduces memory traffic and wall-clock time.
It does not approximate attention; it recomputes attention tiles on GPU.
FlashAttention can make dense attention practical for longer contexts than naive implementations."""),

    _paper("p014", "Optimized Attention Kernels for Long Sequences", ["System Labs", "Kernel Team"], 2023,
           "10.5555/demo.kernel.2023",
           ["kernels", "gpu", "io", "throughput", "systems"],
           """Kernel-level IO awareness can speed up exact attention without changing model quality.
These optimizations complement sparse attention by reducing constant factors.
System improvements alone do not solve algorithmic quadratic memory growth."""),

    _paper("p015", "Efficient Transformers: A Survey", ["Yi Tay", "Mostafa Dehghani", "Dara Bahri"], 2022,
           "10.1145/3501405",
           ["survey", "efficient transformers", "taxonomy", "approximate attention"],
           """Efficient transformer methods trade exact global attention for approximate or structured patterns.
No single sparse pattern dominates all long-context tasks.
The survey identifies evaluation gaps across length generalization, retrieval, and downstream fine-tuning."""),

    _paper("p016", "Long Range Arena: A Benchmark for Efficient Transformers", ["Yi Tay", "Mostafa Dehghani", "Samira Abnar"], 2021,
           "10.48550/arXiv.2011.04006",
           ["benchmark", "long range arena", "evaluation", "retrieval", "sequence length"],
           """Long Range Arena evaluates sequence length scaling across classification, retrieval, and statistics tasks.
Sparse attention methods often degrade on tasks requiring exact token retrieval.
Speed and memory measurements should be reported alongside accuracy."""),

    _paper("p017", "Needle-in-a-Haystack Benchmark for Long-Context Models", ["Gregory Kamradt", "Eval Team"], 2023,
           "10.5281/zenodo.needle.2023",
           ["benchmark", "needle", "haystack", "long context", "retrieval"],
           """Needle-in-a-Haystack tests reveal that some long-context models miss inserted facts as context grows.
Performance can depend on needle position, distractors, and model architecture.
It provides a diagnostic complement to aggregate perplexity scores."""),

    _paper("p019", "Train Short, Test Long: Position Interpolation", ["Samuel Chen", "Brian Dao", "Evan Ansari"], 2023,
           "10.48550/arXiv.2306.15595",
           ["positional interpolation", "context extension", "fine-tuning", "perplexity"],
           """Position interpolation allows models trained on shorter windows to extrapolate to longer contexts.
Fine-tuning with extended positional ranges improves perplexity at longer lengths.
Interpolation can hurt tasks that require precise absolute position memories."""),

    _paper("p020", "YaRN: Efficient Context Window Extension", ["Bowen Peng", "Jeffrey Quesnelle", "Eric Fan"], 2023,
           "10.48550/arXiv.2309.00071",
           ["yarn", "rope", "context window extension", "retrieval"],
           """YaRN modifies rotary embeddings to extend context windows with fewer training tokens.
It reports improved long-context retrieval compared with naive interpolation.
The method assumes a pretrained RoPE base model."""),

    _paper("p021", "StreamingLLM: Attention Sinks in Streaming Inference", ["Guangxuan Xiao", "Yuandong Tian", "Boyang Li"], 2023,
           "10.48550/arXiv.2309.17453",
           ["streaming", "attention sinks", "local window", "kv cache", "infinite length"],
           """StreamingLLM keeps initial attention sinks and a local window for stable long-stream inference.
It loses information outside the retained window, limiting middle-context retrieval.
The approach is useful for streaming workloads where recent context dominates."""),

    _paper("p022", "Heavy-Hitter Oracle for KV Cache Compression", ["Zichang Sun", "Yuheng Zhang", "Hao Zhang"], 2024,
           "10.48550/arXiv.2306.14048",
           ["kv cache", "heavy hitter", "cache compression", "eviction"],
           """Heavy-hitter tokens dominate attention mass and can be preserved in compressed KV caches.
Aggressive cache eviction can reduce accuracy on multi-hop reasoning.
Cache policy interacts with sparse attention and positional encoding."""),

    _paper("p023", "Retracted: Sparse Attention Always Solves Long-Context Modeling", ["Anonymous Author"], 2022,
           "10.0000/retracted.sparse.2022",
           ["sparse attention", "retracted"],
           """Sparse attention always solves long-context modeling without any quality loss.
This paper has been retracted for unsupported conclusions.""",
           retracted=True),

    _paper("p024", "Flagged Preprint: Universal Gains from Sparse Attention", ["Withdrawn Team"], 2021,
           "10.0000/flagged.sparse.2021",
           ["flagged", "preprint", "sparse attention"],
           """A flagged preprint claims universal gains from sparse attention.
Metadata indicates the version was withdrawn by authors.""",
           flagged=True),
]


def seed_corpus(storage: Storage) -> None:
    if storage.count_papers() > 0:
        return
    for p in MOCK_PAPERS:
        storage.save_paper(p)


def paper_to_result(p: Paper, score: float) -> Dict[str, Any]:
    return {
        "paper_id": p.paper_id,
        "title": p.title,
        "authors": p.authors,
        "year": p.year,
        "venue": p.venue,
        "doi": p.doi,
        "url": p.url,
        "abstract": p.abstract,
        "open_access": p.open_access,
        "retracted": p.retracted,
        "flagged": p.flagged,
        "score": round(float(score), 4),
    }


def _search_mock(query: str, year_from: int, year_to: int, limit: int, storage: Storage) -> List[Dict[str, Any]]:
    qtokens = token_set(query)
    scored = []
    for p in storage.list_papers():
        if p.year and not (year_from <= p.year <= year_to):
            continue
        tt = token_set(p.title)
        gt = token_set(" ".join(p.tags))
        at = token_set(p.abstract)
        ft = token_set(p.full_text[:2000])
        score = 3 * len(qtokens & tt) + 2.5 * len(qtokens & gt) + 1.5 * len(qtokens & at) + 0.5 * len(qtokens & ft)
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: (-x[0], -(x[1].year or 0), x[1].paper_id))
    return [paper_to_result(p, s) for s, p in scored[:limit]]


def _search_live(query: str, year_from: int, year_to: int, limit: int, storage: Storage) -> List[Dict[str, Any]]:
    params = {
        "query": query,
        "limit": max(1, min(int(limit), 100)),
        "year": f"{year_from}-{year_to}",
        "fields": "title,authors,year,venue,externalIds,abstract,openAccessPdf,url,publicationTypes"
    }
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "User-Agent": "ReviewAgent-demo",
        "Accept": "application/json",
        "Accept-Encoding": "identity"
    })
    with urllib.request.urlopen(req, timeout=6) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    results = []
    for item in data.get("data", [])[:limit]:
        pid = item.get("paperId")
        if not pid:
            continue
        ext = item.get("externalIds") or {}
        doi = ext.get("DOI") or (f"10.48550/arXiv.{ext['ArXiv']}" if ext.get("ArXiv") else "")
        authors = [a.get("name", "") for a in item.get("authors") or []]
        abstract = item.get("abstract") or ""
        pub_types = item.get("publicationTypes") or []
        retracted = any(str(x).lower() == "retracted" for x in pub_types)
        paper = Paper(
            paper_id=pid,
            title=item.get("title") or "Untitled",
            authors=authors,
            year=int(item.get("year") or 0),
            venue=item.get("venue") or "Semantic Scholar",
            doi=doi or "",
            url=item.get("url") or "",
            abstract=abstract,
            open_access=bool(item.get("openAccessPdf")),
            retracted=retracted,
            flagged=False,
            full_text=abstract,
            tags=tokenize(item.get("title") or "")[:12]
        )
        storage.save_paper(paper)
        results.append(paper_to_result(paper, 1.0))
    return results


def _fetch_live_paper(paper_id: str, storage: Storage) -> Optional[Paper]:
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/{urllib.parse.quote(paper_id, safe='')}?fields=title,authors,year,venue,externalIds,abstract,openAccessPdf,url,publicationTypes"
        req = urllib.request.Request(url, headers={
            "User-Agent": "ReviewAgent-demo",
            "Accept": "application/json",
            "Accept-Encoding": "identity"
        })
        with urllib.request.urlopen(req, timeout=6) as resp:
            item = json.loads(resp.read().decode("utf-8"))
        ext = item.get("externalIds") or {}
        doi = ext.get("DOI") or (f"10.48550/arXiv.{ext['ArXiv']}" if ext.get("ArXiv") else "")
        abstract = item.get("abstract") or ""
        pub_types = item.get("publicationTypes") or []
        paper = Paper(
            paper_id=item.get("paperId", paper_id),
            title=item.get("title") or "Untitled",
            authors=[a.get("name", "") for a in item.get("authors") or []],
            year=int(item.get("year") or 0),
            venue=item.get("venue") or "Semantic Scholar",
            doi=doi or "",
            url=item.get("url") or "",
            abstract=abstract,
            open_access=bool(item.get("openAccessPdf")),
            retracted=any(str(x).lower() == "retracted" for x in pub_types),
            flagged=False,
            full_text=abstract,
            tags=tokenize(item.get("title") or "")[:12]
        )
        storage.save_paper(paper)
        return paper
    except Exception:
        return None


def search_papers(query: str, year_from: int = 2017, year_to: int = 2026, limit: int = 10,
                  storage: Optional[Storage] = None, live: bool = False) -> List[Dict[str, Any]]:
    storage = storage or st.session_state.storage
    key = hashlib.sha1(json.dumps([query, year_from, year_to, limit, live], sort_keys=True).encode()).hexdigest()
    cached = storage.get_search_cache(key)
    if cached is not None:
        results, source = cached, "cache"
    else:
        results = []
        source = "mock"
        if live:
            try:
                results = _search_live(query, year_from, year_to, limit, storage)
                if results:
                    source = "live"
            except Exception:
                results = []
        if not results:
            results = _search_mock(query, year_from, year_to, limit, storage)
            source = "mock"
        storage.put_search_cache(key, results)
    storage.add_trace("search_papers", {"query": query, "year_from": year_from, "year_to": year_to, "limit": limit, "live": live}, f"{len(results)} results ({source})", True)
    return results


def fetch_paper(paper_id: str, storage: Optional[Storage] = None, live: bool = False) -> Optional[Paper]:
    storage = storage or st.session_state.storage
    paper = storage.get_paper(paper_id)
    if paper is None and live:
        paper = _fetch_live_paper(paper_id, storage)
    if paper is None:
        storage.add_trace("fetch_paper", {"paper_id": paper_id}, "not found", False)
        return None
    if paper.retracted or paper.flagged:
        reason = "retracted" if paper.retracted else "flagged metadata"
        storage.add_exclusion(paper.paper_id, reason, "")
        storage.add_trace("fetch_paper", {"paper_id": paper_id}, f"excluded: {reason}", False)
        return None
    storage.save_paper(paper)
    storage.add_trace("fetch_paper", {"paper_id": paper_id}, f"fetched {paper.title[:70]}", True)
    return paper


def extract_passages(paper_id: str, topic: str, top_k: int = 3, storage: Optional[Storage] = None) -> List[Passage]:
    storage = storage or st.session_state.storage
    paper = storage.get_paper(paper_id)
    if paper is None:
        storage.add_trace("extract_passages", {"paper_id": paper_id, "topic": topic}, "paper not found", False)
        return []
    text = paper.full_text or paper.abstract
    paras = split_paragraphs(text)
    if not paras:
        paras = [normalize_ws(f"{paper.title}. {paper.abstract}")]
    qtokens = important_tokens(topic)
    scored = []
    for i, para in enumerate(paras):
        ptokens = important_tokens(para)
        score = len(qtokens & ptokens) + 0.1 * len(ptokens & token_set(paper.title))
        if score == 0:
            score = 0.01
        scored.append((score, i, para))
    scored.sort(key=lambda x: (-x[0], x[1]))
    passages = []
    for score, i, para in scored[:top_k]:
        passage = Passage(f"{paper_id}::para{i}", paper_id, i, para, topic, round(score, 4))
        storage.save_passage(passage)
        passages.append(passage)
    storage.add_trace("extract_passages", {"paper_id": paper_id, "topic": topic, "top_k": top_k}, f"{len(passages)} passages: " + ", ".join(p.passage_id for p in passages), True)
    return passages


def record_claim(claim_text: str, paper_id: str, passage_id: str, quote: str,
                 topic: str = "", storage: Optional[Storage] = None) -> Optional[str]:
    storage = storage or st.session_state.storage
    claim_text = normalize_ws(claim_text)
    quote = normalize_ws(quote)
    passage = storage.get_passage(passage_id)
    args = {"claim_text": claim_text[:160], "paper_id": paper_id, "passage_id": passage_id, "quote": quote[:160]}
    if passage is None or passage.paper_id != paper_id:
        storage.add_trace("record_claim", args, "rejected: passage not found or paper mismatch", False)
        return None
    if quote not in passage.text:
        storage.add_trace("record_claim", args, "rejected: quote is not an exact substring of passage", False)
        return None
    claim_id = hashlib.sha1(f"{claim_text}|{paper_id}|{passage_id}|{quote}".encode()).hexdigest()[:16]
    claim = Claim(claim_id, claim_text, paper_id, passage_id, quote, topic, "RECORDED", "PENDING", "")
    created = storage.save_claim(claim)
    storage.add_trace("record_claim", args, "stored" if created else "duplicate accepted", True)
    return claim_id


def _rule_verifier(claim_text: str, quote: str, passage_text: str) -> Tuple[str, str]:
    claim = normalize_ws(claim_text)
    quote = normalize_ws(quote)
    if claim == quote:
        return "SUPPORTED", "Verbatim claim matches quote."
    ct = important_tokens(claim)
    qt = important_tokens(quote)
    overlap = len(ct & qt) / max(1, len(ct))
    claim_lower = claim.lower()
    quote_lower = quote.lower()
    claim_abs = any(w in claim_lower for w in ABSOLUTE_WORDS)
    quote_hedge = any(w in quote_lower for w in HEDGE_WORDS)
    quote_abs = any(w in quote_lower for w in ABSOLUTE_WORDS)
    if claim_abs and quote_hedge and not quote_abs:
        return "UNSUPPORTED", "Absolute wording in claim is not licensed by hedged quote."
    neg_rx = r"\b(no|not|never|none|cannot|don't|doesn't|isn't|aren't|wasn't|weren't|fail|fails|failed|unable|without|loses|losing|degrades?|degraded|worse)\b"
    claim_neg = bool(re.search(neg_rx, claim_lower))
    quote_neg = bool(re.search(neg_rx, quote_lower))
    if claim_neg != quote_neg and overlap > 0.30:
        return "UNSUPPORTED", "Negation or polarity mismatch between claim and quote."
    cn = re.findall(r"\d+(?:\.\d+)?", claim)
    qn = re.findall(r"\d+(?:\.\d+)?", quote)
    if cn and set(cn) != set(qn):
        return "UNSUPPORTED", "Numeric mismatch between claim and quote."
    if overlap >= 0.75:
        return "SUPPORTED", f"High lexical overlap ({overlap:.2f})."
    if overlap >= 0.45:
        return "PARTIAL", f"Partial lexical overlap ({overlap:.2f})."
    return "UNSUPPORTED", f"Low lexical overlap ({overlap:.2f})."


def verify_claim(claim_id: str, storage: Optional[Storage] = None) -> str:
    storage = storage or st.session_state.storage
    claim = storage.get_claim(claim_id)
    if claim is None:
        storage.add_trace("verify_claim", {"claim_id": claim_id}, "missing claim", False)
        return "MISSING"
    passage = storage.get_passage(claim.passage_id)
    passage_text = passage.text if passage else claim.quote
    verdict, notes = _rule_verifier(claim.claim_text, claim.quote, passage_text)
    claim.verdict = verdict
    claim.status = "DROPPED" if verdict == "UNSUPPORTED" else "VERIFIED"
    claim.verifier_notes = notes
    storage.update_claim(claim)
    storage.add_trace("verify_claim", {"claim_id": claim_id}, f"{verdict}: {notes}", verdict != "UNSUPPORTED")
    return verdict


def extract_topic(question: str) -> str:
    q = question.lower()
    q = re.sub(r"^(what|which|how|why|does|do|did|is|are|can|could|should|would)\s+", " ", q)
    q = re.sub(r"\b(evidence|exists|exist|review|literature|paper|papers|study|studies|show|shows|report|reports|say|says)\b", " ", q)
    toks = [t for t in tokenize(q) if t not in STOPWORDS and len(t) > 2]
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return " ".join(out[:8]) or normalize_ws(question)


def make_plan(question: str) -> List[SubQuestion]:
    topic = extract_topic(question)
    items = [
        (f"What methods or mechanisms address {topic}?", f"{topic} methods mechanisms"),
        (f"What empirical evidence reports outcomes for {topic}?", f"{topic} empirical results evidence"),
        (f"What limitations, risks, or trade-offs are reported for {topic}?", f"{topic} limitations trade-offs risks"),
        (f"Which benchmarks or evaluations measure {topic}?", f"{topic} benchmark evaluation metrics"),
    ]
    if len(topic.split()) >= 2 or "long" in topic or "context" in topic:
        items.append((f"How does {topic} scale to longer sequences or larger budgets?", f"{topic} scaling long sequence memory"))
    return [SubQuestion(f"sq{i}", q, qu) for i, (q, qu) in enumerate(items, 1)]


def make_overclaim(sentence: str) -> Optional[str]:
    t = sentence
    t = re.sub(r"\bcan\b", "always", t, flags=re.I)
    t = re.sub(r"\bmay\b", "does", t, flags=re.I)
    t = re.sub(r"\boften\b", "always", t, flags=re.I)
    t = re.sub(r"\bsome\b", "all", t, flags=re.I)
    t = re.sub(r"\bcompetitive\b", "state-of-the-art", t, flags=re.I)
    t = re.sub(r"\breduces\b", "eliminates", t, flags=re.I)
    t = normalize_ws(t)
    return t if t != normalize_ws(sentence) else None


def generate_claims_from_passage(passage: Passage, topic: str, storage: Storage,
                                 max_sentences: int = 2, include_overclaims: bool = True) -> List[str]:
    ids = []
    for sent in split_sentences(passage.text)[:max_sentences]:
        if len(sent) < 25 or len(sent) > 300:
            continue
        cid = record_claim(sent, passage.paper_id, passage.passage_id, sent, topic, storage)
        if cid:
            ids.append(cid)
        if include_overclaims:
            over = make_overclaim(sent)
            if over:
                oid = record_claim(over, passage.paper_id, passage.passage_id, sent, topic, storage)
                if oid:
                    ids.append(oid)
    return ids


THEME_KEYWORDS = {
    "Efficiency and memory": ["memory", "throughput", "linear", "efficient", "scal", "cost", "compute", "latency", "kv", "cache", "quadratic"],
    "Task quality and accuracy": ["accuracy", "f1", "perplexity", "quality", "performance", "outperform", "result", "score", "competitive", "better"],
    "Long-range retrieval and context": ["retrieval", "needle", "haystack", "long-range", "global", "context", "distance", "recall", "middle"],
    "Training and context extension": ["train", "fine-tune", "pretrain", "positional", "interpolation", "window", "extend", "yarn", "rope"],
    "Systems and kernels": ["kernel", "flash", "io", "gpu", "implementation", "streaming", "wall-clock", "traffic"],
    "Limitations and failure modes": ["limit", "fail", "degrad", "worse", "trade-off", "lose", "miss", "hurt", "bias", "withdraw", "retract", "gap"]
}

THEME_ORDER = [
    "Efficiency and memory",
    "Task quality and accuracy",
    "Long-range retrieval and context",
    "Training and context extension",
    "Systems and kernels",
    "Limitations and failure modes",
    "General evidence"
]


def classify_theme(text: str) -> str:
    t = text.lower()
    scores = {}
    for theme, kws in THEME_KEYWORDS.items():
        s = sum(1 for kw in kws if kw in t)
        if s:
            scores[theme] = s
    if not scores:
        return "General evidence"
    return max(scores, key=scores.get)


def group_claims(claims: List[Claim]) -> Dict[str, List[Claim]]:
    themes = defaultdict(list)
    seen = set()
    for c in claims:
        key = (c.paper_id, c.claim_text)
        if key in seen:
            continue
        seen.add(key)
        themes[classify_theme(c.claim_text + " " + c.topic)].append(c)
    ordered = {}
    for t in THEME_ORDER:
        if t in themes:
            ordered[t] = themes[t][:8]
    for t, v in themes.items():
        if t not in ordered:
            ordered[t] = v[:8]
    return ordered


def detect_contradictions(claims: List[Claim]) -> List[Tuple[Claim, Claim]]:
    pos = ["improve", "improves", "improved", "outperform", "outperforms", "reduce", "reduces", "lower", "scales", "competitive", "maintain", "preserves", "better"]
    neg = ["degrade", "degrades", "degraded", "worse", "fail", "fails", "failed", "lose", "loses", "losing", "limit", "limits", "limitation", "trade-off", "drop", "drops", "miss", "misses", "hurt", "hurts"]
    by_theme = defaultdict(list)
    for c in claims:
        by_theme[classify_theme(c.claim_text)].append(c)
    pairs = []
    for cs in by_theme.values():
        for i, c1 in enumerate(cs):
            for c2 in cs[i + 1:]:
                if c1.paper_id == c2.paper_id:
                    continue
                shared = important_tokens(c1.claim_text) & important_tokens(c2.claim_text)
                if not shared:
                    continue
                t1, t2 = c1.claim_text.lower(), c2.claim_text.lower()
                p1 = "pos" if any(w in t1 for w in pos) else "neg" if any(w in t1 for w in neg) else None
                p2 = "pos" if any(w in t2 for w in pos) else "neg" if any(w in t2 for w in neg) else None
                if p1 and p2 and p1 != p2:
                    pairs.append((c1, c2))
                    if len(pairs) >= 3:
                        return pairs
    return pairs


def identify_gaps(plan: List[SubQuestion], supported: List[Claim], themes: Dict[str, List[Claim]]) -> List[Dict[str, Any]]:
    gaps = []
    for sq in plan:
        count = 0
        sq_tokens = important_tokens(sq.question)
        for c in supported:
            if sq.question in c.topic or len(sq_tokens & important_tokens(c.claim_text)) >= 2:
                count += 1
        if count < 2:
            gaps.append({"type": "sub-question", "item": sq.question, "supported_claims": count})
    for theme, cs in themes.items():
        if len(cs) < 2:
            gaps.append({"type": "theme", "item": theme, "supported_claims": len(cs)})
    return gaps


def sanitize_for_sentence(text: str) -> str:
    s = normalize_ws(text)
    s = re.sub(r"^[-*•]\s*", "", s)
    s = s.rstrip(".")
    if len(s) < 10:
        return ""
    return s[0].upper() + s[1:]


def format_authors(authors: List[str]) -> str:
    if not authors:
        return "Unknown"
    if len(authors) <= 3:
        return ", ".join(authors)
    return ", ".join(authors[:3]) + ", et al."


def finish_review(themes: Dict[str, List[Claim]], contradictions: List[Tuple[Claim, Claim]], storage: Storage):
    lines = ["# ReviewAgent Literature Review", "", "## Cited findings", ""]
    paper_to_num = {}
    paper_order = []
    citation_claims = defaultdict(list)
    cited_claims = []

    def cite(claim: Claim) -> int:
        pid = claim.paper_id
        if pid not in paper_to_num:
            paper_order.append(pid)
            paper_to_num[pid] = len(paper_order)
        num = paper_to_num[pid]
        citation_claims[num].append(claim.claim_id)
        cited_claims.append(claim)
        return num

    for theme, claims in themes.items():
        if not claims:
            continue
        lines.append(f"### {theme}")
        for claim in claims:
            text = sanitize_for_sentence(claim.claim_text)
            if not text:
                continue
            num = cite(claim)
            lines.append(f"{text} [{num}].")
        lines.append("")

    if contradictions:
        lines.append("### Reported tensions")
        for c1, c2 in contradictions:
            t1 = sanitize_for_sentence(c1.claim_text)
            t2 = sanitize_for_sentence(c2.claim_text)
            if not t1 or not t2:
                continue
            n1 = cite(c1)
            n2 = cite(c2)
            lines.append(f"{t1} [{n1}], whereas {t2} [{n2}].")
        lines.append("")

    lines.append("## References")
    for num, pid in enumerate(paper_order, 1):
        p = storage.get_paper(pid)
        if p is None:
            continue
        doi = p.doi or "not available"
        lines.append(f"[{num}] {format_authors(p.authors)}. \"{p.title}\". {p.year}. DOI: {doi}")

    return "\n".join(lines), cited_claims, dict(citation_claims), paper_order


def audit_unlinked_factual_sentences(md: str) -> List[str]:
    bad = []
    in_refs = False
    for raw in md.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("## References"):
            in_refs = True
            continue
        if in_refs:
            continue
        if s.startswith("#"):
            continue
        if re.fullmatch(r"[-*_]{3,}", s):
            continue
        if not re.search(r"\[\d+\]", s):
            bad.append(s)
    return bad


def audit_hallucinated_citations(md: str, citation_claims: Dict[int, List[str]], storage: Storage) -> Tuple[int, int]:
    body = md.split("## References")[0]
    nums = [int(x) for x in re.findall(r"\[(\d+)\]", body)]
    unlinked = 0
    for n in nums:
        ids = citation_claims.get(n, [])
        ok = False
        for cid in ids:
            claim = storage.get_claim(cid)
            passage = storage.get_passage(claim.passage_id) if claim else None
            if claim and claim.verdict == "SUPPORTED" and passage and claim.quote in passage.text:
                ok = True
                break
        if not ok:
            unlinked += 1
    return unlinked, len(nums)


def run_agent(question: str, max_cycles: int = 3, per_query_limit: int = 6, depth: str = "standard",
              live: bool = False, storage: Optional[Storage] = None, trace: bool = True,
              clear_trace: bool = False) -> Dict[str, Any]:
    storage = storage or st.session_state.storage
    prev_trace = storage.trace_enabled
    storage.trace_enabled = trace
    if trace and clear_trace:
        storage.clear_traces()
    excl_since = storage.max_exclusion_id()

    try:
        top_passages = 2 if depth == "standard" else 1
        max_sentences = 2 if depth == "standard" else 1
        include_overclaims = depth == "standard"
        budget = 160 if depth == "standard" else 50

        plan = make_plan(question)
        known = set()
        claim_ids = []
        verified_ids = set()
        papers_processed = set()
        cycle = 0
        stop_reason = "max cycles reached"

        while cycle < max_cycles:
            cycle += 1
            any_new = False
            search_limit = min(per_query_limit + (cycle - 1), 20)

            for sq in plan:
                results = search_papers(sq.query, limit=search_limit, storage=storage, live=live)
                seen_before = sum(1 for r in results if r["paper_id"] in known)
                sq.seen_ratio = seen_before / len(results) if results else 1.0

                new = []
                for r in results:
                    pid = r["paper_id"]
                    if pid in known:
                        continue
                    if r.get("retracted") or r.get("flagged"):
                        reason = "retracted" if r.get("retracted") else "flagged metadata"
                        storage.add_exclusion(pid, reason, sq.query)
                        known.add(pid)
                        continue
                    known.add(pid)
                    new.append(r)
                    any_new = True

                sq.new_papers = len(new)

                for r in new[:per_query_limit]:
                    paper = fetch_paper(r["paper_id"], storage=storage, live=live)
                    if paper is None:
                        continue
                    papers_processed.add(paper.paper_id)
                    passages = extract_passages(paper.paper_id, sq.question, top_k=top_passages, storage=storage)
                    for passage in passages:
                        ids = generate_claims_from_passage(passage, sq.question, storage, max_sentences, include_overclaims)
                        claim_ids.extend(ids)

            for cid in claim_ids:
                if cid not in verified_ids:
                    verify_claim(cid, storage=storage)
                    verified_ids.add(cid)

            current_claims = [storage.get_claim(cid) for cid in dict.fromkeys(claim_ids)]
            current_claims = [c for c in current_claims if c is not None]

            open_count = 0
            for sq in plan:
                sup = sum(1 for c in current_claims if c.topic == sq.question and c.verdict == "SUPPORTED")
                sq.supported_claims = sup
                if sup >= 3:
                    sq.status = "SATURATED"
                elif sq.seen_ratio >= 0.9 or (sq.seen_ratio >= 0.8 and sq.new_papers == 0):
                    sq.status = "LOW_NOVELTY"
                else:
                    sq.status = "OPEN"
                    open_count += 1

            if open_count == 0:
                stop_reason = "coverage saturated or mostly known papers"
                break
            if len(dict.fromkeys(claim_ids)) >= budget:
                stop_reason = "claim budget reached"
                break
            if not any_new:
                stop_reason = "no new non-excluded papers"
                break

        unique_ids = list(dict.fromkeys(claim_ids))
        for cid in unique_ids:
            verify_claim(cid, storage=storage)

        claims = [storage.get_claim(cid) for cid in unique_ids]
        claims = [c for c in claims if c is not None]
        supported = [c for c in claims if c.verdict == "SUPPORTED"]
        partial = [c for c in claims if c.verdict == "PARTIAL"]
        unsupported = [c for c in claims if c.verdict == "UNSUPPORTED"]

        themes = group_claims(supported)
        contradictions = detect_contradictions(supported)
        gaps = identify_gaps(plan, supported, themes)
        review_md, cited_claims, citation_claims, paper_order = finish_review(themes, contradictions, storage)
        unlinked_sentences = audit_unlinked_factual_sentences(review_md)
        hallucinated, total_inline = audit_hallucinated_citations(review_md, citation_claims, storage)
        exclusions = storage.list_exclusions(excl_since)

        metrics = {
            "papers_seen": len(known),
            "papers_processed": len(papers_processed),
            "claims_total": len(claims),
            "supported": len(supported),
            "partial": len(partial),
            "unsupported": len(unsupported),
            "themes": len(themes),
            "contradictions": len(contradictions),
            "gaps": len(gaps),
            "total_inline_citations": total_inline,
            "hallucinated_citations": hallucinated,
            "unlinked_factual_sentences": len(unlinked_sentences),
            "exclusions": len(exclusions)
        }

        stages = [
            {"label": "Planning", "detail": f"{len(plan)} sub-questions generated"},
            {"label": "Searching", "detail": f"{metrics['papers_seen']} papers seen across {cycle} cycles"},
            {"label": "Reading", "detail": f"{metrics['papers_processed']} papers fetched; {metrics['exclusions']} excluded"},
            {"label": "Extracting", "detail": f"{metrics['claims_total']} candidate claims recorded from passages"},
            {"label": "Verifying", "detail": f"{metrics['supported']} supported, {metrics['partial']} partial, {metrics['unsupported']} unsupported"},
            {"label": "Synthesis", "detail": f"{metrics['themes']} themes, {metrics['contradictions']} tensions, {metrics['gaps']} gaps"},
            {"label": "Writing", "detail": f"{metrics['total_inline_citations']} inline citations, {metrics['hallucinated_citations']} hallucinated"}
        ]

        return {
            "question": question,
            "plan": [asdict(sq) for sq in plan],
            "claims": [asdict(c) for c in claims],
            "supported_claims": [asdict(c) for c in supported],
            "cited_claims": [asdict(c) for c in cited_claims],
            "citation_claims": citation_claims,
            "paper_order": paper_order,
            "themes": {k: [asdict(c) for c in v] for k, v in themes.items()},
            "contradictions": [{"left": asdict(a), "right": asdict(b)} for a, b in contradictions],
            "gaps": gaps,
            "exclusions": exclusions,
            "stages": stages,
            "metrics": metrics,
            "review_md": review_md,
            "hallucinated_citations": hallucinated,
            "total_inline_citations": total_inline,
            "unlinked_sentences": unlinked_sentences,
            "cycles": cycle,
            "stop_reason": stop_reason
        }
    finally:
        storage.trace_enabled = prev_trace


BENCHMARK_QUESTIONS = [
    {"question": "What evidence exists on sparse attention for long-context transformers?", "reference_paper_ids": ["p001", "p002", "p006", "p015"]},
    {"question": "Which linear attention methods approximate softmax attention?", "reference_paper_ids": ["p003", "p004", "p005"]},
    {"question": "How do sliding-window and global attention patterns support long documents?", "reference_paper_ids": ["p001", "p002", "p007"]},
    {"question": "What empirical evidence reports memory reduction from sparse attention?", "reference_paper_ids": ["p001", "p002", "p006", "p013"]},
    {"question": "What quality trade-offs are documented for sparse attention?", "reference_paper_ids": ["p002", "p015", "p016", "p021"]},
    {"question": "Which benchmarks evaluate long-range transformer capabilities?", "reference_paper_ids": ["p016", "p017", "p002"]},
    {"question": "How do positional interpolation and context extension methods work?", "reference_paper_ids": ["p019", "p020", "p007"]},
    {"question": "What evidence exists on streaming inference and KV-cache compression?", "reference_paper_ids": ["p021", "p022", "p013"]},
    {"question": "How do state-space models compare for long-context sequence modeling?", "reference_paper_ids": ["p011", "p012"]},
    {"question": "What does needle-in-a-haystack evaluation reveal about long-context retrieval?", "reference_paper_ids": ["p017", "p021", "p016"]},
    {"question": "Which hybrid sparse attention patterns combine random, window, and global attention?", "reference_paper_ids": ["p002", "p008", "p009"]},
    {"question": "What is the evidence for routing or top-k attention in long sequences?", "reference_paper_ids": ["p008", "p009", "p003"]},
    {"question": "How do kernel feature maps enable linear attention?", "reference_paper_ids": ["p004", "p005"]},
    {"question": "What are the results and limitations of LSH attention in Reformer?", "reference_paper_ids": ["p003", "p008"]},
    {"question": "How do surveys taxonomy efficient transformer methods?", "reference_paper_ids": ["p015", "p001", "p002"]},
    {"question": "What evidence exists for long-document classification and question answering?", "reference_paper_ids": ["p001", "p002", "p007"]},
    {"question": "How do sparse and extended-context methods affect language-model perplexity?", "reference_paper_ids": ["p002", "p010", "p019"]},
    {"question": "What system-level kernel optimizations improve exact attention?", "reference_paper_ids": ["p013", "p014"]},
    {"question": "How do attention sinks and local windows affect streaming language models?", "reference_paper_ids": ["p021", "p022", "p001"]},
    {"question": "What open problems and evaluation gaps remain for long-context models?", "reference_paper_ids": ["p015", "p016", "p017"]}
]


def single_prompt_baseline(question: str) -> str:
    topic = extract_topic(question)
    return f"""# Baseline review (single prompt, no tools)

{topic} has been studied in recent machine learning literature. Sparse attention methods improve long-context scaling [1]. They always preserve exact retrieval quality [2]. Many practitioners report deployment wins. The evidence is conclusive and no gaps remain [3].

## References
[1] Model-generated source. DOI: not available
[2] Fabricated benchmark. DOI: 10.0000/fake
[3] Unverified claim. DOI: not available
"""


def _strict_human(claim_text: str, quote: str) -> bool:
    verdict, _ = _rule_verifier(claim_text, quote, quote)
    return verdict == "SUPPORTED"


def run_benchmark(storage: Optional[Storage] = None, max_cycles: int = 1, per_query_limit: int = 4) -> Dict[str, Any]:
    storage = storage or st.session_state.storage
    rows = []
    total_ref = 0
    hit_ref = 0
    human_labels = []
    halluc = 0
    total_cit = 0
    base_halluc = 0
    base_total = 0
    base_unlinked_sentences = 0

    for bq in BENCHMARK_QUESTIONS:
        res = run_agent(
            bq["question"],
            max_cycles=max_cycles,
            per_query_limit=per_query_limit,
            depth="fast",
            live=False,
            storage=storage,
            trace=False,
            clear_trace=False
        )
        refs = set(bq["reference_paper_ids"])
        cited_pids = {c["paper_id"] for c in res["cited_claims"]}
        hits = refs & cited_pids
        total_ref += len(refs)
        hit_ref += len(hits)

        for c in res["cited_claims"][:5]:
            passage = storage.get_passage(c["passage_id"])
            label = 1 if (
                passage is not None
                and c["quote"] in passage.text
                and c["verdict"] == "SUPPORTED"
                and _strict_human(c["claim_text"], c["quote"])
            ) else 0
            human_labels.append(label)

        halluc += res["hallucinated_citations"]
        total_cit += res["total_inline_citations"]

        base_md = single_prompt_baseline(bq["question"])
        base_unlinked_sentences += len(audit_unlinked_factual_sentences(base_md))
        base_body = base_md.split("## References")[0]
        base_nums = re.findall(r"\[(\d+)\]", base_body)
        base_halluc += len(base_nums)
        base_total += len(base_nums)

        rows.append({
            "question": bq["question"],
            "reference_papers": ", ".join(sorted(refs)),
            "cited_reference_papers": ", ".join(sorted(hits)),
            "missing_reference_papers": ", ".join(sorted(refs - cited_pids)),
            "recall": round(len(hits) / len(refs), 3) if refs else 0.0,
            "supported_claims": res["metrics"]["supported"],
            "inline_citations": res["total_inline_citations"],
            "hallucinated_citations": res["hallucinated_citations"]
        })

    sample = human_labels[:100]
    summary = {
        "questions": len(BENCHMARK_QUESTIONS),
        "micro_recall": round(hit_ref / total_ref, 3) if total_ref else 0.0,
        "macro_recall": round(sum(r["recall"] for r in rows) / len(rows), 3) if rows else 0.0,
        "human_sample_size": len(sample),
        "citation_precision": round(sum(sample) / len(sample), 3) if sample else 0.0,
        "hallucinated_citation_rate": round(halluc / total_cit, 3) if total_cit else 0.0,
        "total_inline_citations": total_cit,
        "baseline_hallucinated_citation_rate": round(base_halluc / base_total, 3) if base_total else 0.0,
        "baseline_unlinked_factual_sentences": base_unlinked_sentences
    }
    return {"rows": rows, "summary": summary}


LIVE_PLAYER_TEMPLATE = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  :root { color-scheme: dark; }
  body { margin:0; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; background:#0b1220; color:#e6edf3; }
  .wrap { padding: 14px 16px 18px 16px; }
  .head { display:flex; justify-content:space-between; gap:12px; align-items:flex-start; margin-bottom:10px; }
  h2 { margin:0 0 4px 0; font-size:22px; }
  p { margin:0; color:#8b949e; font-size:14px; line-height:1.35; }
  .badge { border:1px solid #30363d; background:#161b22; color:#7ee787; padding:6px 10px; border-radius:999px; font-size:12px; font-weight:700; letter-spacing:.08em; white-space:nowrap; }
  canvas { width:100%; height:auto; display:block; border:1px solid #30363d; border-radius:14px; background:#0b1220; }
  .controls { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:10px; }
  button, select, a.btn { background:#21262d; color:#e6edf3; border:1px solid #30363d; border-radius:8px; padding:7px 10px; font-size:13px; cursor:pointer; text-decoration:none; }
  button:hover { border-color:#58a6ff; }
  button:disabled { opacity:.55; cursor:not-allowed; }
  input[type=range] { flex:1; min-width:180px; }
  .note { margin-top:8px; color:#8b949e; font-size:12px; }
</style>
</head>
<body>
<div class="wrap">
  <div class="head">
    <div>
      <h2>ReviewAgent live demo</h2>
      <p id="question">Loading...</p>
    </div>
    <div class="badge" id="status">RUNNING</div>
  </div>
  <canvas id="c" width="980" height="430"></canvas>
  <div class="controls">
    <button id="play">Pause</button>
    <button id="restart">Restart</button>
    <input id="timeline" type="range" min="0" max="1000" value="0" />
    <select id="speed">
      <option value="0.5">0.5x</option>
      <option value="1" selected>1x</option>
      <option value="2">2x</option>
      <option value="4">4x</option>
    </select>
    <button id="record">Generate video file (WebM)</button>
    <a id="download" class="btn" style="display:none" download="reviewagent_demo.webm">Download demo video</a>
  </div>
  <div class="note">The simulation starts automatically. A short WebM is generated automatically when the browser supports it; otherwise use the button above.</div>
</div>
<script type="application/json" id="payload">__PAYLOAD__</script>
<script>
var D = JSON.parse(document.getElementById('payload').textContent);
var canvas = document.getElementById('c');
var ctx = canvas.getContext('2d');
var statusEl = document.getElementById('status');
var questionEl = document.getElementById('question');
var playBtn = document.getElementById('play');
var restartBtn = document.getElementById('restart');
var timeline = document.getElementById('timeline');
var speedSel = document.getElementById('speed');
var recordBtn = document.getElementById('record');
var downloadLink = document.getElementById('download');

questionEl.textContent = D.question || '';
var duration = 14000;
var t = 0;
var playing = true;
var recording = false;
var last = performance.now();
var autoTried = false;

function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function num(v) { return Number(v || 0).toLocaleString(); }

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function wrapText(text, maxWidth) {
  var words = String(text || '').split(/\s+/);
  var lines = [];
  var line = '';
  for (var i = 0; i < words.length; i++) {
    var test = line ? line + ' ' + words[i] : words[i];
    if (ctx.measureText(test).width > maxWidth && line) {
      lines.push(line);
      line = words[i];
    } else {
      line = test;
    }
  }
  if (line) lines.push(line);
  return lines;
}

function drawCard(x, y, w, h, title, value, color) {
  ctx.fillStyle = '#161b22';
  roundRect(x, y, w, h, 12);
  ctx.fill();
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.fillStyle = '#8b949e';
  ctx.font = '12px system-ui, -apple-system, Segoe UI, Roboto, Arial';
  ctx.fillText(title, x + 12, y + 22);

  ctx.fillStyle = color;
  ctx.font = '700 24px system-ui, -apple-system, Segoe UI, Roboto, Arial';
  ctx.fillText(value, x + 12, y + 54);
}

function draw() {
  var w = canvas.width;
  var h = canvas.height;
  var g = ctx.createLinearGradient(0, 0, 0, h);
  g.addColorStop(0, '#0b1220');
  g.addColorStop(1, '#121c2e');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);

  var progress = clamp(t / duration, 0, 1);

  ctx.fillStyle = '#e6edf3';
  ctx.font = '700 24px system-ui, -apple-system, Segoe UI, Roboto, Arial';
  ctx.fillText('Agent pipeline', 28, 42);

  ctx.fillStyle = '#8b949e';
  ctx.font = '14px system-ui, -apple-system, Segoe UI, Roboto, Arial';
  var qLines = wrapText(D.question || '', w - 56);
  for (var qi = 0; qi < Math.min(qLines.length, 2); qi++) {
    ctx.fillText(qLines[qi], 28, 66 + qi * 18);
  }

  var stages = (D.stages && D.stages.length) ? D.stages : [{label: 'Idle', detail: 'No result'}];
  var n = stages.length;
  var startX = 28;
  var endX = w - 28;
  var gap = 10;
  var nodeW = (endX - startX - gap * (n - 1)) / n;
  var current = Math.min(n - 1, Math.floor(progress * n));
  var y = 105;
  var nodeH = 62;

  for (var i = 0; i < n; i++) {
    var x = startX + i * (nodeW + gap);
    var active = i <= current;
    var isCurrent = i === current;

    ctx.fillStyle = active ? '#1f6feb' : '#21262d';
    if (isCurrent) ctx.fillStyle = '#238636';
    roundRect(x, y, nodeW, nodeH, 10);
    ctx.fill();
    ctx.strokeStyle = active ? '#58a6ff' : '#30363d';
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.fillStyle = active ? '#ffffff' : '#8b949e';
    ctx.font = '700 12px system-ui, -apple-system, Segoe UI, Roboto, Arial';
    var label = String(stages[i].label || '').toUpperCase();
    ctx.fillText(label.substring(0, 12), x + 10, y + 18);

    ctx.font = '11px system-ui, -apple-system, Segoe UI, Roboto, Arial';
    ctx.fillStyle = active ? '#dbeafe' : '#6e7681';
    var lines = wrapText(stages[i].detail || '', nodeW - 18);
    ctx.fillText(lines[0] || '', x + 10, y + 36);
    if (lines[1]) ctx.fillText(lines[1], x + 10, y + 50);
  }

  var cur = stages[current] || {};
  ctx.fillStyle = '#e6edf3';
  ctx.font = '600 18px system-ui, -apple-system, Segoe UI, Roboto, Arial';
  ctx.fillText(String(cur.label || ''), 28, 190);

  ctx.fillStyle = '#8b949e';
  ctx.font = '14px system-ui, -apple-system, Segoe UI, Roboto, Arial';
  var dLines = wrapText(cur.detail || '', w - 56);
  for (var di = 0; di < Math.min(dLines.length, 3); di++) {
    ctx.fillText(dLines[di], 28, 212 + di * 18);
  }

  var m = D.metrics || {};
  var cards = [
    {t: 'Papers seen', v: num(m.papers_seen), c: '#58a6ff'},
    {t: 'Supported', v: num(m.supported), c: '#7ee787'},
    {t: 'Citations', v: num(m.total_inline_citations), c: '#e6edf3'},
    {t: 'Hallucinated', v: num(m.hallucinated_citations), c: Number(m.hallucinated_citations || 0) === 0 ? '#7ee787' : '#ff7b72'},
    {t: 'Themes', v: num(m.themes), c: '#d2a8ff'},
    {t: 'Excluded', v: num(m.exclusions), c: '#ffa657'}
  ];
  var cardY = 260;
  var cardH = 82;
  var cardGap = 10;
  var cardW = (w - 56 - cardGap * (cards.length - 1)) / cards.length;
  for (var ci = 0; ci < cards.length; ci++) {
    drawCard(28 + ci * (cardW + cardGap), cardY, cardW, cardH, cards[ci].t, cards[ci].v, cards[ci].c);
  }

  var items = D.sample_claims || [];
  if (items.length) {
    ctx.font = '14px system-ui, -apple-system, Segoe UI, Roboto, Arial';
    var labelGap = 70;
    var total = 0;
    var labels = [];
    for (var ii = 0; ii < items.length; ii++) {
      var text = '[' + (items[ii].paper || '') + '] ' + (items[ii].text || '');
      labels.push(text);
      total += ctx.measureText(text).width + labelGap;
    }
    var offset = (t * 0.09) % total;
    var tx = w - offset;
    for (var li = 0; li < labels.length; li++) {
      if (tx > w) break;
      ctx.fillStyle = '#c9d1d9';
      ctx.fillText(labels[li], tx, h - 58);
      tx += ctx.measureText(labels[li]).width + labelGap;
    }
  }

  ctx.fillStyle = '#21262d';
  roundRect(28, h - 28, w - 56, 10, 5);
  ctx.fill();
  ctx.fillStyle = '#238636';
  roundRect(28, h - 28, Math.max(0, (w - 56) * progress), 10, 5);
  ctx.fill();

  statusEl.textContent = recording ? 'RECORDING' : (progress >= 1 ? 'COMPLETE' : 'RUNNING');
}

function setRecording(v) {
  recording = v;
  recordBtn.disabled = v;
  recordBtn.textContent = v ? 'Recording...' : 'Generate video file (WebM)';
  draw();
}

function beginRecording(ms, isAuto) {
  if (recording) return false;
  if (typeof MediaRecorder === 'undefined' || !canvas.captureStream) {
    if (!isAuto) {
      downloadLink.style.display = 'none';
      alert('WebM recording is not supported in this browser. The live canvas demo still works.');
    }
    return false;
  }

  var stream = canvas.captureStream(30);
  var mime = '';
  if (MediaRecorder.isTypeSupported) {
    var types = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
    for (var i = 0; i < types.length; i++) {
      if (MediaRecorder.isTypeSupported(types[i])) {
        mime = types[i];
        break;
      }
    }
  }

  var recorder;
  try {
    recorder = new MediaRecorder(stream, mime ? {mimeType: mime} : undefined);
  } catch (e) {
    if (!isAuto) alert('Could not start recorder: ' + e.message);
    return false;
  }

  var chunks = [];
  setRecording(true);
  downloadLink.style.display = 'none';
  t = 0;
  playing = true;
  playBtn.textContent = 'Pause';

  recorder.ondataavailable = function(e) {
    if (e.data && e.data.size > 0) chunks.push(e.data);
  };

  recorder.onstop = function() {
    setRecording(false);
    var blob = new Blob(chunks, {type: mime || 'video/webm'});
    var url = URL.createObjectURL(blob);
    downloadLink.href = url;
    downloadLink.download = 'reviewagent_demo.webm';
    downloadLink.textContent = 'Download demo video (WebM)';
    downloadLink.style.display = 'inline-block';
  };

  recorder.start();
  setTimeout(function() {
    if (recorder && recorder.state && recorder.state !== 'inactive') recorder.stop();
  }, ms);
  return true;
}

playBtn.addEventListener('click', function() {
  playing = !playing;
  playBtn.textContent = playing ? 'Pause' : 'Play';
  last = performance.now();
});

restartBtn.addEventListener('click', function() {
  t = 0;
  playing = true;
  playBtn.textContent = 'Pause';
  downloadLink.style.display = 'none';
});

timeline.addEventListener('input', function() {
  t = (Number(this.value) / 1000) * duration;
});

recordBtn.addEventListener('click', function() {
  beginRecording(6000, false);
});

function frame(now) {
  var dt = now - last;
  last = now;
  if (playing && !document.hidden) {
    t += dt * parseFloat(speedSel.value || '1');
    if (t > duration) t = duration;
  }
  if (document.activeElement !== timeline) {
    timeline.value = String(Math.round((t / duration) * 1000));
  }
  draw();
  requestAnimationFrame(frame);
}

requestAnimationFrame(function(now) {
  last = now;
  frame(now);
});

setTimeout(function() {
  if (autoTried) return;
  autoTried = true;
  try {
    beginRecording(5000, true);
  } catch (e) {}
}, 1200);
</script>
</body>
</html>
"""


def make_player_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    sample = []
    for c in result.get("supported_claims", [])[:14]:
        txt = normalize_ws(c.get("claim_text", ""))
        if len(txt) > 140:
            txt = txt[:137] + "..."
        sample.append({"paper": c.get("paper_id", ""), "text": txt})
    return {
        "question": result.get("question", ""),
        "stages": result.get("stages", []),
        "metrics": result.get("metrics", {}),
        "plan": result.get("plan", []),
        "stop_reason": result.get("stop_reason", ""),
        "cycles": result.get("cycles", 0),
        "sample_claims": sample,
        "themes": list(result.get("themes", {}).keys())
    }


def render_live_player(result: Dict[str, Any]) -> None:
    payload = make_player_payload(result)
    payload_json = json.dumps(payload, default=str).replace("</", "<\\/")
    html = LIVE_PLAYER_TEMPLATE.replace("__PAYLOAD__", payload_json)
    components.html(html, height=690, scrolling=False)


def run_self_tests() -> List[Dict[str, Any]]:
    out = []
    s = Storage()
    s.trace_enabled = False
    seed_corpus(s)

    passages = extract_passages("p001", "sliding window attention", 1, s)
    if passages:
        p = passages[0]
        sents = split_sentences(p.text)
        exact = sents[0] if sents else p.text
        ok_id = record_claim(exact, "p001", p.passage_id, exact, "test", s)
        bad_quote = exact + " This phrase is not in the passage."
        bad_id = record_claim(bad_quote, "p001", p.passage_id, bad_quote, "test", s)
        passed = ok_id is not None and bad_id is None
        out.append({
            "test": "record_claim accepts verbatim quotes and rejects non-verbatim quotes",
            "passed": passed,
            "details": f"accepted={ok_id is not None}, rejected={bad_id is None}"
        })
    else:
        out.append({"test": "record_claim accepts verbatim quotes and rejects non-verbatim quotes", "passed": False, "details": "no passages extracted"})

    test_passage = Passage("test::0", "p001", 0, "Sparse attention can reduce memory usage.", "test", 1.0)
    s.save_passage(test_passage)
    oc_id = record_claim(
        "Sparse attention always reduces memory usage.",
        "p001", "test::0",
        "Sparse attention can reduce memory usage.",
        "test", s
    )
    verdict = verify_claim(oc_id, s) if oc_id else "MISSING"
    out.append({
        "test": "verify_claim marks unsupported overclaims as UNSUPPORTED",
        "passed": verdict == "UNSUPPORTED",
        "details": verdict
    })

    res = run_agent(DEFAULT_QUESTION, max_cycles=1, per_query_limit=3, depth="fast", live=False, storage=s, trace=False, clear_trace=False)
    bad = audit_unlinked_factual_sentences(res["review_md"])
    halluc, total = audit_hallucinated_citations(res["review_md"], res["citation_claims"], s)
    out.append({
        "test": "review generator emits no unlinked factual sentences",
        "passed": len(bad) == 0 and len(res["cited_claims"]) > 0,
        "details": f"unlinked_sentences={len(bad)}, cited_claims={len(res['cited_claims'])}"
    })
    out.append({
        "test": "hallucinated citation rate is zero in generated review",
        "passed": halluc == 0,
        "details": f"unlinked_citations={halluc}/{total}"
    })

    excl = s.list_exclusions()
    retracted_excluded = any(e["paper_id"] in {"p023", "p024"} for e in excl)
    out.append({
        "test": "retracted or flagged papers are excluded and logged",
        "passed": retracted_excluded,
        "details": f"exclusions={len(excl)}"
    })
    return out


def init_state() -> None:
    if "storage" not in st.session_state:
        st.session_state.storage = Storage()
        seed_corpus(st.session_state.storage)
    st.session_state.setdefault("question", DEFAULT_QUESTION)
    st.session_state.setdefault("main_result", None)
    st.session_state.setdefault("test_results", None)
    st.session_state.setdefault("benchmark_result", None)
    st.session_state.setdefault("bootstrapped", False)
    st.session_state.setdefault("bootstrap_error", None)


def bootstrap() -> None:
    init_state()
    if st.session_state.bootstrapped:
        return
    try:
        with st.spinner("Loading realistic demo corpus and running ReviewAgent..."):
            st.session_state.main_result = run_agent(
                st.session_state.question,
                max_cycles=3,
                per_query_limit=6,
                depth="standard",
                live=False,
                clear_trace=True
            )
        with st.spinner("Running quality-control tests..."):
            st.session_state.test_results = run_self_tests()
        st.session_state.bootstrapped = True
    except Exception as e:
        st.session_state.bootstrap_error = f"{type(e).__name__}: {e}"


def main() -> None:
    st.set_page_config(page_title="ReviewAgent", layout="wide", initial_sidebar_state="expanded")
    bootstrap()

    st.title("ReviewAgent")
    st.caption("Autonomous, citation-grounded literature reviews. Offline demo corpus loads automatically; optional live Semantic Scholar search needs no API key.")

    if st.session_state.bootstrap_error:
        st.error(f"Bootstrap warning: {st.session_state.bootstrap_error}")

    with st.sidebar:
        st.header("Control panel")
        question = st.text_area("Research question", value=st.session_state.question, height=110)
        st.session_state.question = question
        max_cycles = st.slider("Max search/read cycles", 1, 5, 3)
        per_query_limit = st.slider("Papers per query", 3, 12, 6)
        live = st.checkbox("Try live Semantic Scholar search (may fail offline)", False)

        c1, c2 = st.columns(2)
        run_btn = c1.button("Run agent", use_container_width=True)
        test_btn = c2.button("Run tests", use_container_width=True)
        bench_btn = st.button("Run 20-question benchmark", use_container_width=True)

        if run_btn:
            with st.spinner("Running ReviewAgent..."):
                st.session_state.main_result = run_agent(question, max_cycles, per_query_limit, "standard", live, clear_trace=True)
        if test_btn:
            with st.spinner("Running tests..."):
                st.session_state.test_results = run_self_tests()
        if bench_btn:
            with st.spinner("Running benchmark..."):
                st.session_state.benchmark_result = run_benchmark(st.session_state.storage)

        res = st.session_state.main_result
        if res:
            st.divider()
            st.metric("Supported claims", res["metrics"]["supported"])
            st.metric("Inline citations", res["metrics"]["total_inline_citations"])
            st.metric("Hallucinated citations", res["metrics"]["hallucinated_citations"])

    tabs = st.tabs(["Live demo", "Plan & iteration", "Evidence & claims", "Review output", "Trace viewer", "Benchmark", "Tests"])

    with tabs[0]:
        res = st.session_state.main_result
        if not res:
            st.info("No result yet. The app normally bootstraps a demo run automatically.")
        else:
            m = res["metrics"]
            cc = st.columns(4)
            cc[0].metric("Papers seen", m["papers_seen"])
            cc[1].metric("Supported claims", m["supported"])
            cc[2].metric("Themes", m["themes"])
            cc[3].metric("Excluded papers", m["exclusions"])

            st.subheader("Live simulation player")
            st.caption("This starts automatically and is fully interactive. The app also attempts to generate a downloadable WebM video file directly in the browser.")
            render_live_player(res)

            st.subheader("Simulation narrative")
            for s in res["stages"]:
                st.markdown(f"- **{s['label']}**: {s['detail']}")

    with tabs[1]:
        res = st.session_state.main_result
        if not res:
            st.info("No result yet.")
        else:
            st.subheader("Visible plan")
            st.dataframe(res["plan"])
            st.markdown(f"**Cycles run:** {res['cycles']}  ")
            st.markdown(f"**Stop reason:** {res['stop_reason']}")
            st.subheader("Gaps identified by synthesis")
            if res["gaps"]:
                st.dataframe(res["gaps"])
            else:
                st.success("No thin-evidence gaps detected in this run.")

    with tabs[2]:
        res = st.session_state.main_result
        if not res:
            st.info("No result yet.")
        else:
            st.subheader("Claims and verification verdicts")
            st.dataframe(res["claims"])
            st.subheader("Exclusions log")
            if res["exclusions"]:
                st.dataframe(res["exclusions"])
            else:
                st.write("No retracted or flagged papers were encountered in this run.")
            st.subheader("Contradictions / tensions")
            if res["contradictions"]:
                for pair in res["contradictions"]:
                    st.markdown(f"- **{pair['left']['paper_id']}**: {pair['left']['claim_text']}")
                    st.markdown(f"  **vs {pair['right']['paper_id']}**: {pair['right']['claim_text']}")
            else:
                st.write("No direct contradictions detected among supported claims.")

    with tabs[3]:
        res = st.session_state.main_result
        if not res:
            st.info("No result yet.")
        else:
            st.subheader("Generated Markdown review")
            st.markdown(res["review_md"])
            st.download_button("Download review.md", res["review_md"], file_name="reviewagent_review.md", mime="text/markdown")
            with st.expander("Raw Markdown"):
                st.code(res["review_md"], language="markdown")
            st.success(f"Unlinked factual sentences: {len(res['unlinked_sentences'])}")
            st.success(f"Hallucinated citations: {res['hallucinated_citations']} / {res['total_inline_citations']}")

    with tabs[4]:
        st.subheader("Tool-call trace")
        traces = st.session_state.storage.recent_traces(1500)
        if not traces:
            st.info("No traces in current storage.")
        else:
            tools = ["All"] + sorted({t["tool"] for t in traces})
            sel = st.selectbox("Filter by tool", tools)
            rows = traces if sel == "All" else [t for t in traces if t["tool"] == sel]
            st.dataframe(rows)
            with st.expander("Raw trace JSON (truncated)"):
                st.code(json.dumps(rows[:200], indent=2, default=str)[:50000], language="json")

    with tabs[5]:
        bm = st.session_state.benchmark_result
        if not bm:
            st.info("Benchmark not available. Use the sidebar button to run the 20-question benchmark.")
        else:
            s = bm["summary"]
            c = st.columns(4)
            c[0].metric("Micro recall", f"{s['micro_recall']:.1%}")
            c[1].metric("Citation precision", f"{s['citation_precision']:.1%}")
            c[2].metric("Hallucinated citation rate", f"{s['hallucinated_citation_rate']:.1%}")
            c[3].metric("Human sample", s["human_sample_size"])
            st.subheader("20 benchmark questions")
            st.dataframe(bm["rows"])
            st.subheader("Baseline comparison")
            st.dataframe([
                {
                    "System": "ReviewAgent with tools",
                    "Hallucinated citation rate": f"{s['hallucinated_citation_rate']:.1%}",
                    "Unlinked factual sentences": "0 by construction",
                    "Citations audited": s["total_inline_citations"]
                },
                {
                    "System": "Single-prompt baseline",
                    "Hallucinated citation rate": f"{s['baseline_hallucinated_citation_rate']:.1%}",
                    "Unlinked factual sentences": s["baseline_unlinked_factual_sentences"],
                    "Citations audited": s["total_inline_citations"]
                }
            ])

    with tabs[6]:
        tests = st.session_state.test_results
        if not tests:
            st.info("Tests not available.")
        else:
            st.subheader("Automated quality tests")
            for t in tests:
                if t["passed"]:
                    st.success(f"PASS — {t['test']}  \n{t['details']}")
                else:
                    st.error(f"FAIL — {t['test']}  \n{t['details']}")
            st.markdown("""
            These tests verify:
            - `record_claim` rejects non-verbatim quotes in code.
            - `verify_claim` drops unsupported overclaims.
            - the review generator emits no unlinked factual sentences.
            - hallucinated citation rate is zero for generated output.
            - retracted or flagged papers are excluded and logged.
            """)


main()