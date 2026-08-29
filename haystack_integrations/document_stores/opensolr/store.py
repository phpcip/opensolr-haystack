"""Opensolr DocumentStore for Haystack.

Managed, vector-enabled Apache Solr 9.x (knn_vector 1024-dim, cosine) with
embeddings computed **server-side** on Opensolr's GPU infrastructure — no
local embedder component needed at indexing time.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from haystack import Document, default_from_dict, default_to_dict
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.document_stores.opensolr.client import (
    OpensolrClient,
    OpensolrError,
    apply_fresh_bias,
)

_META_KEY_RE = re.compile(r"[^a-z0-9_]+")


def _meta_field(key: str) -> str:
    return f"meta_{_META_KEY_RE.sub('_', key.lower()).strip('_')}"


def _escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _filters_to_fq(filters: Optional[Dict[str, Any]]) -> List[str]:
    """Translate Haystack's standard filter dict into Solr fq clauses.

    Supports field conditions with operators ==, !=, in, not in, and logical
    AND groups (the common cases). OR groups become a single fq with OR.
    """
    if not filters:
        return []

    def _cond(f: Dict[str, Any]) -> str:
        field = f["field"].split("meta.", 1)[-1]
        solr_field = _meta_field(field)
        op = f["operator"]
        value = f.get("value")
        if op == "==":
            return f'{solr_field}:"{_escape(value)}"'
        if op == "!=":
            return f'-{solr_field}:"{_escape(value)}"'
        if op == "in":
            joined = " OR ".join(f'"{_escape(v)}"' for v in value)
            return f"{solr_field}:({joined})"
        if op == "not in":
            joined = " OR ".join(f'"{_escape(v)}"' for v in value)
            return f"-{solr_field}:({joined})"
        if op in (">", ">="):
            return f'{solr_field}:{"{" if op == ">" else "["}"{_escape(value)}" TO *]'
        if op in ("<", "<="):
            return f'{solr_field}:[* TO "{_escape(value)}"{"}" if op == "<" else "]"}'
        raise ValueError(f"Unsupported filter operator: {op}")

    if "operator" in filters and "conditions" in filters:
        parts = []
        for c in filters["conditions"]:
            if "conditions" in c:
                sub = _filters_to_fq(c)
                parts.append("(" + " AND ".join(sub) + ")")
            else:
                parts.append(_cond(c))
        if filters["operator"] == "AND":
            return parts
        if filters["operator"] == "OR":
            return ["(" + " OR ".join(parts) + ")"]
        raise ValueError(f"Unsupported logical operator: {filters['operator']}")
    return [_cond(filters)]




def _build_ingest_doc(index: str, text: str, metadata: dict, doc_id: str):
    """Build one Data Ingestion API document. Returns (doc, solr_id)."""
    import hashlib
    from urllib.parse import quote

    meta = dict(metadata or {})
    uri = meta.get("uri") or meta.get("url")
    if not (isinstance(uri, str) and uri.startswith(("http://", "https://"))):
        uri = f"https://ingest.opensolr.com/{index}/{quote(str(doc_id), safe='')}"
    uri = uri.rstrip("/")
    text = text or " "
    solr_doc = {
        "uri": uri,
        "title": str(meta.get("title") or text[:100] or uri)[:250],
        "description": str(meta.get("description") or text[:200]),
        "text": text,
        "meta_ext_id": str(doc_id),
        "meta_lc_json": json.dumps(meta, ensure_ascii=False),
    }
    if meta.get("rtf"):
        solr_doc["rtf"] = True
    if meta.get("timestamp"):
        solr_doc["timestamp"] = meta["timestamp"]
    for key, value in meta.items():
        if isinstance(value, (str, int, float, bool)) and key not in ("rtf", "uri", "url"):
            solr_doc[_meta_field(str(key))] = str(value)
    return solr_doc, hashlib.md5(uri.encode()).hexdigest()


class OpensolrDocumentStore:
    """Haystack DocumentStore backed by a managed Opensolr vector index.

    Example:
        ```python
        from haystack_integrations.document_stores.opensolr import OpensolrDocumentStore

        store = OpensolrDocumentStore(index="mysite__dense")
        # credentials default to OPENSOLR_EMAIL / OPENSOLR_API_KEY env vars
        ```
    """

    def __init__(
        self,
        index: str,
        email: Secret = Secret.from_env_var("OPENSOLR_EMAIL"),
        api_key: Secret = Secret.from_env_var("OPENSOLR_API_KEY"),
        create_if_missing: bool = False,
        location: str = "us",
        ingest_wait: bool = True,
    ) -> None:
        self.index = index
        self.email = email
        self.api_key = api_key
        self.create_if_missing = create_if_missing
        self.location = location
        self.ingest_wait = ingest_wait
        self._client: Optional[OpensolrClient] = None
        self._checked = False

    # ------------------------------------------------------------------ #

    @property
    def client(self) -> OpensolrClient:
        if self._client is None:
            self._client = OpensolrClient(
                self.email.resolve_value(), self.api_key.resolve_value()
            )
        return self._client

    def _ensure_index(self) -> None:
        if self._checked:
            return
        try:
            self.client.get_core_info(self.index)
        except OpensolrError:
            if not self.create_if_missing:
                raise
            self.client.create_index(self.index, self.location)
            import time

            for _ in range(5):
                time.sleep(2)
                try:
                    self.client.get_core_info(self.index, refresh=True)
                    break
                except OpensolrError:
                    continue
        self._checked = True

    def _doc_from_solr(self, solr_doc: Dict[str, Any]) -> Document:
        def _flat(v: Any) -> Any:
            return v[0] if isinstance(v, list) and len(v) == 1 else v

        meta: Dict[str, Any] = {}
        raw = _flat(solr_doc.get("meta_lc_json"))
        if raw:
            try:
                meta = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                meta = {}
        content = _flat(solr_doc.get("text", "")) or ""
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        score = solr_doc.get("score")
        ext = _flat(solr_doc.get("meta_ext_id"))
        return Document(
            id=str(ext) if ext else str(_flat(solr_doc.get("id", ""))),
            content=str(content),
            meta=meta,
            score=float(_flat(score)) if score is not None else None,
        )

    # ------------------------------------------------------------------ #
    # DocumentStore protocol                                             #
    # ------------------------------------------------------------------ #

    def count_documents(self) -> int:
        self._ensure_index()
        body = self.client.solr_select(self.index, {"q": "*:*", "rows": 0})
        return int(body["response"]["numFound"])

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        self._ensure_index()
        params: Dict[str, Any] = {"q": "*:*", "rows": 1000, "fl": "*"}
        fq = _filters_to_fq(filters)
        if fq:
            params["fq"] = fq
        body = self.client.solr_select(self.index, params)
        return [self._doc_from_solr(d) for d in body["response"]["docs"]]

    def write_documents(
        self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
    ) -> int:
        if not documents:
            return 0
        self._ensure_index()

        if policy in (DuplicatePolicy.SKIP, DuplicatePolicy.FAIL):
            ids = [d.id for d in documents]
            joined = " OR ".join(f'"{_escape(i)}"' for i in ids)
            body = self.client.solr_select(
                self.index,
                {"q": f"id:({joined}) OR meta_ext_id:({joined})", "rows": max(len(ids), 10), "fl": "id,meta_ext_id"},
            )
            existing = set()
            for d in body["response"]["docs"]:
                for f in ("id", "meta_ext_id"):
                    v = d.get(f)
                    v = v[0] if isinstance(v, list) else v
                    if v:
                        existing.add(str(v))
            if existing and policy == DuplicatePolicy.FAIL:
                from haystack.document_stores.errors import DuplicateDocumentError

                raise DuplicateDocumentError(f"IDs already in the store: {sorted(existing)}")
            documents = [d for d in documents if d.id not in existing]
            if not documents:
                return 0

        docs = []
        for doc in documents:
            text = doc.content or " "
            solr_doc, _sid = _build_ingest_doc(self.index, text, doc.meta or {}, doc.id)
            docs.append(solr_doc)

        for i in range(0, len(docs), 50):
            self.client.ingest(self.index, docs[i : i + 50], wait=self.ingest_wait)
        return len(docs)

    def ai_answer(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        # 4 documents is the platform's measured context size (OpensolrClient.RAG_DOCS);
        # this used to pass 3, which quietly overrode the client default with a smaller one.
        rag_docs: int = 4,
        rag_words: int = 1500,
        instruction: Optional[str] = None,
        tuning: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Grounded RAG answer generated only from this index's content.

        Two-step pattern: hybrid (BM25 + kNN) retrieval picks the top
        ``rag_docs`` hits (first ``rag_words`` words of text each), whose
        title/description/text become the LLM context — the same pipeline as
        Opensolr's hosted search UI. Pass ``instruction`` to fully control
        the prompt (e.g. "Answer in German, cite the sources you used").
        Retrieval uses the platform's tuned pipeline: your index's saved
        Search Tuning (Control Panel) applies automatically; ``tuning``
        overrides any knob per call. The list below is the whole set, not a
        sample: an abbreviated one reads as everything that is supported, and
        ``freshness_boost`` was invisible to callers because of it.
        ``fw_title``, ``fw_description``, ``fw_uri``, ``fw_text``,
        ``fw_text_t``, ``lexical_weight``, ``vector_weight``, ``vector_topk``,
        ``search_mode`` (union / keywords_required / meaning_required /
        intersection), ``quality_boost``, ``min_score``, ``freshness_boost``,
        ``fresh_bias``, ``lexical_norm_k``, ``mm`` (flexible / balanced /
        strict or raw Solr mm syntax). ``freshness_boost`` and ``fresh_bias``
        are different knobs despite the names: the first is a hard window in
        DAYS that filters older documents out, the second only re-orders,
        multiplying each score by a recency curve on ``creation_date`` so
        recent documents win ties while nothing becomes unreachable.
        Returns plain text.
        """
        fqs = _filters_to_fq(filters)
        fq = " AND ".join(f"({f})" for f in fqs) if fqs else None
        return self.client.ai_summary(
            self.index, query, filter_query=fq,
            rag_docs=rag_docs, rag_words=rag_words, instruction=instruction,
            tuning=tuning,
            **kwargs,
        )

    def delete_documents(self, document_ids: List[str]) -> None:
        if not document_ids:
            return
        self._ensure_index()
        joined = " OR ".join(f'"{_escape(str(i))}"' for i in document_ids)
        self.client.solr_update(
            self.index,
            {"delete": {"query": f"id:({joined}) OR meta_ext_id:({joined})"}},
        )

    # ------------------------------------------------------------------ #
    # search (used by the retriever component)                           #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        top_k: int = 10,
        hybrid: bool = True,
        alpha: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
        lexical: bool = False,
        fresh_bias: bool = False,
    ) -> List[Document]:
        """Retrieve the top_k documents for a query.

        ``fresh_bias`` biases the ranking toward recent documents by multiplying
        each score by a recency curve on ``creation_date``. It re-orders and never
        filters — the hit count is unchanged, nothing old becomes unreachable, and
        a document with no ``creation_date`` is simply left unboosted. Applies to
        the lexical, hybrid and pure-kNN paths alike. Off by default.
        """
        self._ensure_index()
        params: Dict[str, Any] = {"rows": top_k, "fl": "*,score"}
        if lexical:
            clean = query.replace("{", " ").replace("}", " ").replace('"', " ")
            params["q"] = f'{{!edismax qf="title^100 description^20 text^1"}}{clean}'
            # Wrapped rather than set as an edismax `bf`: edismax is invoked here
            # through local params inside q, not as the request's defType, so a
            # top-level bf is not reliably its own.
            if fresh_bias:
                apply_fresh_bias(params)
            fq = _filters_to_fq(filters)
            if fq:
                params["fq"] = fq
            body = self.client.solr_select(self.index, params)
            return [self._doc_from_solr(d) for d in body["response"]["docs"]]

        vector = self.client.embed(self.index, query, is_query=True)
        compact = json.dumps(vector, separators=(",", ":"))
        knn = f"{{!knn f=embeddings topK={max(top_k, 10)}}}{compact}"
        if hybrid:
            clean = query.replace("{", " ").replace("}", " ").replace('"', " ")
            params["q"] = (
                f"{{!hybrid lexical=$lexicalRaw vector=$vectorQuery "
                f"mode=union alpha={alpha} topN={max(top_k, 10)}}}"
            )
            params["lexicalRaw"] = f'{{!edismax qf="title^100 text^1"}}{clean}'
            params["vectorQuery"] = knn
        else:
            params["q"] = knn
        # Fresh Results Bias wraps whichever shape was just built — fused {!hybrid}
        # or bare {!knn} — so the recency multiplier reaches every candidate,
        # including the vector-only ones an edismax bf never sees.
        if fresh_bias:
            apply_fresh_bias(params)
        fq = _filters_to_fq(filters)
        if fq:
            params["fq"] = fq

        body = self.client.solr_select(self.index, params)
        return [self._doc_from_solr(d) for d in body["response"]["docs"]]

    # ------------------------------------------------------------------ #
    # serialization                                                      #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            index=self.index,
            email=self.email.to_dict(),
            api_key=self.api_key.to_dict(),
            create_if_missing=self.create_if_missing,
            location=self.location,
            # ingest_wait was missing, so a store serialised with ingest_wait=False came back
            # with the default True — a Haystack pipeline saved to YAML silently changed
            # behaviour on reload, turning non-blocking writes into blocking ones. Every
            # constructor parameter has to appear here or from_dict cannot rebuild the object
            # that was saved (2026-08-29).
            ingest_wait=self.ingest_wait,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpensolrDocumentStore":
        deserialize_secrets_inplace(data["init_parameters"], keys=["email", "api_key"])
        return default_from_dict(cls, data)
