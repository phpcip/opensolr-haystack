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
    ) -> None:
        self.index = index
        self.email = email
        self.api_key = api_key
        self.create_if_missing = create_if_missing
        self.location = location
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
        return Document(
            id=str(_flat(solr_doc.get("id", ""))),
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
                self.index, {"q": f"id:({joined})", "rows": len(ids), "fl": "id"}
            )
            existing = {
                str(d["id"][0] if isinstance(d["id"], list) else d["id"])
                for d in body["response"]["docs"]
            }
            if existing and policy == DuplicatePolicy.FAIL:
                from haystack.document_stores.errors import DuplicateDocumentError

                raise DuplicateDocumentError(f"IDs already in the store: {sorted(existing)}")
            documents = [d for d in documents if d.id not in existing]
            if not documents:
                return 0

        texts = [d.content or " " for d in documents]
        embeddings: List[Optional[List[float]]] = [d.embedding for d in documents]
        missing = [i for i, e in enumerate(embeddings) if e is None]
        if missing:
            computed = self.client.batch_embed(self.index, [texts[i] for i in missing])
            for i, vec in zip(missing, computed):
                embeddings[i] = vec

        docs = []
        for doc, text, vector in zip(documents, texts, embeddings):
            meta = dict(doc.meta or {})
            solr_doc: Dict[str, Any] = {
                "id": doc.id,
                "text": text,
                "embeddings": vector,
                "meta_lc_json": json.dumps(meta, ensure_ascii=False),
                "title": str(meta.get("title") or text[:100]),
            }
            for key, value in meta.items():
                if isinstance(value, (str, int, float, bool)):
                    solr_doc[_meta_field(str(key))] = str(value)
            docs.append(solr_doc)

        self.client.solr_update(self.index, docs)
        return len(docs)

    def delete_documents(self, document_ids: List[str]) -> None:
        if not document_ids:
            return
        self._ensure_index()
        self.client.solr_update(self.index, {"delete": list(document_ids)})

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
    ) -> List[Document]:
        self._ensure_index()
        vector = self.client.embed(self.index, query, is_query=True)
        compact = json.dumps(vector, separators=(",", ":"))
        knn = f"{{!knn f=embeddings topK={max(top_k, 10)}}}{compact}"

        params: Dict[str, Any] = {"rows": top_k, "fl": "*,score"}
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
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpensolrDocumentStore":
        deserialize_secrets_inplace(data["init_parameters"], keys=["email", "api_key"])
        return default_from_dict(cls, data)
