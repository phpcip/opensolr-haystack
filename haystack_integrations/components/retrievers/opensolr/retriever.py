"""Hybrid retriever component over :class:`OpensolrDocumentStore`.

Takes a plain-text query — embedding happens server-side on Opensolr's GPU
infrastructure, so no query-embedder component is needed in the pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from haystack import Document, component, default_from_dict, default_to_dict

from haystack_integrations.document_stores.opensolr import OpensolrDocumentStore


@component
class OpensolrHybridRetriever:
    """Retrieve documents from Opensolr with hybrid BM25 + kNN scoring.

    Example:
        ```python
        from haystack import Pipeline
        from haystack_integrations.document_stores.opensolr import OpensolrDocumentStore
        from haystack_integrations.components.retrievers.opensolr import OpensolrHybridRetriever

        store = OpensolrDocumentStore(index="mysite__dense")
        pipe = Pipeline()
        pipe.add_component("retriever", OpensolrHybridRetriever(document_store=store))
        result = pipe.run({"retriever": {"query": "affordable restaurants"}})
        ```
    """

    def __init__(
        self,
        document_store: OpensolrDocumentStore,
        top_k: int = 10,
        hybrid: bool = True,
        alpha: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
        lexical: bool = False,
        fresh_bias: bool = False,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        group_pages: bool = True,
    ) -> None:
        self.document_store = document_store
        self.top_k = top_k
        self.hybrid = hybrid
        self.alpha = alpha
        self.filters = filters
        self.lexical = lexical
        # Threaded through the component because a pipeline never touches the store
        # directly — run() below is the only door onto OpensolrDocumentStore.search()
        # for a Haystack user, so an option the component does not carry is an option
        # that does not exist in a pipeline.
        self.fresh_bias = fresh_bias
        self.date_from = date_from
        self.date_to = date_to
        self.group_pages = group_pages

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: str,
        top_k: Optional[int] = None,
        hybrid: Optional[bool] = None,
        alpha: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        lexical: Optional[bool] = None,
        fresh_bias: Optional[bool] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        group_pages: Optional[bool] = None,
    ) -> Dict[str, List[Document]]:
        """Run the retriever. ``alpha``: 0 = all semantic, 1 = all lexical.
        ``lexical=True`` = pure keyword search, no embedding call.
        ``fresh_bias=True`` biases the ranking toward recent documents by
        multiplying each score by a recency curve on ``creation_date``; it
        re-orders and never filters, so the hit count is unchanged and nothing
        old becomes unreachable. Off by default.
        ``date_from`` / ``date_to`` (``YYYY-MM-DD``) keep only documents dated in that
        period. ``group_pages`` (on by default) returns a PDF indexed page by page once,
        as its best page, with ``page_number``, ``page_url`` and ``more_pages`` in meta."""
        docs = self.document_store.search(
            query=query,
            top_k=top_k if top_k is not None else self.top_k,
            hybrid=hybrid if hybrid is not None else self.hybrid,
            alpha=alpha if alpha is not None else self.alpha,
            filters=filters if filters is not None else self.filters,
            lexical=lexical if lexical is not None else self.lexical,
            fresh_bias=fresh_bias if fresh_bias is not None else self.fresh_bias,
            date_from=date_from if date_from is not None else self.date_from,
            date_to=date_to if date_to is not None else self.date_to,
            group_pages=group_pages if group_pages is not None else self.group_pages,
        )
        return {"documents": docs}

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            document_store=self.document_store.to_dict(),
            top_k=self.top_k,
            hybrid=self.hybrid,
            alpha=self.alpha,
            filters=self.filters,
            lexical=self.lexical,
            # Serialized alongside the other search options so a pipeline saved with
            # the bias on comes back with it on. from_dict() tolerates its absence in
            # dicts written before this option existed — the constructor defaults it.
            fresh_bias=self.fresh_bias,
            date_from=self.date_from,
            date_to=self.date_to,
            group_pages=self.group_pages,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpensolrHybridRetriever":
        data["init_parameters"]["document_store"] = OpensolrDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)
