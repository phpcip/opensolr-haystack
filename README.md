# opensolr-haystack

[Haystack](https://haystack.deepset.ai) integration for
[Opensolr](https://opensolr.com) — managed Apache Solr as a DocumentStore,
with **server-side embeddings** and native **hybrid (BM25 + kNN) retrieval**.

No embedder components needed in your pipeline — texts and queries are
embedded on Opensolr's GPU infrastructure (multilingual E5-large-instruct,
1024 dimensions, cosine).

**Product page:** [opensolr.com/langchain](https://opensolr.com/langchain) ·
free 15-day trial, no card, at [opensolr.com](https://opensolr.com)

```bash
pip install opensolr-haystack
```

## Quickstart

```python
from haystack import Document, Pipeline
from haystack_integrations.document_stores.opensolr import OpensolrDocumentStore
from haystack_integrations.components.retrievers.opensolr import OpensolrHybridRetriever

# credentials default to OPENSOLR_EMAIL / OPENSOLR_API_KEY env vars
store = OpensolrDocumentStore(index="mysite__dense", create_if_missing=True)

store.write_documents([
    Document(content="Hybrid search fuses BM25 with vector similarity"),
    Document(content="Cats sleep sixteen hours a day"),
])

pipe = Pipeline()
pipe.add_component("retriever", OpensolrHybridRetriever(document_store=store))
result = pipe.run({"retriever": {"query": "how do keyword and semantic search combine?"}})
print(result["retriever"]["documents"])
```

Note there is **no embedder** in the pipeline — not for documents, not for
the query. The store embeds server-side at both index and query time.

## Hybrid retrieval

`OpensolrHybridRetriever` fuses BM25 and kNN scores per document via
Opensolr's native `{!hybrid}` Solr query parser:

```python
OpensolrHybridRetriever(
    document_store=store,
    top_k=10,
    hybrid=True,     # False = pure semantic kNN
    alpha=0.5,       # 0 = all semantic … 1 = all lexical
)
```

Standard Haystack filters are supported and map to Solr `fq`:

```python
pipe.run({"retriever": {
    "query": "search engines",
    "filters": {"field": "meta.category", "operator": "==", "value": "docs"},
}})
```

## Notes

- Vector-enabled indexes run on Opensolr's Solr 9.x environments — currently
  `us` (Chicago), `de` (Germany), `fi` (Finland). **Additional dedicated
  regions can be deployed on request** (paid add-on):
  [support@opensolr.com](mailto:support@opensolr.com).
- Every index is also plain Apache Solr with the native `/select` API —
  facets, highlighting, spellcheck included.
- Siblings: [`langchain-opensolr`](https://pypi.org/project/langchain-opensolr/) ·
  [`llama-index-opensolr`](https://pypi.org/project/llama-index-opensolr/) ·
  [`opensolr-mcp`](https://pypi.org/project/opensolr-mcp/)

MIT license.
