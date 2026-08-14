# opensolr-haystack

[Haystack](https://haystack.deepset.ai) integration for
[Opensolr](https://opensolr.com) — managed Apache Solr as a DocumentStore,
with **server-side embeddings** and native **hybrid (BM25 + kNN) retrieval**.

**See it live (real news index, hybrid + AI answer):** https://search.opensolr.com/news__dense?q=how+am+I+supposed+to+save+money%3F

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

## How writing works (Data Ingestion API)

Writes go through Opensolr's [Data Ingestion API](https://opensolr.com/learn/api-data-ingestion/204/data-ingestion-api-push-documents-to-your-opensolr-index-programmatically)
— the same pipeline the Drupal and WordPress connectors use. It is
**asynchronous**: documents are queued, then embeddings, sentiment, language
and all crawler-identical derived fields are computed **server-side**, and
documents become searchable within about a minute. Progress is visible in
**Control Panel → Data Ingestion** — a per-job status board (queued /
processing / completed / failed, with processed / success / failed document
counts per job) — and via the `ingest_status` API. Each document's
identity is its `uri` (the Solr id is `md5(uri)`): pass a real URL in
metadata (`{"uri": "https://..."}`), or a deterministic one is synthesized
from your id. Re-submitting the same `uri` updates the document. Pass
`{"rtf": True, "uri": "https://.../file.pdf"}` and the server extracts the
text from PDF/DOCX/XLSX for you.

## Lexical-only mode

Don't need vectors? Pure keyword search skips the embedding call entirely —
zero AI quota, and it works on **any** Opensolr index, including non-vector
ones and older Solr versions.

## Your index schema

Documents follow the Opensolr document model (`title`, `description`, `text`,
`meta_*` custom fields). To see the full schema: **Control Panel → click your
index → Configuration → Edit File → schema.xml**. Prefer zero-effort data
entry? Configure the **Web Crawler** in the Control Panel (Index Tools →
WebCrawler): add your site URL, validate it, and Opensolr indexes the whole
site for you.

## Grounded RAG answers

One call: hybrid retrieval picks the top hits, whose content becomes the LLM
context, and Opensolr's server-side LLM answers — no generator component,
no LLM key:

```python
answer = store.ai_answer(
    "what does the refund policy say?",
    rag_docs=3,        # how many hybrid hits feed the LLM (default 3)
    rag_words=1500,    # words of text taken from each hit (default 1500)
    # instruction="Answer in German, cite the exact titles you used",  # optional
)
```


### Search tuning

Retrieval (search and RAG grounding) runs through the platform's tuned
pipeline: global defaults → your index's saved **Search Tuning** (Control
Panel → Index Settings → Search Tuning: semantic↔lexical balance, field
weights, minimum match, search mode, vector candidate pool, content quality
boost) → optional per-call overrides via `tuning`:

```
tuning={"search_mode": "keywords_required", "fw_title": 0.2,
        "mm": "strict", "vector_topk": 500, "quality_boost": 0.3}
```

Defaults match the platform's PHP configuration exactly — customize in the
Control Panel once, or per call from code.

## How it's tested

Every release is validated against **live Opensolr infrastructure** — no mocks:

- **Unit tests** (offline): location aliases, filter→fq mapping, query building, escaping.
- **End-to-end suite**: the full write path through the async Data Ingestion
  queue (queued → server-side enrichment → searchable), semantic / hybrid /
  lexical retrieval, metadata round-trip, filters, id round-trip (your ids
  and the Solr `md5(uri)` ids), deletes by id and by query.
- **Real-corpus validation**: searches run against a 340-document replica of
  opensolr.com's own production search index. Verified: pure-semantic hits
  with zero keyword overlap ("how do I get my data back after a disaster" →
  backup &amp; restore docs), cross-lingual queries (Romanian query → English
  content), exact-term surfacing in hybrid mode, all four hybrid modes, and
  the full alpha range 0 → 1.
- **PDF ingestion**: a real PDF ingested via `rtf:true` — server-side text
  extraction (13k+ chars), automatic content-type detection, then retrieved
  with a purely semantic query against its contents.
- **Grounded RAG answers**: `ai_answer` verified end-to-end — a question answerable only from the ingested PDF returns the correct answer, sourced from the PDF's extracted text via hybrid retrieval.

The store is exercised live (write via ingestion, DuplicatePolicy SKIP/FAIL,
hybrid + lexical retrieval, filters, serde round-trip) before every release.

MIT license.
