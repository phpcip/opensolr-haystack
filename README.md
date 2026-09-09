# opensolr-haystack

[Haystack](https://haystack.deepset.ai) integration for
[Opensolr](https://opensolr.com) — managed Apache Solr as a DocumentStore,
with **server-side embeddings** and native **hybrid (BM25 + kNN) retrieval**.

**See it live (real news index, hybrid + AI answer):** https://search.opensolr.com/news__dense?q=how+am+I+supposed+to+save+money%3F

No embedder components needed in your pipeline — texts and queries are
embedded on Opensolr's GPU infrastructure (multilingual E5-large-instruct,
1024 dimensions, cosine).

**Product page:** [opensolr.com/langchain](https://opensolr.com/langchain) ·
free forever, no card, at [opensolr.com](https://opensolr.com)

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

## Try it without an account

There is a public demo account. Point the package at it and everything in this
README works immediately, with no signup:

```bash
export OPENSOLR_EMAIL=mcp@opensolr.com
export OPENSOLR_API_KEY=420b8b23e7b12dc8ab838932145a5065
```

`mcp_demo_d1__dense` is already loaded with 300 news articles, so search, filtering
and grounded answers work the moment you connect. You also get the full write path:
create your own index on the account, ingest into it, query it, delete it.

Know what you are working with:

- **Anything you create there is deleted after 3 days.** Automatically, without warning
  or export. That includes indexes you created and every document in them.
- **The account is shared with everyone reading this.** Your index is visible to them,
  they can change or delete it, and you can do the same to theirs. Never put anything
  real, private or client-owned in it.
- **The limits are per index, and deliberately small.** 200 MB of bandwidth and 50 MB
  of disk per index. Bandwidth is the one you will hit first: it covers a demo, a
  tutorial and a proof of concept, and it will not carry an application.

When you want an index that is private, yours and still there next week, get your own
key — [free forever, no card](https://opensolr.com/register) — and change the two
variables above. Nothing else in your code changes.

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

## Search operators

The query string understands the operators people already expect from a search box. They work
in every mode — keyword, hybrid and pure vector.

| Operator | Meaning | Example |
| --- | --- | --- |
| `"word1 word2"` | Phrase — those words together, in that order | `"machine learning"` |
| `+word` | Required — every result must contain it | `+laptop 15 inch gaming` |
| `+"word1 word2"` | Required phrase | `+"13 inch"` |
| `-word` | Excluded — drop any document containing it | `laptop -refurbished` |
| `-"word1 word2"` | Excluded phrase | `-"open box"` |

They compose: `+laptop +"13 inch" -refurbished` returns only 13-inch laptops and never a
refurbished one.

A prefixed term (`+` or `-`, word or phrase) becomes a **filter**, applied to the whole result
set. That matters as soon as a vector is involved: the semantic side of a hybrid search has no
concept of negation, so left as query text `-refurbished` would actually pull refurbished
listings *towards* the top rather than removing them. As a filter it binds every document,
whichever side of the search found it, and the exclusion is absolute.

An unprefixed phrase (`"machine learning"` with no `+` in front) is a keyword-side relevance
signal rather than a filter — use `+"machine learning"` when you need it enforced.

`+` and `-` only count at the start of a word, so `e-mail`, `covid-19` and `1+1` are searched
for literally.

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

## Search by image

Search your index with a **photo** instead of a text query. Opensolr reads the
picture three ways — visual labels (what it depicts), OCR text (words printed on
it), and any barcode / QR code — and turns that into an ordinary search. Nothing
new is stored in Solr; the picture simply becomes words.

```python
# What the picture reads as (labels, OCR text, barcodes) — no search yet:
read = store.image_to_words("photo.jpg")   # path, bytes, or base64
# {'text': 'red running shoe', 'mode': 'clip',
#  'labels': ['running shoe', 'sneaker'], 'codes': ['0123456789012']}

# Search the index with the picture:
docs = store.search_by_image("photo.jpg", top_k=4)          # engine picks the best reading
docs = store.search_by_image("photo.jpg", using="meaning")  # visual labels
docs = store.search_by_image("photo.jpg", using="text")     # only OCR text
docs = store.search_by_image("photo.jpg", using="code")     # exact barcode / QR match
docs = store.search_by_image("photo.jpg", using="all")      # labels + OCR + codes
```

`search_by_image` forwards the same tuning as `search` — `hybrid`, `lexical`,
`alpha`, `fresh_bias`, `filters` — so an image query runs through the exact
hybrid pipeline a text query does.

## Your index schema

Documents follow the Opensolr document model (`title`, `description`, `text`,
`meta_*` custom fields). The whole schema, every field and every type suffix,
is explained in the [Index Schema Reference](https://opensolr.com/opensolr-platform-user-documentation/schema-reference).
To see your own copy: **Control Panel → click your
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

#### Fresh Results Bias

Rank newer documents higher without hiding anything older. Every score is
multiplied by a recency curve on `creation_date` — full weight for a document
published today, about half after a year:

```python
store.similarity_search_with_score("solar inverter warranty", fresh_bias=True)
client.hybrid_search(index, query, fresh_bias=True)
client.ai_answer(index, question, tuning={"fresh_bias": 1})
```

It **re-orders and never filters**: the hit count is identical either way,
nothing old becomes unreachable, and a document with no `creation_date` simply
keeps its place instead of being pushed to the bottom. It applies to all three
retrieval shapes — vector-only, keyword-only and the fused hybrid ranking —
because the boost wraps the final score rather than one half of it. Off by
default.

This is the same control visitors get as the **Fresh** toggle beside the sort
options on the hosted Opensolr search page, so a query behaves identically here
and there.

> `fresh_bias` and `freshness_boost` are two different knobs and the names
> invite confusion. `freshness_boost` is a hard window in **days** — anything
> older is filtered out and the hit count drops. `fresh_bias` filters nothing.

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
