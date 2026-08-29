#!/usr/bin/env python
"""
Live production test for the ``opensolr-haystack`` package.

Runs EVERY public method of the client layer
(haystack_integrations/document_stores/opensolr/client.py), the Haystack
DocumentStore (store.py) and the retriever component against the REAL
Opensolr production APIs, using the public MCP demo account.

What it touches
---------------
* ``mcp_demo_d1__dense``  — the seeded, read-only demo index (300 news
  articles). Never written to, never deleted from.
* ``mcp_t_hs<rand>__dense`` — a throwaway index this script creates for every
  write path, and deletes again in a ``finally`` block so an exception, a
  Ctrl-C or a failed assertion still leaves the account clean.

Design notes
------------
* Assertions are about VALUES, never merely "no exception was raised": a
  search asserts hits with real scores, an embedding asserts 1024 dimensions
  and unit norm, a RAG answer asserts non-empty text that does not open with
  the two phrasings the shipped instruction forbids (which is what proves the
  package's own prompt — not the server's fallback — reached the model).
* Ingestion is asynchronous (a once-a-minute queue runner), so every wait is a
  POLL of ingest_status plus the live document count, never a fixed sleep, and
  it gives up with a clear message after the deadline.
* The platform rate-limits at 30 requests/minute per account. Every HTTP call
  this script makes — the package's own, through an httpx event hook, and the
  harness's — goes through one sliding-window pacer that keeps the rate under
  that. Direct-to-Solr traffic is not rate limited and is not paced.

Usage:  ./.venv/bin/python test_live.py
Exit code 0 = all green, 1 = at least one check failed.
"""

from __future__ import annotations

import collections
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

import httpx

# --------------------------------------------------------------------------- #
# Credentials — the PUBLIC MCP demo account. Throwaway by design, swept after
# three days. Set as env vars too, because OpensolrDocumentStore resolves its
# secrets from OPENSOLR_EMAIL / OPENSOLR_API_KEY by default and the to_dict()
# round-trip below can only be exercised with env-var secrets (a token secret
# refuses to serialize, by Haystack's design).
# --------------------------------------------------------------------------- #
EMAIL = "mcp@opensolr.com"
API_KEY = "420b8b23e7b12dc8ab838932145a5065"
os.environ["OPENSOLR_EMAIL"] = EMAIL
os.environ["OPENSOLR_API_KEY"] = API_KEY

DEMO_INDEX = "mcp_demo_d1__dense"
#: The __dense suffix is REQUIRED — it is what marks an index vector-enabled
#: across the whole platform. Random suffix so two runs (or two agents) never
#: collide on the same name.
TMP_INDEX = "mcp_t_hs%06x__dense" % random.randrange(16 ** 6)
TMP_LOCATION = "fi"  # same region as the demo index (FINLAND9)

MGMT_HOST = "https://opensolr.com"
AI_HOST = "https://api.opensolr.com"
#: Published, unauthenticated download. Data Ingestion and vector search need
#: this config set (knn_vector schema + {!hybrid} handler) on the index; a
#: freshly created index gets the plain base config, which has neither.
CONFIG_SET_URL = MGMT_HOST + "/configs/mandatory_web_crawler_config_set_solr_9.zip"

#: Seconds to wait for asynchronous ingestion before giving up.
INGEST_DEADLINE = 120.0

# --------------------------------------------------------------------------- #
# Rate pacing                                                                   #
# --------------------------------------------------------------------------- #
# The platform allows 30 requests/minute per account. 24 leaves headroom for
# anything else on this account and for clock skew between us and the server's
# per-calendar-minute bucket.
MAX_PER_MIN = 24
PACED_HOSTS = {"opensolr.com", "api.opensolr.com"}
_REQ_LOG: Dict[str, collections.deque] = collections.defaultdict(collections.deque)


def _pace(request: httpx.Request) -> None:
    """httpx request hook: block until this call fits inside the rate limit.

    Sliding 60-second window per API host. Solr cluster hosts are not rate
    limited, so they fall straight through.
    """
    host = request.url.host
    if host not in PACED_HOSTS:
        return
    window = _REQ_LOG[host]
    now = time.time()
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= MAX_PER_MIN:
        wait = 60 - (now - window[0]) + 0.5
        if wait > 0:
            print("   … pacing: sleeping %.0fs to stay under %d req/min on %s"
                  % (wait, MAX_PER_MIN, host))
            time.sleep(wait)
        now = time.time()
        while window and now - window[0] > 60:
            window.popleft()
    window.append(time.time())


def instrument(client: Any) -> Any:
    """Attach the pacer to an OpensolrClient's underlying httpx client.

    Runtime instrumentation of one instance — the package source is never
    touched, and this is the only way to pace the requests the package makes
    on its own (the ingest wait loop, for one).
    """
    client._http.event_hooks = {"request": [_pace], "response": []}
    return client


#: One httpx client for the harness's own calls (index creation, config-set
#: upload, teardown). Paced by the same hook, so harness traffic and package
#: traffic share one budget.
HARNESS = httpx.Client(timeout=300.0, follow_redirects=True,
                       event_hooks={"request": [_pace]})


# --------------------------------------------------------------------------- #
# Check harness                                                                 #
# --------------------------------------------------------------------------- #
PASSED = 0
FAILED = 0
FAILURES: List[str] = []


def check(label: str, fn) -> bool:
    """Run one assertion block. Prints ✔/✘ and what was actually observed.

    ``fn`` returns a short string of the observed VALUES, which is appended to
    the line — the point being that a passing line still shows evidence, not
    just a tick.
    """
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - a failing check is data, not a crash
        FAILED += 1
        msg = "%s: %s" % (type(exc).__name__, str(exc).replace("\n", " ")[:300])
        FAILURES.append("%s — %s" % (label, msg))
        print("✘ %s — %s" % (label, msg))
        return False
    PASSED += 1
    print("✔ %s%s" % (label, (" — " + str(detail)) if detail else ""))
    return True


def section(name: str) -> None:
    print("\n--- %s " % name + "-" * max(0, 66 - len(name)))


# --------------------------------------------------------------------------- #
# Harness-side platform helpers (NOT part of the package under test)            #
# --------------------------------------------------------------------------- #

def _last_json(raw: str) -> Optional[Any]:
    """Some management endpoints stream progress HTML before their JSON."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    depth = 0
    start = -1
    best = None
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    best = json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    pass
    return best


def apply_config_set(index: str) -> str:
    """Push the mandatory Web Crawler config set onto a fresh index.

    A brand-new Opensolr index is created from the plain base config: no
    ``knn_vector`` field, no ``{!hybrid}`` handler, no enrichment fields — so
    neither Data Ingestion nor vector search can work on it until this zip is
    applied. The platform's own sandbox provisioner does exactly this, and so
    do the Drupal and WordPress integrations. Harness setup, not a package API.
    """
    zip_bytes = HARNESS.get(CONFIG_SET_URL).content
    resp = HARNESS.post(
        MGMT_HOST + "/solr_manager/api/upload_zip_config_files",
        data={"email": EMAIL, "api_key": API_KEY, "core_name": index},
        files={"userfile": ("mandatory_web_crawler_config_set_solr_9.zip",
                            zip_bytes, "application/zip")},
    )
    body = _last_json(resp.text)
    if not isinstance(body, dict) or body.get("status") is not True:
        raise RuntimeError("config set upload failed: %s" % str(body)[:300])
    return "%d KB applied" % (len(zip_bytes) // 1024)


def delete_index(index: str) -> Any:
    """Teardown. Management API, POST body params."""
    resp = HARNESS.post(
        MGMT_HOST + "/solr_manager/api/delete_index",
        data={"email": EMAIL, "api_key": API_KEY, "index_name": index},
    )
    return _last_json(resp.text)


def solr_count(client: Any, index: str) -> int:
    """Live document count straight from Solr (not rate limited)."""
    body = client.solr_select(index, {"q": "*:*", "rows": 0})
    return int(body["response"]["numFound"])


def wait_for_docs(client: Any, index: str, expected: int,
                  job_id: Optional[str] = None,
                  deadline: float = INGEST_DEADLINE) -> str:
    """Poll until the index holds ``expected`` documents.

    Ingestion is asynchronous — the queue runner is a once-a-minute cron — so
    this polls the live document count (free, direct to Solr) and, less often,
    the job's own status so a FAILED job is reported as a failure instead of
    running out the clock. Never a fixed sleep.
    """
    started = time.time()
    last_status = 0.0
    count = -1
    while time.time() - started < deadline:
        count = solr_count(client, index)
        if count >= expected:
            return "%d docs searchable after %.0fs" % (count, time.time() - started)
        if job_id and time.time() - last_status > 25:
            last_status = time.time()
            st = client.ingest_status(job_id)
            job = st.get("job") or {}
            label = job.get("state_label")
            if label in ("failed", "stopped"):
                raise AssertionError(
                    "ingest job %s reported state=%s error=%s"
                    % (job_id, label, job.get("error"))
                )
        time.sleep(5)
    raise AssertionError(
        "gave up after %.0fs: index %s holds %d docs, expected %d "
        "(ingestion queue did not complete in time)"
        % (deadline, index, count, expected)
    )


# --------------------------------------------------------------------------- #

def main() -> int:
    # Declared up front: the teardown block below counts a failed cleanup as a
    # failure, and a `global` has to precede every use of the name in a scope.
    global FAILED

    from haystack import Document, Pipeline
    from haystack.document_stores.errors import DuplicateDocumentError
    from haystack.document_stores.types import DuplicatePolicy
    from haystack.utils import Secret

    from haystack_integrations.components.retrievers.opensolr import (
        OpensolrHybridRetriever,
    )
    from haystack_integrations.document_stores.opensolr import OpensolrDocumentStore
    from haystack_integrations.document_stores.opensolr.client import (
        BATCH_EMBED_MAX,
        DOC_FENCE,
        FRESH_BIAS_FUNCTION,
        OpensolrClient,
        OpensolrError,
        VECTOR_LOCATIONS,
        apply_fresh_bias,
        build_context,
        build_instruction,
        resolve_location,
    )

    print("opensolr-haystack — LIVE production test")
    print("account : %s" % EMAIL)
    print("read    : %s" % DEMO_INDEX)
    print("write   : %s (created and deleted by this run)" % TMP_INDEX)

    client = instrument(OpensolrClient(EMAIL, API_KEY))
    created = False
    state: Dict[str, Any] = {}

    try:
        # ------------------------------------------------------------------ #
        section("pure builders (no network)")
        # ------------------------------------------------------------------ #

        def t_resolve():
            assert resolve_location("us") == "CHICAGO-96", resolve_location("us")
            assert resolve_location(" FI ") == "FINLAND9"
            assert resolve_location("DE-SOLR-9") == "DE-SOLR-9", "unknown passes through"
            assert set(VECTOR_LOCATIONS) == {"us", "de", "fi"}
            assert issubclass(OpensolrError, RuntimeError)
            return "us→CHICAGO-96, fi→FINLAND9, unknown passes through"

        check("resolve_location maps aliases and passes unknowns through", t_resolve)

        def t_fresh_bias():
            params = {"q": "{!hybrid lexical=$l vector=$v}", "rows": 5}
            out = apply_fresh_bias(params)
            assert out is params, "must mutate in place and return the same dict"
            assert out["q"] == "{!boost b=$freshBias v=$freshBiasInner}", out["q"]
            assert out["freshBias"] == FRESH_BIAS_FUNCTION
            assert out["freshBiasInner"] == "{!hybrid lexical=$l vector=$v}"
            assert out["rows"] == 5, "untouched keys survive"
            assert "recip(max(0,ms(NOW,creation_date))" in FRESH_BIAS_FUNCTION
            return "q wrapped in {!boost}, inner query moved to $freshBiasInner"

        check("apply_fresh_bias wraps q in {!boost} with the recency function",
              t_fresh_bias)

        def t_build_context():
            docs = [
                {"id": "d1", "score": 1.0, "title": "Solar power",
                 "description": "Panels", "text": "one two three four five six"},
                {"id": "d2", "score": 0.9, "title": "Wind power",
                 "description": "Turbines", "text": "alpha beta gamma"},
                {"id": "d3", "score": 0.2, "title": "Unrelated",
                 "description": "Noise", "text": "should be dropped"},
            ]
            hl = {"d1": {"text": ["<em>Solar</em> panels cut bills"]}}
            ctx = build_context(docs, hl, top_n=4, max_words=4)
            assert ctx.count(DOC_FENCE) == 2, \
                "the 0.2-scored doc is below half the top score and must be dropped"
            assert "Unrelated" not in ctx
            assert "===== END OF DOCUMENT 2 =====" in ctx
            assert "MOST RELEVANT EXCERPTS:" in ctx
            assert "<em>" not in ctx and "</em>" not in ctx, "tags must be stripped"
            assert "Solar panels cut bills ..." in ctx, \
                "a fragment ending mid-sentence gets the trailing ellipsis"
            assert "one two three four" in ctx and "five" not in ctx, \
                "max_words=4 cuts after the fourth word"
            return "2 of 3 docs kept, tags stripped, fragment marked, body cut at 4 words"

        check("build_context filters weak hits, strips tags, cuts at max_words",
              t_build_context)

        def t_build_instruction():
            ctx = build_context(
                [{"id": "a", "score": 1.0, "title": "T", "description": "D",
                  "text": "body"}], {}, top_n=4, max_words=50)
            prompt = build_instruction(ctx, "who won?")
            assert prompt.startswith(DOC_FENCE), "documents come first"
            assert "Those were the 1 documents." in prompt
            assert prompt.rstrip().endswith("Answer:"), "question last"
            assert "Question: who won?" in prompt
            assert 'Never begin with "Based on" or "According to"' in prompt
            assert "There is no information about" in prompt
            empty = build_instruction("", "x")
            assert "Those were the 1 documents." in empty, \
                "document count never drops below 1"
            return "count=1, documents first, 'Question: … Answer:' last, bans present"

        check("build_instruction assembles the canonical prompt", t_build_instruction)

        # ------------------------------------------------------------------ #
        section("client — management API")
        # ------------------------------------------------------------------ #

        def t_index_list():
            lst = client.get_index_list()
            assert isinstance(lst, list) and lst, "expected a non-empty list"
            names = [r["index_name"] for r in lst]
            assert DEMO_INDEX in names, names
            state["index_list"] = names
            return "%d indexes, %s present" % (len(names), DEMO_INDEX)

        check("get_index_list returns this account's indexes", t_index_list)

        def t_core_info():
            info = client.get_core_info(DEMO_INDEX)
            url = info["connection_url"]
            assert url.startswith("https://"), url
            assert url.endswith("/solr/" + DEMO_INDEX), url
            assert info["auth_username"], "HTTP basic auth username missing"
            assert info["solr_version"].startswith("9."), info["solr_version"]
            assert info["environment_identifier"], info
            state["demo_url"] = url
            return "%s, Solr %s, env %s, auth user %r" % (
                url.split("/solr/")[0], info["solr_version"],
                info["environment_identifier"], info["auth_username"])

        check("get_core_info resolves the index's Solr endpoint + auth", t_core_info)

        def t_core_info_cache():
            before = len(_REQ_LOG["opensolr.com"])
            again = client.get_core_info(DEMO_INDEX)
            after = len(_REQ_LOG["opensolr.com"])
            assert after == before, "cached call must not hit the network"
            assert again["connection_url"] == state["demo_url"]
            return "no HTTP request on the second call"

        check("get_core_info caches per client", t_core_info_cache)

        def t_regions():
            regions = client.vector_regions()
            assert isinstance(regions, list) and regions
            envs = {r["environment"] for r in regions}
            assert "FINLAND9" in envs, envs
            for r in regions:
                assert r["country"] and r["solr_version"].startswith("9."), r
            state["regions"] = sorted(envs)
            return "%d regions: %s" % (len(envs), ", ".join(sorted(envs)))

        check("vector_regions lists live vector-enabled environments", t_regions)

        def t_bad_location():
            try:
                client.create_index("mcp_t_never_created__dense", "mars")
            except ValueError as exc:
                assert "not a vector-enabled" in str(exc), str(exc)
                return "ValueError raised before any network call"
            raise AssertionError("an unknown location must be rejected")

        check("create_index rejects a non-vector location", t_bad_location)

        def t_create():
            nonlocal_created = client.create_index(TMP_INDEX, TMP_LOCATION)
            # Flag the index for teardown the moment the API claims it exists —
            # before the assertions below — so a later failure cannot leak it.
            state["created"] = True
            assert isinstance(nonlocal_created, dict), nonlocal_created
            assert nonlocal_created.get("status") is True, nonlocal_created
            assert "CREATED" in str(nonlocal_created.get("msg", "")).upper(), \
                nonlocal_created
            # The API reads core ownership from a MySQL replica, so a lookup
            # issued in the same second as the insert can still answer
            # NOT_OWNER_ERROR. The package's own _ensure_index() retries for the
            # same reason; do the same here rather than race it.
            started = time.time()
            info = None
            while time.time() - started < 30:
                try:
                    info = client.get_core_info(TMP_INDEX, refresh=True)
                    break
                except OpensolrError as exc:
                    if "NOT_OWNER" not in str(exc):
                        raise
                    time.sleep(3)
            assert info is not None, \
                "index created but never became resolvable within 30s"
            assert info["connection_url"].endswith("/solr/" + TMP_INDEX)
            assert info["environment_identifier"] == \
                resolve_location(TMP_LOCATION), info
            return "%s on %s, msg=%s, resolvable after %.0fs" % (
                TMP_INDEX, info["environment_identifier"],
                nonlocal_created.get("msg"), time.time() - started)

        ok_create = check("create_index creates a vector index in a named region",
                          t_create)
        created = bool(state.get("created"))
        if not ok_create:
            print("\n✘ cannot continue without the temporary index — "
                  "the write-path checks are skipped.")
            raise SystemExit  # the finally block still runs cleanup

        def t_config():
            detail = apply_config_set(TMP_INDEX)
            base, auth = client.solr_endpoint(TMP_INDEX)
            sch = HARNESS.get(base + "/schema/fields", params={"wt": "json"},
                              auth=auth).json()
            fields = {f["name"]: f for f in sch["fields"]}
            assert fields["embeddings"]["type"] == "knn_vector", fields.get("embeddings")
            assert "creation_date" in fields and "text" in fields
            return "%s; embeddings is knn_vector, %d fields" % (detail, len(fields))

        check("[setup] Web Crawler config set applied (knn_vector schema)", t_config)

        # ------------------------------------------------------------------ #
        section("client — ingestion (async, kicked off now, verified later)")
        # ------------------------------------------------------------------ #

        INGEST_DOCS = [
            {"uri": "https://example.com/opensolr-haystack-test/aurora",
             "title": "Aurora borealis over Lapland",
             "description": "Green curtains of light above the Finnish arctic.",
             "text": "The aurora borealis is caused by charged particles from the "
                     "sun striking the upper atmosphere. Lapland offers some of "
                     "the clearest viewing in Europe during the winter months.",
             "timestamp": 1756400000},
            {"uri": "https://example.com/opensolr-haystack-test/sourdough",
             "title": "A beginner's guide to sourdough",
             "description": "Flour, water, salt and patience.",
             "text": "Sourdough bread rises with a wild yeast starter rather than "
                     "commercial yeast. The fermentation takes twelve to eighteen "
                     "hours and gives the loaf its characteristic sour flavour.",
             "timestamp": 1756400000},
            {"uri": "https://example.com/opensolr-haystack-test/harbour",
             "title": "Rebuilding the old harbour",
             "description": "A port town restores its nineteenth century quay.",
             "text": "The harbour restoration replaced rotted timber piles with "
                     "concrete caissons while keeping the original granite copings "
                     "that give the quay its appearance.",
             "timestamp": 1756400000},
        ]

        def t_ingest():
            body = client.ingest(TMP_INDEX, INGEST_DOCS, wait=False)
            assert body.get("status") is True, body
            assert body.get("msg") == "QUEUED", body
            assert body.get("total_docs") == 3, body
            job_id = body.get("job_id")
            assert isinstance(job_id, str) and len(job_id) == 32, job_id
            ids = body.get("doc_ids")
            import hashlib
            expect = [hashlib.md5(d["uri"].encode()).hexdigest() for d in INGEST_DOCS]
            assert ids == expect, "doc ids must be md5(uri): %s" % ids
            state["job_id"] = job_id
            return "job %s queued, 3 docs, ids = md5(uri)" % job_id[:12]

        check("ingest queues documents and returns a job id", t_ingest)

        def t_ingest_status():
            st = client.ingest_status(state["job_id"])
            assert st.get("status") is True, st
            job = st["job"]
            assert job["id"] == state["job_id"]
            assert job["core_name"] == TMP_INDEX, job
            assert int(job["total_docs"]) == 3, job
            assert job["state_label"] in (
                "pending", "processing", "completed"), job["state_label"]
            return "state=%s, total_docs=%s" % (job["state_label"], job["total_docs"])

        check("ingest_status reports the job by id", t_ingest_status)

        # ------------------------------------------------------------------ #
        section("client — AI API + direct Solr (read-only, demo index)")
        # ------------------------------------------------------------------ #

        def t_embed():
            vec = client.embed(DEMO_INDEX, "climate change in the Alps", is_query=True)
            assert isinstance(vec, list) and len(vec) == 1024, \
                "expected 1024 dims, got %d" % len(vec)
            assert all(isinstance(x, float) for x in vec)
            norm = math.sqrt(sum(x * x for x in vec))
            assert 0.9 < norm < 1.1, "expected an L2-normalised vector, |v|=%.3f" % norm
            state["qvec"] = vec
            return "1024 dims, all floats, |v|=%.4f" % norm

        check("embed returns a 1024-dim query vector", t_embed)

        def t_batch_embed():
            texts = ["a football match report",
                     "a recipe for lentil soup",
                     "a football match report"]
            vecs = client.batch_embed(DEMO_INDEX, texts)
            assert len(vecs) == 3, len(vecs)
            assert all(len(v) == 1024 for v in vecs), [len(v) for v in vecs]

            def cos(a, b):
                return sum(x * y for x, y in zip(a, b))

            same = cos(vecs[0], vecs[2])
            diff = cos(vecs[0], vecs[1])
            assert same > 0.99, "identical texts must embed identically (%.4f)" % same
            assert diff < same, "unrelated texts must be less similar (%.4f)" % diff
            assert BATCH_EMBED_MAX == 50
            return "3×1024 dims; cos(same)=%.4f > cos(different)=%.4f" % (same, diff)

        check("batch_embed embeds a list and preserves order", t_batch_embed)

        def t_endpoint():
            url, auth = client.solr_endpoint(DEMO_INDEX)
            assert url == state["demo_url"], url
            assert isinstance(auth, tuple) and len(auth) == 2, auth
            assert auth[0] and auth[1], auth
            return "%s with basic auth as %r" % (url.split("//")[1][:28], auth[0])

        check("solr_endpoint returns (url, basic-auth) for the index", t_endpoint)

        def t_select():
            body = client.solr_select(DEMO_INDEX, {"q": "*:*", "rows": 0})
            assert body["responseHeader"]["status"] == 0, body["responseHeader"]
            n = int(body["response"]["numFound"])
            assert n >= 100, "the seeded demo index should hold ~300 docs, saw %d" % n
            state["demo_docs"] = n
            # A facet gives us a real field VALUE to filter on further down.
            f = client.solr_select(DEMO_INDEX, {
                "q": "*:*", "rows": 0, "facet": "true",
                "facet.field": "meta_domain", "facet.limit": 1, "facet.mincount": 5})
            pairs = f["facet_counts"]["facet_fields"]["meta_domain"]
            state["domain"] = pairs[0]
            return "numFound=%d; top meta_domain=%r (%d docs)" % (
                n, pairs[0], pairs[1])

        check("solr_select runs a native Solr query on the index", t_select)

        def hybrid_case(mode, **kw):
            body = client.hybrid_search(DEMO_INDEX, "storms and flooding in Europe",
                                        rows=5, mode=mode, **kw)
            assert body["responseHeader"]["status"] == 0, body["responseHeader"]
            docs = body["response"]["docs"]
            assert docs, "no hits for mode=%s" % mode
            for d in docs:
                assert float(d["score"]) > 0, "hit without a score: %s" % d.get("id")
                assert d.get("title"), "hit without a title"
            return body, docs

        def t_hybrid_union():
            body, docs = hybrid_case("union")
            state["union_ids"] = [d["id"] for d in docs]
            state["union_found"] = int(body["response"]["numFound"])
            return "%d hits, top score %.4f, %r" % (
                len(docs), float(docs[0]["score"]), docs[0]["title"][:52])

        check("hybrid_search mode=union → fused BM25+kNN hits with scores",
              t_hybrid_union)

        def t_hybrid_kw():
            body, docs = hybrid_case("keywords_required")
            return "%d hits, top score %.4f, %r" % (
                len(docs), float(docs[0]["score"]), docs[0]["title"][:52])

        check("hybrid_search mode=keywords_required → lexical candidate set",
              t_hybrid_kw)

        def t_hybrid_meaning():
            body, docs = hybrid_case("meaning_required")
            return "%d hits, top score %.4f, %r" % (
                len(docs), float(docs[0]["score"]), docs[0]["title"][:52])

        check("hybrid_search mode=meaning_required → vector candidate set",
              t_hybrid_meaning)

        def t_hybrid_fq():
            domain = state["domain"]
            body, docs = hybrid_case("union", fq='meta_domain:"%s"' % domain)
            for d in docs:
                got = d.get("meta_domain")
                got = got[0] if isinstance(got, list) else got
                assert got == domain, "fq leaked a %r doc" % got
            return "%d hits, every one from %s" % (len(docs), domain)

        check("hybrid_search honours fq (every hit matches the filter)", t_hybrid_fq)

        def t_fresh_bias_live():
            off = client.hybrid_search(DEMO_INDEX, "storms and flooding in Europe",
                                       rows=5, fresh_bias=False)
            on = client.hybrid_search(DEMO_INDEX, "storms and flooding in Europe",
                                      rows=5, fresh_bias=True)
            n_off = int(off["response"]["numFound"])
            n_on = int(on["response"]["numFound"])
            assert n_off == n_on, \
                "fresh_bias must re-order, never filter: numFound %d → %d" % (
                    n_off, n_on)
            ids_off = {d["id"] for d in off["response"]["docs"]}
            ids_on = {d["id"] for d in on["response"]["docs"]}
            assert ids_off == ids_on, \
                "the same documents must remain reachable: %d vs %d in common" % (
                    len(ids_off), len(ids_off & ids_on))
            assert all(float(d["score"]) > 0 for d in on["response"]["docs"])
            return "numFound %d with bias off and on; same %d documents returned" % (
                n_off, len(ids_on))

        check("hybrid_search fresh_bias=True leaves numFound unchanged",
              t_fresh_bias_live)

        def t_eas():
            body = client.embed_and_search(DEMO_INDEX, "glacier collapse", rows=3)
            assert body.get("status") is True, str(body)[:200]
            res = body["results"]
            assert int(res["num"]) > 0, res["num"]
            docs = res["docs"]
            assert 0 < len(docs) <= 3, len(docs)
            for d in docs:
                assert d.get("score") is not None, "hit without a score"
                assert d.get("title"), "hit without a title"
            assert isinstance(res.get("hl"), dict) and res["hl"], \
                "highlight fragments are what the RAG context is built from"
            assert len(body.get("embeddings", [])) == 1024
            return "num=%s, %d docs, %d highlighted, query vector 1024-dim" % (
                res["num"], len(docs), len(res["hl"]))

        check("embed_and_search runs the platform's tuned pipeline", t_eas)

        def t_ai_summary():
            answer = client.ai_summary(
                DEMO_INDEX,
                "What do the articles say about glaciers and extreme weather?")
            assert isinstance(answer, str), type(answer)
            body = answer.lstrip("#*- \n\t")
            assert len(body) > 60, "answer too short to be grounded: %r" % answer[:120]
            low = body.lower()
            assert not low.startswith("based on"), \
                "the shipped instruction forbids opening with 'Based on': %r" % body[:90]
            assert not low.startswith("according to"), \
                "the shipped instruction forbids opening with 'According to': %r" % body[:90]
            assert not low.startswith("there is no information about"), \
                "retrieval returned nothing usable: %r" % body[:120]
            assert "glacier" in low, \
                "answer is not grounded in the retrieved documents: %r" % body[:120]
            state["answer"] = body
            return "%d chars, opens %r" % (len(body), body[:58])

        check("ai_summary returns a grounded answer obeying the shipped prompt",
              t_ai_summary)

        def t_ai_summary_not_fragile():
            # A question the corpus DEMONSTRABLY covers, so a refusal here would be a
            # wrong answer rather than an honest one.
            #
            # This assertion was wrong on 2026-08-29 and is worth the note: it used to ask
            # "why are glacier disasters a growing risk in the Alps?" while every retrieved
            # article was about Nepal and the Himalayas. The model answered "There is no
            # information about glacier disasters in the Alps; the documents discuss the
            # Himalayas" — which is exactly right, and exactly what the shipped instruction
            # asks for. The test was failing correct behaviour. When a refusal check fires,
            # check the CORPUS before blaming the model.
            question = "What did the Nepal flood reveal about glacier disaster risk?"
            answer = client.ai_summary(DEMO_INDEX, question).lstrip("#*- \n\t")
            if answer.lower().startswith("there is no information about"):
                hits = client.embed_and_search(DEMO_INDEX, question, rows=4)
                titles = [str(d.get("title"))[:60]
                          for d in hits.get("results", {}).get("docs", [])]
                raise AssertionError(
                    "refused a question its own retrieval covers. Answer: %r. "
                    "Retrieved: %s" % (answer[:160], titles))
            assert len(answer) > 60, "answer too short: %r" % answer[:120]
            return "%d chars, opens %r" % (len(answer), answer[:58])

        check("ai_summary answers a question the retrieved documents cover",
              t_ai_summary_not_fragile)

        def t_ai_summary_override():
            answer = client.ai_summary(
                DEMO_INDEX, "glaciers",
                instruction="Reply with exactly one word: OPENSOLR",
                rag_docs=1, rag_words=60)
            assert isinstance(answer, str) and answer.strip(), "empty override answer"
            assert "OPENSOLR" in answer.upper(), \
                "the caller's own instruction did not reach the model: %r" % answer[:120]
            return "custom instruction honoured: %r" % answer.strip()[:58]

        check("ai_summary honours a caller-supplied instruction override",
              t_ai_summary_override)

        # ------------------------------------------------------------------ #
        section("client — writes on the temporary index")
        # ------------------------------------------------------------------ #

        check("ingested documents become searchable (polled, not slept)",
              lambda: wait_for_docs(client, TMP_INDEX, 3, state.get("job_id")))

        def t_ingest_completed():
            st = client.ingest_status(state["job_id"])
            job = st["job"]
            assert job["state_label"] == "completed", job
            assert int(job["success_docs"]) == 3, job
            assert int(job["failed_docs"]) == 0, job
            return "state=completed, success=3, failed=0"

        check("ingest_status shows the job completed with 3 successes",
              t_ingest_completed)

        def t_ingested_content():
            body = client.solr_select(TMP_INDEX, {
                "q": 'title:aurora', "rows": 5, "fl": "id,title,text,creation_date"})
            docs = body["response"]["docs"]
            assert docs, "the ingested aurora document is not searchable"
            d = docs[0]
            assert "Aurora" in d["title"], d["title"]
            assert "charged particles" in str(d["text"]), d["text"][:80]
            assert d.get("creation_date"), \
                "timestamp should have produced a creation_date"
            # A {!knn} query is the only way to prove the vectors landed —
            # embeddings is stored=false, so it can never be read back. Reuses
            # the query vector embedded earlier, so this costs no API call.
            vec = json.dumps(state["qvec"], separators=(",", ":"))
            emb = client.solr_select(TMP_INDEX, {
                "q": "{!knn f=embeddings topK=10}" + vec, "rows": 10, "fl": "id"})
            assert emb["responseHeader"]["status"] == 0, emb["responseHeader"]
            n = int(emb["response"]["numFound"])
            assert n == 3, "server-side embeddings missing: kNN found %d of 3" % n
            return "title/text/creation_date stored; kNN reaches 3/3 docs"

        check("ingested documents keep their fields and got server-side embeddings",
              t_ingested_content)

        def t_solr_update():
            payload = {"add": {"doc": {
                "id": "hs_raw_update_1",
                "uri": "https://example.com/opensolr-haystack-test/raw",
                "title": "Raw solr_update probe",
                "description": "written through solr_update",
                "text": "this document was written by the raw update endpoint"}}}
            res = client.solr_update(TMP_INDEX, payload, commit=True)
            assert res["responseHeader"]["status"] == 0, res
            assert solr_count(client, TMP_INDEX) == 4, "document was not committed"
            res2 = client.solr_update(
                TMP_INDEX, {"delete": {"query": 'id:"hs_raw_update_1"'}}, commit=True)
            assert res2["responseHeader"]["status"] == 0, res2
            assert solr_count(client, TMP_INDEX) == 3, "delete was not committed"
            return "add → 4 docs, delete → 3 docs, both committed"

        check("solr_update adds and deletes through the native update handler",
              t_solr_update)

        def t_close():
            throwaway = OpensolrClient(EMAIL, API_KEY)
            assert throwaway._http.is_closed is False
            throwaway.close()
            assert throwaway._http.is_closed is True, "transport still open"
            return "underlying httpx transport reports is_closed=True"

        check("close releases the HTTP transport", t_close)

        # ------------------------------------------------------------------ #
        section("OpensolrDocumentStore")
        # ------------------------------------------------------------------ #

        # ingest_wait=False so this script owns the polling (and so the store's
        # own blocking loop cannot outrun the rate limit).
        store = OpensolrDocumentStore(index=TMP_INDEX, ingest_wait=False)
        _ = store.client  # force construction so the pacer can be attached
        instrument(store.client)

        def t_count():
            n = store.count_documents()
            assert n == 3, "expected the 3 ingested docs, got %d" % n
            return "count_documents() == 3"

        check("count_documents counts the live index", t_count)

        HS_DOCS = [
            Document(id="hs_doc_alpha",
                     content="Narwhals are arctic whales with a single long tusk "
                             "that is in fact an elongated canine tooth.",
                     meta={"category": "alpha", "source": "unit-test"}),
            Document(id="hs_doc_beta",
                     content="The espresso machine forces hot water through finely "
                             "ground coffee at nine bars of pressure.",
                     meta={"category": "beta", "source": "unit-test"}),
        ]

        def t_write_empty():
            assert store.write_documents([]) == 0, "empty write must be a no-op"
            return "write_documents([]) == 0, no request issued"

        check("write_documents short-circuits on an empty list", t_write_empty)

        def t_write():
            n = store.write_documents(HS_DOCS)
            assert n == 2, n
            return "2 Documents queued for ingestion"

        check("write_documents pushes Haystack Documents through ingestion", t_write)

        check("written Documents become searchable (polled, not slept)",
              lambda: wait_for_docs(client, TMP_INDEX, 5))

        def t_write_skip():
            n = store.write_documents(HS_DOCS, policy=DuplicatePolicy.SKIP)
            assert n == 0, "both ids already exist, expected 0 written, got %d" % n
            assert store.count_documents() == 5, "SKIP must not add anything"
            return "0 written, count still 5"

        check("write_documents DuplicatePolicy.SKIP skips existing ids", t_write_skip)

        def t_write_fail():
            try:
                store.write_documents(HS_DOCS, policy=DuplicatePolicy.FAIL)
            except DuplicateDocumentError as exc:
                assert "hs_doc_alpha" in str(exc), str(exc)
                return "DuplicateDocumentError naming hs_doc_alpha"
            raise AssertionError("DuplicatePolicy.FAIL must raise on existing ids")

        check("write_documents DuplicatePolicy.FAIL raises on existing ids",
              t_write_fail)

        def t_filter_all():
            docs = store.filter_documents()
            assert len(docs) == 5, len(docs)
            assert all(isinstance(d, Document) for d in docs)
            assert all(d.content for d in docs), "a Document came back with no content"
            by_id = {d.id: d for d in docs}
            assert "hs_doc_alpha" in by_id, sorted(by_id)
            assert "Narwhals" in by_id["hs_doc_alpha"].content
            assert by_id["hs_doc_alpha"].meta.get("category") == "alpha", \
                by_id["hs_doc_alpha"].meta
            return "5 Documents, ids and meta round-tripped (hs_doc_alpha → alpha)"

        check("filter_documents(None) returns every Document with content + meta",
              t_filter_all)

        def t_filter_cond():
            docs = store.filter_documents(
                {"field": "meta.category", "operator": "==", "value": "beta"})
            assert len(docs) == 1, [d.id for d in docs]
            assert docs[0].id == "hs_doc_beta", docs[0].id
            assert "espresso" in docs[0].content
            neg = store.filter_documents(
                {"field": "meta.source", "operator": "!=", "value": "unit-test"})
            assert all(d.id.startswith("hs_doc_") is False for d in neg), \
                "!= returned a unit-test document"
            grp = store.filter_documents({"operator": "AND", "conditions": [
                {"field": "meta.source", "operator": "==", "value": "unit-test"},
                {"field": "meta.category", "operator": "in", "value": ["alpha", "beta"]},
            ]})
            assert {d.id for d in grp} == {"hs_doc_alpha", "hs_doc_beta"}, \
                {d.id for d in grp}
            return "==→1 doc, !=→%d docs, AND+in→2 docs" % len(neg)

        check("filter_documents translates ==, !=, in and AND groups to fq",
              t_filter_cond)

        def t_delete():
            store.delete_documents(["hs_doc_alpha", "hs_doc_beta"])
            n = store.count_documents()
            assert n == 3, "expected 3 docs left after deleting 2, got %d" % n
            left = store.filter_documents()
            assert not [d for d in left if d.id.startswith("hs_doc_")], \
                "a deleted document is still there"
            store.delete_documents([])  # must be a silent no-op
            return "2 removed, 3 remain, delete_documents([]) is a no-op"

        check("delete_documents removes by Haystack id", t_delete)

        # -- read-only store over the seeded demo index ---------------------- #
        demo_store = OpensolrDocumentStore(index=DEMO_INDEX)
        _ = demo_store.client
        instrument(demo_store.client)

        def t_search_lexical():
            docs = demo_store.search("Bundesliga football", top_k=4, lexical=True)
            assert docs, "no lexical hits"
            assert len(docs) <= 4
            for d in docs:
                assert isinstance(d, Document)
                assert d.content, "hit with empty content"
                assert d.score and d.score > 0, "hit without a score"
            return "%d Documents, top score %.4f, %r" % (
                len(docs), docs[0].score, docs[0].content[:46].replace("\n", " "))

        check("store.search(lexical=True) → keyword hits with content and score",
              t_search_lexical)

        def t_search_hybrid():
            docs = demo_store.search("melting glaciers in the alps", top_k=4)
            assert docs, "no hybrid hits"
            for d in docs:
                assert d.content and d.score and d.score > 0
            state["hybrid_ids"] = [d.id for d in docs]
            return "%d Documents, top score %.4f, %r" % (
                len(docs), docs[0].score, docs[0].content[:46].replace("\n", " "))

        check("store.search(hybrid=True) → fused hits with content and score",
              t_search_hybrid)

        def t_search_knn():
            docs = demo_store.search("melting glaciers in the alps", top_k=4,
                                     hybrid=False)
            assert docs, "no kNN hits"
            for d in docs:
                assert d.content and d.score and d.score > 0
            return "%d Documents, top score %.4f, %r" % (
                len(docs), docs[0].score, docs[0].content[:46].replace("\n", " "))

        check("store.search(hybrid=False) → pure kNN hits with content and score",
              t_search_knn)

        def t_search_fresh():
            off = demo_store.search("european energy policy", top_k=5)
            on = demo_store.search("european energy policy", top_k=5, fresh_bias=True)
            assert len(off) == len(on), \
                "fresh_bias changed the hit count: %d → %d" % (len(off), len(on))
            assert {d.id for d in off} == {d.id for d in on}, \
                "fresh_bias dropped documents instead of re-ordering them"
            assert all(d.score and d.score > 0 for d in on)
            return "%d hits with the bias off and on, identical id set" % len(on)

        check("store.search(fresh_bias=True) re-orders without filtering",
              t_search_fresh)

        def t_ai_answer():
            answer = demo_store.ai_answer(
                "What are the main climate risks the articles describe?")
            body = answer.lstrip("#*- \n\t")
            assert len(body) > 60, "answer too short: %r" % answer[:120]
            low = body.lower()
            assert not low.startswith("based on"), body[:90]
            assert not low.startswith("according to"), body[:90]
            return "%d chars, opens %r" % (len(body), body[:58])

        check("store.ai_answer returns a grounded RAG answer", t_ai_answer)

        def t_store_serialization():
            data = demo_store.to_dict()
            assert data["type"].endswith("OpensolrDocumentStore"), data["type"]
            p = data["init_parameters"]
            assert p["index"] == DEMO_INDEX, p
            assert p["email"]["type"] == "env_var", p["email"]
            assert p["email"]["env_vars"] == ["OPENSOLR_EMAIL"], p["email"]
            assert p["api_key"]["env_vars"] == ["OPENSOLR_API_KEY"], p["api_key"]
            back = OpensolrDocumentStore.from_dict(json.loads(json.dumps(data)))
            assert isinstance(back, OpensolrDocumentStore)
            instrument(back.client)
            assert back.index == DEMO_INDEX
            assert back.location == demo_store.location
            assert back.create_if_missing == demo_store.create_if_missing
            assert isinstance(back.email, Secret)
            assert back.email.resolve_value() == EMAIL
            assert back.count_documents() == state["demo_docs"], \
                "the deserialized store must query the same index"
            return "round-tripped; deserialized store counts %d docs" % \
                state["demo_docs"]

        check("store.to_dict/from_dict round-trips and stays usable",
              t_store_serialization)

        def t_store_serialization_full():
            # Haystack's contract is that from_dict(to_dict(x)) reproduces x.
            # ingest_wait is a constructor parameter that changes behaviour
            # (blocking vs fire-and-forget writes), so it has to survive.
            src = OpensolrDocumentStore(index=TMP_INDEX, ingest_wait=False)
            back = OpensolrDocumentStore.from_dict(src.to_dict())
            assert back.ingest_wait == src.ingest_wait, (
                "ingest_wait is dropped by to_dict(): saved as %r, restored as %r"
                % (src.ingest_wait, back.ingest_wait))
            return "every constructor parameter survives the round-trip"

        check("store.to_dict carries every constructor parameter",
              t_store_serialization_full)

        # ------------------------------------------------------------------ #
        section("OpensolrHybridRetriever")
        # ------------------------------------------------------------------ #

        retriever = OpensolrHybridRetriever(document_store=demo_store, top_k=3)

        def t_retriever_run():
            out = retriever.run(query="wildfires and heatwaves")
            assert set(out) == {"documents"}, list(out)
            docs = out["documents"]
            assert docs, "retriever returned no documents"
            assert len(docs) <= 3, len(docs)
            for d in docs:
                assert isinstance(d, Document)
                assert d.content, "Document with empty content"
                assert isinstance(d.score, float) and d.score > 0, d.score
            return "%d Documents, scores %s" % (
                len(docs), ", ".join("%.4f" % d.score for d in docs))

        check("retriever.run returns scored Documents with content", t_retriever_run)

        def t_retriever_overrides():
            out = retriever.run(query="Bundesliga football", top_k=2, lexical=True)
            docs = out["documents"]
            assert 0 < len(docs) <= 2, len(docs)
            assert all(d.content and d.score for d in docs)
            return "per-call top_k=2 and lexical=True honoured (%d hits)" % len(docs)

        check("retriever.run honours per-call overrides", t_retriever_overrides)

        def t_retriever_serialization():
            data = retriever.to_dict()
            assert data["type"].endswith("OpensolrHybridRetriever"), data["type"]
            p = data["init_parameters"]
            assert p["top_k"] == 3 and p["hybrid"] is True and p["alpha"] == 0.5, p
            assert p["lexical"] is False and p["fresh_bias"] is False, p
            back = OpensolrHybridRetriever.from_dict(json.loads(json.dumps(data)))
            assert back.top_k == 3 and back.alpha == 0.5
            assert back.fresh_bias is False and back.lexical is False
            assert isinstance(back.document_store, OpensolrDocumentStore)
            assert back.document_store.index == DEMO_INDEX
            return "top_k/hybrid/alpha/lexical/fresh_bias + nested store all restored"

        check("retriever.to_dict/from_dict round-trips the component",
              t_retriever_serialization)

        def t_pipeline():
            pipe = Pipeline()
            pipe.add_component("retriever",
                               OpensolrHybridRetriever(document_store=demo_store,
                                                       top_k=2))
            result = pipe.run({"retriever": {"query": "renewable energy in Europe"}})
            docs = result["retriever"]["documents"]
            assert docs, "the pipeline produced no documents"
            assert all(d.content and d.score > 0 for d in docs)
            return "%d Documents out of a real Haystack Pipeline, top score %.4f" % (
                len(docs), docs[0].score)

        check("retriever works inside a Haystack Pipeline", t_pipeline)

    except SystemExit:
        pass
    except KeyboardInterrupt:
        print("\ninterrupted — cleaning up")
    finally:
        # ---------------------------------------------------------------- #
        # Teardown. Runs on success, on assertion failure, on an unexpected
        # exception and on Ctrl-C — the temporary index never survives a run.
        # ---------------------------------------------------------------- #
        section("teardown")
        if created:
            try:
                body = delete_index(TMP_INDEX)
                ok = isinstance(body, dict) and body.get("status") is True
                print("%s deleted temporary index %s%s" % (
                    "✔" if ok else "✘", TMP_INDEX,
                    "" if ok else " — %s" % str(body)[:200]))
                if not ok:
                    FAILED += 1
                    FAILURES.append("teardown — could not delete %s" % TMP_INDEX)
            except Exception as exc:  # noqa: BLE001
                FAILED += 1
                FAILURES.append("teardown — %s: %s" % (type(exc).__name__, exc))
                print("✘ could not delete temporary index %s — %s: %s"
                      % (TMP_INDEX, type(exc).__name__, exc))
        else:
            print("nothing to clean up (temporary index was never created)")
        try:
            client.close()
            HARNESS.close()
        except Exception:  # noqa: BLE001
            pass

    print()
    if FAILURES:
        print("failures:")
        for f in FAILURES:
            print("  ✘ %s" % f)
        print()
    print("%d passed, %d failed" % (PASSED, FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
