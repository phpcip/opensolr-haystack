"""Opensolr DocumentStore for Haystack — managed Apache Solr with server-side
embeddings and native hybrid (BM25 + kNN) search."""

from haystack_integrations.document_stores.opensolr.store import OpensolrDocumentStore

__all__ = ["OpensolrDocumentStore"]
