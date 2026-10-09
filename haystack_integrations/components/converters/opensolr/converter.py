"""Converter component: local files read on Opensolr's servers (text, OCR, a PDF page by page)."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.dataclasses import ByteStream

from haystack_integrations.document_stores.opensolr import OpensolrDocumentStore
from haystack_integrations.document_stores.opensolr.client import list_files

logger = logging.getLogger(__name__)


@component
class OpensolrConverter:
    """Turn files into Documents, read on Opensolr's servers.

    Each file is read there: its text, the text in its pictures (OCR, more than 100
    languages), a short description of the pictures that have no text, scanned pages
    and picture files (JPG, PNG, TIFF...) included. A PDF gives one Document per page
    (``meta["page_number"]``), any other file one Document. Nothing is indexed: put an
    embedder and a writer after it, or use ``OpensolrDocumentStore.write_files`` to
    index files directly.

    Example:
        ```python
        from haystack_integrations.document_stores.opensolr import OpensolrDocumentStore
        from haystack_integrations.components.converters.opensolr import OpensolrConverter

        store = OpensolrDocumentStore(index="mysite__dense")
        converter = OpensolrConverter(document_store=store)
        docs = converter.run(sources=["./contracts"])["documents"]
        ```

    The work is counted on the store's index: each picture read costs 0.5 of an AI
    request; a file this account sent before is read for free. ``sources`` takes file
    paths, folders (every file in them, sub-folders included) and ``ByteStream``s.
    Files that could not be read, or were still being read when ``timeout`` ran out,
    are logged and left out; run again to get the late ones (they are not sent twice).
    """

    #: Files sent, and waited for, together.
    BATCH = 20

    def __init__(self, document_store: OpensolrDocumentStore, timeout: float = 600.0) -> None:
        self.document_store = document_store
        self.timeout = timeout

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ) -> Dict[str, List[Document]]:
        """``meta``: added to every Document, one dict for all sources or one per source."""
        if isinstance(meta, list) and len(meta) != len(sources):
            raise ValueError("meta must be one dict, or one dict per source")
        jobs: List[tuple] = []
        temps: List[str] = []
        try:
            for i, source in enumerate(sources):
                extra = dict((meta[i] if isinstance(meta, list) else meta) or {})
                if isinstance(source, ByteStream):
                    name = Path(str(source.meta.get("file_path") or "document")).name
                    fd, tmp = tempfile.mkstemp(suffix="-" + name)
                    with os.fdopen(fd, "wb") as fh:
                        fh.write(source.data)
                    temps.append(tmp)
                    jobs.append((tmp, {**(source.meta or {}), "file_path": source.meta.get("file_path") or name, **extra}))
                else:
                    for f in list_files(source):
                        jobs.append((str(f), {"file_path": str(f), **extra}))
            documents: List[Document] = []
            client = self.document_store.client
            for i in range(0, len(jobs), self.BATCH):
                batch = jobs[i : i + self.BATCH]
                entries = client.extract_files(self.document_store.index, [p for p, _ in batch], pages=True, timeout=self.timeout)
                for entry, (_, base) in zip(entries, batch):
                    documents.extend(self._documents(entry, base))
            return {"documents": documents}
        finally:
            for tmp in temps:
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    @staticmethod
    def _documents(entry: Dict[str, Any], base: Dict[str, Any]) -> List[Document]:
        state = entry.get("state")
        if state != "done":
            logger.warning("Opensolr did not read %s (%s): %s", base.get("file_path"), state, entry.get("error") or "")
            return []
        base = {**base, "content_type": entry.get("mime") or ""}
        pages = entry.get("pages") or []
        if pages:
            return [
                Document(content=str(p.get("text") or ""), meta={
                    **base,
                    "page_number": int(p.get("page") or 0),
                    "is_ocr": bool(p.get("is_ocr")),
                    "language": str(p.get("lang") or ""),
                })
                for p in pages if str(p.get("text") or "").strip()
            ]
        text = str(entry.get("text") or "")
        return [Document(content=text, meta=base)] if text.strip() else []

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(self, document_store=self.document_store.to_dict(), timeout=self.timeout)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpensolrConverter":
        data["init_parameters"]["document_store"] = OpensolrDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)
