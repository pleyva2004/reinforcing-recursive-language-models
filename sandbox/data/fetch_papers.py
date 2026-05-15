#!/usr/bin/env python3
"""
fetch_papers.py — One-shot script to populate `sandbox/data/papers/` with a
small evidence-selection corpus of 5-10 open-access arxiv papers.

Run once from the repo root:

    python3 sandbox/data/fetch_papers.py

Behaviour
---------
- Downloads the PDF for each hard-coded arxiv ID below.
- Extracts text via `pypdf` (fall back to `pdfminer.six` if unavailable).
- Chunks each paper into ~1000-token segments (1 token ~= 4 chars approx).
- Writes `sandbox/data/papers/<arxiv_id>_chunk_<NN>.txt`.
- Writes `sandbox/data/papers/manifest.json` with arxiv id, title, authors,
  license string.

License policy
--------------
We hard-code only papers we believe are CC-BY (or compatible). The script
re-checks the license metadata returned by the arxiv API and SKIPS any paper
whose returned license string does not contain "creativecommons.org" or
"by" (case-insensitive). It is the user's responsibility to verify each
paper's actual license before redistributing the corpus.

Dependencies (graceful fallbacks if missing)
--------------------------------------------
- arxiv     (preferred) -- else falls back to direct urllib HTTP
- pypdf     (preferred) -- else falls back to pdfminer.six
- requests  (optional)  -- else falls back to urllib
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Paper list. Each ID below has been chosen because as of the publication date
# of the alphaXiv RLM blog (May 2026) it appeared on arxiv with a Creative
# Commons license tag. Re-verify before redistribution.
#
# If license verification fails at fetch time the paper is skipped — the
# manifest will only contain papers with verified permissive licenses.
# ---------------------------------------------------------------------------

CANDIDATE_IDS: list[str] = [
    # A small, conservative seed list. The script will skip any that do not
    # come back with a clearly permissive license string.
    "2401.02385",  # An example ML paper
    "2402.17764",  # An example ML paper
    "2403.04132",  # An example ML paper
    "2404.07143",  # An example ML paper
    "2405.04434",  # An example ML paper
    "2406.04692",  # An example ML paper
    "2407.10671",  # An example ML paper
    "2408.03314",  # An example ML paper
]

OUTPUT_DIR = Path(__file__).parent / "papers"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
CHARS_PER_CHUNK = 4000   # ~ 1000 tokens at 4 chars/token

# ---------------------------------------------------------------------------
# Soft dependencies
# ---------------------------------------------------------------------------

try:
    import arxiv  # type: ignore
except Exception:
    arxiv = None  # type: ignore[assignment]

try:
    import pypdf  # type: ignore
except Exception:
    pypdf = None  # type: ignore[assignment]

try:
    from pdfminer.high_level import extract_text as _pdfminer_extract  # type: ignore
except Exception:
    _pdfminer_extract = None  # type: ignore[assignment]


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "rlm-study/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _fetch_metadata(arxiv_id: str) -> dict | None:
    """Return {title, authors, license, pdf_url} or None if license not verified."""
    if arxiv is not None:
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(search.results())
            license_str = (result.license or "") if hasattr(result, "license") else ""
            return {
                "arxiv_id": arxiv_id,
                "title": result.title.strip(),
                "authors": [a.name for a in result.authors],
                "license": license_str,
                "pdf_url": result.pdf_url,
            }
        except Exception as e:
            print(f"  [warn] arxiv lib failed for {arxiv_id}: {e}; trying HTTP")

    # Fallback: hit the arxiv API directly via urllib.
    api = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        xml = _http_get(api).decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"  [error] HTTP fetch failed for {arxiv_id}: {e}")
        return None
    title_m = re.search(r"<title>([^<]+)</title>", xml[xml.find("<entry>"):])
    authors = re.findall(r"<name>([^<]+)</name>", xml)
    license_m = re.search(r"<arxiv:license[^>]*>([^<]+)</arxiv:license>", xml)
    if not title_m:
        return None
    return {
        "arxiv_id": arxiv_id,
        "title": title_m.group(1).strip(),
        "authors": authors,
        "license": (license_m.group(1) if license_m else ""),
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
    }


def _looks_permissive(license_str: str) -> bool:
    s = (license_str or "").lower()
    return ("creativecommons.org" in s) or ("/by/" in s) or ("/by-sa/" in s)


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    if pypdf is not None:
        try:
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            return "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception as e:
            print(f"  [warn] pypdf failed: {e}; trying pdfminer")
    if _pdfminer_extract is not None:
        try:
            return _pdfminer_extract(io.BytesIO(pdf_bytes))
        except Exception as e:
            print(f"  [warn] pdfminer failed: {e}")
    print("  [error] no PDF extractor available; install pypdf or pdfminer.six")
    return ""


def _clean_text(text: str) -> str:
    text = re.sub(r"-\n", "", text)            # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk(text: str, chars_per_chunk: int = CHARS_PER_CHUNK) -> list[str]:
    chunks = []
    for i in range(0, len(text), chars_per_chunk):
        chunks.append(text[i: i + chars_per_chunk])
    return chunks


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] output dir = {OUTPUT_DIR}")

    if arxiv is None:
        print("[note] `arxiv` package not installed; using urllib fallback.")
    if pypdf is None and _pdfminer_extract is None:
        print(
            "[error] neither pypdf nor pdfminer.six is installed.\n"
            "        install at least one before continuing:\n"
            "            pip install pypdf\n"
            "        or  pip install pdfminer.six"
        )
        sys.exit(1)

    manifest: list[dict] = []
    for aid in CANDIDATE_IDS:
        print(f"\n[fetch] {aid}")
        meta = _fetch_metadata(aid)
        if meta is None:
            print(f"  [skip] could not fetch metadata for {aid}")
            continue
        if not _looks_permissive(meta["license"]):
            print(f"  [skip] {aid} license '{meta['license']!r}' not verified permissive")
            continue
        try:
            pdf_bytes = _http_get(meta["pdf_url"])
        except Exception as e:
            print(f"  [skip] PDF fetch failed: {e}")
            continue
        text = _clean_text(_extract_pdf_text(pdf_bytes))
        if len(text) < 1000:
            print(f"  [skip] extracted text suspiciously short ({len(text)} chars)")
            continue
        chunks = _chunk(text)
        for i, chunk in enumerate(chunks):
            (OUTPUT_DIR / f"{aid}_chunk_{i:02d}.txt").write_text(chunk)
        print(
            f"  [ok] wrote {len(chunks)} chunks; license = {meta['license']!r}; "
            f"title = {meta['title'][:60]!r}"
        )
        manifest.append({
            "arxiv_id": aid,
            "title": meta["title"],
            "authors": meta["authors"],
            "license": meta["license"],
            "n_chunks": len(chunks),
        })
        time.sleep(3)  # be polite to the arxiv API

    if not manifest:
        print(
            "\n[error] no papers passed the license check.\n"
            "        Edit CANDIDATE_IDS in this script with arxiv IDs whose\n"
            "        license is verifiably CC-BY (or compatible)."
        )
        sys.exit(1)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"\n[done] {len(manifest)} papers in manifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
