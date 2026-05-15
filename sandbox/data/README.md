# sandbox/data — evidence-selection corpus

Small set of open-access arxiv papers used by `real_rlm_lora.py` as the
evidence-selection environment. Each episode pairs a synthesized research
question with one gold paper and two distractors; the model must (a) pick
the correct paper, (b) extract the relevant passage.

## Provenance

The corpus is **not committed to this repository**. To populate it locally:

```bash
pip install arxiv pypdf
python3 sandbox/data/fetch_papers.py
```

This will:

1. Hit the arxiv API for each ID in `CANDIDATE_IDS`.
2. Skip any paper whose returned license string is not clearly Creative
   Commons (`creativecommons.org` / `by` / `by-sa`).
3. Download the PDF, extract text via `pypdf` (or `pdfminer.six`), chunk it
   into ~1000-token segments.
4. Write `papers/<arxiv_id>_chunk_<NN>.txt` and a top-level
   `papers/manifest.json` with the verified metadata.

## License notes

Re-verify every license string in `papers/manifest.json` before
redistributing the corpus or any derivative artifacts. arxiv's license
metadata can be incomplete or wrong; if in doubt, drop the paper.

The curated `CANDIDATE_IDS` list in `fetch_papers.py` is intentionally
small (~8 IDs) and deliberately conservative: it is the seed, not the
authority. The script will only keep papers whose license string passes
`_looks_permissive`.

## Per-paper licensing table

After running `fetch_papers.py`, this table is reflected in
`papers/manifest.json`. Example shape:

| arxiv id     | title (abridged)             | license                                      |
|--------------|------------------------------|----------------------------------------------|
| 24XX.XXXXX   | Foo Bar: A Method            | http://creativecommons.org/licenses/by/4.0/  |
| 24YY.YYYYY   | Baz Quux: Another Result     | http://creativecommons.org/licenses/by-sa/4.0/|

(Run the fetch script to populate.)

## Refreshing the corpus

Edit `CANDIDATE_IDS` in `fetch_papers.py`, delete `papers/*.txt` and
`papers/manifest.json`, then re-run the script.
