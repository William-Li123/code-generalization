# Stage 04 metadata

`reproducibility_manifest.json` is intentionally compact. It contains source
provenance, expected row counts, the public record schema, and hash placeholders
for externally rebuilt arm files. It contains no examples, sample identifiers,
dataset payloads, model artifacts, or evaluation results.

After rebuilding the nine JSONL files, replace each
`TO_BE_FILLED_AFTER_REBUILD` value with the SHA-256 of the exact file bytes.
Keep the large files under `CG_DATA_ROOT`; do not copy them into this directory.
