# Stage 01 progress

**Status:** completed and used as the paper's main evidence.

- 35,974 train + 100 validation rows.
- Six reported backbones, 12 trained arms, three seeds: 216 adapters.
- Base plus trained conditions under three evaluation seeds: 234 result rows.
- Qwen2.5 aggregate values reconstructed from source metrics match the paper
  table exactly (Base 43.3529/66.2646; Full 46.2150/65.9490; matched control
  46.8014/67.3094 for transfer/in-domain aggregates).
- DeepSeek was excluded because archived code generations contain tokenizer
  decoding artifacts and are not comparable model-quality results.

Evidence: active Stage-01 namespaces and the 2026-08-13 AOSS snapshot listed in
`docs/ARCHIVE_PROVENANCE.md`.

