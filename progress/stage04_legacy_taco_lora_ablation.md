# Stage 04 progress

**Status:** completed and used in Appendix D.

- Full data 21,879; matched arms 14,043.
- Two backbones, nine arms, and seeds 20260603/20260604/20260605 completed under
  the legacy pipeline: 54 LoRA training combinations.
- Formal evaluation sources are components `03769--03771`; corrected HumanEval
  summaries must be preferred over earlier intermediate output.
- Qwen3-8B-Base showed substantial in-domain code degradation after SFT.
- The public reconstruction now exposes a portable 54-row training-matrix
  builder and a 56-row formal evaluation matrix (54 adapters plus two base-once
  rows), using only external `CG_*` roots.
- The multi-seed runner defaults to all three formal seeds and derives its final
  completion summary from the actual submitted model/arm/seed rows.
- Formal reconstructed evaluation is pinned to the shared 11-task `scoreable`
  contract; APPS-Hard and open-ended/diagnostic suites are not in the overall.
- No archived datasets, weights, responses, per-example IDs, or historical
  result blobs are committed. `metadata/stage04/` contains compact provenance
  and hash placeholders only.

Evidence: data component `03644`, checkpoint components `03624--03629`.
