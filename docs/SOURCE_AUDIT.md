# Paper / active tree / AOSS source audit

This ledger records the final 2026-09-02 cross-check used to assemble the
public code tree. It distinguishes executable provenance from later exploratory
work and from large artifacts that must remain external.

| Stage | Paper evidence | Active `/data` evidence | Read-only `/mnt` evidence | Public target |
|---|---|---|---|---|
| 01 | Main text, Tables 1/2/5/6 and Figures 2--6, 10--12 | current arm builder, LoRA trainer, evaluator, result packager, verified v2 builder and taxonomy | `archive_20260813_full` snapshot | `stages/stage01_*`, current `evaluation/`, `analysis/`, `reference_results/stage01` |
| 02 | Appendix B SFT rows in Tables 3/4 and Figures 8/9 | current diagnostic utilities | legacy component 03664 (`diagnose_sft`) | `stages/stage02_*`, `reference_results/appendix_b/source/stage02_work` |
| 03 | Appendix B DAPO rows | current shared DAPO/runtime code | components 03661 (`diagnose_dapo`) and 03662 (`diagnose_dapo_full37881`) | `stages/stage03_*`, `dapo/`, remaining Appendix-B reference inputs |
| 04 | Appendix D LoRA ablation | later portable copies where present | data component 03644; checkpoints 03624--03629; evaluation 03769--03771 | `stages/stage04_*`, `analysis/package_legacy_results.py` |
| 05 | Appendix D DAPO ablation | later portable copies where present | data component 03649; checkpoints 03611--03619; evaluation 03890 | `stages/stage05_*`, `reference_results/legacy` |
| 06 | Appendix A full-SFT probe, Figure 7 | later portable copies where present | evaluation component 03962; scripts 03975; plans/progress 03969/03970 | `stages/stage06_*`, `reference_results/stage06` |

## File-level findings that changed the release

- The active August snapshot contains 50 post-archive additions and two changed
  main-stage scripts; those current Stage-01 versions are used. The additions
  are mostly candidate-model, public-code-data, and pass@8 experiments produced
  after the manuscript and are not relabelled as one of the six paper stages.
- The legacy archive contains a deterministic Gemma/native/final MBPP+ repair
  (`repair_gemma2_native_final_mbpp_plus.py`) and a template-neutral full-corpus
  exporter (`build_generic_dataset.py`) that were missing from the first clean
  draft. Both are now restored in Stages 02 and 03.
- Component 03962 proves that Stage-06 Figure 7 must be rebuilt from ten tasks.
  Its raw metrics include GSM8K n=250 and APPS, but the published aggregate
  excludes both. The public Stage-06 evaluator and packager enforce the ten-task
  no-GSM8K contract, and `reference_results/stage06` is derived from all 36
  archived final metrics rather than manuscript values.
- The five Stage-03 selection JSONs were checked directly. Full-corpus Qwen2.5
  selected step 397 on the frozen 100-row validation set; it is no longer left
  unresolved or inferred from checkpoint recency.
- Independent Appendix-B Base reruns differ slightly because evaluation used
  nonzero decoding temperature. Their hashes and drift are retained, while the
  paper's displayed DAPO deltas use the shared frozen Stage-02 Base.

## Deliberate exclusions

Models, checkpoints, optimizer states, raw datasets, benchmark records/hidden
tests, prompts, generations, grader traces, health-evaluation details, logs,
ACP metadata, and environment archives are not Git content. Their counts,
hashes, selection records, and aggregate scores are represented by compact
manifests where recoverable.

The release also excludes experiments that postdate or fall outside the
manuscript's six-stage chain: candidate-model screens, pass@8 reruns, public
code-data comparisons, DeepSeek replication attempts, and unrelated
`codeislogic_v1` or Magicoder/UltraChat GRPO directories. Exclusion means “not
claimed by this paper,” not “unimportant.”
