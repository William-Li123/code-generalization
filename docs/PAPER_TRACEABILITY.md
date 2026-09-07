# Paper-to-code traceability

This ledger maps every empirical table/figure in the manuscript to its public
input contract and executable entry point. Figure 1 is conceptual and has no
experiment artifact.

## Main six-backbone experiment

| Paper item | Reproduction source |
|---|---|
| Table 1, ten categories | `stages/stage01_main_kodcode_lora_ablation/taxonomy.json` and `metadata/stage01/arm_manifest.json` |
| Table 2, two-family arm scores | `analysis/make_overall_table.py` (`overall_table.*`, `family_tradeoff_vs_base.csv`) |
| Figure 2, three structural results | `analysis/make_headline_figure.py` (`headline_figure.*`) |
| Figure 3, transfer/in-domain quadrants | `analysis/make_paper_figures.py` (`in_domain_vs_transfer_tradeoff.*`) |
| Figure 4, per-backbone support maps | `analysis/make_paper_figures.py` (`support_map_by_backbone.*`) |
| Figure 5, raw support null | `analysis/make_extra_figures.py` (`raw_support_permutation_null.*`) |
| Figure 6, two-family practitioner map | `analysis/make_extra_figures.py` (`practitioner_group_support_map.*`) |
| Tables 5/6, screened support cells | `analysis/make_support_table.py` (`support_table_top_residuals.*`, `support_table_bias_absorbed.*`) |
| Figure 10, pooled map and residual | `analysis/make_residual_null.py` (`pooled_support_and_residual.*`) |
| Figure 11, task-wise net effect | `analysis/make_paper_figures.py` (`net_effect_by_task.*`) |
| Figure 12, per-arm delta heatmaps | `analysis/make_paper_figures.py` (`arm_delta_vs_base_by_backbone.*`) |

`analysis/reproduce_paper.py` invokes every main-paper entry point in one run.
The included `reference_results/stage01` package is sufficient for this CPU
analysis reproduction. A full GPU rerun begins with the strict Stage-01 driver
and its frozen data/evaluation manifests.

## Appendix A and Appendix B

| Paper item | Reproduction source |
|---|---|
| Figure 7, full-parameter SFT | `stages/stage06_legacy_full_sft_robustness/package_appendix_a.py`; exact ten-task no-GSM8K contract and `reference_results/stage06` |
| Tables 3/4, KodCode diagnosis | `stages/stage03_kodcode_dapo_diagnosis/make_appendix_b_outputs.py` |
| Figures 8/9, diagnosis heatmaps | same Appendix-B entry point |

The Appendix-B entry point joins the Stage-02 SFT and two Stage-03 DAPO
branches only after checking prompt mode, the 11-task mean, every Base-replicate
hash/drift, 100-row validation selection, and the rounded manuscript values.
The displayed DAPO deltas use the shared frozen Stage-02 Base, matching Tables
3/4; independent nonzero-temperature Base reruns remain audit evidence. The
included `reference_results/appendix_b` package makes this CPU analysis
self-contained.

## Legacy Appendix-D experiment

| Paper item | Reproduction source |
|---|---|
| Table 7, harness calibration | `analysis/make_calibration_table.py` after the main and legacy packages exist |
| Figures 13/15, legacy SFT delta/absolute | `analysis/package_legacy_results.py` |
| Figures 14/16, legacy DAPO delta/absolute | same legacy package entry point |
| Figure 17, DAPO support map | same legacy package entry point |
| Figure 18, SFT/DAPO sensitivity agreement | same legacy package entry point |

Stages 04 and 05 now provide explicit training/evaluation matrices. The legacy
packager rejects the obsolete 250-row GSM8K result unless its smoke-only flag
is supplied; formal output requires the canonical 1,319-row result.
`reference_results/legacy` provides the released aggregate tables for a
raw-generation-free rebuild.

## Evaluation-suite table and audit trail

Table 8 is fixed by `metadata/evaluation/paper_suite_reference.json`,
`evaluation/prepare_standard11.py`, and the strict manifest validation in both
evaluation drivers. Counts and SHA256 values identify the benchmark snapshot;
benchmark records and hidden tests remain external.

The six manuscript-named main-analysis scripts were not present in the active
tree or either archive. Their public implementations are equation-based
reconstructions sharing `analysis/paper_artifacts.py`; known numerical
differences from manuscript Tables 5/6 are recorded in
`docs/REPRODUCIBILITY.md` and emitted in `null_statistics.json`.
