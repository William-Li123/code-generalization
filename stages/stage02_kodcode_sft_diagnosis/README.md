# Stage 02 — KodCode SFT diagnosis

`run_all.sh` is the portable top-level driver for the Appendix-B LoRA-SFT
diagnosis. It requires the external roots documented in `.env.example`, audits
plain/native prompt contracts, trains the fixed 0.5/1.0-epoch reporting
checkpoints, evaluates them, and writes results outside the repository.

Native tokenization and evaluation both pass `enable_thinking=False`. The full
plain/native sweep is retained for diagnosis; `select_paper_rows.py` selects
only Qwen2.5/native, Qwen3/plain, Llama/native, and Gemma/native at the frozen
half/final checkpoints for the Appendix-B rows.

The actual archived source contains 37,881 answer rows (36,446 unique original
questions). This stage must not be silently pointed at the later 35,974-row
main split. See the matching plan, progress record, and config before running.

From the repository root, configure all external roots and run the complete
prepare → prompt/length audit → 8 LoRA trainings → 24 evaluations → paper-row
selection chain:

```bash
export CG_DATA_ROOT=/external/code-generalization-data
export CG_DATA_TEST_ROOT=/external/paper-standard11
export CG_MODEL_ROOT=/external/models
export CG_OUTPUT_ROOT=/external/outputs
export CG_WORK_ROOT=/external/scratch

bash stages/stage02_kodcode_sft_diagnosis/run_all.sh
```

The driver accepts either two or four visible GPUs. On four GPUs it runs two
independent two-GPU DDP lanes; the effective batch remains 16 per training.
The source JSONL and Parquet are both required and checked against the hashes
in `configs/stage02_kodcode_sft_diagnosis.json`. Results, checkpoints, token
audits, and generated CSVs stay under the external output root.

The archive also contains a one-off deterministic recovery for the formal
Gemma/native/final MBPP+ cell. The original evaluator had generated all 378
answers but its child process could not load `libcblas.so.3`; therefore
`repair_gemma2_native_final_mbpp_plus.py` re-executes the stored generations
without regenerating or changing decoding. Its audit and pre-repair backups
must remain with the external result tree.
