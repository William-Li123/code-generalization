# DAPO / VERL compatibility contract

The archived DAPO jobs used `verl==0.8.0` with the external recipe repository
at Git commit `e0f4dc2ccdd51e8e445a683b828090577c625534`.  The recipe repository and its
large environment archive are deliberately not vendored here.

`runtime_patches/` contains the small compatibility modules recovered from the
canonical AOSS archive. They cover:

- attention implementation selection;
- model-specific chat-template handling;
- accelerator/device normalization;
- worker construction and LoRA adapter loading;
- FSDP transformer integration.

The historical launcher copied these files over five exact VERL modules and
made one controlled vLLM LoRA-path edit. Merely adding `sitecustomize/` to
`PYTHONPATH` does **not** install those module replacements. The clean runner
therefore calls `apply_runtime_patches.py` against the explicitly configured,
job-local `CG_VERL_ENV`, backs up each original once, verifies all source and
installed SHA256 hashes, and checks the external recipe Git commit. It refuses
an environment whose `site-packages` is not inside that directory.

`sitecustomize/sitecustomize.py` separately supplies the recovered filelock and
tokenizer compatibility guards. Both mechanisms are required by
`run_training.sh`.

To audit an already prepared isolated environment without changing it:

```bash
python dapo/apply_runtime_patches.py check \
  --environment "$CG_VERL_ENV" \
  --recipe "$CG_VERL_RECIPE_DIR"
```

`CG_VERL_PATCH_ACTION=check` makes the training runner verify rather than
install. `CG_ALLOW_UNVERIFIED_VERL=1` exists only for explicitly exploratory
runs; outputs from it are not historical reproductions.

These patches are historical compatibility code, not a claim that every newer
VERL release requires the same changes. Changing the VERL revision, Ray
version, attention backend, or chat-template behavior changes the experiment
contract and should be recorded as a new run rather than silently mixed with
the archived results.
