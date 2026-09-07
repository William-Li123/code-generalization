# Evaluation contracts

## Current paper suite

`evaluation/paper_suite.py` evaluates the 11 scoreable tasks used by the main
experiment and Appendix B:

- ARC-Challenge, ScienceQA, LegalBench;
- GSM8K, MATH500-Medium, MATH500-High;
- FinQA, MedCalc;
- HumanEval, MBPP+, MBPP-Simple.

Generation uses temperature 0.2 and top-p 0.95. Code responses use a 2,048
token budget in the final current contract. Qwen thinking is disabled. Base
checkpoints are training-free but are evaluated under all three evaluation
seeds in Stage 01.

HealthBench, HealthBench Professional, PlanBench, and APPS-Hard are excluded
from the paper aggregate. Pass@8 is a later project experiment and is not part
of this release's paper results.

## Legacy suite

`evaluation/legacy_suite.py` preserves the older evaluator required to
reconstruct Stages 04--06. It is intentionally not silently replaced by the
current harness. The same Qwen2.5 base checkpoint differed by up to about 30
percentage points between historical and current task implementations, so
absolute scores must not be combined across the two suites.

## Safety

HumanEval and MBPP evaluation execute generated Python. The historical scripts
apply process timeouts and resource limits, but they are not a security
boundary. Run them in a disposable container or restricted worker with no
credentials and no writable access to valuable files.

## External test layout

Both harnesses expect a `data_test` root with `main_test`, `hard_test`, and
`diagnostic_test` subdirectories. Supply it with `--data-root`; the repository
does not redistribute benchmark data or hidden tests. The current harness also
requires a strict `manifest.json` created by `prepare_standard11.py`; it checks
the contract name, all eleven row counts, and each SHA256 before generation.
Exploratory evaluation on a different snapshot requires the explicit
`--allow-unverified-data` flag and must not be reported as the paper suite.

`prepare_standard11.py` builds and validates that external layout. Seven
historically normalized task files cannot be regenerated exactly from the
recovered code alone, so the command deliberately fails rather than changing
the calibration unless matching files are supplied with `--source-root`.
The expected identities are versioned in
`metadata/evaluation/paper_suite_reference.json`.

Resume protection is part of the evaluation contract. Existing metrics are
reused only when the resolved base/adapter paths, model, seed, prompt mode,
sampling settings, task list, and data-manifest hash match the requested run.
