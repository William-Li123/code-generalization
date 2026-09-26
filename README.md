# Code-to-Reasoning Transfer Experiments

Code for **Why Does Code Improve Reasoning? On Code-to-Reasoning Transfer in
LLM Post-Training Through a Decompositional Lens**.

This checkout contains implementation code, configuration templates and unit
tests only. Datasets, weights, run records, results, figures and retrospective
analyses are not distributed here.

## Experimental stages

| Stage | Method | Data family |
|---|---|---|
| 01 | Category-controlled LoRA SFT | Labeled code problems |
| 02 | LoRA SFT and validation-based checkpoint selection | Code question–answer pairs |
| 03 | DAPO reward and dataset adapters | Code problems with executable tests |
| 04 | LoRA SFT and category data preparation | TACO and LeetCode-style problems |
| 05 | Execution verification and DAPO | TACO and LeetCode-style problems |
| 06 | Full-parameter SFT | Code question–answer pairs |

## Layout

```text
configs/                 editable, path-only and hyperparameter examples
dapo/                    VERL runner and version-checked compatibility patches
evaluation/              separate current and legacy evaluation harnesses
experiments/category_sft/ portable category sampling and multi-model job runner
stages/                  reusable preparation, training and reward implementations
tests/                   CPU-only unit and repository checks
```

## Setup

Use Linux with CUDA for training. Install [Pixi](https://pixi.sh/), then run
`pixi install` for the SFT/evaluation environment. The dependency specification
is a setup starting point, not a claim of an exact historical runtime. No
environment or model checkpoint is bundled.

Copy `.env.example` to a local `.env` and export the relevant variables, or
provide paths through command-line arguments. Keep all data and outputs outside
the checkout. Models must already be available locally.

```bash
pixi run test
pixi run validate-repo
pixi run compile
```

## Category-controlled LoRA SFT

The portable runner supports whole-category training, minimum-size matched
single categories, balanced or proportional mixtures, random controls and
leave-category-out arms. Selected mixtures accept any caller-supplied category
list, including two or three categories. No model-specific best-category
ranking is embedded.

1. Provide a JSONL or Parquet training file with `problem_id`,
   `primary_category`, `prompt` and `response`.
2. Copy `configs/category_recipe.example.json` outside the checkout and replace
   the placeholder categories with labels from your data.
3. Prepare the arms in a new external directory:

```bash
python -m experiments.category_sft.build_data \
  --source "$TRAIN_FILE" --validation "$VALIDATION_FILE" \
  --recipe "$RECIPE_FILE" --output-dir "$PREPARED_DIR" \
  --seed "${SAMPLING_RANDOM_STATE:?set explicitly}"
```

`per_category: "minimum"` uses the smallest category in the source pool;
`"all"` retains the selected category in full. A numeric `per_category`
uses that many examples from each selected category. Alternatively, `total`
specifies a total mixture budget. For matched single and mixture arms, shared
category rankings reuse the same examples. The builder rejects duplicate IDs,
train/validation overlap, insufficient capacity and existing output directories.
Omitting `--validation` also omits the overlap check.

Copy `configs/category_run.example.json` to an external location and set your
model directories, learning rates, prepared arm names and evaluation-data root.
Its numerical values are configurable defaults, not a record of a completed run.
All model entries require an explicit training prompt mode. Check the plan first:

```bash
python -m experiments.category_sft.run \
  --config "$RUN_CONFIG" --manifest "$PREPARED_DIR/manifest.json" \
  --output-dir "$RUN_OUTPUT" \
  --seeds "${TRAIN_RANDOM_STATE:?set explicitly}" \
  --eval-seed "${EVAL_RANDOM_STATE:?set explicitly}" \
  --template-date "${TEMPLATE_DATE:?use DD Mon YYYY}" --include-base
```

Add `--execute` only after checking the plan and environment. Each configured
GPU runs one independent job at a time; a free lane takes the next model/arm.
The example exposes four lanes without using four-rank DDP for a single arm.
This command runs inside an already allocated machine or cluster job; it does
not create an ACP allocation.

Native templates receive the same caller-supplied date during training and
evaluation. The Transformers resume-tail compatibility patch applies only to its
explicitly supported version and only within the running Python process.
Completed outputs are checked before reuse. Incomplete training checkpoints
are rejected, and partial evaluation is not silently resumed or overwritten.
Runtime manifests are written only to the external output directory.

## Evaluation warning

`evaluation/paper_suite.py` and `evaluation/legacy_suite.py` remain separate.
Prompt construction, answer extraction and scoring are retained in their
respective cores. Do not mix their scores as if they were one protocol.
The current suite's GSM8K generation budget can be supplied independently using
`--max-new-tokens-gsm8k`; its default remains unchanged.

Use `evaluation/prepare_standard11.py --help` for the required benchmark
layout and external source inputs. Some normalized source files must be
provided by the caller; this repository does not distribute them.

The portable category runner requires bubblewrap, libseccomp, a Linux x86-64
system Python 3.10, and a matching NumPy package directory for isolated code
execution. Set `CG_BWRAP` and `CG_SANDBOX_PACKAGES` if their locations differ
from the defaults. Run `python -m experiments.category_sft.sandbox_runner`
to verify isolation before GPU work. It fails closed when unavailable.
Direct legacy evaluation, verification and reward scripts execute supplied code:
run those only inside a separately isolated, disposable environment.

## DAPO and other stages

Stage scripts expose their own command-line arguments through `--help`.
DAPO additionally needs a separately provisioned VERL recipe and compatible
runtime. `dapo/apply_runtime_patches.py --help` describes the version-checked
patch interface. Configure the external paths in `.env.example` and supply
runtime randomness explicitly before using `dapo/run_training.sh`.
Patch application is opt-in; upstream license notices are retained.

## Release status

No results or analysis claims are supplied with this code-only checkout.
Dataset and model access remain subject to their upstream terms. A repository-wide
source-code license has not yet been selected.
