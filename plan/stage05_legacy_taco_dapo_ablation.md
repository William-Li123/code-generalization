# Stage 05 plan — legacy TACO DAPO ablation

## Objective

Measure seven-category support under execution-reward DAPO and compare task
sensitivity ordering with Stage-04 LoRA-SFT.

## Frozen design

- Verified pool: 17,249 train + 100 shared validation problems.
- Eight matched 12,168-row arms: random control plus seven removals.
- Two complete-corpus references use archived seed 20260609; the 16 matched
  model/arm jobs use seed 20260610. The manuscript reports DAPO as seed
  20260610 and does not expose this complete-reference distinction.
- LoRA DAPO, LR 1e-6, eight rollouts, batch 32, one PPO epoch.
- Reward: 1.0 for full pass, otherwise 0.8 times test-pass fraction.

## Procedure

Build and freeze verified data and matched arms, apply the isolated VERL
compatibility patches, and generate the strict 18-job training manifest. Train
two complete references plus all 16 matched cells, select the best saved
nonzero step using validation `pass_frac`, and evaluate Base plus all trained
cells with `canonical_11_20260719`. Smoke/debug runs and test-based checkpoint
selection are excluded.
