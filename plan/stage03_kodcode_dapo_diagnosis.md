# Stage 03 plan — KodCode DAPO diagnosis

## Objective

Compare execution-reward DAPO with the Stage-02 SFT diagnosis without claiming
that their training sets are identical.

## Frozen design

- RL10K branch: four backbones, 9,501 train + 100 validation problems.
- Full branch: Qwen2.5 only, 35,974 train + 100 validation problems.
- Seed 20260725; eight rollouts; train/generation batch 32; LR 1e-6.
- LoRA rank/alpha 16/32; one PPO epoch; no KL or length reward.
- Strict binary execution reward: 1 only when all tests pass.
- Runtime LoRA dropout is 0.0: the archived invocation omitted the field and
  therefore used PEFT's default. The paper prose value 0.05 is not substituted.

## Procedure

1. Verify the frozen source SHA-256/count and build each external branch.
2. Iterate prepare → exact-reward preflight → cumulative reject merge →
   rebuild until a zero-failure pass, then validate generated file/ID hashes
   and atomically write `.data_ready`.
3. Train each RL10K model serially and record the frozen adapter-selection rule.
4. Run the full branch only for Qwen2.5.
5. Evaluate selected adapters and matching Base with the current suite.
6. Rebuild Tables 3/4 and Figures 8/9 only from the strict raw-metric mapping;
   reject incomplete 100-row adapter selections or inconsistent Base metrics.
