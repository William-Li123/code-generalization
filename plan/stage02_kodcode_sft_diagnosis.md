# Stage 02 plan — KodCode SFT diagnosis

## Objective

Test whether half- and full-epoch LoRA-SFT effects depend on model family and
prompt contract while keeping the frozen 37,881 answer rows unchanged.

## Frozen design

- Four backbones; plain and native prompt modes.
- Seed 20260724.
- LoRA-SFT LR 2e-5, effective batch 16, one epoch, 4096-token strict limit.
- Save and report adapters at 0.5 and 1.0 epoch without test-based selection.
- Compare Base and adapters only under matching prompt modes.
- For the paper rows use Qwen2.5/native, Qwen3/plain, Llama/native, and
  Gemma/native; native rendering always supplies `enable_thinking=False`.

## Procedure

1. Verify the source hash and 37,881-row schema with `prepare_data.py`.
2. Audit all eight tokenizer/mode combinations with `audit_token_lengths.py`.
3. Train with `train_sft_lora.py`.
4. Evaluate the two formal adapters and matching Base on the current suite.
5. Summarize all reported checkpoints, including negative results.
6. Run `select_paper_rows.py` to emit exactly the eight Appendix-B SFT rows;
   fail when a mapped model/mode/checkpoint is missing or duplicated.
