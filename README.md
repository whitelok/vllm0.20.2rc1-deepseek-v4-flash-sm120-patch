# vLLM × DeepSeek-V4-Flash on SM120 (Blackwell Workstation)

> 📖 **[中文版本 / Chinese Version](./README_cn.md)**

A complete, end-to-end deployment guide and patch set that enables **DeepSeek-V4-Flash** inference on **NVIDIA RTX PRO 5000 72 GB (SM120 / consumer Blackwell)** using **vLLM 0.20.2rc1.dev246**.

**Base image:** `vllm/vllm-openai:cu129-nightly-28ee78af543c563a2fbf78829a7688120e4e4eb5`

---

## Table of Contents

- [TL;DR](#tldr--the-whole-flow-at-a-glance)
- [1. Background & Hardware Constraints](#1-background--hardware-constraints)
- [2. Repository Layout](#2-repository-layout)
- [3. Launch Command](#3-launch-command-full-vllm-serve)
- [4. Phase-by-Phase Walkthrough](#4-phase-by-phase-walkthrough)
- [5. Troubleshooting](#5-troubleshooting)
- [6. Pushing `max-model-len` to 262144](#6-advanced-pushing-max-model-len-to-262144)
- [7. Per-Patch Operations](#7-per-patch-operations-fine-grained-control)
- [8. Emergency Rollback Playbook](#8-full-rollback-playbook-emergency)
- [9. Design Philosophy & Trade-offs](#9-design-philosophy--known-trade-offs)
- [10. Quick-Reference Cheat Sheet](#10-quick-reference-cheat-sheet)

---

## TL;DR — the whole flow at a glance

```bash
# Run inside the remote vllm container:
cd ${PROJECT_ROOT}     # assumes the repo has been uploaded

# Phase 0 — environment pre-flight check
./scripts/apply_all.sh check

# Phase 1 — run local mock tests (validate patch math, modifies nothing)
./scripts/apply_all.sh selftest

# Phase 2 — one-shot apply of all 5 patches
./scripts/apply_all.sh apply

# Phase 3 — export the env-var block printed by Phase 2, then `vllm serve`
#           (full command in §3)

# Phase 4 — health check + progressive smoke test
./scripts/smoke_test.sh

# Rollback (emergency)
./scripts/apply_all.sh revert
```

---

## 1. Background & Hardware Constraints

### 1.1 Target Hardware

| Item        | Value |
|-------------|-------|
| GPU         | NVIDIA RTX PRO 5000 Blackwell × 4 (72 GB per card) |
| SM version  | **SM120 / 12.0** (consumer Blackwell — distinct from SM100 = B100/B200/GB200) |
| CUDA        | 12.9 |
| PyTorch     | 2.11.0 + cu129 |
| Python      | 3.12 |

### 1.2 Model & Framework

- **Model:** DeepSeek-V4-Flash (this model is a hard requirement — it cannot be swapped out)
- **vLLM:** 0.20.2rc1.dev246 (stock pip install; resolves to `/usr/local/lib/python3.12/dist-packages/vllm/`)

### 1.3 Why are these patches needed?

DeepSeek-V4-Flash relies on a suite of CUDA kernels that only run on **SM90a (H100)** and **SM100f (B100/B200/GB200 data-center Blackwell)**:

| Kernel | Provider | Hardware requirement | Symptom on SM120 |
|---|---|---|---|
| `fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits` | DeepGEMM | tcgen05 (SM100) | `NotImplementedError` / compile failure |
| `cutlass_scaled_mm` (UE8M0 scale) | CUTLASS  | SM90a/SM100 mixed-dtype matmul | `RuntimeError` |
| `fp8_einsum` (DeepGEMM) | DeepGEMM | tcgen05 | `NotImplementedError` |
| `flash_mla_with_kvcache` (sparse decode) | FlashMLA | SM90a/SM100f | `RuntimeError: Sparse Attention Decode Kernel is only supported on SM90a and SM100f architectures` |
| `flash_mla_sparse_fwd` (sparse prefill) | FlashMLA | SM90a/SM100f | `RuntimeError: Sparse Attention Forward Kernel is only supported on SM90a and SM100f architectures` |

The five patches each wrap the corresponding kernel with an **env-var-gated, pure-PyTorch fallback**. The performance hit is roughly 5–20× (acceptable — the goal is "runs at all").

---

## 2. Repository Layout

```
vllm-deepseek-v4-flash-sm120/
├── README.md                                      # this document (English)
├── README_cn.md                                   # Chinese version
├── patches/                                       # 5 patches; apply order is enforced by apply_all.sh
│   ├── indexer_mqa_logits_sm120_fallback.py       # #1 → vllm/utils/deep_gemm.py
│   ├── cutlass_scaled_mm_ue8m0_fallback_v3.py     # #2 → vllm/_custom_ops.py
│   ├── fp8_einsum_sm120_fallback.py               # #3 → vllm/utils/deep_gemm.py
│   ├── sparse_decode_fwd_sm120_fallback.py        # #4 → flash_mla_interface.py
│   └── sparse_prefill_fwd_sm120_fallback.py       # #5 → flash_mla_interface.py
├── tests/                                         # local mock unit tests (CPU or GPU)
│   ├── test_sparse_decode_fallback.py             # decode: 10 cases
│   └── test_sparse_prefill_fallback.py            # prefill v4.1: 12 cases
└── scripts/                                       # deploy / verify helpers (path-aware, run from anywhere)
    ├── apply_all.sh                               # recommended: one-shot apply/revert/check/selftest for all 5
    ├── deploy_sparse_decode.sh                    # single-patch deploy (fine-grained)
    ├── deploy_sparse_prefill.sh                   # single-patch deploy (includes v4 → v4.1 redeploy)
    └── smoke_test.sh                              # 7-step progressive HTTP smoke test
```

### 2.1 Patch dependency & conflict analysis

| # | Patch | Target file | Backup suffix | Env switch(es) |
|---|---|---|---|---|
| 1 | indexer_mqa_logits          | `vllm/utils/deep_gemm.py` | `.bak.indexer_sm120` | `VLLM_HC_FALLBACK=1`, `VLLM_INDEXER_FALLBACK=1` |
| 2 | cutlass_scaled_mm_ue8m0 v3  | `vllm/_custom_ops.py` | `.bak_v3` | `VLLM_UE8M0_FALLBACK=1` |
| 3 | fp8_einsum                  | `vllm/utils/deep_gemm.py` | `.bak_fp8einsum` | `VLLM_FP8_EINSUM_FALLBACK=1` |
| 4 | sparse_decode v3            | `vllm/third_party/flashmla/flash_mla_interface.py` | `.bak_sparse_decode_fb` | `VLLM_SPARSE_DECODE_FALLBACK=1` |
| 5 | sparse_prefill v4.1         | same as above | `.bak_sparse_prefill_fb` | `VLLM_SPARSE_PREFILL_FALLBACK=1` |

> **Important:** Patches 1 & 3 both modify `deep_gemm.py`; patches 4 & 5 both modify `flash_mla_interface.py`. Each pair uses fully independent backup suffixes and sentinel comments, so they can be reverted individually without affecting one another. **However, apply order must be 1 → 2 → 3 → 4 → 5** (each patch snapshots the current target file as its backup source).

### 2.2 How each patch works (shared template)

All five patches follow the same recipe:

1. **Backup.** On apply, copy the target file to a uniquely suffixed backup (e.g. `flash_mla_interface.py.bak_sparse_prefill_fb`).
2. **Inject helpers block.** Insert the fallback function bodies as a string literal at module scope, fenced by paired sentinels `# SM120_xxx_BEGIN` / `# SM120_xxx_END`.
3. **Inject dispatch swap.** Prepend an env-var branch immediately before the original CUDA kernel call site:
   ```python
   if _SM120_xxx_FALLBACK_ENABLED:
       return _sm120_xxx_fallback(...)
   # original CUDA kernel call
   ```
4. **Quadruple self-check.** Immediately after apply (any failure triggers automatic revert):
   - `ast.parse` syntax check passes
   - dispatch was injected at the correct indentation
   - clear `__pycache__/*.pyc` to prevent stale bytecode
   - `inspect.getsource` live-verifies the marker is actually present in the runtime function body
5. **Env var defaults to OFF.** With no env var set the patch is dormant and the original kernel is still called.

---

## 3. Launch Command (full `vllm serve`)

After all patches are applied, you must export **the entire env-var block below** before running `vllm serve`. Missing any one of them will fall back to the original SM90/SM100 kernel:

```bash
# ── SM120 fallback switches ───────────────────────────────────────
export VLLM_HC_FALLBACK=1                  # patch 1 (indexer HC)
export VLLM_INDEXER_FALLBACK=1             # patch 1 (indexer mqa_logits)
export VLLM_UE8M0_FALLBACK=1               # patch 2 (ue8m0 scaled mm)
export VLLM_FP8_EINSUM_FALLBACK=1          # patch 3 (fp8 einsum)
export VLLM_FP8_EINSUM_FALLBACK_DEBUG=1    #   debug: print 1 log line on first forward
export VLLM_USE_DEEP_GEMM=0                # disable DeepGEMM path (SM100-only)
export VLLM_FUSED_MOE_BACKEND=triton       # MoE uses triton (not Marlin)
export VLLM_SPARSE_DECODE_FALLBACK=1       # patch 4 (sparse decode)
export VLLM_SPARSE_DECODE_FALLBACK_DEBUG=1
export VLLM_SPARSE_PREFILL_FALLBACK=1      # patch 5 (sparse prefill v4.1)
export VLLM_SPARSE_PREFILL_FALLBACK_DEBUG=1

# ── Optional performance tuning ───────────────────────────────────
# Reduce prefill fallback chunk size → lowers per-chunk peak memory (default 256)
# export VLLM_SPARSE_PREFILL_FALLBACK_CHUNK=128
# Mitigate PyTorch CUDA allocator fragmentation
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Start vllm serve ──────────────────────────────────────────────
vllm serve /workspace/models/DeepSeek-V4-Flash \
  --served-model-name deepseek-v4-flash \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-expert-parallel \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --port 8081 \
  --enable-prompt-tokens-details \
  --enforce-eager
```

**Flag notes:**
- `--enforce-eager` — **required**: CUDA Graph would capture incorrect tensor addresses on the fallback path.
- `--kv-cache-dtype fp8` — KV cache in fp8 to save memory (the fallback handles dequant).
- `--tensor-parallel-size 4` — saturate all 4 cards.
- `--max-model-len 32768` — conservative starting point; pushing to 262144 requires step-by-step validation (see §6).

---

## 4. Phase-by-Phase Walkthrough

### Phase 0 — Pre-flight check (read-only, no file changes)

```bash
cd /vllm-workspace/vllm-deepseek-v4-flash-sm120
./scripts/apply_all.sh check
```

**Expected output (key lines):**
```
==> repo root: /vllm-workspace/vllm-deepseek-v4-flash-sm120
==> checking presence of patch files in patches/:
  [OK]   indexer_mqa_logits_sm120_fallback.py   (15071 bytes)
  [OK]   cutlass_scaled_mm_ue8m0_fallback_v3.py   (11584 bytes)
  [OK]   fp8_einsum_sm120_fallback.py   (22549 bytes)
  [OK]   sparse_decode_fwd_sm120_fallback.py   (26140 bytes)
  [OK]   sparse_prefill_fwd_sm120_fallback.py   (26458 bytes)

==> target file presence:
  [OK]   /usr/local/lib/python3.12/dist-packages/vllm/utils/deep_gemm.py
  [OK]   /usr/local/lib/python3.12/dist-packages/vllm/_custom_ops.py
  [OK]   /usr/local/lib/python3.12/dist-packages/vllm/third_party/flashmla/flash_mla_interface.py
```

**On failure:**
- `[MISS] patches/xxx.py` → incomplete upload; `scp` the entire `vllm-deepseek-v4-flash-sm120/` directory again.
- `[MISS] /usr/local/.../vllm/...` → vLLM is installed elsewhere. Run `python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))"` to find the real path, then update the `TARGET = Path(...)` at the top of each of the 5 patch files.

### Phase 1 — Local mock tests (validate math, vllm untouched)

```bash
./scripts/apply_all.sh selftest
```

**Expected output:**
```
==> mock test: sparse decode fallback (10 cases)
... 10 PASS ...

==> mock test: sparse prefill fallback v4.1 (12 cases)
[PASS] 01 basic_smoke
[PASS] 02 matches_reference
[PASS] 03 with_attn_sink
[PASS] 04 invalid_indices_negative
[PASS] 05 invalid_indices_overflow
[PASS] 06 topk_length
[PASS] 07 topk_length_zero
[PASS] 08 preallocated_out
[PASS] 09 attn_sink_with_invalid
[PASS] 10 shape_assertions
[PASS] 11 chunk_equivalence            ← v4.1 new: chunk size ∈ {1,17,256,600,9999} bit-equivalent
[PASS] 12 chunk_with_invalid_and_sink  ← v4.1 new: chunk boundary + invalid + sink joint correctness

==== 12/12 PASS, 0 FAIL ====
```

**On failure:**
- `chunk_equivalence FAIL` → a patch file was corrupted; re-pull from the repo.
- `matches_reference FAIL` → torch version mismatch; confirm torch 2.11.0 + cu129.

Phase 1 must be fully green before Phase 2.

### Phase 2 — Apply all 5 patches

```bash
./scripts/apply_all.sh apply
```

**Expected output (each patch prints these three sections):**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[1/5] indexer_mqa_logits_sm120_fallback.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[CHECK] patch NOT applied
[APPLY] backup -> .../deep_gemm.py.bak.indexer_sm120
[APPLY] injected SM120_HC_FALLBACK helpers + dispatch
[OK] self-check: ast.parse PASSED
[OK] self-check: dispatch at correct indent
[OK] self-check: live import + inspect.getsource verified
[CHECK] patch APPLIED

... [2/5] ... [3/5] ... [4/5] ... [5/5] ...

==> all 5 patches applied. Byte-compiling targets...
OK: all target files compile

==============================================================================
ADD THESE ENV VARS TO YOUR `vllm serve` COMMAND, then restart vllm:
==============================================================================
export VLLM_HC_FALLBACK=1
...
```

**On failure (depends on where it fails):**
- `[SKIP] patch already applied` → previously applied; either continue, or `./scripts/apply_all.sh revert` and re-apply.
- `self-check ast.parse FAILED` → rare; the patch auto-reverts itself. Inspect stderr for the exact `SyntaxError` — usually a vllm version mismatch (the patch uses regex anchors; missing anchors cause incorrect insertion).
- `live import check skipped: ImportError` → not fatal; the vllm module simply hasn't been imported yet. The patch is already on disk; proceed to Phase 3.

### Phase 3 — Start vllm

Copy the env-var block printed at the end of Phase 2 (or copy the full command from §3) and launch:

```bash
# Run the full export ... + vllm serve ... block from §3
```

**Expected vllm startup log lines** (you should see all six fallback markers, proving the env vars took effect):
```
[SM120_HC_FALLBACK] engaged
[SM120_INDEXER_FALLBACK] engaged
[SM120_UE8M0_FALLBACK] engaged ...
[SM120_FP8_EINSUM_FALLBACK] engaged ...
[SM120_SPARSE_DECODE_FALLBACK] engaged ...
[SM120_SPARSE_PREFILL_FALLBACK] engaged: q=... chunk=256
```

> **Note:** the two `[SM120_SPARSE_*_FALLBACK] engaged` markers are printed lazily — on the **first inference request**, not at vllm startup. The other four markers appear during the capture/profile phase.

**Common startup failures:**

| Error | Cause | Fix |
|---|---|---|
| `RuntimeError: Sparse Attention Decode Kernel is only supported on SM90a and SM100f` | `VLLM_SPARSE_DECODE_FALLBACK` not exported | Re-export the full env block |
| `RuntimeError: ... DeepGEMM ...` | `VLLM_USE_DEEP_GEMM=0` not set | Same as above |
| `ImportError: cannot import _flashmla_C` | flash_mla C++ extension not built / wheel-installed | Reinstall vllm or build from source |
| `CUDA out of memory` during startup | TP=4 has insufficient memory for weights | Verify `nvidia-smi` shows all 4 cards free; model is ~140 GB / 4 = 35 GB per card, plus KV cache ~50 GB per card; 72 GB is enough |

### Phase 4 — Smoke test (end-to-end validation)

```bash
./scripts/smoke_test.sh
```

7-step progressive test:

| Step | Action | Expected |
|---|---|---|
| 1 | `GET /health` | HTTP 200 |
| 2 | `GET /v1/models` | JSON listing `deepseek-v4-flash`, `max_model_len=32768` |
| 3 | `/v1/completions` — generate 1 token | "Hello" → any non-empty token (validates the decode patch) |
| 4 | `/v1/chat/completions` — generate 16 tokens | Short answer like "1 + 1 = 2." (first time the prefill path is exercised) |
| 5 | `/v1/chat/completions` — generate 256 tokens | 200+ token long answer; record decode tok/s |
| 6 | SSE streaming chat completions | Continuous streamed output (a `curl (23) Failure writing output` is stdout blocking, **not** a server error) |
| 7 | 4 k prompt single inference | **Critical v4.1 test**: full prompt processing + answer, no OOM |

**If Step 7 fails (memory pressure):**
1. `export VLLM_SPARSE_PREFILL_FALLBACK_CHUNK=128` (default 256 → 128, halve again)
2. `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
3. Restart vllm
4. If still OOM, temporarily reduce `--max-model-len 32768` to `16384` to see if KV cache is the culprit

---

## 5. Troubleshooting

### 5.1 Patch deployment

#### Error: `[SKIP] patch already (partially) applied`
**Cause:** a previous apply attempt left partial sentinels behind.
**Fix:**
```bash
# revert a single patch
python3 patches/<patch_name>.py --revert
# or revert everything in one shot
./scripts/apply_all.sh revert
# then re-apply
./scripts/apply_all.sh apply
```

#### Error: `SyntaxError` / `IndentationError` on vllm startup after apply
**Cause:** rare — likely a vllm version mismatch causing the regex anchor to find the wrong line.
**Fix:**
```bash
# 1. Immediately revert all patches
./scripts/apply_all.sh revert
# 2. Manually compare against backups
ls -la /usr/local/lib/python3.12/dist-packages/vllm/utils/deep_gemm.py*
# 3. Manually `cp` the oldest .bak.* back into place
# 4. File an issue including the vllm version and Python version
```

#### Error: apply succeeds but vllm behavior unchanged (fallback not engaged)
**Cause:** stale `__pycache__/*.pyc` files.
**Fix:**
```bash
# Patches normally clean these automatically; do it again to be safe:
find /usr/local/lib/python3.12/dist-packages/vllm -name '*.pyc' -delete
# Then restart vllm
```

### 5.2 Inference

#### Error: `RuntimeError: ... SM90a and SM100f` during chat/completions
**Root cause:** the corresponding env var was not exported, or was exported *after* `vllm serve`.
**Fix:**
```bash
# In the same shell session: export first, vllm serve after
# Verify vllm actually saw the env:
python3 -c "import os; print({k:v for k,v in os.environ.items() if k.startswith('VLLM_')})"
```

#### Error: Step 7 OOM
**Root cause:** the default `chunk=256` sparse-prefill fallback peaks at ~260 MiB per layer; 60 layers + KV cache can squeeze memory.
**Fix (in priority order):**
```bash
# 1. Lower chunk size
export VLLM_SPARSE_PREFILL_FALLBACK_CHUNK=128
# 2. Enable expandable segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 3. Lower max-model-len
# Change --max-model-len in the launch command from 32768 to 16384
# 4. Restart vllm
```

#### Error: decode output is gibberish / repeated tokens
**Root cause:** rare — potentially a bug in the sparse-decode fallback's fp8 dequant path.
**Diagnosis:**
```bash
# Look for SM120_SPARSE_DECODE_FALLBACK debug lines in worker logs:
# expect: "engaged: q=(B, S, H, D=512) cache=..."
grep SM120 /path/to/vllm.log | head -20
# If values are wrong, temporarily revert just the decode patch to see if the original kernel also errors
python3 patches/sparse_decode_fwd_sm120_fallback.py --revert
# Restart vllm — expect the SM90a error (confirming the env is genuinely SM120),
# then re-apply and file an issue with the worker log attached.
```

### 5.3 Performance

#### Symptom: prefill is slow (> 5 s for 1 k tokens)
**Expected:** the fallback is 5–20× slower than native SM100 kernels. Prefill of 1 k tokens may take 2–5 s; 4 k may take 10–30 s. This is the documented trade-off.

**Light optimization:**
- `VLLM_SPARSE_PREFILL_FALLBACK_CHUNK=512` — when memory permits, larger chunks reduce launch overhead.
- The first inference after startup pays the cuDNN benchmark cost; subsequent ones are stable.

---

## 6. Advanced: Pushing `max-model-len` to 262144

The default guide uses `--max-model-len 32768` as a safe value. To push to 262144 (DSV4-Flash's design ceiling):

```bash
# 1. Make sure chunk=128 + expandable_segments are already on
export VLLM_SPARSE_PREFILL_FALLBACK_CHUNK=128
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. Adjust launch flags. Note: with fp8 KV cache, each token is ~256 B/layer × 60 layers ≈ 15 KB;
#    a single 262 144-token sequence is ~4 GB of KV cache, so concurrency must drop.
#    --max-model-len 262144
#    --max-num-seqs 1                      # drop concurrency to 1 for long-context
#    --gpu-memory-utilization 0.92         # leave room for KV cache

# 3. After restart, validate progressively: 16 k → 64 k → 128 k → 262 144 prompts
```

**Known risk:** MARLIN MoE may become the next SM120 hot-spot under long prompts (high probability, not yet validated). If so, add a 6th patch using the same template.

---

## 7. Per-Patch Operations (fine-grained control)

If you only want to apply/revert a single patch (e.g. while iterating on the v4 → v4.1 upgrade):

```bash
# Drive the patch CLI directly
python3 patches/sparse_prefill_fwd_sm120_fallback.py --check
python3 patches/sparse_prefill_fwd_sm120_fallback.py --apply
python3 patches/sparse_prefill_fwd_sm120_fallback.py --revert

# Or use the deploy wrapper (adds byte-compile + decode-v3 dependency check)
./scripts/deploy_sparse_prefill.sh check
./scripts/deploy_sparse_prefill.sh selftest
./scripts/deploy_sparse_prefill.sh apply
./scripts/deploy_sparse_prefill.sh redeploy   # = revert + clear pycache + apply
```

Each patch file is **self-contained**: it includes a full docstring (background, error symptoms, fix rationale), the helpers string, the dispatch-injection logic, the quadruple self-check, and the apply/revert/check CLI. You can `cp` any patch file to another environment and use it on its own.

---

## 8. Full Rollback Playbook (emergency)

If the patched vllm misbehaves and you need a clean state:

```bash
# Revert all 5 patches in reverse order (5 → 4 → 3 → 2 → 1)
./scripts/apply_all.sh revert

# Clear bytecode cache
find /usr/local/lib/python3.12/dist-packages/vllm -name '*.pyc' -delete

# Verify all backups were restored
./scripts/apply_all.sh check
# Expected: every patch shows "NOT applied"

# Restart vllm without the SM120 env vars — you should hit the original
# SM90/SM100 errors, proving the revert is clean.
```

If even revert fails (worst case — backups deleted), reinstall from a clean vllm wheel:
```bash
pip install --force-reinstall vllm==0.20.2rc1.dev246
```

---

## 9. Design Philosophy & Known Trade-offs

### 9.1 Why pure-PyTorch fallback instead of a triton rewrite?
- **Correctness first.** A pure-PyTorch op is the reference implementation — unit tests are easy to write.
- **Readability.** 60 lines of PyTorch ≪ 600 lines of triton; debuggable when patches misbehave.
- **Coverage.** All 5 kernels use the same template — uniform maintenance.

### 9.2 Approximate performance loss

| Path | Native SM90/SM100 | SM120 fallback | Ratio |
|---|---|---|---|
| sparse decode      | 0.5 ms        | ~3–5 ms          | 6–10× slower  |
| sparse prefill     | 2 ms / 1 k tok | 20–50 ms / 1 k tok | 10–25× slower |
| ue8m0 scaled mm    | < 1 ms        | ~5 ms            | 5× slower     |
| fp8 einsum         | 1 ms          | 10–20 ms         | 10–20× slower |

**Result:** end-to-end DSV4-Flash throughput on SM120 is roughly 1/10 of SM100 — but it runs (which is the user's hard constraint: the model cannot be swapped).

### 9.3 v4.1 vs v4 (prefill patch history)
- **v4 problem.** fp32 `K_f` upcast + a single bmm over the entire `Sq` peaked at 4.2 GiB per layer; 4 k prompts OOM'd.
- **v4.1 fix.** Native bf16 bmm (only logits are promoted to fp32 for numerical stability) + a streaming chunk loop over `Sq` (default 256), bringing the per-chunk peak down to ~260 MiB (16× improvement).
- **Mathematical equivalence.** Chunks are independent (per-query), and `test_chunk_equivalence` uses `torch.equal` to verify outputs are bit-equivalent across chunk sizes ∈ {1, 17, 256, 600, 9999}.

---

## 10. Quick-Reference Cheat Sheet

```bash
# repo
cd /vllm-workspace/vllm-deepseek-v4-flash-sm120

# status
./scripts/apply_all.sh check

# math validation
./scripts/apply_all.sh selftest

# full deploy
./scripts/apply_all.sh apply
# then export env vars + vllm serve as printed

# validation
./scripts/smoke_test.sh

# single-patch ops
python3 patches/<name>.py --{check,apply,revert}

# prefill v4 → v4.1 upgrade
./scripts/deploy_sparse_prefill.sh redeploy

# emergency rollback
./scripts/apply_all.sh revert
find /usr/local/lib/python3.12/dist-packages/vllm -name '*.pyc' -delete
```

---

## Version Information

This guide tracks the following patch set:

- indexer_mqa_logits — **v1**
- cutlass_scaled_mm_ue8m0 — **v3**
- fp8_einsum — **v2**
- sparse_decode — **v3**
- sparse_prefill — **v4.1** (latest, includes OOM fix)

## Contributing / Feedback

If you discover a new SM120 hot-spot (e.g. MARLIN MoE, kvcache routing, etc.), add a new patch following the same template. The most complete reference template is `patches/sparse_prefill_fwd_sm120_fallback.py` — `cp` and rename to bootstrap.

## License

This patch set is intended for research and deployment use on consumer Blackwell hardware. Refer to upstream vLLM and DeepSeek-V4-Flash licenses for the underlying components.
