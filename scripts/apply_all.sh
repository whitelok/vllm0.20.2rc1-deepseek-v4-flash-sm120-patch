#!/usr/bin/env bash
# apply_all.sh — one-shot installer for all 5 SM120 fallback patches.
# ===================================================================
#
# Applies the patches in dependency order on a fresh
# vllm 0.20.2rc1.dev246 install.  Stops on first failure.
#
# Order matters:
#   1. indexer_mqa_logits_sm120_fallback.py     → vllm/utils/deep_gemm.py
#   2. cutlass_scaled_mm_ue8m0_fallback_v3.py   → vllm/_custom_ops.py
#   3. fp8_einsum_sm120_fallback.py             → vllm/utils/deep_gemm.py
#      (patches 1 & 3 touch the same file with separate backups & sentinels)
#   4. sparse_decode_fwd_sm120_fallback.py      → flash_mla_interface.py
#   5. sparse_prefill_fwd_sm120_fallback.py     → flash_mla_interface.py
#      (patches 4 & 5 touch the same file with separate backups & sentinels)
#
# Each individual patch has its own --apply / --revert / --check CLI.
# If a single patch fails, you can either:
#   - re-run apply_all.sh after fixing the issue (each patch is idempotent
#     and refuses to double-apply); OR
#   - drop down to per-patch deploy scripts in this directory.
#
# Usage:
#   ./apply_all.sh check       # show status of all 5 patches without changing anything
#   ./apply_all.sh apply       # apply all 5 patches in order
#   ./apply_all.sh revert      # revert all 5 patches in REVERSE order
#   ./apply_all.sh selftest    # run the two python mock test suites (decode + prefill)
#
# After 'apply', RESTART vllm with the env-var block printed by this
# script (or copy from USAGE.md → 'phase 3: start vllm').

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PATCH_DIR="$REPO_ROOT/patches"
TEST_DIR="$REPO_ROOT/tests"

# patches in dependency-apply order
PATCHES=(
  "indexer_mqa_logits_sm120_fallback.py"
  "cutlass_scaled_mm_ue8m0_fallback_v3.py"
  "fp8_einsum_sm120_fallback.py"
  "sparse_decode_fwd_sm120_fallback.py"
  "sparse_prefill_fwd_sm120_fallback.py"
)

DEEP_GEMM="/usr/local/lib/python3.12/dist-packages/vllm/utils/deep_gemm.py"
CUSTOM_OPS="/usr/local/lib/python3.12/dist-packages/vllm/_custom_ops.py"
FLASH_MLA="/usr/local/lib/python3.12/dist-packages/vllm/third_party/flashmla/flash_mla_interface.py"

cmd="${1:-}"

print_env_block() {
  cat <<'ENV'
==============================================================================
ADD THESE ENV VARS TO YOUR `vllm serve` COMMAND, then restart vllm:
==============================================================================
export VLLM_HC_FALLBACK=1                  # patch 1 (indexer HC)
export VLLM_INDEXER_FALLBACK=1             # patch 1 (indexer mqa_logits)
export VLLM_UE8M0_FALLBACK=1               # patch 2 (ue8m0 scaled mm)
export VLLM_FP8_EINSUM_FALLBACK=1          # patch 3 (fp8 einsum)
export VLLM_FP8_EINSUM_FALLBACK_DEBUG=1    #   (debug first-fire log)
export VLLM_USE_DEEP_GEMM=0                # disable DeepGEMM (SM100-only)
export VLLM_FUSED_MOE_BACKEND=triton       # triton MoE (not Marlin)
export VLLM_SPARSE_DECODE_FALLBACK=1       # patch 4 (sparse decode)
export VLLM_SPARSE_DECODE_FALLBACK_DEBUG=1
export VLLM_SPARSE_PREFILL_FALLBACK=1      # patch 5 (sparse prefill)
export VLLM_SPARSE_PREFILL_FALLBACK_DEBUG=1
# Optional v4.1 tuning (default chunk=256; lower → less peak GPU mem):
# export VLLM_SPARSE_PREFILL_FALLBACK_CHUNK=256
==============================================================================
ENV
}

case "$cmd" in
  check)
    echo "==> repo root: $REPO_ROOT"
    echo "==> checking presence of patch files in patches/:"
    for p in "${PATCHES[@]}"; do
      f="$PATCH_DIR/$p"
      if [ -f "$f" ]; then
        echo "  [OK]   $p   ($(wc -c < "$f") bytes)"
      else
        echo "  [MISS] $p"
        exit 2
      fi
    done
    echo
    echo "==> target file presence:"
    for f in "$DEEP_GEMM" "$CUSTOM_OPS" "$FLASH_MLA"; do
      if [ -f "$f" ]; then
        echo "  [OK]   $f"
      else
        echo "  [MISS] $f   (wrong vllm install? check python3 -c 'import vllm')"
        exit 3
      fi
    done
    echo
    echo "==> per-patch --check status:"
    for p in "${PATCHES[@]}"; do
      echo "----- $p -----"
      python3 "$PATCH_DIR/$p" --check || true
    done
    echo
    echo "==> sentinel count summary:"
    echo "  deep_gemm.py    SM120_HC_FALLBACK       : $(grep -c 'SM120_HC_FALLBACK\|VLLM_HC_FALLBACK' "$DEEP_GEMM" 2>/dev/null || echo 0)"
    echo "  deep_gemm.py    SM120_FP8_EINSUM_*      : $(grep -c 'SM120_FP8_EINSUM' "$DEEP_GEMM" 2>/dev/null || echo 0)"
    echo "  _custom_ops.py  UE8M0 backup .bak_v3    : $([ -f "$CUSTOM_OPS.bak_v3" ] && echo yes || echo no)"
    echo "  flash_mla       SM120_SPARSE_DECODE_*   : $(grep -c 'SM120_SPARSE_DECODE_FALLBACK' "$FLASH_MLA" 2>/dev/null || echo 0)"
    echo "  flash_mla       SM120_SPARSE_PREFILL_*  : $(grep -c 'SM120_SPARSE_PREFILL_FALLBACK' "$FLASH_MLA" 2>/dev/null || echo 0)"
    ;;

  selftest)
    echo "==> mock test: sparse decode fallback (10 cases)"
    python3 "$TEST_DIR/test_sparse_decode_fallback.py"
    echo
    echo "==> mock test: sparse prefill fallback v4.1 (12 cases)"
    python3 "$TEST_DIR/test_sparse_prefill_fallback.py"
    echo
    echo "==> if both suites are all-PASS, ready for: $0 apply"
    ;;

  apply)
    echo "==> applying ${#PATCHES[@]} patches in dependency order"
    echo "    repo: $REPO_ROOT"
    echo
    for i in "${!PATCHES[@]}"; do
      n=$((i + 1))
      p="${PATCHES[$i]}"
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      echo "[$n/${#PATCHES[@]}] $p"
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      python3 "$PATCH_DIR/$p" --check || true
      echo
      python3 "$PATCH_DIR/$p" --apply
      echo
      python3 "$PATCH_DIR/$p" --check
      echo
    done
    echo "==> all ${#PATCHES[@]} patches applied. Byte-compiling targets..."
    python3 -c "import py_compile; \
      py_compile.compile('$DEEP_GEMM',   doraise=True); \
      py_compile.compile('$CUSTOM_OPS',  doraise=True); \
      py_compile.compile('$FLASH_MLA',   doraise=True); \
      print('OK: all target files compile')"
    echo
    print_env_block
    ;;

  revert)
    echo "==> reverting all 5 patches in REVERSE order"
    for ((i=${#PATCHES[@]}-1; i>=0; i--)); do
      n=$((i + 1))
      p="${PATCHES[$i]}"
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      echo "[$n/${#PATCHES[@]}] revert $p"
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      python3 "$PATCH_DIR/$p" --revert || echo "(revert returned non-zero; continuing)"
      echo
    done
    echo "==> done. Restart vllm to pick up reverted files (clear __pycache__ if needed)."
    ;;

  *)
    cat <<USAGE
usage: $0 {check|selftest|apply|revert}

One-shot installer for all 5 SM120 fallback patches in this repo.

  check     -- inspect status of each patch & target file (read-only)
  selftest  -- run the 2 mock test suites (decode 10, prefill v4.1 12)
  apply     -- apply all 5 patches in dependency order
  revert    -- revert all 5 patches in REVERSE order

After 'apply', restart vllm with the env-var block printed at the end.
For per-patch control (e.g. only re-apply the prefill patch), use:
  scripts/deploy_sparse_decode.sh
  scripts/deploy_sparse_prefill.sh
or run the patches directly:
  python3 patches/<patch>.py {--check, --apply, --revert}
USAGE
    exit 2
    ;;
esac
