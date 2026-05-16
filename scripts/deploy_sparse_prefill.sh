#!/usr/bin/env bash
# deploy_sparse_prefill.sh
# =========================
#
# Patch #5 in the SM120 fallback chain: sparse MLA prefill fallback (v4.1).
# Targets vllm/third_party/flashmla/flash_mla_interface.py (same file as
# the decode patch, but separate backup suffix .bak_sparse_prefill_fb).
#
# Run THIS SCRIPT ON THE REMOTE BOX (in the vllm container terminal).
# The script is path-aware — it locates patches/tests relative to its
# own location, so just upload the whole repo and run from anywhere.
#
# PRE-REQUISITE: decode patch v3 (deploy_sparse_decode.sh apply) must
# already be applied. This prefill patch piggybacks on the SAME vllm
# file but uses a SEPARATE backup suffix (.bak_sparse_prefill_fb) so
# the two patches don't clobber each other.
#
# Usage (run each phase manually, stop on any FAIL):
#   ./deploy_sparse_prefill.sh check     # 1) verify files + decode v3 status
#   ./deploy_sparse_prefill.sh selftest  # 2) run local mock test (v4.1: 12 cases)
#   ./deploy_sparse_prefill.sh apply     # 3) apply patch + verify markers + byte-compile
#   ./deploy_sparse_prefill.sh revert    # rollback prefill patch only
#   ./deploy_sparse_prefill.sh redeploy  # revert + clear pycache + apply (v4 → v4.1)
#
# After 'apply', RESTART vllm with these env vars added to your usual command:
#   export VLLM_SPARSE_PREFILL_FALLBACK=1
#   export VLLM_SPARSE_PREFILL_FALLBACK_DEBUG=1
#   # optional v4.1 tuning (default chunk=256):
#   # export VLLM_SPARSE_PREFILL_FALLBACK_CHUNK=256
# (keep all existing env vars including VLLM_SPARSE_DECODE_FALLBACK=1)
#
# Then run smoke_test.sh; step 4-7 should now pass. v4.1 specifically
# fixes step 7 (4k prompt) OOM caused by v4's fp32 K_f bmm.

set -euo pipefail

# ── path resolution: this script lives in scripts/; patches in ../patches/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PATCH_DIR="$REPO_ROOT/patches"
TEST_DIR="$REPO_ROOT/tests"

PATCH_FILE="$PATCH_DIR/sparse_prefill_fwd_sm120_fallback.py"
TEST_FILE="$TEST_DIR/test_sparse_prefill_fallback.py"
VLLM_INTERFACE="/usr/local/lib/python3.12/dist-packages/vllm/third_party/flashmla/flash_mla_interface.py"

cmd="${1:-}"

case "$cmd" in
  check)
    echo "==> [1/3] verify files on this box"
    ls -la "$PATCH_FILE" "$TEST_FILE"
    md5sum "$PATCH_FILE" "$TEST_FILE"
    echo
    echo "==> vllm flash_mla_interface.py target file:"
    ls -la "$VLLM_INTERFACE"
    echo "    DECODE  patch v3 markers in target: $(grep -c 'SM120_SPARSE_DECODE_FALLBACK' "$VLLM_INTERFACE" || echo 0)   (expected: > 0)"
    echo "    PREFILL patch v4 markers in target: $(grep -c 'SM120_SPARSE_PREFILL_FALLBACK' "$VLLM_INTERFACE" || echo 0)  (expected: 0 before apply)"
    echo
    echo "==> patch self --check:"
    python3 "$PATCH_FILE" --check || true
    echo
    echo "==> ready for: $0 selftest"
    echo "    then     : $0 apply"
    ;;

  selftest)
    echo "==> [2/3] running local mock test on this box (v4.1: 12 cases)"
    echo "    (verifies fallback math matches naive ref + chunk equivalence)"
    python3 "$TEST_FILE"
    echo
    echo "==> if all 12 PASS, ready for: $0 apply"
    ;;

  apply)
    echo "==> [3/3] apply prefill patch v4.1 on this box"
    echo "--- pre-apply --check ---"
    python3 "$PATCH_FILE" --check || true
    echo
    echo "--- decode patch v3 sanity (must already be applied) ---"
    if ! grep -q SM120_SPARSE_DECODE_FALLBACK_BEGIN "$VLLM_INTERFACE"; then
      echo "[FAIL] decode patch v3 NOT applied. Apply it FIRST:"
      echo "       $SCRIPT_DIR/deploy_sparse_decode.sh apply"
      exit 3
    fi
    echo "[OK] decode patch v3 already wired"
    echo
    echo "--- applying prefill patch v4.1 ---"
    python3 "$PATCH_FILE" --apply
    echo
    echo "--- post-apply --check ---"
    python3 "$PATCH_FILE" --check
    echo
    echo "--- grep markers in target file ---"
    grep -n SM120_SPARSE_PREFILL_FALLBACK "$VLLM_INTERFACE" | head -20
    echo
    echo "--- byte-compile target file (catches syntax errors) ---"
    python3 -c "import py_compile; py_compile.compile('$VLLM_INTERFACE', doraise=True); print('OK: target file compiles')"
    echo
    echo "==> NEXT: restart vllm serve manually with these env vars added:"
    echo "      export VLLM_SPARSE_PREFILL_FALLBACK=1"
    echo "      export VLLM_SPARSE_PREFILL_FALLBACK_DEBUG=1"
    echo "      # optional v4.1 tuning (default chunk=256):"
    echo "      # export VLLM_SPARSE_PREFILL_FALLBACK_CHUNK=256"
    echo "    (keep existing: VLLM_HC_FALLBACK=1 VLLM_INDEXER_FALLBACK=1"
    echo "                    VLLM_UE8M0_FALLBACK=1 VLLM_FP8_EINSUM_FALLBACK=1"
    echo "                    VLLM_USE_DEEP_GEMM=0 VLLM_FUSED_MOE_BACKEND=triton"
    echo "                    VLLM_SPARSE_DECODE_FALLBACK=1)"
    echo "    Then: $SCRIPT_DIR/smoke_test.sh and check steps 4-7 output."
    echo "    v4.1 specifically targets step 7 (4k prompt) OOM fix."
    ;;

  revert)
    echo "==> reverting PREFILL patch v4.1 on this box (leaves decode v3 intact)"
    python3 "$PATCH_FILE" --revert
    python3 "$PATCH_FILE" --check
    echo "PREFILL markers remaining: $(grep -c SM120_SPARSE_PREFILL_FALLBACK "$VLLM_INTERFACE" || echo 0)"
    echo "DECODE  markers remaining: $(grep -c SM120_SPARSE_DECODE_FALLBACK "$VLLM_INTERFACE" || echo 0)  (should be > 0)"
    python3 -c "import py_compile; py_compile.compile('$VLLM_INTERFACE', doraise=True); print('OK: target file compiles after revert')"
    ;;

  redeploy)
    echo "==> redeploy = revert + clear pycache + apply (v4 → v4.1 upgrade path)"
    echo "--- revert (prefill only; decode v3 untouched) ---"
    python3 "$PATCH_FILE" --revert || echo "(revert returned non-zero; continuing)"
    echo "--- clear stale .pyc ---"
    PYC_DIR="$(dirname "$VLLM_INTERFACE")/__pycache__"
    rm -fv "$PYC_DIR"/flash_mla_interface*.pyc 2>/dev/null || true
    echo "--- decode v3 sanity ---"
    if ! grep -q SM120_SPARSE_DECODE_FALLBACK_BEGIN "$VLLM_INTERFACE"; then
      echo "[FAIL] decode patch v3 NOT applied (after revert). Apply it FIRST:"
      echo "       $SCRIPT_DIR/deploy_sparse_decode.sh apply"
      exit 3
    fi
    echo "[OK] decode patch v3 still wired"
    echo "--- apply prefill v4.1 ---"
    python3 "$PATCH_FILE" --apply
    echo "--- post-apply --check ---"
    python3 "$PATCH_FILE" --check
    echo "--- byte-compile target ---"
    python3 -c "import py_compile; py_compile.compile('$VLLM_INTERFACE', doraise=True); print('OK: target file compiles after apply')"
    echo ""
    echo "==> done. RESTART vllm now, then run smoke step 4-7."
    ;;

  *)
    cat <<USAGE
usage: $0 {check|selftest|apply|revert|redeploy}

Run on the REMOTE vllm container terminal, in this order, stop on FAIL:
  1) check     -- verify files present + show vllm target state
                  (also shows decode patch v3 status, which must be present)
  2) selftest  -- run local mock test on this box's torch
                  (v4.1: 12 PASS expected, incl. chunk equivalence)
  3) apply     -- apply patch + verify markers injected + byte-compile target
                  (after this, manually restart vllm; run smoke step 4-7)

  redeploy     -- revert + clear .pyc + apply (use for any version upgrade)
  revert       -- emergency rollback of prefill patch (decode v3 left intact)

Patch resolved from:
  $PATCH_FILE
Test resolved from:
  $TEST_FILE
USAGE
    exit 2
    ;;
esac
