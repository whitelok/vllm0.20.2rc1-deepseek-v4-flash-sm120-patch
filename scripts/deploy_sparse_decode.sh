#!/usr/bin/env bash
# deploy_sparse_decode.sh
# ========================
#
# Patch #4 in the SM120 fallback chain: sparse MLA decode fallback (v3).
# Targets vllm/third_party/flashmla/flash_mla_interface.py.
#
# Run THIS SCRIPT ON THE REMOTE BOX (in the vllm container terminal).
# The script is path-aware — it locates patches/tests relative to its
# own location, so just upload the whole repo and run from anywhere.
#
# Usage (run each phase manually, stop on any FAIL):
#   ./deploy_sparse_decode.sh check      # 1) verify files present + show current state
#   ./deploy_sparse_decode.sh selftest   # 2) (optional) run local mock test (10 cases)
#   ./deploy_sparse_decode.sh apply      # 3) apply patch + verify markers + byte-compile target
#   ./deploy_sparse_decode.sh revert     # rollback if anything bad
#   ./deploy_sparse_decode.sh redeploy   # revert + clear pycache + apply (version upgrade)
#
# After 'apply', RESTART vllm with these env vars added to your usual command:
#   export VLLM_SPARSE_DECODE_FALLBACK=1
#   export VLLM_SPARSE_DECODE_FALLBACK_DEBUG=1
#
# Then run smoke_test.sh step 3 to verify decode produces text.

set -euo pipefail

# ── path resolution: this script lives in scripts/; patches in ../patches/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PATCH_DIR="$REPO_ROOT/patches"
TEST_DIR="$REPO_ROOT/tests"

PATCH_FILE="$PATCH_DIR/sparse_decode_fwd_sm120_fallback.py"
TEST_FILE="$TEST_DIR/test_sparse_decode_fallback.py"
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
    echo "    markers currently in target: $(grep -c 'SM120_SPARSE_DECODE_FALLBACK' "$VLLM_INTERFACE" || echo 0)"
    echo
    echo "==> patch self --check:"
    python3 "$PATCH_FILE" --check || true
    echo
    echo "==> ready for: $0 selftest   (optional)"
    echo "    then     : $0 apply"
    ;;

  selftest)
    echo "==> [2/3] running local mock test on this box (10 cases)"
    echo "    (verifies fp8_e4m3fn / bf16 byte ops work on this torch)"
    python3 "$TEST_FILE"
    echo
    echo "==> if all 10 PASS, ready for: $0 apply"
    ;;

  apply)
    echo "==> [3/3] apply patch on this box"
    echo "--- pre-apply --check ---"
    python3 "$PATCH_FILE" --check || true
    echo
    echo "--- applying ---"
    python3 "$PATCH_FILE" --apply
    echo
    echo "--- post-apply --check ---"
    python3 "$PATCH_FILE" --check
    echo
    echo "--- grep markers in target file ---"
    grep -n SM120_SPARSE_DECODE_FALLBACK "$VLLM_INTERFACE" | head -20
    echo
    echo "--- byte-compile target file (catches syntax errors) ---"
    python3 -c "import py_compile; py_compile.compile('$VLLM_INTERFACE', doraise=True); print('OK: target file compiles')"
    echo
    echo "==> NEXT: restart vllm serve manually with these env vars added:"
    echo "      export VLLM_SPARSE_DECODE_FALLBACK=1"
    echo "      export VLLM_SPARSE_DECODE_FALLBACK_DEBUG=1"
    echo "    (keep existing: VLLM_HC_FALLBACK=1 VLLM_INDEXER_FALLBACK=1"
    echo "                    VLLM_UE8M0_FALLBACK=1 VLLM_FP8_EINSUM_FALLBACK=1"
    echo "                    VLLM_USE_DEEP_GEMM=0 VLLM_FUSED_MOE_BACKEND=triton)"
    echo "    Then: $SCRIPT_DIR/smoke_test.sh and check step 3 output."
    ;;

  revert)
    echo "==> reverting patch on this box"
    python3 "$PATCH_FILE" --revert
    python3 "$PATCH_FILE" --check
    echo "markers remaining: $(grep -c SM120_SPARSE_DECODE_FALLBACK "$VLLM_INTERFACE" || echo 0)"
    python3 -c "import py_compile; py_compile.compile('$VLLM_INTERFACE', doraise=True); print('OK: target file compiles after revert')"
    ;;

  redeploy)
    echo "==> redeploy = revert + clear pycache + apply (use for version upgrade, e.g. v2 → v3)"
    echo "--- revert ---"
    python3 "$PATCH_FILE" --revert || echo "(revert returned non-zero; continuing)"
    echo "--- clear stale .pyc ---"
    PYC_DIR="$(dirname "$VLLM_INTERFACE")/__pycache__"
    rm -fv "$PYC_DIR"/flash_mla_interface*.pyc 2>/dev/null || true
    echo "--- apply ---"
    python3 "$PATCH_FILE" --apply
    echo "--- post-apply --check ---"
    python3 "$PATCH_FILE" --check
    echo "--- byte-compile target ---"
    python3 -c "import py_compile; py_compile.compile('$VLLM_INTERFACE', doraise=True); print('OK: target file compiles after apply')"
    echo ""
    echo "==> done. RESTART vllm now, then run smoke step 3."
    ;;

  *)
    cat <<USAGE
usage: $0 {check|selftest|apply|revert|redeploy}

Run on the REMOTE vllm container terminal, in this order, stop on FAIL:
  1) check     -- verify files present in repo + show vllm target state
  2) selftest  -- (optional) run local mock test on this box's torch
                  (v3: 10 PASS expected)
  3) apply     -- apply patch + verify markers injected + byte-compile target
                  (after this, manually restart vllm; run smoke step 3)

  redeploy     -- revert + clear .pyc + apply (use for any version upgrade)
  revert       -- emergency rollback of patch

Patch resolved from:
  $PATCH_FILE
Test resolved from:
  $TEST_FILE
USAGE
    exit 2
    ;;
esac
