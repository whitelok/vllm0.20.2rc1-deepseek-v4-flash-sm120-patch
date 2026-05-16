#!/usr/bin/env python3
"""
test_sparse_prefill_fallback.py
================================

Mock unit-test for `_sm120_sparse_prefill_fallback`, the pure-PyTorch
replacement for `flash_mla_cuda.sparse_prefill_fwd` on SM120 boxes.

Goal: verify that the fallback matches a naive reference implementation
on a small handcrafted setup BEFORE we apply the patch to the real
flash_mla_interface.py.

We re-import the helper functions textually from the patch file by
exec-ing the HELPERS_BLOCK so we test the EXACT bytes that will be
injected into vllm. This guards against drift between this test and the
patch shipped to production.

Test coverage (12 cases):
  1.  basic_smoke              : tiny shape, no sink, no topk_length, no invalid
  2.  matches_reference        : random shapes, compare against naive einsum ref
  3.  with_attn_sink           : sink applied → out *= sigmoid(lse - sink)
  4.  invalid_indices_negative : -1 indices are masked
  5.  invalid_indices_overflow : idx >= Skv are masked
  6.  topk_length              : per-query effective topk truncates
  7.  topk_length_zero         : len=0 → all_invalid → out=0, lse=-inf
  8.  preallocated_out         : pass `out` buffer, assert in-place write
  9.  attn_sink_with_invalid   : combined masking + sink doesn't NaN
  10. shape_assertions         : Dqk != 512, d_v != 512, Hkv != 1 all rejected
  11. chunk_equivalence        : (v4.1) Sq=600, vary chunk ∈ {1,17,256,600,9999}
                                 → all bit-equivalent (per-query independence)
  12. chunk_with_invalid_and_sink: (v4.1) chunked path keeps all-invalid rows
                                 zeroed and sink scaling intact across chunk boundary

Run on the SAME box (remote container) so torch/CUDA flavor matches:
    python3 test_sparse_prefill_fallback.py
"""

from __future__ import annotations

import re
import sys
import traceback
from pathlib import Path

import torch


# ────────────────────────────────────────────────────────────────────
# Extract HELPERS_BLOCK from the patch file and exec it in a sandbox
# so we test the literal injected bytes (not a copy).
# ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
# Look in: same dir (legacy flat layout), then ../patches/ (repo layout)
_CANDIDATES = [
    _HERE / "sparse_prefill_fwd_sm120_fallback.py",
    _HERE.parent / "patches" / "sparse_prefill_fwd_sm120_fallback.py",
]
PATCH_PATH = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])
assert PATCH_PATH.exists(), "missing patch file. Tried:\n  " + "\n  ".join(
    str(c) for c in _CANDIDATES
)

_patch_src = PATCH_PATH.read_text()


def _extract_helpers_block(src: str) -> str:
    """Pull out the BEGIN…END block exactly as it'll be injected."""
    m = re.search(
        r"^(# SM120_SPARSE_PREFILL_FALLBACK_BEGIN.*?# SM120_SPARSE_PREFILL_FALLBACK_END)\s*$",
        src,
        re.MULTILINE | re.DOTALL,
    )
    assert m, "could not locate HELPERS_BLOCK sentinels in patch file"
    return m.group(1)


HELPERS_SRC = _extract_helpers_block(_patch_src)

# Build a sandbox namespace mirroring how the helpers will live inside
# flash_mla_interface.py: torch is in scope at module level.
_sandbox: dict[str, object] = {"torch": torch, "__name__": "_sm120_test_sandbox"}
exec(compile(HELPERS_SRC, str(PATCH_PATH), "exec"), _sandbox)

_sm120_sparse_prefill_fallback = _sandbox["_sm120_sparse_prefill_fallback"]


# ────────────────────────────────────────────────────────────────────
# Reference implementation — straightforward, unoptimized, fp32 path.
# Spec-matches the flash_mla_sparse_fwd docstring exactly.
# ────────────────────────────────────────────────────────────────────
def reference_sparse_prefill(
    q: torch.Tensor,  # [Sq, Hq, Dqk] bf16
    kv: torch.Tensor,  # [Skv, Hkv=1, Dqk] bf16
    indices: torch.Tensor,  # [Sq, Hkv=1, topk] int32
    sm_scale: float,
    d_v: int,
    attn_sink: torch.Tensor | None,  # [Hq] fp32
    topk_length: torch.Tensor | None,  # [Sq] int32
):
    """Naive ref. Returns (out_bf16, max_logits_fp32, lse_fp32)."""
    Sq, Hq, Dqk = q.shape
    Skv, Hkv, _ = kv.shape
    _, _, topk = indices.shape
    assert Hkv == 1
    device = q.device

    q_f = q.to(torch.float32)
    kv_f = kv.squeeze(1).to(torch.float32)  # [Skv, Dqk]
    idx = indices.squeeze(1).to(torch.int64)  # [Sq, topk]

    invalid = (idx < 0) | (idx >= Skv)  # [Sq, topk]
    if topk_length is not None:
        pos = torch.arange(topk, device=device).unsqueeze(0)
        tlen = topk_length.to(torch.int64).unsqueeze(1)
        invalid = invalid | (pos >= tlen)

    safe_idx = torch.where(invalid, torch.zeros_like(idx), idx)
    gathered = kv_f[safe_idx]  # [Sq, topk, Dqk]

    # logits[s, h, t] = sum_d q_f[s, h, d] * gathered[s, t, d] * sm_scale
    logits = torch.einsum("shd,std->sht", q_f, gathered) * float(sm_scale)
    logits = logits.masked_fill(invalid.unsqueeze(1), float("-inf"))

    max_logits = logits.max(dim=-1).values  # [Sq, Hq]
    lse = torch.logsumexp(logits, dim=-1)  # [Sq, Hq]

    all_invalid = torch.isinf(lse) & (lse < 0)
    safe_lse = torch.where(all_invalid, torch.zeros_like(lse), lse)
    attn = torch.exp(logits - safe_lse.unsqueeze(-1))  # [Sq, Hq, topk]
    attn = torch.where(all_invalid.unsqueeze(-1), torch.zeros_like(attn), attn)

    out = torch.einsum("sht,std->shd", attn, gathered)  # [Sq, Hq, Dqk]
    out = out[..., :d_v]

    if attn_sink is not None:
        sink = attn_sink.to(torch.float32).reshape(1, Hq)
        scale_f = torch.sigmoid(lse - sink)  # [Sq, Hq]
        scale_f = torch.where(all_invalid, torch.zeros_like(scale_f), scale_f)
        out = out * scale_f.unsqueeze(-1)

    return out.to(q.dtype), max_logits, lse


# ────────────────────────────────────────────────────────────────────
# Test runner harness
# ────────────────────────────────────────────────────────────────────
PASS, FAIL = 0, 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run(name: str, fn):
    global PASS, FAIL
    try:
        fn()
    except Exception as e:
        FAIL += 1
        print(f"[FAIL] {name}: {e}")
        traceback.print_exc()
        return
    PASS += 1
    print(f"[PASS] {name}")


def _close(a: torch.Tensor, b: torch.Tensor, atol=1e-2, rtol=1e-2, msg=""):
    """bf16 + sigmoid compose loosely; default ~1% tolerance."""
    if not torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol, equal_nan=False):
        diff = (a.float() - b.float()).abs()
        raise AssertionError(
            f"{msg}\n  max_abs_diff = {diff.max().item():.4e}\n"
            f"  mean_abs_diff = {diff.mean().item():.4e}\n"
            f"  a[0:3] = {a.flatten()[:3].tolist()}\n"
            f"  b[0:3] = {b.flatten()[:3].tolist()}"
        )


def _make_inputs(
    Sq=4,
    Hq=8,
    Dqk=512,
    Skv=32,
    topk=16,
    seed=0,
    device=DEVICE,
):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = (
        torch.randn(Sq, Hq, Dqk, generator=g, dtype=torch.float32).to(
            device=device, dtype=torch.bfloat16
        )
        * 0.1
    )
    kv = (
        torch.randn(Skv, 1, Dqk, generator=g, dtype=torch.float32).to(
            device=device, dtype=torch.bfloat16
        )
        * 0.1
    )
    # valid random indices into [0, Skv)
    indices = torch.randint(0, Skv, (Sq, 1, topk), generator=g, dtype=torch.int32).to(
        device
    )
    return q, kv, indices


# ────────────────────────────────────────────────────────────────────
# Test cases
# ────────────────────────────────────────────────────────────────────
def test_basic_smoke():
    q, kv, idx = _make_inputs(Sq=2, Hq=4, Dqk=512, Skv=8, topk=4, seed=1)
    sm_scale = 1.0 / (512**0.5)
    out, ml, lse = _sm120_sparse_prefill_fallback(
        q=q,
        kv=kv,
        indices=idx,
        sm_scale=sm_scale,
        d_v=512,
        attn_sink=None,
        topk_length=None,
        out=None,
    )
    assert out.shape == (2, 4, 512), out.shape
    assert out.dtype == torch.bfloat16, out.dtype
    assert ml.shape == (2, 4), ml.shape
    assert lse.shape == (2, 4), lse.shape
    assert ml.dtype == torch.float32, ml.dtype
    assert lse.dtype == torch.float32, lse.dtype
    assert torch.isfinite(out).all(), "out has NaN/Inf"


def test_matches_reference():
    q, kv, idx = _make_inputs(Sq=6, Hq=16, Dqk=512, Skv=64, topk=32, seed=2)
    sm_scale = 1.0 / (512**0.5)
    out_a, ml_a, lse_a = _sm120_sparse_prefill_fallback(
        q=q,
        kv=kv,
        indices=idx,
        sm_scale=sm_scale,
        d_v=512,
        attn_sink=None,
        topk_length=None,
        out=None,
    )
    out_b, ml_b, lse_b = reference_sparse_prefill(
        q=q,
        kv=kv,
        indices=idx,
        sm_scale=sm_scale,
        d_v=512,
        attn_sink=None,
        topk_length=None,
    )
    _close(out_a, out_b, msg="out mismatch vs ref")
    _close(ml_a, ml_b, atol=1e-3, rtol=1e-3, msg="max_logits mismatch vs ref")
    _close(lse_a, lse_b, atol=1e-3, rtol=1e-3, msg="lse mismatch vs ref")


def test_with_attn_sink():
    q, kv, idx = _make_inputs(Sq=4, Hq=8, Dqk=512, Skv=32, topk=16, seed=3)
    sm_scale = 1.0 / (512**0.5)
    sink = torch.zeros(8, dtype=torch.float32, device=DEVICE)  # sigmoid(lse - 0)

    out_a, ml_a, lse_a = _sm120_sparse_prefill_fallback(
        q=q,
        kv=kv,
        indices=idx,
        sm_scale=sm_scale,
        d_v=512,
        attn_sink=sink,
        topk_length=None,
        out=None,
    )
    out_b, ml_b, lse_b = reference_sparse_prefill(
        q=q,
        kv=kv,
        indices=idx,
        sm_scale=sm_scale,
        d_v=512,
        attn_sink=sink,
        topk_length=None,
    )
    _close(out_a, out_b, msg="out mismatch with attn_sink")
    _close(lse_a, lse_b, atol=1e-3, rtol=1e-3, msg="lse mismatch with attn_sink")
    # sanity: with sink=0, scale = sigmoid(lse), bounded in (0,1) → magnitude reduced
    out_nosink, _, _ = _sm120_sparse_prefill_fallback(
        q=q,
        kv=kv,
        indices=idx,
        sm_scale=sm_scale,
        d_v=512,
        attn_sink=None,
        topk_length=None,
        out=None,
    )
    assert out_a.abs().mean() < out_nosink.abs().mean() + 1e-6, (
        "attn_sink should generally shrink output magnitude"
    )


def test_invalid_indices_negative():
    q, kv, idx = _make_inputs(Sq=3, Hq=4, Dqk=512, Skv=16, topk=8, seed=4)
    # mark first 3 indices of query 0 as invalid (-1)
    idx[0, 0, :3] = -1
    sm_scale = 1.0 / (512**0.5)

    out_a, ml_a, lse_a = _sm120_sparse_prefill_fallback(
        q=q,
        kv=kv,
        indices=idx,
        sm_scale=sm_scale,
        d_v=512,
        attn_sink=None,
        topk_length=None,
        out=None,
    )
    out_b, ml_b, lse_b = reference_sparse_prefill(
        q=q,
        kv=kv,
        indices=idx,
        sm_scale=sm_scale,
        d_v=512,
        attn_sink=None,
        topk_length=None,
    )
    _close(out_a, out_b, msg="negative invalid index masking")
    _close(lse_a, lse_b, atol=1e-3, rtol=1e-3, msg="lse mismatch w/ negative invalids")


def test_invalid_indices_overflow():
    q, kv, idx = _make_inputs(Sq=3, Hq=4, Dqk=512, Skv=16, topk=8, seed=5)
    # mark indices of query 1 as out-of-range
    idx[1, 0, 4:] = 9999
    sm_scale = 1.0 / (512**0.5)

    out_a, ml_a, lse_a = _sm120_sparse_prefill_fallback(
        q=q,
        kv=kv,
        indices=idx,
        sm_scale=sm_scale,
        d_v=512,
        attn_sink=None,
        topk_length=None,
        out=None,
    )
    out_b, ml_b, lse_b = reference_sparse_prefill(
        q=q,
        kv=kv,
        indices=idx,
        sm_scale=sm_scale,
        d_v=512,
        attn_sink=None,
        topk_length=None,
    )
    _close(out_a, out_b, msg="overflow invalid index masking")
    _close(lse_a, lse_b, atol=1e-3, rtol=1e-3, msg="lse mismatch w/ overflow invalids")


def test_topk_length():
    q, kv, idx = _make_inputs(Sq=4, Hq=8, Dqk=512, Skv=32, topk=16, seed=6)
    # per-query effective topk: [2, 16, 8, 0]
    tlen = torch.tensor([2, 16, 8, 0], dtype=torch.int32, device=DEVICE)
    sm_scale = 1.0 / (512**0.5)

    out_a, ml_a, lse_a = _sm120_sparse_prefill_fallback(
        q=q,
        kv=kv,
        indices=idx,
        sm_scale=sm_scale,
        d_v=512,
        attn_sink=None,
        topk_length=tlen,
        out=None,
    )
    out_b, ml_b, lse_b = reference_sparse_prefill(
        q=q,
        kv=kv,
        indices=idx,
        sm_scale=sm_scale,
        d_v=512,
        attn_sink=None,
        topk_length=tlen,
    )
    _close(out_a, out_b, msg="topk_length truncation mismatch")
    _close(lse_a, lse_b, atol=1e-3, rtol=1e-3, msg="lse mismatch w/ topk_length")
    # query 3 has tlen=0 → all invalid → out must be 0
    assert out_a[3].abs().max().item() == 0.0, "tlen=0 row must be zero"


def test_topk_length_zero():
    q, kv, idx = _make_inputs(Sq=2, Hq=4, Dqk=512, Skv=8, topk=4, seed=7)
    tlen = torch.tensor([0, 0], dtype=torch.int32, device=DEVICE)
    sm_scale = 1.0 / (512**0.5)
    out, ml, lse = _sm120_sparse_prefill_fallback(
        q=q,
        kv=kv,
        indices=idx,
        sm_scale=sm_scale,
        d_v=512,
        attn_sink=None,
        topk_length=tlen,
        out=None,
    )
    assert out.abs().max().item() == 0.0, "all-zero topk_length must give zero out"
    # lse must be -inf when no valid positions
    assert torch.isinf(lse).all() and (lse < 0).all(), (
        f"lse should be -inf when tlen=0, got {lse}"
    )


def test_preallocated_out():
    q, kv, idx = _make_inputs(Sq=3, Hq=8, Dqk=512, Skv=16, topk=8, seed=8)
    sm_scale = 1.0 / (512**0.5)
    out_buf = torch.empty(3, 8, 512, dtype=torch.bfloat16, device=DEVICE)
    out_buf.fill_(float("nan"))  # sentinel: ensure we actually write

    ret_out, ml, lse = _sm120_sparse_prefill_fallback(
        q=q,
        kv=kv,
        indices=idx,
        sm_scale=sm_scale,
        d_v=512,
        attn_sink=None,
        topk_length=None,
        out=out_buf,
    )
    assert ret_out.data_ptr() == out_buf.data_ptr(), (
        "preallocated out buffer should be returned in place"
    )
    assert not torch.isnan(out_buf).any(), "out_buf still NaN after write"

    # cross-check with the no-preallocated path
    out_b, _, _ = _sm120_sparse_prefill_fallback(
        q=q,
        kv=kv,
        indices=idx,
        sm_scale=sm_scale,
        d_v=512,
        attn_sink=None,
        topk_length=None,
        out=None,
    )
    _close(out_buf, out_b, msg="preallocated path differs from fresh path")


def test_attn_sink_with_invalid():
    q, kv, idx = _make_inputs(Sq=3, Hq=4, Dqk=512, Skv=16, topk=8, seed=9)
    idx[0, 0, :] = -1  # query 0 all invalid
    sink = torch.zeros(4, dtype=torch.float32, device=DEVICE)
    sm_scale = 1.0 / (512**0.5)
    out_a, ml_a, lse_a = _sm120_sparse_prefill_fallback(
        q=q,
        kv=kv,
        indices=idx,
        sm_scale=sm_scale,
        d_v=512,
        attn_sink=sink,
        topk_length=None,
        out=None,
    )
    out_b, ml_b, lse_b = reference_sparse_prefill(
        q=q,
        kv=kv,
        indices=idx,
        sm_scale=sm_scale,
        d_v=512,
        attn_sink=sink,
        topk_length=None,
    )
    assert torch.isfinite(out_a).all(), (
        "out should not have NaN/Inf even when a row is all-invalid"
    )
    assert out_a[0].abs().max().item() == 0.0, (
        "all-invalid row with sink should still produce zero (sink branch guarded)"
    )
    _close(out_a, out_b, msg="mismatch with invalid+sink combo")


def test_chunk_equivalence():
    """v4.1: Sq-dim chunking must be numerically equivalent across chunk sizes.

    We import the sandbox module-level CHUNK var and override it per run.
    Per-query independence guarantees bit equivalence (no reduction across
    queries), so we require torch.equal (not allclose).
    """
    q, kv, idx = _make_inputs(Sq=600, Hq=8, Dqk=512, Skv=128, topk=32, seed=11)
    # add some invalid indices to exercise masked_fill_ path inside chunk loop
    idx[100, 0, :5] = -1
    idx[517, 0, 10:] = 9999
    tlen = torch.full((600,), 32, dtype=torch.int32, device=DEVICE)
    tlen[10] = 0  # all-invalid row mid-chunk-1
    tlen[300] = 4  # partial truncate spanning a chunk boundary
    sm_scale = 1.0 / (512**0.5)
    sink = torch.zeros(8, dtype=torch.float32, device=DEVICE)

    results = {}
    for cs in (1, 17, 256, 600, 9999):
        _sandbox["_SM120_SPARSE_PREFILL_FALLBACK_CHUNK"] = cs
        out, ml, lse = _sm120_sparse_prefill_fallback(
            q=q,
            kv=kv,
            indices=idx,
            sm_scale=sm_scale,
            d_v=512,
            attn_sink=sink,
            topk_length=tlen,
            out=None,
        )
        results[cs] = (out.clone(), ml.clone(), lse.clone())

    # restore default (256) for downstream tests
    _sandbox["_SM120_SPARSE_PREFILL_FALLBACK_CHUNK"] = 256

    ref_out, ref_ml, ref_lse = results[600]  # whole-Sq-as-one-chunk baseline
    for cs, (o, m, l) in results.items():
        # bit-equivalent: same kernel, same order of ops per query
        assert torch.equal(o, ref_out), (
            f"chunk={cs}: out diverged from chunk=600 baseline\n"
            f"  max_abs_diff = {(o.float() - ref_out.float()).abs().max().item()}"
        )
        assert torch.equal(m, ref_ml), f"chunk={cs}: max_logits diverged"
        assert torch.equal(l, ref_lse), f"chunk={cs}: lse diverged"

    # sanity: tlen=0 row truly zero across all chunk sizes
    for cs, (o, _, l) in results.items():
        assert o[10].abs().max().item() == 0.0, f"chunk={cs}: tlen=0 row not zero"
        assert torch.isinf(l[10]).all() and (l[10] < 0).all(), (
            f"chunk={cs}: tlen=0 row lse should be -inf"
        )


def test_chunk_with_invalid_and_sink():
    """v4.1: spot-check chunk-boundary correctness against ref impl.

    Sq=400 with chunk=128 means 4 chunks (3 full + 1 partial 16). Each
    chunk must independently produce output matching the reference.
    """
    q, kv, idx = _make_inputs(Sq=400, Hq=8, Dqk=512, Skv=64, topk=24, seed=12)
    # invalid indices straddling chunk boundary 128
    idx[125, 0, :3] = -1
    idx[126, 0, :] = -1  # all-invalid row at the boundary
    idx[129, 0, 10:] = 9999
    tlen = torch.full((400,), 24, dtype=torch.int32, device=DEVICE)
    tlen[256] = 0  # all-invalid row at next boundary
    tlen[383] = 5  # partial in last chunk
    sm_scale = 1.0 / (512**0.5)
    sink = torch.randn(8, dtype=torch.float32, device=DEVICE) * 0.5

    _sandbox["_SM120_SPARSE_PREFILL_FALLBACK_CHUNK"] = 128
    out_a, ml_a, lse_a = _sm120_sparse_prefill_fallback(
        q=q,
        kv=kv,
        indices=idx,
        sm_scale=sm_scale,
        d_v=512,
        attn_sink=sink,
        topk_length=tlen,
        out=None,
    )
    _sandbox["_SM120_SPARSE_PREFILL_FALLBACK_CHUNK"] = 256  # restore

    out_b, ml_b, lse_b = reference_sparse_prefill(
        q=q,
        kv=kv,
        indices=idx,
        sm_scale=sm_scale,
        d_v=512,
        attn_sink=sink,
        topk_length=tlen,
    )
    _close(out_a, out_b, msg="chunked path differs from ref at boundary")
    _close(ml_a, ml_b, atol=1e-3, rtol=1e-3, msg="max_logits diverged at boundary")
    _close(lse_a, lse_b, atol=1e-3, rtol=1e-3, msg="lse diverged at boundary")
    # all-invalid rows must be zero
    assert out_a[126].abs().max().item() == 0.0, "row 126 (idx=-1 all) not zero"
    assert out_a[256].abs().max().item() == 0.0, "row 256 (tlen=0) not zero"
    # preallocated out + chunked path
    out_buf = torch.empty_like(out_a)
    out_buf.fill_(float("nan"))
    _sandbox["_SM120_SPARSE_PREFILL_FALLBACK_CHUNK"] = 128
    ret_out, _, _ = _sm120_sparse_prefill_fallback(
        q=q,
        kv=kv,
        indices=idx,
        sm_scale=sm_scale,
        d_v=512,
        attn_sink=sink,
        topk_length=tlen,
        out=out_buf,
    )
    _sandbox["_SM120_SPARSE_PREFILL_FALLBACK_CHUNK"] = 256
    assert ret_out.data_ptr() == out_buf.data_ptr(), "preallocated buffer not returned"
    assert not torch.isnan(out_buf).any(), "out_buf still NaN after chunked write"
    _close(out_buf, out_a, msg="preallocated chunked path differs from fresh")


def test_shape_assertions():
    """Patch should reject configs outside V4-Flash spec (Dqk=512, d_v=512, Hkv=1)."""
    q, kv, idx = _make_inputs(Sq=2, Hq=4, Dqk=512, Skv=8, topk=4, seed=10)
    sm_scale = 1.0 / (512**0.5)

    # 1) d_v != 512 should assert
    raised = False
    try:
        _sm120_sparse_prefill_fallback(
            q=q,
            kv=kv,
            indices=idx,
            sm_scale=sm_scale,
            d_v=256,
            attn_sink=None,
            topk_length=None,
            out=None,
        )
    except AssertionError:
        raised = True
    assert raised, "d_v != 512 should trigger AssertionError"

    # 2) Dqk != 512 should assert
    q_bad = torch.randn(2, 4, 256, dtype=torch.bfloat16, device=DEVICE)
    kv_bad = torch.randn(8, 1, 256, dtype=torch.bfloat16, device=DEVICE)
    raised = False
    try:
        _sm120_sparse_prefill_fallback(
            q=q_bad,
            kv=kv_bad,
            indices=idx,
            sm_scale=sm_scale,
            d_v=512,
            attn_sink=None,
            topk_length=None,
            out=None,
        )
    except AssertionError:
        raised = True
    assert raised, "Dqk != 512 should trigger AssertionError"

    # 3) Hkv != 1 should assert
    kv_h2 = torch.randn(8, 2, 512, dtype=torch.bfloat16, device=DEVICE)
    idx_h2 = torch.randint(0, 8, (2, 2, 4), dtype=torch.int32, device=DEVICE)
    raised = False
    try:
        _sm120_sparse_prefill_fallback(
            q=q,
            kv=kv_h2,
            indices=idx_h2,
            sm_scale=sm_scale,
            d_v=512,
            attn_sink=None,
            topk_length=None,
            out=None,
        )
    except AssertionError:
        raised = True
    assert raised, "Hkv != 1 should trigger AssertionError"


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────
def main() -> int:
    print(f"device: {DEVICE}")
    print(f"torch:  {torch.__version__}")
    if torch.cuda.is_available():
        print(
            f"cuda:   {torch.cuda.get_device_name(0)}  "
            f"capability={torch.cuda.get_device_capability(0)}"
        )
    print()

    _run("01 basic_smoke", test_basic_smoke)
    _run("02 matches_reference", test_matches_reference)
    _run("03 with_attn_sink", test_with_attn_sink)
    _run("04 invalid_indices_negative", test_invalid_indices_negative)
    _run("05 invalid_indices_overflow", test_invalid_indices_overflow)
    _run("06 topk_length", test_topk_length)
    _run("07 topk_length_zero", test_topk_length_zero)
    _run("08 preallocated_out", test_preallocated_out)
    _run("09 attn_sink_with_invalid", test_attn_sink_with_invalid)
    _run("10 shape_assertions", test_shape_assertions)
    _run("11 chunk_equivalence", test_chunk_equivalence)
    _run("12 chunk_with_invalid_and_sink", test_chunk_with_invalid_and_sink)

    total = PASS + FAIL
    print()
    print(f"==== {PASS}/{total} PASS, {FAIL} FAIL ====")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
