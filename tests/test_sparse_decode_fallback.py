#!/usr/bin/env python3
"""
test_sparse_decode_fallback.py
==============================

本地 mock test，验证 sparse_decode_fwd_sm120_fallback.py 里的两个核心函数：
  1) _sm120_dequant_kv_token: cache uint8 → bf16 (N, 512), NoPE+RoPE 正确还原
  2) _sm120_sparse_mla_decode_fallback: 端到端 shape/dtype/mask/sink 语义

设计原则：
  - 不依赖远程或 SM100 kernel，本地 CPU/CUDA 都能跑（CPU 优先，方便快速迭代）
  - 每个 test 输出 PASS/FAIL + 简明 diff
  - 测 layout 的方式：手工构造已知 byte pattern → 验证 dequant 结果与"用 numpy 单独算的预期"一致

注意：
  - torch.float8_e4m3fn 在 CPU 上 dtype 对象存在但实际算子有限；我们只用 view+to(fp32) 的 bitcast 转换，应该可用
  - 如果 CPU 不支持 fp8，自动切到 CUDA
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch


# ── load module ─────────────────────────────────────────────────────────
HERE = Path(__file__).parent
# Look in: same dir (legacy flat layout), then ../patches/ (repo layout)
_CANDIDATES = [
    HERE / "sparse_decode_fwd_sm120_fallback.py",
    HERE.parent / "patches" / "sparse_decode_fwd_sm120_fallback.py",
]
PATCH_PATH = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])
assert PATCH_PATH.exists(), "missing patch file. Tried:\n  " + "\n  ".join(
    str(c) for c in _CANDIDATES
)

# We can't `import` the patch file directly (it's a CLI tool), so we exec the
# helpers block in a stub module that has `torch` defined.
import types

# Read the patch file and extract just the helpers block
src = PATCH_PATH.read_text()
begin_marker = (
    "# SM120_SPARSE_DECODE_FALLBACK_BEGIN  (do NOT hand-edit; managed by patch)"
)
end_marker = "# SM120_SPARSE_DECODE_FALLBACK_END"
i = src.index(begin_marker)
# end marker also appears in the docstring example near top; pick the one AFTER i
j = src.index(end_marker, i) + len(end_marker)
helpers_code = src[i:j]

# Build a stub module
mod = types.ModuleType("sm120_fallback_mod")
mod.__dict__["torch"] = torch
exec(compile(helpers_code, "<helpers>", "exec"), mod.__dict__)

_dequant = mod._sm120_dequant_kv_token
_decode = mod._sm120_sparse_mla_decode_fallback


# ── helpers ─────────────────────────────────────────────────────────────
def _device() -> torch.device:
    # CPU first for fast iteration; fall back to CUDA only if needed.
    return torch.device("cpu")


def _make_cache(
    num_blocks: int,
    ps: int,
    *,
    head_bytes: int = 584,
    device=None,
) -> torch.Tensor:
    """Build an uninitialized fp8_ds_mla SWA cache, all zeros."""
    device = device or _device()
    return torch.zeros(
        (num_blocks, ps, 1, head_bytes), dtype=torch.uint8, device=device
    )


def _fill_random_clean(cache: torch.Tensor, seed: int) -> None:
    """Fill cache with synthetic but well-behaved bytes for end-to-end tests.

    We need to avoid:
      - fp8_e4m3fn NaN bytes (0x7F, 0xFF) in the NoPE data segment
      - huge ue8m0 scale bytes (e.g. 0xFE = 2^127) in the scale segment,
        which produce Inf when multiplied with fp8 values

    Real vLLM cache-write CUDA kernels never produce these; we synthesise here
    so that NaN/Inf in output signals a fallback bug, not test data poisoning.
    """
    g = torch.Generator().manual_seed(seed)
    num_blocks, ps, _, head_bytes = cache.shape
    TOKEN_STRIDE = 576
    SCALE_DIM = head_bytes - TOKEN_STRIDE  # 8 for SWA

    # Data segment is split per-token: [0:448] fp8 NoPE | [448:576] bf16 RoPE.
    # NoPE: random fp8_e4m3fn bytes, avoid 0x7F / 0xFF (NaN).
    nope_seg = torch.randint(
        0, 256, (num_blocks, ps, 448), generator=g, dtype=torch.int32
    )
    nope_seg = torch.where(
        (nope_seg == 127) | (nope_seg == 255), torch.zeros_like(nope_seg), nope_seg
    )
    # RoPE: synthesise bf16 values in [-1, 1] then view as bytes (LE).
    rope_vals = (torch.rand((num_blocks, ps, 64), generator=g) * 2.0 - 1.0).to(
        torch.bfloat16
    )
    rope_bytes = rope_vals.view(torch.uint8).reshape(num_blocks, ps, 128)
    # Combine NoPE (448) + RoPE (128) = 576 per token
    data_per_token = torch.cat(
        [nope_seg.to(torch.uint8), rope_bytes], dim=-1
    )  # (nb, ps, 576)
    data_seg = data_per_token.reshape(num_blocks, ps * TOKEN_STRIDE)

    # Scale segment: ue8m0 byte = exponent + 127; restrict exponent to [-4, 4]
    # → byte in [123, 131] — gives scales in [2^-4, 2^4] = [0.0625, 16]
    scale_seg = (
        torch.randint(
            123, 132, (num_blocks, ps * SCALE_DIM), generator=g, dtype=torch.int32
        )
    ).to(torch.uint8)

    flat = torch.cat([data_seg, scale_seg], dim=-1)
    assert flat.shape == (num_blocks, ps * head_bytes), flat.shape
    cache.copy_(flat.view(num_blocks, ps, 1, head_bytes))


def _ue8m0_byte(exponent: int) -> int:
    """encode scale 2^e as ue8m0 byte (e + 127)."""
    v = exponent + 127
    assert 0 <= v <= 255, exponent
    return v


def _f32_to_fp8e4m3_byte(x: float) -> int:
    """Convert one fp32 value to fp8_e4m3fn byte via torch's cast."""
    t = torch.tensor([x], dtype=torch.float32).to(torch.float8_e4m3fn)
    return int(t.view(torch.uint8).item())


def _fp8e4m3_byte_to_f32(b: int) -> float:
    t = torch.tensor([b], dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
    return float(t.item())


def _bf16_to_bytes(x: float) -> tuple[int, int]:
    """Return 2 bytes (little-endian) for one bf16 value."""
    t = torch.tensor([x], dtype=torch.bfloat16)
    raw = t.view(torch.uint8)  # 2 bytes
    return int(raw[0].item()), int(raw[1].item())


def _write_token(
    cache: torch.Tensor,
    block_idx: int,
    pos_idx: int,
    nope_f: list[float],  # length 448, will be fp8-quantized with given scales
    nope_scales_exp: list[int],  # length 7, exponent for 2^e per 64-elem block
    rope_f: list[float],  # length 64, will be bf16 quantized
) -> None:
    """Write one token's data + scales into the segregated cache layout.

    Pretend layout per block:
      bytes [0,             ps*576):    token data
        per token at offset = pos*576, 576 bytes:
          [0:448]   fp8 NoPE
          [448:576] bf16 RoPE (64 elements * 2 bytes)
      bytes [ps*576, ps*584):           ue8m0 scales
        per token at offset = ps*576 + pos*8, 8 bytes (7 real + 1 pad)
    """
    num_blocks, ps, _, head_bytes = cache.shape
    TOKEN_STRIDE = 576
    SCALE_DIM = head_bytes - TOKEN_STRIDE
    assert len(nope_f) == 448
    assert len(nope_scales_exp) == 7
    assert len(rope_f) == 64

    flat = cache.view(num_blocks, ps * head_bytes)

    # Quantize NoPE per 64-elem block: divide by 2^e then cast to fp8.
    fp8_bytes = []
    for blk in range(7):
        e = nope_scales_exp[blk]
        inv = 2.0 ** (-e)
        for i in range(64):
            x = nope_f[blk * 64 + i] * inv
            fp8_bytes.append(_f32_to_fp8e4m3_byte(x))
    assert len(fp8_bytes) == 448

    # Write NoPE 448 bytes
    data_off = pos_idx * TOKEN_STRIDE
    for i, b in enumerate(fp8_bytes):
        flat[block_idx, data_off + i] = b

    # Write RoPE 128 bytes (64 bf16 LE)
    for i, x in enumerate(rope_f):
        b0, b1 = _bf16_to_bytes(x)
        flat[block_idx, data_off + 448 + i * 2] = b0
        flat[block_idx, data_off + 448 + i * 2 + 1] = b1

    # Write scales (7 + 1 pad)
    scale_off = ps * TOKEN_STRIDE + pos_idx * SCALE_DIM
    for i, e in enumerate(nope_scales_exp):
        flat[block_idx, scale_off + i] = _ue8m0_byte(e)
    flat[block_idx, scale_off + 7] = 0  # pad


def _passfail(name: str, ok: bool, info: str = "") -> bool:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}{':  ' + info if info else ''}")
    return ok


# ── tests ───────────────────────────────────────────────────────────────
def test_dequant_single_token_known_pattern() -> bool:
    """Construct cache with known NoPE/RoPE values and verify dequant inverts the quant."""
    print("\ntest_dequant_single_token_known_pattern")
    ps = 64
    num_blocks = 4
    cache = _make_cache(num_blocks, ps, head_bytes=584)

    # NoPE: per block, use the block index +1 as scale (2^1..2^7), values like 1.0, -2.5, ...
    nope_f = []
    nope_scales_exp = [1, 2, 3, 4, 5, 6, 7]  # block 0 uses 2^1, block 6 uses 2^7
    for blk in range(7):
        for i in range(64):
            # Values within FP8 e4m3 range when divided by scale: keep abs < 448
            x = (i - 32) * 0.5 * (2.0 ** nope_scales_exp[blk]) / 32.0
            nope_f.append(x)
    rope_f = [float((i - 32) * 0.03125) for i in range(64)]

    block_idx, pos_idx = 2, 17
    _write_token(cache, block_idx, pos_idx, nope_f, nope_scales_exp, rope_f)

    # Dequant via the function under test
    global_idx = torch.tensor([block_idx * ps + pos_idx], dtype=torch.int32)
    K = _dequant(cache, global_idx)  # (1, 512)
    K_np = K.float().cpu().numpy()[0]
    assert K_np.shape == (512,)

    nope_out = K_np[:448]
    rope_out = K_np[448:]

    # Compute "expected" by independently re-doing quant/dequant
    expected_nope = np.zeros(448, dtype=np.float32)
    for blk in range(7):
        e = nope_scales_exp[blk]
        inv = 2.0 ** (-e)
        scale = 2.0**e
        for i in range(64):
            x = nope_f[blk * 64 + i] * inv
            # round-trip through fp8_e4m3fn
            b = _f32_to_fp8e4m3_byte(x)
            x_q = _fp8e4m3_byte_to_f32(b)
            expected_nope[blk * 64 + i] = x_q * scale

    expected_rope = np.zeros(64, dtype=np.float32)
    for i in range(64):
        b0, b1 = _bf16_to_bytes(rope_f[i])
        t = torch.tensor([b0, b1], dtype=torch.uint8).view(torch.bfloat16).float()
        expected_rope[i] = float(t.item())

    nope_diff = np.max(np.abs(nope_out - expected_nope))
    rope_diff = np.max(np.abs(rope_out - expected_rope))

    ok1 = _passfail(
        "nope_max_abs_diff < 1e-2", nope_diff < 1e-2, f"got {nope_diff:.6f}"
    )
    ok2 = _passfail(
        "rope_max_abs_diff < 1e-2", rope_diff < 1e-2, f"got {rope_diff:.6f}"
    )
    return ok1 and ok2


def test_dequant_multiple_tokens_across_blocks() -> bool:
    """Verify gather across different (block_idx, pos_idx) combinations works."""
    print("\ntest_dequant_multiple_tokens_across_blocks")
    ps = 64
    num_blocks = 8
    cache = _make_cache(num_blocks, ps, head_bytes=584)

    # Put a unique signature in each of 5 tokens
    targets = [(0, 0), (1, 33), (3, 7), (5, 63), (7, 12)]
    expected_K = np.zeros((len(targets), 512), dtype=np.float32)

    for k, (bi, pi) in enumerate(targets):
        nope_scales_exp = [k % 7] * 7  # use simple scale per token
        nope_f = [(k * 13.0 + i * 0.1) for i in range(448)]
        rope_f = [(k * 7.0 + i * 0.05) for i in range(64)]
        _write_token(cache, bi, pi, nope_f, nope_scales_exp, rope_f)
        # expected via independent round-trip
        for blk in range(7):
            e = nope_scales_exp[blk]
            inv = 2.0 ** (-e)
            s = 2.0**e
            for i in range(64):
                b = _f32_to_fp8e4m3_byte(nope_f[blk * 64 + i] * inv)
                expected_K[k, blk * 64 + i] = _fp8e4m3_byte_to_f32(b) * s
        for i in range(64):
            b0, b1 = _bf16_to_bytes(rope_f[i])
            t = torch.tensor([b0, b1], dtype=torch.uint8).view(torch.bfloat16).float()
            expected_K[k, 448 + i] = float(t.item())

    indices = torch.tensor([bi * ps + pi for bi, pi in targets], dtype=torch.int32)
    K = _dequant(cache, indices).float().cpu().numpy()
    diff = np.max(np.abs(K - expected_K))
    return _passfail("multi-token max_abs_diff < 1e-2", diff < 1e-2, f"got {diff:.6f}")


def test_decode_end_to_end_shape_and_dtype(dq: int = 576) -> bool:
    """End-to-end fallback should not crash; output should have right shape/dtype.

    `dq` is the last-dim of Q. Hypotheses:
      - dq=576: upper layer passes NoPE+RoPE concatenated (original assumption)
      - dq=512: upper layer pre-splits and passes only NoPE  (recon hypothesis 1)
    """
    print(f"\ntest_decode_end_to_end_shape_and_dtype  [Dq={dq}]")
    ps = 64
    num_blocks = 16
    Hq = 64  # padded_heads = 64 for DSV4-Flash (64 attention heads)
    B = 2
    Sq = 1
    topk = 64
    cache = _make_cache(num_blocks, ps, head_bytes=584)
    # Fill cache with random data, avoiding fp8_e4m3fn NaN bytes (0x7F/0xFF)
    _fill_random_clean(cache, 0)

    q = torch.randn(B, Sq, Hq, dq, dtype=torch.bfloat16)
    # Valid indices in range
    idx_max = num_blocks * ps
    indices = torch.randint(0, idx_max, (B, Sq, topk), dtype=torch.int32)
    # Make some invalid
    indices[0, 0, -3:] = -1
    topk_length = torch.tensor([topk - 3, topk], dtype=torch.int32)

    out_buf = torch.empty(B, Sq, Hq, 512, dtype=torch.bfloat16)
    out, lse = _decode(
        q=q,
        k_cache=cache,
        indices_in_kvcache=indices,
        topk_length=topk_length,
        attn_sink=None,
        extra_k_cache=None,
        extra_indices_in_kvcache=None,
        extra_topk_length=None,
        head_dim_v=512,
        softmax_scale=1.0 / math.sqrt(dq),
        out=out_buf,
    )
    ok1 = _passfail(
        "out shape", out.shape == (B, Sq, Hq, 512), f"got {tuple(out.shape)}"
    )
    ok2 = _passfail(
        "out dtype bfloat16", out.dtype == torch.bfloat16, f"got {out.dtype}"
    )
    ok3 = _passfail("lse shape", lse.shape == (B, Hq, Sq), f"got {tuple(lse.shape)}")
    ok4 = _passfail("no NaN in out", not torch.isnan(out).any().item())
    return ok1 and ok2 and ok3 and ok4


def test_decode_with_extra_cache_and_sink(dq: int = 576) -> bool:
    """Verify path with extra_k_cache + extra_indices + attn_sink completes."""
    print(f"\ntest_decode_with_extra_cache_and_sink  [Dq={dq}]")
    ps_main = 64
    ps_extra = 64
    num_blocks_main = 8
    num_blocks_extra = 8
    Hq = 64
    B = 1
    Sq = 1
    topk_main = 32
    topk_extra = 64

    main_cache = _make_cache(num_blocks_main, ps_main, head_bytes=584)
    extra_cache = _make_cache(num_blocks_extra, ps_extra, head_bytes=584)
    _fill_random_clean(main_cache, 1)
    _fill_random_clean(extra_cache, 2)

    q = torch.randn(B, Sq, Hq, dq, dtype=torch.bfloat16)
    idx_main = torch.randint(
        0, num_blocks_main * ps_main, (B, Sq, topk_main), dtype=torch.int32
    )
    idx_extra = torch.randint(
        0, num_blocks_extra * ps_extra, (B, Sq, topk_extra), dtype=torch.int32
    )

    # First call without sink to get lse magnitude
    out_nosink, lse_nosink = _decode(
        q=q,
        k_cache=main_cache,
        indices_in_kvcache=idx_main,
        topk_length=None,
        attn_sink=None,
        extra_k_cache=extra_cache,
        extra_indices_in_kvcache=idx_extra,
        extra_topk_length=None,
        head_dim_v=512,
        softmax_scale=1.0 / math.sqrt(dq),
        out=None,
    )
    # Choose sink near lse so sigmoid is not saturated. lse shape (B, Hq, Sq).
    lse_mean = lse_nosink.float().mean().item()
    sink = torch.full((Hq,), lse_mean, dtype=torch.float32)  # → sigmoid≈0.5

    out, lse = _decode(
        q=q,
        k_cache=main_cache,
        indices_in_kvcache=idx_main,
        topk_length=None,
        attn_sink=sink,
        extra_k_cache=extra_cache,
        extra_indices_in_kvcache=idx_extra,
        extra_topk_length=None,
        head_dim_v=512,
        softmax_scale=1.0 / math.sqrt(dq),
        out=None,
    )
    ok1 = _passfail(
        "with extra: shape", out.shape == (B, Sq, Hq, 512), f"got {tuple(out.shape)}"
    )
    ok2 = _passfail("with extra: no NaN", not torch.isnan(out).any().item())
    # Sink should multiply out by sigmoid(lse - sink) elementwise per head.
    # Verify: out ≈ out_nosink * sigmoid(lse - sink) (broadcast over last dim).
    # lse shape is (B, Hq, Sq); reshape to (B, Sq, Hq, 1) to match out (B,Sq,Hq,512).
    expected_scale = torch.sigmoid(
        lse.float().permute(0, 2, 1).unsqueeze(-1) - sink.reshape(1, 1, Hq, 1)
    )
    expected_out = out_nosink.float() * expected_scale
    diff = (out.float() - expected_out).abs().max().item()
    # Use relative error: bf16 has ~0.8% precision; two independent bf16
    # round-trips (out and out_nosink) can compound up to a few %.
    rel = diff / max(out_nosink.float().abs().max().item(), 1e-6)
    ok3 = _passfail(
        "sink matches sigmoid(lse - sink)",
        rel < 0.05,
        f"max_abs_diff={diff:.5f} rel={rel:.4f}",
    )
    mag_with = out.float().abs().mean().item()
    mag_no = out_nosink.float().abs().mean().item()
    ok4 = _passfail(
        "sink near lse_mean reduces magnitude ~0.5x",
        mag_with < mag_no * 0.9,
        f"with={mag_with:.4f} no={mag_no:.4f} (expect ~0.5x)",
    )
    return ok1 and ok2 and ok3 and ok4


def test_all_invalid_indices_returns_zero_no_nan(dq: int = 576) -> bool:
    """If all indices for a query are invalid, output must be 0, no NaN."""
    print(f"\ntest_all_invalid_indices_returns_zero_no_nan  [Dq={dq}]")
    ps = 64
    num_blocks = 4
    Hq = 64
    B = 1
    Sq = 1
    topk = 16
    cache = _make_cache(num_blocks, ps, head_bytes=584)
    _fill_random_clean(cache, 2)

    q = torch.randn(B, Sq, Hq, dq, dtype=torch.bfloat16)
    # All invalid (negative)
    indices = torch.full((B, Sq, topk), -1, dtype=torch.int32)

    out, lse = _decode(
        q=q,
        k_cache=cache,
        indices_in_kvcache=indices,
        topk_length=None,
        attn_sink=None,
        extra_k_cache=None,
        extra_indices_in_kvcache=None,
        extra_topk_length=None,
        head_dim_v=512,
        softmax_scale=1.0 / math.sqrt(dq),
        out=None,
    )
    ok1 = _passfail("all-invalid: no NaN", not torch.isnan(out).any().item())
    ok2 = _passfail(
        "all-invalid: out is zero",
        out.float().abs().max().item() < 1e-6,
        f"max_abs={out.float().abs().max().item():.6f}",
    )
    return ok1 and ok2


def test_topk_length_masking_equiv_invalid(dq: int = 576) -> bool:
    """topk_length=N should be equivalent to setting indices[N:] = -1."""
    print(f"\ntest_topk_length_masking_equiv_invalid  [Dq={dq}]")
    ps = 64
    num_blocks = 4
    Hq = 64
    B = 1
    Sq = 1
    topk = 16
    cache = _make_cache(num_blocks, ps, head_bytes=584)
    _fill_random_clean(cache, 3)

    q = torch.randn(B, Sq, Hq, dq, dtype=torch.bfloat16)
    indices_full = torch.randint(0, num_blocks * ps, (B, Sq, topk), dtype=torch.int32)
    valid_len = 10

    # Path A: topk_length=10
    out_a, _ = _decode(
        q=q,
        k_cache=cache,
        indices_in_kvcache=indices_full.clone(),
        topk_length=torch.tensor([valid_len], dtype=torch.int32),
        attn_sink=None,
        extra_k_cache=None,
        extra_indices_in_kvcache=None,
        extra_topk_length=None,
        head_dim_v=512,
        softmax_scale=1.0 / math.sqrt(dq),
        out=None,
    )
    # Path B: indices[10:] = -1, topk_length=None
    indices_masked = indices_full.clone()
    indices_masked[..., valid_len:] = -1
    out_b, _ = _decode(
        q=q,
        k_cache=cache,
        indices_in_kvcache=indices_masked,
        topk_length=None,
        attn_sink=None,
        extra_k_cache=None,
        extra_indices_in_kvcache=None,
        extra_topk_length=None,
        head_dim_v=512,
        softmax_scale=1.0 / math.sqrt(dq),
        out=None,
    )
    diff = (out_a.float() - out_b.float()).abs().max().item()
    return _passfail(
        "topk_length ≡ -1 masking", diff < 1e-3, f"max_abs_diff={diff:.6f}"
    )


def main() -> int:
    """Run all tests.

    recon_q_shape.sh resolved the Q.last_dim question:
      * V4-Flash sparse decode receives Q as NoPE-only latent, Dq=512.
      * Old V3-style hypothesis (Dq=576 = NoPE_512+RoPE_64 concat) is wrong.

    So patch v3's fallback asserts Dq==512. We still parametrize tests over
    Dq ∈ {512, 576} as a regression guard:
      * Dq=512 group → MUST all PASS (this is the canonical path)
      * Dq=576 group → MUST all FAIL with AssertionError (the fallback should
        explicitly reject the old layout to surface upstream changes early)

    Dequant tests are Dq-independent (they only validate K-cache decoding).
    """
    print("=" * 64)
    print("test_sparse_decode_fallback.py")
    print("device:", _device())
    print("=" * 64)
    results = []

    # ── Dq-independent tests (K-cache dequant only) ─────────────────────
    print("\n--- Dq-independent: K-cache dequant ---")
    for fn in (
        test_dequant_single_token_known_pattern,
        test_dequant_multiple_tokens_across_blocks,
    ):
        try:
            results.append((fn.__name__, fn()))
        except Exception:
            import traceback

            traceback.print_exc()
            results.append((fn.__name__, False))

    # ── Dq-parametric tests (end-to-end decode) ─────────────────────────
    decode_tests = (
        test_decode_end_to_end_shape_and_dtype,
        test_decode_with_extra_cache_and_sink,
        test_all_invalid_indices_returns_zero_no_nan,
        test_topk_length_masking_equiv_invalid,
    )

    # Canonical: Dq=512 must all PASS
    print("\n--- Dq-canonical e2e (Dq=512, MUST PASS) ---")
    for fn in decode_tests:
        label = f"{fn.__name__}[Dq=512]"
        try:
            results.append((label, fn(dq=512)))
        except Exception:
            import traceback

            traceback.print_exc()
            results.append((label, False))

    # Regression guard: Dq=576 must all REJECT (AssertionError expected)
    print("\n--- Dq-rejection guard (Dq=576, MUST raise AssertionError) ---")
    for fn in decode_tests:
        label = f"{fn.__name__}[Dq=576-rejected]"
        try:
            fn(dq=576)
            # If we reach here, fallback did NOT reject Dq=576 → regression
            print(f"  [FAIL] {label}: fallback accepted Dq=576 (should reject)")
            results.append((label, False))
        except AssertionError as e:
            if "512" in str(e):
                print(f"  [PASS] {label}: correctly raised AssertionError ({e})")
                results.append((label, True))
            else:
                print(f"  [FAIL] {label}: wrong AssertionError: {e}")
                results.append((label, False))
        except Exception as e:
            print(f"  [FAIL] {label}: unexpected {type(e).__name__}: {e}")
            results.append((label, False))

    print("\n" + "=" * 64)
    print("summary:")
    for n, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    print("=" * 64)
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())
