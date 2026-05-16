#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/third_party/flashmla/"
    "flash_mla_interface.py"
)
BACKUP_SUFFIX = ".bak_sparse_decode_fb"

# ------------------------------------------------------------------
# 注入的 fallback 代码块。整体策略：
#   1. 提供 _sm120_unpack_kv() 把 segregated 2D uint8 cache 还原成 bf16 K (576) + V (512)。
#   2. 提供 _sm120_sparse_mla_decode_fallback() 完成 indices gather + concat extra +
#      MLA attention (softmax(QK^T) @ V) + attn_sink + topk_length mask。
#   3. 在 sparse 分支前 (line 154 `if topk is not None:`) 插一个 env 拦截：
#      if VLLM_SPARSE_DECODE_FALLBACK=1: 用 fallback 路径；否则原 cuda 调用。
#
# 这段代码会用 sentinel 注释包裹，便于 --revert / --check：
#     # SM120_SPARSE_DECODE_FALLBACK_BEGIN
#     ... (helpers + import-guard env load) ...
#     # SM120_SPARSE_DECODE_FALLBACK_END
# ------------------------------------------------------------------

HELPERS_BLOCK = (
    '''
# SM120_SPARSE_DECODE_FALLBACK_BEGIN  (do NOT hand-edit; managed by patch)
import os as _sm120_os
import math as _sm120_math


_SM120_SPARSE_DECODE_FALLBACK_ENABLED = (
    _sm120_os.environ.get("VLLM_SPARSE_DECODE_FALLBACK", "0") == "1"
)
_SM120_SPARSE_DECODE_FALLBACK_DEBUG = (
    _sm120_os.environ.get("VLLM_SPARSE_DECODE_FALLBACK_DEBUG", "0") == "1"
)
_SM120_SPARSE_DECODE_FALLBACK_LOGGED = False


def _sm120_log_once(msg: str) -> None:
    global _SM120_SPARSE_DECODE_FALLBACK_LOGGED
    if _SM120_SPARSE_DECODE_FALLBACK_DEBUG and not _SM120_SPARSE_DECODE_FALLBACK_LOGGED:
        _SM120_SPARSE_DECODE_FALLBACK_LOGGED = True
        print(f"[SM120_SPARSE_DECODE_FALLBACK] {msg}", flush=True)


def _sm120_dequant_kv_token(
    k_cache: torch.Tensor,      # (num_blocks, page_block_size, 1, head_bytes), uint8
    global_indices: torch.Tensor,  # (N,), int32: block_idx * page_block_size + offset
) -> torch.Tensor:
    """Gather N tokens from a segregated 2D uint8 cache and dequant to bf16 (N, 512).

    Layout per cache block (page_block_size tokens):
        bytes [0,                 ps*576):    token data (448B fp8 NoPE + 128B bf16 RoPE)
        bytes [ps*576, ps*576+ps*8):          ue8m0 scales (7 real + 1 pad)/token
      total = ps * (576 + 8) = ps * 584
    Where ps = page_block_size (e.g. 64 for SWA, 64 for C4A inner block).

    Returns:
        out: (N, 512) bf16. First 448 cols = NoPE (dequant fp8 with ue8m0 scales),
                            last  64 cols = RoPE (bf16 in place).
    """
    assert k_cache.dim() == 4 and k_cache.shape[2] == 1, k_cache.shape
    num_blocks, ps, _, head_bytes = k_cache.shape
    # head_bytes carries both data (576) and scales (8) interleaved by block.
    # head_bytes might be 584 (SWA) or 656 (some other configs). We rely on TOKEN_STRIDE=576.
    TOKEN_STRIDE = 576
    SCALE_DIM = head_bytes - TOKEN_STRIDE   # SWA: 8;  also fits other DSV4 configs
    assert SCALE_DIM >= 7, (
        f"Unexpected fp8_ds_mla head_bytes={head_bytes}, expected >= {TOKEN_STRIDE+7}"
    )

    # Flatten cache view to a 2D buffer [num_blocks, ps * head_bytes] for raw byte access.
    # head_bytes is the per-token *segment* including scale tail, but real layout is:
    #   [0 : ps*TOKEN_STRIDE)               -> token data segment
    #   [ps*TOKEN_STRIDE : ps*TOKEN_STRIDE + ps*SCALE_DIM)  -> scale segment
    # so per-block total is ps*TOKEN_STRIDE + ps*SCALE_DIM, must equal ps*head_bytes.
    # Sanity:
    expected = ps * TOKEN_STRIDE + ps * SCALE_DIM
    actual = ps * head_bytes
    assert expected == actual, (
        f"layout mismatch: ps={ps} head_bytes={head_bytes} expected_total={expected} got={actual}"
    )

    cache_flat = k_cache.view(num_blocks, ps * head_bytes)  # uint8

    block_idx = (global_indices // ps).to(torch.int64)
    pos_idx = (global_indices % ps).to(torch.int64)

    # Token data byte range per token:
    data_off = pos_idx * TOKEN_STRIDE          # [N]
    scale_off = ps * TOKEN_STRIDE + pos_idx * SCALE_DIM   # [N]

    N = global_indices.shape[0]
    # Gather 576-byte token data
    data_byte_idx = data_off.unsqueeze(1) + torch.arange(TOKEN_STRIDE, device=k_cache.device, dtype=torch.int64)
    token_bytes = torch.gather(
        cache_flat[block_idx],            # [N, ps*head_bytes]
        1,
        data_byte_idx,                    # [N, 576]
    )                                     # [N, 576] uint8

    # NoPE = first 448 bytes as fp8_e4m3
    nope_u8 = token_bytes[:, :448].contiguous()
    # RoPE = next 128 bytes as bf16 (64 elems)
    rope_bytes = token_bytes[:, 448:576].contiguous()
    rope = rope_bytes.view(torch.bfloat16).view(N, 64)   # (N, 64) bf16

    # Gather 7 ue8m0 scale bytes (we ignore the 8th pad byte)
    scale_byte_idx = scale_off.unsqueeze(1) + torch.arange(7, device=k_cache.device, dtype=torch.int64)
    scale_u8 = torch.gather(cache_flat[block_idx], 1, scale_byte_idx)  # [N, 7] uint8

    # Dequant fp8 NoPE: reshape (N, 7, 64) and apply per-block scale 2^(byte-127)
    fp8 = nope_u8.view(torch.float8_e4m3fn).view(N, 7, 64).to(torch.float32)
    scale_exp = scale_u8.to(torch.float32) - 127.0
    scale_f = torch.exp2(scale_exp).unsqueeze(-1)        # (N, 7, 1)
    nope = (fp8 * scale_f).view(N, 448).to(torch.bfloat16)

    # Assemble K = [NoPE_448 (dequant fp8) | RoPE_64 (bf16)] = 512-dim bf16.
    # This matches what V4-Flash's sparse attention consumes:
    #   * recon_q_shape.sh + runtime log confirmed: Q comes in as (..., 512), already
    #     projected to NoPE-only latent space by the upper attention layer.
    #   * Cache layout per DeepseekV4SWACacheSpec: 448B fp8 NoPE + 128B bf16 RoPE.
    #   * Attention is a single Q(512) @ K(512).T dot product in latent space —
    #     no separate RoPE concat/contraction like V3 MLA.
    K = torch.empty(N, 512, dtype=torch.bfloat16, device=k_cache.device)
    K[:, :448] = nope
    K[:, 448:] = rope
    return K


def _sm120_sparse_mla_decode_fallback(
    q,                 # (B, Sq, Hq, Dq) bf16, Dq = 512 (NoPE-only latent)
    k_cache,           # (num_blocks, ps, 1, head_bytes) uint8 — SWA cache
    indices_in_kvcache,    # (B, Sq, topk) int32, global slot = block_idx * ps + offset
    topk_length,           # (B,) int32 or None
    attn_sink,             # (Hq,) fp32 or None
    extra_k_cache,         # (num_blocks2, ps2, 1, head_bytes2) uint8 or None
    extra_indices_in_kvcache,  # (B, Sq, extra_topk) int32 or None
    extra_topk_length,         # (B,) int32 or None
    head_dim_v,            # int = 512
    softmax_scale,         # float
    out,                   # (B, Sq, Hq, head_dim_v) bf16 or None
):
    """Pure-PyTorch sparse MLA decode for V4-Flash.

    Recon-verified semantics (recon_q_shape.sh + runtime log):
      * Q arrives as (B, Sq, Hq, 512), already projected to NoPE-only latent
        space by the upper attention layer. There is NO RoPE concat at this
        layer (unlike V3 MLA which sends 576-d Q = NoPE_512+RoPE_64).
      * K is gathered from the segregated fp8_ds_mla cache, dequantized to
        (T, 512) bf16 = [NoPE_448 from fp8 | RoPE_64 from bf16].
      * Attention is plain latent-space dot product:
            logits = Q(512) @ K(512).T * softmax_scale
            attn   = softmax(logits, dim=-1)
            out    = attn @ K(512)                # V == K in latent attention
            lse    = logsumexp(logits, dim=-1)
      * head_dim_v == 512 (the caller asserts this).
      * attn_sink (when given) multiplies out by sigmoid(lse - sink) per head.
      * invalid index convention: idx < 0 → masked with -inf in logits.
    """
    _sm120_log_once(
        f"engaged: q={tuple(q.shape)} ps={k_cache.shape[1]} "
        f"head_bytes={k_cache.shape[-1]} sink={attn_sink is not None} "
        f"extra={extra_k_cache is not None}"
    )

    assert q.dim() == 4, q.shape
    B, Sq, Hq, Dq = q.shape
    device = q.device
    assert head_dim_v == 512, head_dim_v
    assert Dq == 512, (
        f"DSV4-Flash sparse decode expects Q.last_dim == 512 "
        f"(NoPE-only latent); got {Dq}. If you see Dq=576, the upper layer "
        f"may have changed to V3-style concat — re-run recon_q_shape.sh."
    )

    # ── Gather K from main (SWA) cache by indices ─────────────────────
    # indices_in_kvcache shape (B, Sq, topk); flatten across BxSq for gather.
    BSq = B * Sq
    flat_idx = indices_in_kvcache.reshape(BSq, -1).to(torch.int32)  # (BSq, topk)
    topk = flat_idx.shape[-1]

    # Valid mask: index >= 0 (kernel convention: -1 / >=N means invalid)
    invalid = (flat_idx < 0)
    safe_idx = torch.where(invalid, torch.zeros_like(flat_idx), flat_idx)

    K_swa = _sm120_dequant_kv_token(k_cache, safe_idx.reshape(-1))   # (BSq*topk, 512)
    K_swa = K_swa.view(BSq, topk, 512)                                # (BSq, topk, 512)

    # Optional extra cache (C4A / C128A compressed cache)
    if extra_k_cache is not None and extra_indices_in_kvcache is not None:
        flat_eidx = extra_indices_in_kvcache.reshape(BSq, -1).to(torch.int32)
        e_topk = flat_eidx.shape[-1]
        e_invalid = (flat_eidx < 0)
        e_safe = torch.where(e_invalid, torch.zeros_like(flat_eidx), flat_eidx)
        K_extra = _sm120_dequant_kv_token(extra_k_cache, e_safe.reshape(-1))
        K_extra = K_extra.view(BSq, e_topk, 512)
        K_all = torch.cat([K_swa, K_extra], dim=1)                    # (BSq, topk+e_topk, 512)
        invalid_all = torch.cat([invalid, e_invalid], dim=1)
        # topk_length / extra_topk_length apply per (B,) — broadcast to per-(BSq).
        if topk_length is not None or extra_topk_length is not None:
            pos = torch.arange(topk, device=device)
            e_pos = torch.arange(e_topk, device=device)
            # per-batch lens, expand to (BSq, ...)
            if topk_length is not None:
                tl_per = topk_length.repeat_interleave(Sq) if topk_length.shape[0] == B else topk_length
                len_mask = pos.unsqueeze(0) >= tl_per.unsqueeze(1)        # (BSq, topk)
                invalid_all[:, :topk] = invalid_all[:, :topk] | len_mask
            if extra_topk_length is not None:
                etl_per = extra_topk_length.repeat_interleave(Sq) if extra_topk_length.shape[0] == B else extra_topk_length
                e_len_mask = e_pos.unsqueeze(0) >= etl_per.unsqueeze(1)   # (BSq, e_topk)
                invalid_all[:, topk:] = invalid_all[:, topk:] | e_len_mask
    else:
        K_all = K_swa
        invalid_all = invalid
        if topk_length is not None:
            pos = torch.arange(topk, device=device)
            tl_per = topk_length.repeat_interleave(Sq) if topk_length.shape[0] == B else topk_length
            len_mask = pos.unsqueeze(0) >= tl_per.unsqueeze(1)
            invalid_all = invalid_all | len_mask

    Ttot = K_all.shape[1]

    # ── Q for inner product: NoPE-only latent, 512-dim, matches K layout ──
    # No RoPE concat / split needed — V4-Flash sparse attention is fully in
    # latent space (recon-verified).
    q_f = q.reshape(BSq, Hq, 512).to(torch.float32)
    K_f = K_all.to(torch.float32)                                  # (BSq, Ttot, 512)

    # ── Attention logits: (BSq, Hq, Ttot) ────────────────────────────
    logits = torch.bmm(q_f, K_f.transpose(1, 2)) * float(softmax_scale)
    # Mask invalid positions
    neg_inf = float("-inf")
    if invalid_all.any():
        # invalid_all: (BSq, Ttot); broadcast over heads
        logits = logits.masked_fill(invalid_all.unsqueeze(1), neg_inf)

    # ── softmax + lse ────────────────────────────────────────────────
    # Numerically stable: lse = logsumexp(logits, dim=-1); attn = exp(logits - lse)
    lse = torch.logsumexp(logits, dim=-1)                          # (BSq, Hq)
    # Where ALL positions invalid (lse = -inf), guard against NaN: set attn=0
    all_invalid = torch.isinf(lse) & (lse < 0)
    safe_lse = torch.where(all_invalid, torch.zeros_like(lse), lse)
    attn = torch.exp(logits - safe_lse.unsqueeze(-1))              # (BSq, Hq, Ttot)
    attn = torch.where(all_invalid.unsqueeze(-1), torch.zeros_like(attn), attn)

    # ── Output: attn @ V, where V = K[..., :512] (full 512 dims used as value) ──
    out_fp = torch.bmm(attn, K_f)                                  # (BSq, Hq, 512)

    # ── attn_sink: out *= exp(lse) / (exp(lse) + exp(sink)) = sigmoid(lse - sink) ──
    if attn_sink is not None:
        sink = attn_sink.to(torch.float32).reshape(1, Hq)          # (1, Hq)
        # sigmoid(lse - sink), but when all_invalid -> use 0 (no contribution)
        scale_factor = torch.sigmoid(lse - sink)                   # (BSq, Hq)
        scale_factor = torch.where(all_invalid, torch.zeros_like(scale_factor), scale_factor)
        out_fp = out_fp * scale_factor.unsqueeze(-1)

    out_bf = out_fp.to(q.dtype).reshape(B, Sq, Hq, 512)

    if out is not None:
        out.copy_(out_bf)
        return out, lse.reshape(B, Hq, Sq)
    return out_bf, lse.reshape(B, Hq, Sq)
# SM120_SPARSE_DECODE_FALLBACK_END
'''.strip("\n")
    + "\n"
)


# Sentinel for the in-line dispatch swap, inserted right BEFORE the
# `if topk is not None:` line (function body, 4-space indent).
#
# CRITICAL: 4-space indent (function body level). 8-space would land inside
# the preceding `else: have_initialized` block (the v1 bug).
DISPATCH_BLOCK = (
    """
    # SM120_SPARSE_DECODE_FALLBACK_DISPATCH_BEGIN
    if _SM120_SPARSE_DECODE_FALLBACK_ENABLED and topk is not None:
        out, lse = _sm120_sparse_mla_decode_fallback(
            q=q,
            k_cache=k_cache,
            indices_in_kvcache=indices_in_kvcache,
            topk_length=topk_length,
            attn_sink=attn_sink,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices_in_kvcache,
            extra_topk_length=extra_topk_length,
            head_dim_v=head_dim_v,
            softmax_scale=softmax_scale,
            out=out,
        )
        # tile_scheduler_metadata / num_splits are NOT used by callers when
        # sched_meta is FlashMLASchedMeta; we just keep whatever was set.
        return (out, lse)
    # SM120_SPARSE_DECODE_FALLBACK_DISPATCH_END
""".strip("\n")
    + "\n"
)


def _find_helpers_anchor(src: str) -> int:
    """Locate insertion point for helpers: right after `flash_mla_cuda = ...` line."""
    m = re.search(
        r"^flash_mla_cuda\s*=\s*torch\.ops\._flashmla_C\s*$", src, re.MULTILINE
    )
    if not m:
        raise RuntimeError(
            "Could not find `flash_mla_cuda = torch.ops._flashmla_C` anchor"
        )
    # Insert after the line break following the match
    return src.index("\n", m.end()) + 1


def _find_dispatch_anchor(src: str) -> int:
    """Locate insertion point for dispatch swap: right BEFORE `if topk is not None:`
    inside flash_mla_with_kvcache (the only such line that is at 4-space indent)."""
    # We want the unique line `    if topk is not None:`
    m = re.search(r"^    if topk is not None:\s*$", src, re.MULTILINE)
    if not m:
        raise RuntimeError("Could not find `if topk is not None:` dispatch anchor")
    return m.start()


def cmd_check() -> int:
    if not TARGET.exists():
        print(f"[FAIL] target not found: {TARGET}")
        return 2
    src = TARGET.read_text()
    has_helpers = "SM120_SPARSE_DECODE_FALLBACK_BEGIN" in src
    has_dispatch = "SM120_SPARSE_DECODE_FALLBACK_DISPATCH_BEGIN" in src
    has_backup = TARGET.with_suffix(TARGET.suffix + BACKUP_SUFFIX).exists()
    print(f"target:           {TARGET}")
    print(f"backup exists:    {has_backup} ({TARGET.name + BACKUP_SUFFIX})")
    print(f"helpers injected: {has_helpers}")
    print(f"dispatch swap:    {has_dispatch}")
    if has_helpers and has_dispatch:
        print("[OK] patch is APPLIED")
        return 0
    if not has_helpers and not has_dispatch:
        print("[OK] patch is NOT applied (pristine)")
        return 0
    print("[WARN] patch is PARTIALLY applied — recommend --revert then --apply")
    return 1


def cmd_apply() -> int:
    if not TARGET.exists():
        print(f"[FAIL] target not found: {TARGET}")
        return 2
    src = TARGET.read_text()

    if (
        "SM120_SPARSE_DECODE_FALLBACK_BEGIN" in src
        or "SM120_SPARSE_DECODE_FALLBACK_DISPATCH_BEGIN" in src
    ):
        print(
            "[SKIP] patch already (partially) applied. Run --revert first if you want to re-apply."
        )
        return 1

    # backup
    backup = TARGET.with_suffix(TARGET.suffix + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(TARGET, backup)
        print(f"[OK] backup written: {backup}")
    else:
        print(f"[OK] backup already exists: {backup}")

    # 1) inject helpers near top
    anchor = _find_helpers_anchor(src)
    src = src[:anchor] + "\n" + HELPERS_BLOCK + "\n" + src[anchor:]

    # 2) inject dispatch swap before `if topk is not None:`
    anchor2 = _find_dispatch_anchor(src)
    src = src[:anchor2] + DISPATCH_BLOCK + src[anchor2:]

    TARGET.write_text(src)
    print(f"[OK] patch applied to {TARGET}")

    # ── self-check 1: AST parse must succeed (catches indent bugs)
    import ast

    try:
        ast.parse(src, filename=str(TARGET))
        print("[OK] self-check: ast.parse PASSED")
    except SyntaxError as e:
        print(f"[FAIL] self-check: ast.parse FAILED: {e}")
        print("       reverting…")
        backup = TARGET.with_suffix(TARGET.suffix + BACKUP_SUFFIX)
        shutil.copy2(backup, TARGET)
        return 4

    # ── self-check 2: dispatch block must be at 4-space indent (function body)
    #    NOT 8-space (inside else block). This is the v1→v2 fix.
    bad = re.search(
        r"^        # SM120_SPARSE_DECODE_FALLBACK_DISPATCH_BEGIN\s*$",
        src,
        re.MULTILINE,
    )
    good = re.search(
        r"^    # SM120_SPARSE_DECODE_FALLBACK_DISPATCH_BEGIN\s*$",
        src,
        re.MULTILINE,
    )
    if bad and not good:
        print(
            "[FAIL] self-check: dispatch block at 8-space indent (v1 bug). Reverting…"
        )
        backup = TARGET.with_suffix(TARGET.suffix + BACKUP_SUFFIX)
        shutil.copy2(backup, TARGET)
        return 5
    if not good:
        print("[FAIL] self-check: dispatch block not found at expected indent.")
        backup = TARGET.with_suffix(TARGET.suffix + BACKUP_SUFFIX)
        shutil.copy2(backup, TARGET)
        return 6
    print("[OK] self-check: dispatch block at 4-space indent (function body)")

    # ── self-check 3: clear stale .pyc so next import picks up our edit
    pycache = TARGET.parent / "__pycache__"
    if pycache.exists():
        n = 0
        for f in pycache.glob("flash_mla_interface*.pyc"):
            f.unlink()
            n += 1
        print(f"[OK] cleared {n} stale .pyc files in {pycache}")

    # ── self-check 4: live import + introspect via inspect.getsource
    try:
        import importlib

        sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
        # purge any pre-loaded module first
        for mod_name in list(sys.modules):
            if "flash_mla_interface" in mod_name:
                del sys.modules[mod_name]
        m = importlib.import_module("vllm.third_party.flashmla.flash_mla_interface")
        import inspect

        fn_src = inspect.getsource(m.flash_mla_with_kvcache)
        # verify dispatch block sits OUTSIDE the `else:` block by checking
        # it appears AFTER the closing of the else (i.e., at function-body indent)
        lines = fn_src.splitlines()
        for i, line in enumerate(lines):
            if "SM120_SPARSE_DECODE_FALLBACK_DISPATCH_BEGIN" in line:
                # the marker line should start with exactly 4 spaces
                indent = len(line) - len(line.lstrip(" "))
                if indent != 4:
                    print(
                        f"[FAIL] runtime dispatch marker at indent={indent}, "
                        f"expected 4 (line {i + 1} of fn)"
                    )
                    return 7
                print(
                    f"[OK] self-check: runtime dispatch marker at indent=4 "
                    f"(fn line {i + 1})"
                )
                break
        else:
            print("[FAIL] runtime dispatch marker not found in live function!")
            return 8
        flag = getattr(m, "_SM120_SPARSE_DECODE_FALLBACK_ENABLED", None)
        print(f"[OK] self-check: _SM120_SPARSE_DECODE_FALLBACK_ENABLED = {flag}")
    except Exception as e:
        print(f"[WARN] live import check skipped: {e!r}")

    # final check via cmd_check
    code = cmd_check()
    return 0 if code == 0 else 3


def cmd_revert() -> int:
    backup = TARGET.with_suffix(TARGET.suffix + BACKUP_SUFFIX)
    if not backup.exists():
        print(f"[FAIL] no backup at {backup}; cannot revert safely.")
        return 2
    shutil.copy2(backup, TARGET)
    print(f"[OK] reverted {TARGET} from {backup}")
    return cmd_check()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true")
    g.add_argument("--revert", action="store_true")
    g.add_argument("--check", action="store_true")
    args = p.parse_args(argv)

    if args.apply:
        return cmd_apply()
    if args.revert:
        return cmd_revert()
    if args.check:
        return cmd_check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
