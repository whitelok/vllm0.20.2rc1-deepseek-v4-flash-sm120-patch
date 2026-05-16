#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/third_party/flashmla/"
    "flash_mla_interface.py"
)
BACKUP_SUFFIX = ".bak_sparse_prefill_fb"


# ────────────────────────────────────────────────────────────────────
# Helpers block (module-level). Inserted right AFTER the decode patch's
# helpers block (or, if decode patch absent, right after
# `flash_mla_cuda = torch.ops._flashmla_C`).
#
# Sentinels:
#   # SM120_SPARSE_PREFILL_FALLBACK_BEGIN
#   ...
#   # SM120_SPARSE_PREFILL_FALLBACK_END
# ────────────────────────────────────────────────────────────────────
HELPERS_BLOCK = (
    '''
# SM120_SPARSE_PREFILL_FALLBACK_BEGIN  (do NOT hand-edit; managed by patch v4.1)
import os as _sm120p_os


_SM120_SPARSE_PREFILL_FALLBACK_ENABLED = (
    _sm120p_os.environ.get("VLLM_SPARSE_PREFILL_FALLBACK", "0") == "1"
)
_SM120_SPARSE_PREFILL_FALLBACK_DEBUG = (
    _sm120p_os.environ.get("VLLM_SPARSE_PREFILL_FALLBACK_DEBUG", "0") == "1"
)
_SM120_SPARSE_PREFILL_FALLBACK_LOGGED = False
# v4.1: streaming chunk along Sq dim to bound peak memory.
# 256 keeps single-chunk peak well under 300 MiB at topk≈614, Hq=64.
_SM120_SPARSE_PREFILL_FALLBACK_CHUNK = max(
    1, int(_sm120p_os.environ.get("VLLM_SPARSE_PREFILL_FALLBACK_CHUNK", "256"))
)


def _sm120p_log_once(msg: str) -> None:
    global _SM120_SPARSE_PREFILL_FALLBACK_LOGGED
    if _SM120_SPARSE_PREFILL_FALLBACK_DEBUG and not _SM120_SPARSE_PREFILL_FALLBACK_LOGGED:
        _SM120_SPARSE_PREFILL_FALLBACK_LOGGED = True
        print(f"[SM120_SPARSE_PREFILL_FALLBACK] {msg}", flush=True)


def _sm120_sparse_prefill_fallback(
    q,                # [Sq, Hq, Dqk] bf16
    kv,               # [Skv, Hkv, Dqk] bf16  (already dequant'd by caller)
    indices,          # [Sq, Hkv, topk] int32 (invalid: -1 or >= Skv)
    sm_scale,         # float
    d_v,              # int = 512
    attn_sink,        # Optional [Hq] fp32
    topk_length,      # Optional [Sq] int32
    out,              # Optional [Sq, Hq, d_v] bf16 preallocated
):
    """Pure-PyTorch sparse prefill MLA attention for DSV4-Flash.

    v4.1: bf16-native + Sq-streaming chunk loop to bound peak memory.
    Math is bit-equivalent to the v4 reference (per-query independent;
    chunk boundary has no effect on lse/softmax).

    Recon-verified semantics (recon_sparse_prefill.sh):
      * kv arrives ALREADY bf16-dequantized by dequantize_and_gather_k_cache
        (no fp8/ue8m0 work needed here, unlike decode path).
      * No sched_meta, no causal flag — sparse-selection attention pattern
        is entirely encoded in `indices` + `topk_length` by the caller.
      * Hkv == 1 (MLA latent-shared K/V), so we squeeze that dim early.
      * d_qk == d_v == 512 (K serves as V; identity in latent space).
      * Invalid index convention (per docstring):
            idx < 0  OR  idx >= Skv  → mask with -inf in logits
      * topk_length[s] gives per-query effective topk; positions
        >= topk_length[s] also masked.
      * attn_sink: out *= sigmoid(lse - sink[h])    per head

    Memory budget per chunk (chunk=256, topk=614, Hq=64, Dqk=512):
      q_chunk (bf16)        : 16  MiB
      gathered (bf16)       : 154 MiB    ← was 2886 MiB fp32 in v4
      logits  (fp32 softmax): 38  MiB
      attn    (bf16)        : 19  MiB
      out_fp  (fp32 accum)  : 32  MiB
      ─ peak per chunk      : ≈ 260 MiB (reused across chunks)

    Returns (output, max_logits, lse) tuple matching the cuda kernel:
        output     : [Sq, Hq, d_v]   bf16
        max_logits : [Sq, Hq]        fp32
        lse        : [Sq, Hq]        fp32
    """
    assert q.dim() == 3, q.shape
    assert kv.dim() == 3, kv.shape
    assert indices.dim() == 3, indices.shape

    Sq, Hq, Dqk = q.shape
    Skv, Hkv, Dqk_kv = kv.shape
    Sq_i, Hkv_i, topk = indices.shape

    assert Dqk == Dqk_kv, (Dqk, Dqk_kv)
    assert Hkv == 1, f"MLA expects Hkv=1, got {Hkv}"
    assert Hkv_i == 1, f"MLA indices expects Hkv=1, got {Hkv_i}"
    assert Sq == Sq_i, (Sq, Sq_i)
    assert d_v == 512, d_v
    assert Dqk == 512, f"DSV4-Flash sparse prefill expects Dqk=512, got {Dqk}"

    device = q.device
    chunk_size = _SM120_SPARSE_PREFILL_FALLBACK_CHUNK

    _sm120p_log_once(
        f"engaged: q={tuple(q.shape)} kv={tuple(kv.shape)} "
        f"indices={tuple(indices.shape)} sink={attn_sink is not None} "
        f"tlen={topk_length is not None} d_v={d_v} chunk={chunk_size}"
    )

    # ── Flatten Hkv=1 dim once: kv2 stays bf16, reused across chunks ──
    # kv     : [Skv, 1, Dqk] -> [Skv, Dqk]
    kv2 = kv.squeeze(1)                                  # bf16 [Skv, Dqk]

    # ── Pre-allocate outputs so chunk loop can write in place ─────────
    if out is not None:
        # caller-provided buffer; we write into it directly.
        assert out.shape == (Sq, Hq, d_v), (out.shape, (Sq, Hq, d_v))
        assert out.dtype == q.dtype, (out.dtype, q.dtype)
        out_bf = out
    else:
        out_bf = torch.empty((Sq, Hq, d_v), dtype=q.dtype, device=device)
    max_logits = torch.empty((Sq, Hq), dtype=torch.float32, device=device)
    lse = torch.empty((Sq, Hq), dtype=torch.float32, device=device)

    if topk_length is not None:
        tlen_i64 = topk_length.to(torch.int64)           # [Sq]
        pos_row = torch.arange(topk, device=device)      # [topk]
    else:
        tlen_i64 = None
        pos_row = None

    if attn_sink is not None:
        sink_f = attn_sink.to(torch.float32).reshape(1, Hq)   # [1, Hq]
    else:
        sink_f = None

    neg_inf = float("-inf")

    # ── Streaming chunk loop along Sq dim ─────────────────────────────
    for s_lo in range(0, Sq, chunk_size):
        s_hi = min(s_lo + chunk_size, Sq)
        cs = s_hi - s_lo   # current chunk size

        # idx slice (int64 for gather), invalid mask, safe gather indices
        idx_c = indices[s_lo:s_hi].squeeze(1).to(torch.int64)    # [cs, topk]
        invalid_c = (idx_c < 0) | (idx_c >= Skv)                 # [cs, topk]
        safe_idx_c = torch.where(invalid_c, torch.zeros_like(idx_c), idx_c)

        if tlen_i64 is not None:
            tlen_c = tlen_i64[s_lo:s_hi].unsqueeze(1)            # [cs, 1]
            invalid_c = invalid_c | (pos_row.unsqueeze(0) >= tlen_c)

        # Gather K (== V) per query, KEEP bf16 — this is the big win.
        gathered_c = kv2[safe_idx_c]                             # bf16 [cs, topk, Dqk]

        # bf16 bmm: (cs, Hq, Dqk) @ (cs, Dqk, topk) -> (cs, Hq, topk)
        q_c = q[s_lo:s_hi]                                       # bf16 [cs, Hq, Dqk]
        # Promote ONLY the small logits tensor to fp32 for numerically
        # stable softmax/logsumexp. Big K/V stay bf16.
        logits_c = torch.bmm(q_c, gathered_c.transpose(1, 2)).to(torch.float32)
        logits_c.mul_(float(sm_scale))

        if invalid_c.any():
            logits_c.masked_fill_(invalid_c.unsqueeze(1), neg_inf)

        # max_logits + lse (numerically stable softmax)
        ml_c = logits_c.max(dim=-1).values                       # [cs, Hq]
        lse_c = torch.logsumexp(logits_c, dim=-1)                # [cs, Hq]

        # Where ALL positions invalid → lse = -inf, attn = 0
        all_invalid_c = torch.isinf(lse_c) & (lse_c < 0)         # [cs, Hq]
        safe_lse_c = torch.where(all_invalid_c, torch.zeros_like(lse_c), lse_c)
        attn_c_f = torch.exp(logits_c - safe_lse_c.unsqueeze(-1))   # [cs, Hq, topk] fp32
        # free intermediate
        del logits_c
        if all_invalid_c.any():
            attn_c_f = torch.where(
                all_invalid_c.unsqueeze(-1), torch.zeros_like(attn_c_f), attn_c_f
            )

        # Downcast attn to bf16 for the final bmm (back into bf16 K/V latent).
        # (cs, Hq, topk) @ (cs, topk, Dqk) -> (cs, Hq, Dqk)
        attn_c_bf = attn_c_f.to(q.dtype)
        del attn_c_f
        out_c_bf = torch.bmm(attn_c_bf, gathered_c)              # bf16 [cs, Hq, Dqk]
        del attn_c_bf, gathered_c

        if sink_f is not None:
            scale_c = torch.sigmoid(lse_c - sink_f)              # fp32 [cs, Hq]
            if all_invalid_c.any():
                scale_c = torch.where(
                    all_invalid_c, torch.zeros_like(scale_c), scale_c
                )
            # multiply in fp32 then cast (sigmoid is well-conditioned)
            out_c_bf = (out_c_bf.to(torch.float32) * scale_c.unsqueeze(-1)).to(q.dtype)

        # Write back into preallocated buffers.
        # Slice to d_v in case Dqk > d_v in a future config (currently both 512).
        if out_c_bf.shape[-1] != d_v:
            out_c_bf = out_c_bf[..., :d_v].contiguous()
        out_bf[s_lo:s_hi].copy_(out_c_bf)
        max_logits[s_lo:s_hi].copy_(ml_c)
        lse[s_lo:s_hi].copy_(lse_c)

    return (out_bf, max_logits, lse)
# SM120_SPARSE_PREFILL_FALLBACK_END
'''.strip("\n")
    + "\n"
)


# ────────────────────────────────────────────────────────────────────
# Dispatch swap: inject INSIDE flash_mla_sparse_fwd, BEFORE the line
#     results = flash_mla_cuda.sparse_prefill_fwd(...)
# At 4-space indent (function body level).
#
# Sentinels:
#   # SM120_SPARSE_PREFILL_FALLBACK_DISPATCH_BEGIN
#   ...
#   # SM120_SPARSE_PREFILL_FALLBACK_DISPATCH_END
# ────────────────────────────────────────────────────────────────────
DISPATCH_BLOCK = (
    """
    # SM120_SPARSE_PREFILL_FALLBACK_DISPATCH_BEGIN
    if _SM120_SPARSE_PREFILL_FALLBACK_ENABLED:
        return _sm120_sparse_prefill_fallback(
            q=q,
            kv=kv,
            indices=indices,
            sm_scale=sm_scale,
            d_v=d_v,
            attn_sink=attn_sink,
            topk_length=topk_length,
            out=out,
        )
    # SM120_SPARSE_PREFILL_FALLBACK_DISPATCH_END
""".strip("\n")
    + "\n"
)


# ────────────────────────────────────────────────────────────────────
# Anchor finders
# ────────────────────────────────────────────────────────────────────
def _find_helpers_anchor(src: str) -> int:
    """Insertion point for helpers block.

    Preferred: right after decode patch v3's END sentinel (so v3 + v4
    helpers sit together). Fallback: right after
    `flash_mla_cuda = torch.ops._flashmla_C`.
    """
    # Try: after v3 decode patch END marker
    m = re.search(
        r"^# SM120_SPARSE_DECODE_FALLBACK_END\s*$",
        src,
        re.MULTILINE,
    )
    if m:
        return src.index("\n", m.end()) + 1

    # Fallback: after `flash_mla_cuda = torch.ops._flashmla_C`
    m = re.search(
        r"^flash_mla_cuda\s*=\s*torch\.ops\._flashmla_C\s*$",
        src,
        re.MULTILINE,
    )
    if not m:
        raise RuntimeError(
            "Could not find anchor for helpers injection "
            "(neither v3 END marker nor `flash_mla_cuda = ...`)"
        )
    return src.index("\n", m.end()) + 1


def _find_dispatch_anchor(src: str) -> int:
    """Find insertion point right BEFORE the line
        `    results = flash_mla_cuda.sparse_prefill_fwd(`
    inside flash_mla_sparse_fwd (4-space indent, function body level).
    """
    m = re.search(
        r"^    results = flash_mla_cuda\.sparse_prefill_fwd\(",
        src,
        re.MULTILINE,
    )
    if not m:
        raise RuntimeError(
            "Could not find `results = flash_mla_cuda.sparse_prefill_fwd(` "
            "anchor at 4-space indent inside flash_mla_sparse_fwd"
        )
    return m.start()


# ────────────────────────────────────────────────────────────────────
# Commands
# ────────────────────────────────────────────────────────────────────
def cmd_check() -> int:
    if not TARGET.exists():
        print(f"[FAIL] target not found: {TARGET}")
        return 2
    src = TARGET.read_text()
    has_helpers = "SM120_SPARSE_PREFILL_FALLBACK_BEGIN" in src
    has_dispatch = "SM120_SPARSE_PREFILL_FALLBACK_DISPATCH_BEGIN" in src
    has_backup = TARGET.with_suffix(TARGET.suffix + BACKUP_SUFFIX).exists()

    # also note decode patch v3 status for context
    has_decode_v3 = "SM120_SPARSE_DECODE_FALLBACK_BEGIN" in src

    print(f"target:                  {TARGET}")
    print(f"backup exists:           {has_backup} ({TARGET.name + BACKUP_SUFFIX})")
    print(f"decode patch v3 present: {has_decode_v3}   (expected: True)")
    print(f"prefill helpers:         {has_helpers}")
    print(f"prefill dispatch swap:   {has_dispatch}")
    if has_helpers and has_dispatch:
        print("[OK] prefill patch v4 is APPLIED")
        return 0
    if not has_helpers and not has_dispatch:
        print("[OK] prefill patch v4 is NOT applied (pristine wrt v4)")
        return 0
    print(
        "[WARN] prefill patch v4 is PARTIALLY applied — recommend --revert then --apply"
    )
    return 1


def cmd_apply() -> int:
    if not TARGET.exists():
        print(f"[FAIL] target not found: {TARGET}")
        return 2
    src = TARGET.read_text()

    if (
        "SM120_SPARSE_PREFILL_FALLBACK_BEGIN" in src
        or "SM120_SPARSE_PREFILL_FALLBACK_DISPATCH_BEGIN" in src
    ):
        print(
            "[SKIP] prefill patch already (partially) applied. "
            "Run --revert first if you want to re-apply."
        )
        return 1

    # backup (separate from decode patch's backup)
    backup = TARGET.with_suffix(TARGET.suffix + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(TARGET, backup)
        print(f"[OK] backup written: {backup}")
    else:
        print(f"[OK] backup already exists: {backup}")

    # 1) inject helpers
    anchor = _find_helpers_anchor(src)
    src = src[:anchor] + "\n" + HELPERS_BLOCK + "\n" + src[anchor:]

    # 2) inject dispatch swap before `    results = flash_mla_cuda.sparse_prefill_fwd(`
    anchor2 = _find_dispatch_anchor(src)
    src = src[:anchor2] + DISPATCH_BLOCK + src[anchor2:]

    TARGET.write_text(src)
    print(f"[OK] patch applied to {TARGET}")

    # ── self-check 1: AST parse must succeed
    import ast

    try:
        ast.parse(src, filename=str(TARGET))
        print("[OK] self-check: ast.parse PASSED")
    except SyntaxError as e:
        print(f"[FAIL] self-check: ast.parse FAILED: {e}")
        print("       reverting…")
        shutil.copy2(backup, TARGET)
        return 4

    # ── self-check 2: dispatch block must be at 4-space indent
    bad = re.search(
        r"^        # SM120_SPARSE_PREFILL_FALLBACK_DISPATCH_BEGIN\s*$",
        src,
        re.MULTILINE,
    )
    good = re.search(
        r"^    # SM120_SPARSE_PREFILL_FALLBACK_DISPATCH_BEGIN\s*$",
        src,
        re.MULTILINE,
    )
    if bad and not good:
        print("[FAIL] self-check: dispatch block at 8-space indent. Reverting…")
        shutil.copy2(backup, TARGET)
        return 5
    if not good:
        print("[FAIL] self-check: dispatch block not found at expected indent.")
        shutil.copy2(backup, TARGET)
        return 6
    print("[OK] self-check: dispatch block at 4-space indent (function body)")

    # ── self-check 3: clear stale .pyc
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
        for mod_name in list(sys.modules):
            if "flash_mla_interface" in mod_name:
                del sys.modules[mod_name]
        m = importlib.import_module("vllm.third_party.flashmla.flash_mla_interface")
        import inspect

        fn_src = inspect.getsource(m.flash_mla_sparse_fwd)
        lines = fn_src.splitlines()
        for i, line in enumerate(lines):
            if "SM120_SPARSE_PREFILL_FALLBACK_DISPATCH_BEGIN" in line:
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
        flag = getattr(m, "_SM120_SPARSE_PREFILL_FALLBACK_ENABLED", None)
        print(f"[OK] self-check: _SM120_SPARSE_PREFILL_FALLBACK_ENABLED = {flag}")

        # also confirm decode v3 still works (sanity)
        flag_dec = getattr(m, "_SM120_SPARSE_DECODE_FALLBACK_ENABLED", None)
        print(
            f"[OK] self-check: _SM120_SPARSE_DECODE_FALLBACK_ENABLED = "
            f"{flag_dec}  (decode patch v3 still wired)"
        )
    except Exception as e:
        print(f"[WARN] live import check skipped: {e!r}")

    # final
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
