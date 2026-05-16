"""
SM120 fallback for DeepSeek-V4 Indexer (NSA / sparse attention) MQA logits kernels.

================================================================================
背景
================================================================================
DeepSeek-V4-Flash 的 sparse attention (NSA) 在 vLLM 里走 DeepseekV4Indexer 后端，
最终调到 DeepGEMM 的 3 个 kernel 来算 selection logits：

    1. get_paged_mqa_logits_metadata   —— build 阶段，调度元数据
    2. fp8_fp4_paged_mqa_logits        —— decode 阶段算 logits
    3. fp8_fp4_mqa_logits              —— prefill 阶段算 logits

这 3 个 kernel 在 DeepGEMM 里是 SM100 (B100/B200/GB200 数据中心 Blackwell) 专属
（用了 tcgen05 / 5th-gen Tensor Core 指令），SM120 (RTX PRO 5000 / 消费级
Blackwell) 物理上没有这些指令，**编译都过不去，跑不起来**。

vLLM 在 indexer.py 已经检测到 SM120 走 use_flattening 路径绕过 fp4 indexer cache，
但是真正算 logits 的 fp8_fp4_*_mqa_logits 这俩 DeepGEMM kernel 还是无条件被调用，
启动起来必撞 "Unsupported architecture" 或 NotImplementedError。

本补丁在 vllm/utils/deep_gemm.py 里把这 3 个 wrapper 替换成纯 PyTorch 等价实现：
    - 把 FP8 KV cache (UE4M3 + scale) dequant 回 fp32
    - 用 torch.einsum / matmul 算 q @ k^T
    - 按 cu_seqlen_ks/ke 或 context_lens 做 mask
比 DeepGEMM 慢 5-20x，但是能跑。

================================================================================
启动命令（应用本补丁后）
================================================================================
export VLLM_HC_FALLBACK=1                 # 第一个补丁开关
export VLLM_INDEXER_FALLBACK=1            # 本补丁开关
export VLLM_USE_DEEP_GEMM=0
export VLLM_FUSED_MOE_BACKEND=triton

vllm serve /workspace/models/DeepSeek-V4-Flash \\
  --served-model-name deepseek-v4-flash \\
  --trust-remote-code \\
  --kv-cache-dtype fp8 \\
  --block-size 256 \\
  --enable-expert-parallel \\
  --tensor-parallel-size 4 \\
  --max-model-len 32768 \\
  --tokenizer-mode deepseek_v4 \\
  --tool-call-parser deepseek_v4 \\
  --enable-auto-tool-choice \\
  --reasoning-parser deepseek_v4 \\
  --port 8081 \\
  --enable-prompt-tokens-details \\
  --enforce-eager

================================================================================
应用方式
================================================================================
    # 一键打补丁（备份原文件 + 写入 fallback）
    python indexer_mqa_logits_sm120_fallback.py --apply

    # 还原
    python indexer_mqa_logits_sm120_fallback.py --revert

    # 也可以手动：把 _patched_* 三个函数 + _fallback_* 实现拷到
    # /usr/local/lib/python3.12/dist-packages/vllm/utils/deep_gemm.py
    # 替换原同名 wrapper 即可。
================================================================================
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# ============================================================================
# Part 1: 真正的 fallback 实现（这部分会被注入到 vllm/utils/deep_gemm.py）
# ============================================================================

FALLBACK_CODE = r'''
# -------------------------- SM120 INDEXER FALLBACK BEGIN --------------------------
# Auto-injected by indexer_mqa_logits_sm120_fallback.py
# 用 VLLM_INDEXER_FALLBACK=1 强制启用；不设或设为 0 则保持原 DeepGEMM 行为
import os as _os_sm120
import torch as _torch_sm120


def _sm120_indexer_fallback_enabled() -> bool:
    return _os_sm120.environ.get("VLLM_INDEXER_FALLBACK", "0") == "1"


def _sm120_dequant_fp8_kv_block(kv_cache: _torch_sm120.Tensor) -> tuple:
    """把 paged FP8 KV cache 拆成 (k_fp8, scale_fp32)。

    DeepGEMM FP8 paged layout: [num_blocks, block_size, 1, D+4]，dtype=uint8。
    最后 4 字节 per (block, pos) 存 fp32 dequant scale，前 D 字节是 fp8_e4m3fn k。
    """
    assert kv_cache.dtype == _torch_sm120.uint8, (
        f"expected uint8 paged FP8 KV cache, got {kv_cache.dtype}"
    )
    assert kv_cache.dim() == 4 and kv_cache.size(2) == 1, (
        f"expected [num_blocks, block_size, 1, D+4], got {tuple(kv_cache.shape)}"
    )
    num_blocks, block_size, _, D_plus_4 = kv_cache.shape
    D = D_plus_4 - 4
    flat = kv_cache.view(num_blocks, block_size, D_plus_4)
    k_bytes = flat[..., :D].contiguous()
    scale_bytes = flat[..., D:].contiguous()
    k_fp8 = k_bytes.view(_torch_sm120.float8_e4m3fn)
    scale_fp32 = scale_bytes.view(_torch_sm120.float32).squeeze(-1)
    return k_fp8, scale_fp32  # k_fp8: [B,blk,D]  scale: [B,blk]


def _sm120_get_paged_mqa_logits_metadata_fallback(
    context_lens: _torch_sm120.Tensor, block_size: int, num_sms: int
) -> _torch_sm120.Tensor:
    """metadata 在我们 fallback 里其实没用（fallback 不切 SM 调度），
    返回一个跟 DeepGEMM 形状兼容的占位 tensor 即可，下游 fallback 不会读它。
    DeepGEMM 真实形状是 [num_sms+1, 2] int32（vllm/v1/attention/backends/mla/indexer.py
    里的 self.scheduler_metadata_buffer 验证）。
    """
    return _torch_sm120.zeros(
        (num_sms + 1, 2), dtype=_torch_sm120.int32, device=context_lens.device
    )


def _sm120_fp8_fp4_mqa_logits_fallback(
    q,
    kv,
    weights: _torch_sm120.Tensor,
    cu_seqlen_ks: _torch_sm120.Tensor,
    cu_seqlen_ke: _torch_sm120.Tensor,
    clean_logits: bool,
) -> _torch_sm120.Tensor:
    """Prefill MQA logits, 纯 PyTorch fallback。

    Inputs (FP8 path):
        q       : tuple (q_values, q_scale_or_None)
                  q_values: [M, H, D] float8_e4m3fn (per-token scale 已折进 weights)
        kv      : tuple (k_packed, k_scales)
                  k_packed: [N, D] float8_e4m3fn
                  k_scales: [N]    float32
        weights : [M, H] float32  (= per-token q_scale * head_weight)
        cu_seqlen_ks/ke : [M] int32, 每个 query 位 valid K 的 [start, end)

    Output: logits [M, N] float32
    """
    q_values = q[0] if isinstance(q, tuple) else q
    k_packed, k_scales = kv

    # 走 FP4 路径的话 q_values dtype 是 uint8，本 fallback 仅支持 FP8，因为
    # SM120 + DeepSeek-V4-Flash 走 FP8 indexer cache（vllm 已 gate fp4_indexer_cache
    # 在 SM100 only）。
    assert q_values.dtype == _torch_sm120.float8_e4m3fn, (
        f"sm120 fallback only supports FP8 q, got {q_values.dtype}"
    )
    assert k_packed.dtype == _torch_sm120.float8_e4m3fn, (
        f"sm120 fallback only supports FP8 k, got {k_packed.dtype}"
    )

    # Dequant 到 fp32（fp8_e4m3fn -> fp32 是无损升精度）
    q_fp32 = q_values.to(_torch_sm120.float32)            # [M, H, D]
    k_fp32 = k_packed.to(_torch_sm120.float32)            # [N, D]
    k_fp32 = k_fp32 * k_scales.to(_torch_sm120.float32).unsqueeze(-1)  # [N, D]

    # logits = sum_h weights[m,h] * (q[m,h] @ k.T)[m,n]
    # = (weights[:,:,None] * q_fp32) -> [M,H,D] reduce H 后再 matmul
    qw = (q_fp32 * weights.to(_torch_sm120.float32).unsqueeze(-1)).sum(dim=1)  # [M, D]
    logits = qw @ k_fp32.t()  # [M, N]

    # mask: valid K 范围之外置 -inf
    M, N = logits.shape
    n_idx = _torch_sm120.arange(N, device=logits.device).unsqueeze(0)  # [1, N]
    ks = cu_seqlen_ks.to(logits.device).unsqueeze(1)  # [M, 1]
    ke = cu_seqlen_ke.to(logits.device).unsqueeze(1)  # [M, 1]
    valid = (n_idx >= ks) & (n_idx < ke)
    if clean_logits:
        logits = _torch_sm120.where(valid, logits, _torch_sm120.full_like(logits, float("-inf")))
    else:
        logits = logits.masked_fill(~valid, 0.0)
    return logits


def _sm120_fp8_fp4_paged_mqa_logits_fallback(
    q,
    kv_cache: _torch_sm120.Tensor,
    weights: _torch_sm120.Tensor,
    context_lens: _torch_sm120.Tensor,
    block_tables: _torch_sm120.Tensor,
    schedule_metadata: _torch_sm120.Tensor,
    max_model_len: int,
    clean_logits: bool,
) -> _torch_sm120.Tensor:
    """Decode MQA logits over paged KV-cache, 纯 PyTorch fallback。

    Inputs:
        q            : tuple (q_values, q_scale_or_None)
                       q_values: [B, next_n, H, D] float8_e4m3fn
        kv_cache     : [num_blocks, block_size, 1, D+4] uint8
                       (last 4 bytes per (block,pos) = fp32 scale)
        weights      : [B*next_n, H] float32
        context_lens : [B] int32  effective context length
        block_tables : [B, max_blocks] int32
        schedule_metadata : ignored in fallback
        max_model_len : output N dim

    Output: logits [B*next_n, max_model_len] float32
    """
    q_values = q[0] if isinstance(q, tuple) else q
    assert q_values.dtype == _torch_sm120.float8_e4m3fn, (
        f"sm120 fallback only supports FP8 q, got {q_values.dtype}"
    )
    B, next_n, H, D = q_values.shape
    num_blocks, block_size, one, D_plus_4 = kv_cache.shape
    assert one == 1 and D_plus_4 == D + 4, (
        f"unexpected kv_cache shape {tuple(kv_cache.shape)} vs q D={D}"
    )

    device = q_values.device
    q_fp32 = q_values.to(_torch_sm120.float32)  # [B, next_n, H, D]
    # qw[b, t, d] = sum_h weights[b*next_n+t, h] * q_fp32[b, t, h, d]
    w = weights.to(_torch_sm120.float32).view(B, next_n, H)  # [B, next_n, H]
    qw = _torch_sm120.einsum("bthd,bth->btd", q_fp32, w)  # [B, next_n, D]

    # Dequant 整个 KV cache 一次（简单实现；如果显存吃紧可以按 block 切片，但
    # max_model_len 比 num_blocks*block_size 小一般无所谓）
    k_fp8_all, scale_all = _sm120_dequant_fp8_kv_block(kv_cache)  # [B,blk,D] [B,blk]
    k_fp32_all = k_fp8_all.to(_torch_sm120.float32) * scale_all.unsqueeze(-1)  # [num_blocks, blk, D]

    # 按 block_tables gather 出每个 batch 的 K
    # block_tables: [B, max_blocks]
    bt = block_tables.to(_torch_sm120.long)
    max_blocks = bt.size(1)
    # gathered: [B, max_blocks, blk, D]
    gathered = k_fp32_all[bt]  # advanced indexing
    # reshape -> [B, max_blocks*blk, D] = [B, N_phys, D]
    K_per_batch = gathered.view(B, max_blocks * block_size, D)

    N_phys = K_per_batch.size(1)
    N_out = max_model_len
    if N_phys < N_out:
        pad = _torch_sm120.zeros(B, N_out - N_phys, D, device=device, dtype=_torch_sm120.float32)
        K_per_batch = _torch_sm120.cat([K_per_batch, pad], dim=1)
    elif N_phys > N_out:
        K_per_batch = K_per_batch[:, :N_out, :]

    # logits[b, t, n] = qw[b, t, :] · K_per_batch[b, n, :]
    logits = _torch_sm120.einsum("btd,bnd->btn", qw, K_per_batch)  # [B, next_n, N_out]
    logits = logits.reshape(B * next_n, N_out)

    # mask: 每个 batch b 只 valid 前 context_lens[b] 个位置
    # （per-token context length 已经在 vllm 上层算好；如果传入是 2D
    # [B, next_n] 也兼容）
    n_idx = _torch_sm120.arange(N_out, device=device).unsqueeze(0)  # [1, N_out]
    if context_lens.dim() == 1:
        ctx = context_lens.to(device).repeat_interleave(next_n).unsqueeze(1)  # [B*next_n, 1]
    else:
        ctx = context_lens.to(device).reshape(B * next_n, 1)
    valid = n_idx < ctx
    if clean_logits:
        logits = _torch_sm120.where(valid, logits, _torch_sm120.full_like(logits, float("-inf")))
    else:
        logits = logits.masked_fill(~valid, 0.0)
    return logits


# ---- Monkey-patch the public wrappers ----
_orig_get_paged_mqa_logits_metadata = get_paged_mqa_logits_metadata
_orig_fp8_fp4_mqa_logits = fp8_fp4_mqa_logits
_orig_fp8_fp4_paged_mqa_logits = fp8_fp4_paged_mqa_logits


def get_paged_mqa_logits_metadata(context_lens, block_size, num_sms):  # noqa: F811
    if _sm120_indexer_fallback_enabled():
        return _sm120_get_paged_mqa_logits_metadata_fallback(
            context_lens, block_size, num_sms
        )
    return _orig_get_paged_mqa_logits_metadata(context_lens, block_size, num_sms)


def fp8_fp4_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits):  # noqa: F811
    if _sm120_indexer_fallback_enabled():
        return _sm120_fp8_fp4_mqa_logits_fallback(
            q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits
        )
    return _orig_fp8_fp4_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits)


def fp8_fp4_paged_mqa_logits(  # noqa: F811
    q, kv_cache, weights, context_lens, block_tables, schedule_metadata,
    max_model_len, clean_logits,
):
    if _sm120_indexer_fallback_enabled():
        return _sm120_fp8_fp4_paged_mqa_logits_fallback(
            q, kv_cache, weights, context_lens, block_tables,
            schedule_metadata, max_model_len, clean_logits,
        )
    return _orig_fp8_fp4_paged_mqa_logits(
        q, kv_cache, weights, context_lens, block_tables,
        schedule_metadata, max_model_len, clean_logits,
    )
# --------------------------- SM120 INDEXER FALLBACK END ---------------------------
'''


# ============================================================================
# Part 2: --apply / --revert installer
# ============================================================================

DEFAULT_TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/utils/deep_gemm.py"
SENTINEL_BEGIN = "# -------------------------- SM120 INDEXER FALLBACK BEGIN --------------------------"
SENTINEL_END = "# --------------------------- SM120 INDEXER FALLBACK END ---------------------------"


def apply_patch(target: Path) -> None:
    src = target.read_text(encoding="utf-8")
    if SENTINEL_BEGIN in src:
        print(f"[skip] patch already applied to {target}")
        return
    backup = target.with_suffix(target.suffix + ".bak.indexer_sm120")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"[ok] backup -> {backup}")
    new = src.rstrip() + "\n\n" + FALLBACK_CODE.lstrip() + "\n"
    target.write_text(new, encoding="utf-8")
    print(f"[ok] patched: {target}")
    print("[hint] set VLLM_INDEXER_FALLBACK=1 to enable the fallback at runtime")


def revert_patch(target: Path) -> None:
    backup = target.with_suffix(target.suffix + ".bak.indexer_sm120")
    if not backup.exists():
        print(f"[err] no backup found at {backup}; cannot revert")
        sys.exit(1)
    shutil.copy2(backup, target)
    print(f"[ok] reverted {target} from {backup}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--apply", action="store_true", help="apply patch in-place")
    p.add_argument(
        "--revert", action="store_true", help="restore from .bak.indexer_sm120"
    )
    p.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help=f"target file (default: {DEFAULT_TARGET})",
    )
    args = p.parse_args()

    if not (args.apply ^ args.revert):
        p.print_help()
        sys.exit(1)
    target = Path(args.target)
    if not target.exists():
        print(f"[err] target not found: {target}")
        sys.exit(1)

    if args.apply:
        apply_patch(target)
    else:
        revert_patch(target)


if __name__ == "__main__":
    main()
