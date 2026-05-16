#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fp8_einsum_sm120_fallback.py  (v2)
==================================

目的:
    v3 UE8M0 补丁 (cutlass_scaled_mm_ue8m0_fallback_v3) 解决了
    cutlass_scaled_mm 的 ScalarType 44 报错后, DeepSeek-V4-Flash 在
    RTX PRO 5000 (SM120) 上 profile_run 时撞到下一个 DeepGEMM kernel
    断言:

        File ".../vllm/utils/deep_gemm.py", line 302, in fp8_einsum
            return _fp8_einsum_impl(*args, **kwargs)
        RuntimeError: Assertion error
          (csrc/apis/.../utils/layout.hpp:39): t.dim() == N

    DeepGEMM 的 fp8_einsum 是 SM100 (B100/B200) 专属内核, 在 SM120 上
    虽然 _fp8_einsum_impl 不是 None (DeepGEMM 编译时把入口符号放进去了),
    但 kernel 内部 layout 校验失败.

    本补丁在 vllm/utils/deep_gemm.py 中:
    1) 末尾追加一个纯 PyTorch fallback `_fp8_einsum_pytorch_fallback`,
       走 dequant -> torch.einsum -> 写回 out 的路径
    2) 把 line 302 的 `return _fp8_einsum_impl(*args, **kwargs)` 替换为
       受 VLLM_FP8_EINSUM_FALLBACK 环境变量控制的分支:
           if VLLM_FP8_EINSUM_FALLBACK 启用:
               return _fp8_einsum_pytorch_fallback(*args, **kwargs)
           else:
               return _fp8_einsum_impl(*args, **kwargs)

    fp8_einsum 的调用约定 (从 deepseek_v4_attention.py:584-593 推断):
        fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=(...))
      - equation: str, 例如 "bhr,hdr->bhd"
      - a, b    : float8_e4m3fn 张量
      - a_scale, b_scale : block / per-token scale; dtype 可能是 float32
                          / float8_e8m0fnu (UE8M0) / **torch.int32 packed
                          UE8M0**(每个 int32 含 4 个 UE8M0 字节, 沿最后
                          一维 ×4 展开)
      - out     : preallocated bf16/fp16 张量, 函数 in-place 写入
      - recipe  : DeepGEMM block size 元组, fallback 路径主要用于推断
                  block_size, 通常 = 128
      - return  : None (原 API 副作用写 out, 无返回值)

v2 相对 v1 的关键修复:
    v1 dispatch 已工作, 但 dequant 后 einsum 报错:
        einsum(...) operand 1 has 2 dims, equation expects 3
    根因 3 个 bug:

    Bug-A) a_scale.dtype 是 torch.int32, 实际是 packed UE8M0:
           每个 int32 容纳 4 个 UE8M0 指数字节, 沿 last dim ×4 展开.
           v1 把 int32 直接 .to(float32) 当数值, 算出来根本不是 scale.
           v2: 如果 dtype == int32 -> view(uint8) -> 沿 last dim 自动 ×4
                -> 2^(uint8 - 127) -> fp32.

    Bug-B) b 进来是 2D (H*D, R), 但 equation 写的是 "hdr,...",
           torch.einsum 不会自动把 (H*D, R) 还原成 (H, D, R), 报
           "operand has 2 dims, equation expects 3".
           v2: 解析 equation, 用 out.shape / a.shape 反推每个字母
               对应的真实长度, 然后把 b reshape 成 equation 要求的 ndim
               (e.g. (H, D, R)).

    Bug-C) b_scale 是 (H*D // block, R // block) 二维, b reshape 成
           (H, D, R) 之后 b_scale 也得同步 reshape 成
           (H, D // block, R // block) 才能 broadcast.
           v2: 在 b reshape 之后按相同字母维度对 b_scale 做 reshape.

    另外 v2:
      - 加 VLLM_FP8_EINSUM_FALLBACK_DEBUG=1 时, dequant 路径每次打印
        a/b/a_scale/b_scale 的 shape 和 dtype 以及 reshape 过程, 便于
        诊断"下一个 shape 不一样的 layer".
      - 在 reshape b / b_scale 失败时, 抛 RuntimeError 并把所有相关
        shape 都打出来.

完整 vLLM 启动命令:
----------------------------------------------------------------------
export VLLM_HC_FALLBACK=1
export VLLM_INDEXER_FALLBACK=1
export VLLM_UE8M0_FALLBACK=1
export VLLM_FP8_EINSUM_FALLBACK=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_FUSED_MOE_BACKEND=triton

vllm serve /workspace/models/DeepSeek-V4-Flash \\
  --served-model-name deepseek-v4-flash --trust-remote-code \\
  --kv-cache-dtype fp8 --block-size 256 --enable-expert-parallel \\
  --tensor-parallel-size 4 --max-model-len 32768 \\
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 \\
  --enable-auto-tool-choice --reasoning-parser deepseek_v4 \\
  --port 8081 --enable-prompt-tokens-details --enforce-eager
----------------------------------------------------------------------

用法:
    sudo python3 fp8_einsum_sm120_fallback.py --apply
    sudo python3 fp8_einsum_sm120_fallback.py --revert
    python3      fp8_einsum_sm120_fallback.py --check
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

TARGET_FILE = Path("/usr/local/lib/python3.12/dist-packages/vllm/utils/deep_gemm.py")
BACKUP_FILE = TARGET_FILE.with_suffix(TARGET_FILE.suffix + ".bak_fp8einsum")

V4_MARKER_BEGIN = "# >>> SM120_FP8_EINSUM_FALLBACK_BEGIN <<<"
V4_MARKER_END = "# <<< SM120_FP8_EINSUM_FALLBACK_END >>>"

# 要被替换的原始 fp8_einsum 函数体 (line 298-302).
# 注意: 严格匹配原文本(包括缩进/换行), 这是 5-line block (含 def).
ORIGINAL_FP8_EINSUM = (
    "def fp8_einsum(*args, **kwargs):\n"
    "    _lazy_init()\n"
    "    if _fp8_einsum_impl is None:\n"
    "        return _missing(*args, **kwargs)\n"
    "    return _fp8_einsum_impl(*args, **kwargs)\n"
)

# 替换后的新 fp8_einsum: 多一层 env-controlled fallback dispatch
PATCHED_FP8_EINSUM = (
    "def fp8_einsum(*args, **kwargs):\n"
    "    _lazy_init()\n"
    "    if _fp8_einsum_impl is None:\n"
    "        return _missing(*args, **kwargs)\n"
    "    # >>> SM120_FP8_EINSUM_FALLBACK_DISPATCH <<<\n"
    "    import os as _os_fp8e\n"
    "    if _os_fp8e.environ.get('VLLM_FP8_EINSUM_FALLBACK', '1') != '0':\n"
    "        return _fp8_einsum_pytorch_fallback(*args, **kwargs)\n"
    "    # <<< SM120_FP8_EINSUM_FALLBACK_DISPATCH >>>\n"
    "    return _fp8_einsum_impl(*args, **kwargs)\n"
)

# 末尾追加的 fallback 实现 (v2)
V4_APPEND_CODE = f'''

{V4_MARKER_BEGIN}
# Auto-injected by fp8_einsum_sm120_fallback.py (v2)
# Pure-PyTorch fallback for DeepGEMM's fp8_einsum, used on SM120
# (RTX PRO 5000) where the C++ kernel asserts on layout (B100/B200-only).
#
# Calling convention (from deepseek_v4_attention.py:584-593):
#   fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=(...))
#
# v2 修复了 3 个真实生产 bug, 详见文件顶部说明.
import os as _os_fp8e_v4
import torch as _torch_fp8e_v4
from vllm.logger import logger as _logger_fp8e_v4

_FP8E_DEBUG = _os_fp8e_v4.environ.get("VLLM_FP8_EINSUM_FALLBACK_DEBUG", "0") != "0"
_FP8E_LOG_ONCE = {{"done": False}}

try:
    _UE8M0_DTYPE_FP8E = _torch_fp8e_v4.float8_e8m0fnu  # type: ignore[attr-defined]
except AttributeError:
    _UE8M0_DTYPE_FP8E = None


def _fp8e_unpack_int32_ue8m0(scale_i32):
    """packed UE8M0 in int32 -> float32 scale tensor.

    每个 int32 容纳 4 个 UE8M0 指数字节 (little-endian byte order).
    `scale_i32` shape (..., K) -> output shape (..., K*4).
    UE8M0 的数值含义: x = 2^(uint8 - 127).
    """
    # view as uint8: last dim expands by 4 (assumes contiguous, last-dim packed)
    if not scale_i32.is_contiguous():
        scale_i32 = scale_i32.contiguous()
    scale_u8 = scale_i32.view(_torch_fp8e_v4.uint8)
    # 2^(uint8 - 127)
    exp = scale_u8.to(_torch_fp8e_v4.int32) - 127
    return _torch_fp8e_v4.pow(
        _torch_fp8e_v4.tensor(2.0, dtype=_torch_fp8e_v4.float32, device=scale_i32.device),
        exp.to(_torch_fp8e_v4.float32),
    )


def _fp8e_scale_to_fp32(scale):
    """把 scale 张量统一转成 float32, 处理 UE8M0 / int32 packed UE8M0 / 已是 float32."""
    if scale is None:
        return None
    if scale.dtype == _torch_fp8e_v4.int32:
        # Packed UE8M0 (4 bytes per int32, expands last dim ×4)
        return _fp8e_unpack_int32_ue8m0(scale)
    if (_UE8M0_DTYPE_FP8E is not None) and scale.dtype == _UE8M0_DTYPE_FP8E:
        return scale.to(_torch_fp8e_v4.float32)
    if scale.dtype != _torch_fp8e_v4.float32:
        return scale.to(_torch_fp8e_v4.float32)
    return scale


def _fp8e_dequant_broadcast(t_fp32, s_fp32):
    """把 s 通过 per-axis repeat_interleave broadcast 到 t.shape, 然后乘."""
    if s_fp32 is None:
        return t_fp32

    # cheap path
    try:
        return t_fp32 * s_fp32
    except RuntimeError:
        pass

    # 维度对齐 (前补 1)
    while s_fp32.dim() < t_fp32.dim():
        s_fp32 = s_fp32.unsqueeze(0)
    while s_fp32.dim() > t_fp32.dim():
        if s_fp32.shape[0] == 1:
            s_fp32 = s_fp32.squeeze(0)
        else:
            raise RuntimeError(
                f"[fp8_einsum fallback] scale has more dims than tensor and "
                f"leading dim != 1: scale={{tuple(s_fp32.shape)}} t={{tuple(t_fp32.shape)}}"
            )

    if s_fp32.shape == t_fp32.shape:
        return t_fp32 * s_fp32

    new_s = s_fp32
    for dim in range(t_fp32.dim()):
        tgt = t_fp32.shape[dim]
        cur = new_s.shape[dim]
        if cur == tgt or cur == 1:
            continue
        if tgt % cur != 0:
            raise RuntimeError(
                f"[fp8_einsum fallback] scale shape {{tuple(s_fp32.shape)}} "
                f"cannot broadcast to tensor shape {{tuple(t_fp32.shape)}} on "
                f"dim {{dim}} (cur={{cur}}, tgt={{tgt}})"
            )
        factor = tgt // cur
        new_s = new_s.repeat_interleave(factor, dim=dim)
    return t_fp32 * new_s


def _fp8e_parse_equation(equation):
    """'bhr,hdr->bhd' -> (['b','h','r'], ['h','d','r'], ['b','h','d'])."""
    lhs, rhs = equation.split("->")
    a_lbl, b_lbl = lhs.split(",")
    return list(a_lbl.strip()), list(b_lbl.strip()), list(rhs.strip())


def _fp8e_infer_dim_map(a_lbl, b_lbl, out_lbl, a_shape, out_shape):
    """根据 equation + a.shape + out.shape 反推每个字母 -> 长度."""
    dim_map = {{}}
    for i, lab in enumerate(a_lbl):
        dim_map[lab] = a_shape[i]
    for i, lab in enumerate(out_lbl):
        if lab in dim_map and dim_map[lab] != out_shape[i]:
            raise RuntimeError(
                f"[fp8_einsum fallback] dim mismatch for '{{lab}}': "
                f"from a={{dim_map[lab]}} vs out={{out_shape[i]}}"
            )
        dim_map[lab] = out_shape[i]
    # b 的字母此时全部应该已被 a 或 out 覆盖
    for lab in b_lbl:
        if lab not in dim_map:
            raise RuntimeError(
                f"[fp8_einsum fallback] cannot infer length of '{{lab}}' "
                f"from a_shape={{a_shape}} out_shape={{out_shape}}"
            )
    return dim_map


def _fp8e_reshape_b_to_equation(b, b_lbl, dim_map):
    """b 可能是 2D (H*D, R), reshape 成 equation 要求的 ndim (e.g. (H, D, R))."""
    target_shape = tuple(dim_map[l] for l in b_lbl)
    if tuple(b.shape) == target_shape:
        return b
    total = 1
    for s in target_shape:
        total *= s
    if b.numel() != total:
        raise RuntimeError(
            f"[fp8_einsum fallback] cannot reshape b from {{tuple(b.shape)}} "
            f"to {{target_shape}}: numel mismatch ({{b.numel()}} vs {{total}})"
        )
    return b.reshape(target_shape)


def _fp8e_reshape_b_scale(b_scale, b_lbl, dim_map, b_target_shape):
    """b_scale 是 (H*D // block, R // block) 二维, 同步 reshape 成
    (H, D // block, R // block) 等. 通过推断每维的 block size 完成.

    具体做法:
      target_b_shape: (H, D, R)
      b_scale.shape:  (M_s, R_s)  (M_s = H*D / block_m, R_s = R / block_r)

      已知 b 是 reshape 自 (M, R) -> (H, D, R), 其中 M = H*D.
      所以 b_scale (M_s, R_s) 应该 reshape 成 (H, D_s, R_s),
      其中 D_s = M_s / H = (H*D / block_m) / H = D / block_m.

      若 H 不整除 M_s, 退化为保持 2D, 在 dequant_broadcast 时让
      per-axis repeat_interleave 兜底.
    """
    if b_scale is None:
        return None

    # b_scale 维数若已等于 b_target_shape 维数, 不动
    if b_scale.dim() == len(b_target_shape):
        return b_scale

    if b_scale.dim() != 2 or len(b_target_shape) < 2:
        return b_scale  # 不知道怎么 reshape, 交给 broadcast 兜底

    # 假设 b_lbl 的第 0 维是要拆分的 (e.g. "hdr"), 拆成第 0 维 (H) +
    # 后续若干维. 找到 H 在 dim_map 里的值, b_scale 的第 0 维必须能被 H 整除.
    head_lab = b_lbl[0]
    head_len = dim_map[head_lab]
    m_s = b_scale.shape[0]
    if m_s % head_len != 0:
        # 退化: 保持 2D, 交给 broadcast
        return b_scale
    remain_s = m_s // head_len  # block-reduced 中间维
    tail_s = b_scale.shape[1]
    # 目标: (head_len, remain_s, tail_s)  (假设 b_lbl 是 3 个字母)
    if len(b_target_shape) == 3:
        return b_scale.reshape(head_len, remain_s, tail_s)
    # 否则保持原样, 让 broadcast 兜底
    return b_scale


def _fp8_einsum_pytorch_fallback(*args, **kwargs):
    """Drop-in replacement for DeepGEMM's fp8_einsum on SM120 (v2)."""
    # 解包
    if len(args) >= 4:
        equation, a_pack, b_pack, out = args[0], args[1], args[2], args[3]
    else:
        equation = kwargs.get("equation") or args[0]
        a_pack = kwargs.get("a") or args[1]
        b_pack = kwargs.get("b") or args[2]
        out = kwargs.get("out") or args[3]

    a, a_scale = a_pack if isinstance(a_pack, (tuple, list)) else (a_pack, None)
    b, b_scale = b_pack if isinstance(b_pack, (tuple, list)) else (b_pack, None)

    a_lbl, b_lbl, out_lbl = _fp8e_parse_equation(equation)
    dim_map = _fp8e_infer_dim_map(a_lbl, b_lbl, out_lbl, tuple(a.shape), tuple(out.shape))

    # 一次性日志 (第一次 dispatch 或 DEBUG=1 时)
    if _FP8E_DEBUG or not _FP8E_LOG_ONCE["done"]:
        try:
            _logger_fp8e_v4.info(
                "[SM120_FP8_EINSUM_FALLBACK v2] dispatch: eq=%s a=%s/%s a_s=%s/%s "
                "b=%s/%s b_s=%s/%s out=%s/%s dim_map=%s",
                equation,
                tuple(a.shape), a.dtype,
                None if a_scale is None else tuple(a_scale.shape),
                None if a_scale is None else a_scale.dtype,
                tuple(b.shape), b.dtype,
                None if b_scale is None else tuple(b_scale.shape),
                None if b_scale is None else b_scale.dtype,
                tuple(out.shape), out.dtype,
                dim_map,
            )
        except Exception:  # noqa: BLE001
            pass
        _FP8E_LOG_ONCE["done"] = True

    # === 处理 b: reshape 2D -> equation-required ndim ===
    b_target_shape = tuple(dim_map[l] for l in b_lbl)
    b_reshaped = _fp8e_reshape_b_to_equation(b, b_lbl, dim_map)
    b_scale_reshaped = _fp8e_reshape_b_scale(b_scale, b_lbl, dim_map, b_target_shape)

    # === scale -> fp32 ===
    a_scale_f = _fp8e_scale_to_fp32(a_scale)
    b_scale_f = _fp8e_scale_to_fp32(b_scale_reshaped)

    if _FP8E_DEBUG:
        try:
            _logger_fp8e_v4.info(
                "[SM120_FP8_EINSUM_FALLBACK v2] after reshape/unpack: "
                "b=%s a_scale_f=%s b_scale_f=%s",
                tuple(b_reshaped.shape),
                None if a_scale_f is None else (tuple(a_scale_f.shape), a_scale_f.dtype),
                None if b_scale_f is None else (tuple(b_scale_f.shape), b_scale_f.dtype),
            )
        except Exception:  # noqa: BLE001
            pass

    # === dequant ===
    a_fp32 = a.to(_torch_fp8e_v4.float32)
    b_fp32 = b_reshaped.to(_torch_fp8e_v4.float32)
    a_deq = _fp8e_dequant_broadcast(a_fp32, a_scale_f)
    b_deq = _fp8e_dequant_broadcast(b_fp32, b_scale_f)

    # === einsum ===
    result = _torch_fp8e_v4.einsum(equation, a_deq, b_deq)
    out.copy_(result.to(out.dtype))
    return None


_logger_fp8e_v4.info(
    "[SM120_FP8_EINSUM_FALLBACK v2] installed (env VLLM_FP8_EINSUM_FALLBACK=%s, DEBUG=%s)",
    _os_fp8e_v4.environ.get("VLLM_FP8_EINSUM_FALLBACK", "1"),
    _os_fp8e_v4.environ.get("VLLM_FP8_EINSUM_FALLBACK_DEBUG", "0"),
)
{V4_MARKER_END}
'''


def _has_v4(text: str) -> bool:
    return V4_MARKER_BEGIN in text and V4_MARKER_END in text


def _has_dispatch_marker(text: str) -> bool:
    return "SM120_FP8_EINSUM_FALLBACK_DISPATCH" in text


def _has_v2_marker(text: str) -> bool:
    return "SM120_FP8_EINSUM_FALLBACK v2" in text


def cmd_check() -> int:
    if not TARGET_FILE.exists():
        print(f"[CHECK] target NOT FOUND: {TARGET_FILE}")
        return 2
    text = TARGET_FILE.read_text(encoding="utf-8")
    has_v4 = _has_v4(text)
    has_dispatch = _has_dispatch_marker(text)
    has_orig = ORIGINAL_FP8_EINSUM in text
    has_v2 = _has_v2_marker(text)
    print(f"[CHECK] file                                : {TARGET_FILE}")
    print(f"[CHECK] v4 trailer marker                   : {has_v4}")
    print(f"[CHECK] dispatch in fp8_einsum body         : {has_dispatch}")
    print(f"[CHECK] original 5-line body still present  : {has_orig}")
    print(f"[CHECK] v2 fallback implementation present  : {has_v2}")
    print(f"[CHECK] backup .bak_fp8einsum exists        : {BACKUP_FILE.exists()}")
    if has_v4 and has_dispatch and has_v2 and not has_orig:
        print("[CHECK] STATE = v2 fully applied ✓")
        return 0
    if has_v4 and has_dispatch and not has_v2 and not has_orig:
        print("[CHECK] STATE = v1 applied (need re-apply for v2)")
        return 4
    if not has_v4 and not has_dispatch and has_orig:
        print("[CHECK] STATE = clean (no v4)")
        return 1
    print("[CHECK] STATE = INCONSISTENT")
    return 3


def cmd_apply() -> int:
    if not TARGET_FILE.exists():
        print(f"[APPLY][ERR] target NOT FOUND: {TARGET_FILE}", file=sys.stderr)
        return 2
    if not os.access(TARGET_FILE.parent, os.W_OK):
        print(
            f"[APPLY][ERR] no write permission. retry with: sudo python3 {sys.argv[0]} --apply",
            file=sys.stderr,
        )
        return 3

    text = TARGET_FILE.read_text(encoding="utf-8")

    # 备份 (仅在第一次, 保留最初 vanilla 版)
    if not BACKUP_FILE.exists():
        shutil.copy2(TARGET_FILE, BACKUP_FILE)
        print(f"[APPLY] backup -> {BACKUP_FILE}")
    else:
        print(f"[APPLY] backup already exists, keep it: {BACKUP_FILE}")

    # 幂等 re-apply: 若已有 v4 trailer (无论 v1/v2), 全部清掉
    if _has_v4(text):
        pat = re.compile(
            re.escape(V4_MARKER_BEGIN) + r".*?" + re.escape(V4_MARKER_END) + r"\s*",
            re.DOTALL,
        )
        text = pat.sub("", text).rstrip() + "\n"
        print("[APPLY] removed previous v4 trailer (re-apply)")

    # 若当前是被改写后的 dispatch 版本, 先恢复原 body
    if _has_dispatch_marker(text):
        if PATCHED_FP8_EINSUM in text:
            text = text.replace(PATCHED_FP8_EINSUM, ORIGINAL_FP8_EINSUM)
            print("[APPLY] reverted prior dispatch body to original")
        else:
            print(
                "[APPLY][ERR] dispatch marker present but PATCHED body not "
                "found verbatim; file may have been hand-edited",
                file=sys.stderr,
            )
            return 5

    # 验证原始 5-line block 存在
    if ORIGINAL_FP8_EINSUM not in text:
        print(
            "[APPLY][ERR] original fp8_einsum body not found verbatim; "
            "file structure may differ from expected vLLM 0.20.2rc1.dev246",
            file=sys.stderr,
        )
        print("[APPLY][ERR] expected:\n" + ORIGINAL_FP8_EINSUM, file=sys.stderr)
        return 6

    # 替换 dispatch
    text = text.replace(ORIGINAL_FP8_EINSUM, PATCHED_FP8_EINSUM, 1)
    n_dispatch = text.count("SM120_FP8_EINSUM_FALLBACK_DISPATCH")
    print(f"[APPLY] inserted dispatch into fp8_einsum body (markers={n_dispatch})")

    # 追加 v2 fallback impl
    text = text.rstrip() + "\n" + V4_APPEND_CODE + "\n"
    TARGET_FILE.write_text(text, encoding="utf-8")
    print(f"[APPLY] wrote patch to {TARGET_FILE}")

    # 自检
    verify = TARGET_FILE.read_text(encoding="utf-8")
    ok = _has_v4(verify) and _has_dispatch_marker(verify) and _has_v2_marker(verify)
    print(
        f"[APPLY] self-check: v4={_has_v4(verify)} "
        f"dispatch={_has_dispatch_marker(verify)} "
        f"v2={_has_v2_marker(verify)}"
    )
    if not ok:
        print("[APPLY][ERR] self-check FAILED", file=sys.stderr)
        return 4
    print("[APPLY] self-check OK ✓")

    # grep 让人眼看
    try:
        out = subprocess.check_output(
            [
                "grep",
                "-n",
                "SM120_FP8_EINSUM_FALLBACK\\|_fp8_einsum_pytorch_fallback\\|_fp8e_unpack_int32_ue8m0",
                str(TARGET_FILE),
            ],
            text=True,
        )
        print("[APPLY] grep verification:")
        print(out)
    except subprocess.CalledProcessError:
        print("[APPLY][WARN] grep found nothing (unexpected)")

    return 0


def cmd_revert() -> int:
    if BACKUP_FILE.exists():
        if not os.access(TARGET_FILE.parent, os.W_OK):
            print(f"[REVERT][ERR] no write permission, use sudo", file=sys.stderr)
            return 3
        shutil.copy2(BACKUP_FILE, TARGET_FILE)
        print(f"[REVERT] restored from {BACKUP_FILE}")
        return 0

    print(f"[REVERT][WARN] backup not found, doing in-place revert ...")
    if not TARGET_FILE.exists():
        print(f"[REVERT][ERR] target NOT FOUND", file=sys.stderr)
        return 2
    if not os.access(TARGET_FILE.parent, os.W_OK):
        print(f"[REVERT][ERR] no write permission, use sudo", file=sys.stderr)
        return 3
    text = TARGET_FILE.read_text(encoding="utf-8")
    if _has_v4(text):
        pat = re.compile(
            re.escape(V4_MARKER_BEGIN) + r".*?" + re.escape(V4_MARKER_END) + r"\s*",
            re.DOTALL,
        )
        text = pat.sub("", text).rstrip() + "\n"
        print("[REVERT] removed v4 trailer")
    if PATCHED_FP8_EINSUM in text:
        text = text.replace(PATCHED_FP8_EINSUM, ORIGINAL_FP8_EINSUM)
        print("[REVERT] restored fp8_einsum body to original")
    TARGET_FILE.write_text(text, encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true")
    g.add_argument("--revert", action="store_true")
    g.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.apply:
        return cmd_apply()
    if args.revert:
        return cmd_revert()
    if args.check:
        return cmd_check()
    return 1


if __name__ == "__main__":
    sys.exit(main())
