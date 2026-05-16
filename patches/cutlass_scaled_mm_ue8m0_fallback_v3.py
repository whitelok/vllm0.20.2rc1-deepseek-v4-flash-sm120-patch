#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cutlass_scaled_mm_ue8m0_fallback_v3.py
========================================

v2 失败原因复盘:
    v2 用 `torch.ops._C.cutlass_scaled_mm = _patched_xxx` 直接给
    OpOverloadPacket 赋值, 把它替换成了普通 Python function.
    后续 torch._export.utils._collect_all_valid_cia_ops_for_namespace
    遍历 torch.ops._C 时调 `.overloads()` 失败, 把 CustomDecompTable
    搞成半初始化坏态, 导致 torch._inductor.decomposition import 时
    `{**core_aten_decompositions()}` 抛 TypeError:
        'CustomDecompTable' object is not a mapping.

    Traceback 在 v2 表面上指向"补丁污染 inductor", 实际根因是
    "破坏了 torch.ops._C 的 OpPacket 遍历".

v3 策略:
    完全不触碰 torch.ops._C 名字空间, 改成:
    1) 在 _custom_ops.py 末尾追加一个 helper `_ue8m0_safe_cutlass_scaled_mm`
    2) 用 sed-like 文本替换, 把文件里所有
           torch.ops._C.cutlass_scaled_mm(out, a, b, scale_a, scale_b, ...)
       的直接调用改成
           _ue8m0_safe_cutlass_scaled_mm(out, a, b, scale_a, scale_b, ...)
    3) helper 内部把 UE8M0 dtype 的 scale 张量 .to(torch.float32) 再
       转发到 torch.ops._C.cutlass_scaled_mm (一次性调用, 不污染名字空间)

完整 vLLM 启动命令:
----------------------------------------------------------------------
export VLLM_HC_FALLBACK=1
export VLLM_INDEXER_FALLBACK=1
export VLLM_UE8M0_FALLBACK=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_FUSED_MOE_BACKEND=triton

vllm serve /workspace/models/DeepSeek-V4-Flash \
  --served-model-name deepseek-v4-flash --trust-remote-code \
  --kv-cache-dtype fp8 --block-size 256 --enable-expert-parallel \
  --tensor-parallel-size 4 --max-model-len 32768 \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
  --port 8081 --enable-prompt-tokens-details --enforce-eager
----------------------------------------------------------------------

用法:
    sudo python3 cutlass_scaled_mm_ue8m0_fallback_v3.py --apply
    sudo python3 cutlass_scaled_mm_ue8m0_fallback_v3.py --revert
    python3      cutlass_scaled_mm_ue8m0_fallback_v3.py --check

注意:
    必须先把 v2 (如果还在) 完全 revert 干净再 apply v3, 不然两个补丁
    会同时存在. v3 的 --apply 会先检测 v2 残留, 自动清理.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

TARGET_FILE = Path("/usr/local/lib/python3.12/dist-packages/vllm/_custom_ops.py")
BACKUP_FILE = TARGET_FILE.with_suffix(TARGET_FILE.suffix + ".bak_v3")

V2_MARKER_BEGIN = "# >>> SM120_UE8M0_FALLBACK_PATCH_BEGIN <<<"
V2_MARKER_END = "# <<< SM120_UE8M0_FALLBACK_PATCH_END >>>"

V3_MARKER_BEGIN = "# >>> SM120_UE8M0_FALLBACK_V3_BEGIN <<<"
V3_MARKER_END = "# <<< SM120_UE8M0_FALLBACK_V3_END >>>"

# 末尾追加: 安全的 helper, 不污染 torch.ops._C
V3_APPEND_CODE = f'''

{V3_MARKER_BEGIN}
# Auto-injected by cutlass_scaled_mm_ue8m0_fallback_v3.py
# 不替换 torch.ops._C.cutlass_scaled_mm, 只提供一个 module-level helper
# 供本文件其他位置调用. 这样不会破坏 torch.ops._C 的 OpPacket 遍历.
import os as _os_ue8m0_v3
import torch as _torch_ue8m0_v3

_UE8M0_FALLBACK_ENABLED_V3 = _os_ue8m0_v3.environ.get("VLLM_UE8M0_FALLBACK", "1") != "0"

try:
    _UE8M0_DTYPE_V3 = _torch_ue8m0_v3.float8_e8m0fnu  # type: ignore[attr-defined]
except AttributeError:
    _UE8M0_DTYPE_V3 = None

def _ue8m0_to_fp32_v3(t):
    if (t is None) or (_UE8M0_DTYPE_V3 is None) or (not _UE8M0_FALLBACK_ENABLED_V3):
        return t
    if isinstance(t, _torch_ue8m0_v3.Tensor) and t.dtype == _UE8M0_DTYPE_V3:
        return t.to(_torch_ue8m0_v3.float32)
    return t

def _ue8m0_safe_cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias=None):
    """UE8M0-aware shim around torch.ops._C.cutlass_scaled_mm.

    Converts scale_a / scale_b to float32 if they are UE8M0 (float8_e8m0fnu),
    because the underlying C++ kernel does not yet support ScalarType 44
    (UE8M0) and would raise RuntimeError.
    """
    scale_a = _ue8m0_to_fp32_v3(scale_a)
    scale_b = _ue8m0_to_fp32_v3(scale_b)
    return _torch_ue8m0_v3.ops._C.cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)

print("[SM120_UE8M0_FALLBACK_V3] helper installed (no torch.ops._C namespace pollution)")
{V3_MARKER_END}
'''


def _has_v2(text: str) -> bool:
    return V2_MARKER_BEGIN in text or V2_MARKER_END in text


def _has_v3(text: str) -> bool:
    return V3_MARKER_BEGIN in text and V3_MARKER_END in text


def _strip_v2(text: str) -> tuple[str, bool]:
    """Remove the v2 injected block (markers + content). Returns (new_text, removed?)."""
    if V2_MARKER_BEGIN not in text:
        return text, False
    pattern = re.compile(
        re.escape(V2_MARKER_BEGIN) + r".*?" + re.escape(V2_MARKER_END) + r"\s*",
        re.DOTALL,
    )
    new_text, n = pattern.subn("", text)
    return new_text.rstrip() + "\n", n > 0


# 把所有直接调用 torch.ops._C.cutlass_scaled_mm( -> _ue8m0_safe_cutlass_scaled_mm(
_INPLACE_PATTERN = re.compile(r"torch\.ops\._C\.cutlass_scaled_mm\(")
_INPLACE_REPLACEMENT = "_ue8m0_safe_cutlass_scaled_mm("


def _rewrite_call_sites(text: str) -> tuple[str, int]:
    new_text, n = _INPLACE_PATTERN.subn(_INPLACE_REPLACEMENT, text)
    return new_text, n


def _restore_call_sites(text: str) -> tuple[str, int]:
    """Revert rewrite: _ue8m0_safe_cutlass_scaled_mm( -> torch.ops._C.cutlass_scaled_mm("""
    pat = re.compile(r"_ue8m0_safe_cutlass_scaled_mm\(")
    new_text, n = pat.subn("torch.ops._C.cutlass_scaled_mm(", text)
    return new_text, n


def cmd_check() -> int:
    if not TARGET_FILE.exists():
        print(f"[CHECK] target NOT FOUND: {TARGET_FILE}")
        return 2
    text = TARGET_FILE.read_text(encoding="utf-8")
    v2 = _has_v2(text)
    v3 = _has_v3(text)
    sites = len(_INPLACE_PATTERN.findall(text))
    safe_calls = text.count("_ue8m0_safe_cutlass_scaled_mm(")
    print(f"[CHECK] file                : {TARGET_FILE}")
    print(f"[CHECK] v2 marker present   : {v2}")
    print(f"[CHECK] v3 marker present   : {v3}")
    print(f"[CHECK] raw  torch.ops._C.cutlass_scaled_mm( sites: {sites}")
    print(f"[CHECK] safe _ue8m0_safe_cutlass_scaled_mm(  sites: {safe_calls}")
    print(f"[CHECK] backup .bak_v3 exists: {BACKUP_FILE.exists()}")
    if v3 and sites == 0 and safe_calls >= 1:
        print("[CHECK] STATE = v3 fully applied ✓")
        return 0
    if not v3 and sites >= 1 and safe_calls == 0:
        print("[CHECK] STATE = clean (no v3)")
        return 1
    print("[CHECK] STATE = INCONSISTENT (manual inspection recommended)")
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

    # 备份原始(无补丁)的文件. 若已经存在 .bak_v3 备份, 保留它(更原始).
    if not BACKUP_FILE.exists():
        # 先尝试从 v2 的 .bak_ue8m0 拿真正的干净版本
        v2_bak = TARGET_FILE.with_suffix(TARGET_FILE.suffix + ".bak_ue8m0")
        if v2_bak.exists():
            shutil.copy2(v2_bak, BACKUP_FILE)
            print(f"[APPLY] backup from existing v2 .bak_ue8m0 -> {BACKUP_FILE}")
        else:
            shutil.copy2(TARGET_FILE, BACKUP_FILE)
            print(f"[APPLY] backup created from current file -> {BACKUP_FILE}")
    else:
        print(f"[APPLY] backup already exists, keep it: {BACKUP_FILE}")

    # 1) 如果有 v2 残留, 先 strip 掉
    text, v2_removed = _strip_v2(text)
    if v2_removed:
        print("[APPLY] stripped existing v2 injection")

    # 2) 如果已经有 v3, 也先 strip
    if _has_v3(text):
        pattern = re.compile(
            re.escape(V3_MARKER_BEGIN) + r".*?" + re.escape(V3_MARKER_END) + r"\s*",
            re.DOTALL,
        )
        text = pattern.sub("", text).rstrip() + "\n"
        print("[APPLY] stripped existing v3 injection (will re-apply)")

    # 3) 改写所有调用点
    text, n_sites = _rewrite_call_sites(text)
    print(f"[APPLY] rewrote {n_sites} call site(s) of torch.ops._C.cutlass_scaled_mm(")

    # 4) 末尾追加 helper
    text = text.rstrip() + "\n" + V3_APPEND_CODE + "\n"
    TARGET_FILE.write_text(text, encoding="utf-8")
    print(f"[APPLY] patch v3 written to {TARGET_FILE}")

    # 自检
    verify = TARGET_FILE.read_text(encoding="utf-8")
    leftover_raw = len(_INPLACE_PATTERN.findall(verify))
    safe_calls = verify.count("_ue8m0_safe_cutlass_scaled_mm(")
    has_v3 = _has_v3(verify)
    print(
        f"[APPLY] self-check: v3_marker={has_v3} raw_left={leftover_raw} safe_calls={safe_calls}"
    )
    if not has_v3 or leftover_raw != 0 or safe_calls < 1:
        print("[APPLY][ERR] self-check FAILED", file=sys.stderr)
        return 4
    print("[APPLY] self-check OK ✓")

    # grep 给人看
    try:
        out = subprocess.check_output(
            [
                "grep",
                "-n",
                "SM120_UE8M0_FALLBACK_V3\\|_ue8m0_safe_cutlass_scaled_mm",
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

    # 没有备份, 尝试 in-place 反向改写
    print(f"[REVERT][WARN] backup not found, attempting in-place revert ...")
    if not TARGET_FILE.exists():
        print(f"[REVERT][ERR] target NOT FOUND", file=sys.stderr)
        return 2
    if not os.access(TARGET_FILE.parent, os.W_OK):
        print(f"[REVERT][ERR] no write permission, use sudo", file=sys.stderr)
        return 3
    text = TARGET_FILE.read_text(encoding="utf-8")
    text, v2_removed = _strip_v2(text)
    if v2_removed:
        print("[REVERT] removed v2 block")
    if _has_v3(text):
        pattern = re.compile(
            re.escape(V3_MARKER_BEGIN) + r".*?" + re.escape(V3_MARKER_END) + r"\s*",
            re.DOTALL,
        )
        text = pattern.sub("", text).rstrip() + "\n"
        print("[REVERT] removed v3 block")
    text, n = _restore_call_sites(text)
    if n:
        print(
            f"[REVERT] restored {n} call site(s) back to torch.ops._C.cutlass_scaled_mm("
        )
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
