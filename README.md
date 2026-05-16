# vLLM × DeepSeek-V4-Flash on SM120 (Blackwell Workstation)

完整端到端部署手册：在 **NVIDIA RTX PRO 5000 72GB (SM120 / Blackwell)** 上用 **vLLM 0.20.2rc1.dev246** 跑 **DeepSeek-V4-Flash** 推理。

运行镜像：vllm/vllm-openai:cu129-nightly-28ee78af543c563a2fbf78829a7688120e4e4eb5

---

## 0. TL;DR — 完整流程一览

```bash
# 在远程 vllm 容器内执行：
cd ${PROJECT_ROOT}     # 假设你已上传整个 repo

# Phase 0  环境预检
./scripts/apply_all.sh check

# Phase 1  跑本地 mock test（验证 patch 数学正确性，不改任何文件）
./scripts/apply_all.sh selftest

# Phase 2  一键 apply 全部 5 个 patch
./scripts/apply_all.sh apply

# Phase 3  按打印的 env-var 块设置环境变量，然后启动 vllm serve
#         （完整命令见 §3）

# Phase 4  健康检查 + 渐进 smoke test
./scripts/smoke_test.sh

# Rollback（紧急）
./scripts/apply_all.sh revert
```

---

## 1. 背景与硬件约束

### 1.1 硬件目标

| 项 | 值 |
|---|---|
| GPU | NVIDIA RTX PRO 5000 Blackwell × 4 (72 GB / 卡) |
| SM 版本 | **SM120 / 12.0** (消费级 Blackwell，区别于 SM100 = B100/B200/GB200) |
| CUDA | 12.9 |
| PyTorch | 2.11.0+cu129 |
| Python | 3.12 |

### 1.2 模型 & 框架

- **模型**：DeepSeek-V4-Flash（必须用这个，不能换；硬性需求）
- **vLLM**：0.20.2rc1.dev246（pip install 标准版即可，安装路径 `/usr/local/lib/python3.12/dist-packages/vllm/`）

### 1.3 为什么需要这些 patch？

DeepSeek-V4-Flash 用了一系列只在 **SM90a (H100)** / **SM100f (B100/B200/GB200 数据中心 Blackwell)** 上跑的 CUDA kernel：

| Kernel | 提供方 | 物理硬件要求 | 在 SM120 上的现象 |
|---|---|---|---|
| `fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits` | DeepGEMM | tcgen05 (SM100) | NotImplementedError / 编译失败 |
| `cutlass_scaled_mm` (UE8M0 scale) | CUTLASS | SM90a/SM100 mixed-dtype matmul | RuntimeError |
| `fp8_einsum` (DeepGEMM) | DeepGEMM | tcgen05 | NotImplementedError |
| `flash_mla_with_kvcache` (sparse decode) | FlashMLA | SM90a/SM100f | RuntimeError: "Sparse Attention Decode Kernel is only supported on SM90a and SM100f architectures" |
| `flash_mla_sparse_fwd` (sparse prefill) | FlashMLA | SM90a/SM100f | RuntimeError: "Sparse Attention Forward Kernel is only supported on SM90a and SM100f architectures" |

5 个 patch 各自给对应的 wrapper 加 **env-var 控制的纯 PyTorch fallback**，性能下降 5-20×（acceptable，能跑就行）。

---

## 2. 仓库结构

```
vllm-deepseek-v4-flash-sm120/
├── USAGE.md                                       # 本文档
├── patches/                                       # 5 个补丁，apply 顺序由 apply_all.sh 控制
│   ├── indexer_mqa_logits_sm120_fallback.py       # #1 → vllm/utils/deep_gemm.py
│   ├── cutlass_scaled_mm_ue8m0_fallback_v3.py     # #2 → vllm/_custom_ops.py
│   ├── fp8_einsum_sm120_fallback.py               # #3 → vllm/utils/deep_gemm.py
│   ├── sparse_decode_fwd_sm120_fallback.py        # #4 → flash_mla_interface.py
│   └── sparse_prefill_fwd_sm120_fallback.py       # #5 → flash_mla_interface.py
├── tests/                                         # 本地 mock 单测（CPU/GPU 任意）
│   ├── test_sparse_decode_fallback.py             # decode 10 cases
│   └── test_sparse_prefill_fallback.py            # prefill v4.1 12 cases
└── scripts/                                       # 部署 / 验证脚本（path-aware，可在任意目录运行）
    ├── apply_all.sh                               # 推荐：一键 apply/revert/check/selftest 全部 5 个
    ├── deploy_sparse_decode.sh                    # 单 patch 部署（细粒度控制）
    ├── deploy_sparse_prefill.sh                   # 单 patch 部署（含 v4 → v4.1 redeploy）
    └── smoke_test.sh                              # 7 步渐进 HTTP smoke test
```

### 2.1 5 个 patch 的依赖关系与冲突分析

| # | Patch | Target 文件 | Backup 后缀 | env 开关 |
|---|---|---|---|---|
| 1 | indexer_mqa_logits | `vllm/utils/deep_gemm.py` | `.bak.indexer_sm120` | `VLLM_HC_FALLBACK=1`, `VLLM_INDEXER_FALLBACK=1` |
| 2 | cutlass_scaled_mm_ue8m0 v3 | `vllm/_custom_ops.py` | `.bak_v3` | `VLLM_UE8M0_FALLBACK=1` |
| 3 | fp8_einsum | `vllm/utils/deep_gemm.py` | `.bak_fp8einsum` | `VLLM_FP8_EINSUM_FALLBACK=1` |
| 4 | sparse_decode v3 | `vllm/third_party/flashmla/flash_mla_interface.py` | `.bak_sparse_decode_fb` | `VLLM_SPARSE_DECODE_FALLBACK=1` |
| 5 | sparse_prefill v4.1 | 同上 | `.bak_sparse_prefill_fb` | `VLLM_SPARSE_PREFILL_FALLBACK=1` |

**重要：patch 1 & 3 改同一个 `deep_gemm.py`；patch 4 & 5 改同一个 `flash_mla_interface.py`。** 各自的 backup suffix 完全独立，sentinel 注释也独立，可以单独 revert 而不影响对方。但是 **apply 顺序必须是 1 → 2 → 3 → 4 → 5**（每个 patch 都会把当前 target 文件作为 backup 源）。

### 2.2 每个 patch 的工作机制（共同模式）

所有 5 个 patch 走同一套模板：

1. **Backup**：apply 时把目标文件复制一份到独立后缀（如 `flash_mla_interface.py.bak_sparse_prefill_fb`）
2. **Helpers block 注入**：把 fallback 函数体作为字符串字面量插到目标文件 module 级，由 `# SM120_xxx_BEGIN` / `# SM120_xxx_END` 双 sentinel 包裹
3. **Dispatch swap 注入**：在原 CUDA kernel 调用点之前插入 env-var 分支：
   ```python
   if _SM120_xxx_FALLBACK_ENABLED:
       return _sm120_xxx_fallback(...)
   # 原 CUDA kernel 调用
   ```
4. **4 重 self-check**：apply 后立刻验证（任何一步 fail 就自动 revert）：
   - `ast.parse` 语法 OK
   - dispatch 注入到正确缩进位置
   - 清理 `__pycache__/*.pyc` 防止旧字节码污染
   - `inspect.getsource` live 验证 marker 真的在 runtime function 体里
5. **Env var 默认 OFF**：env var 不设的话 patch 是惰性的，原 kernel 仍会被调用

---

## 3. 启动命令（完整 vllm serve）

所有 patch apply 完毕后，**严格按下面这一段** export 环境变量再 `vllm serve`。少一个 env var 都会撞回原始 SM90/SM100 kernel：

```bash
# ── SM120 fallback 开关 ──────────────────────────────────────────────
export VLLM_HC_FALLBACK=1                  # patch 1 (indexer HC)
export VLLM_INDEXER_FALLBACK=1             # patch 1 (indexer mqa_logits)
export VLLM_UE8M0_FALLBACK=1               # patch 2 (ue8m0 scaled mm)
export VLLM_FP8_EINSUM_FALLBACK=1          # patch 3 (fp8 einsum)
export VLLM_FP8_EINSUM_FALLBACK_DEBUG=1    #   debug: 首次 forward 打 1 行日志
export VLLM_USE_DEEP_GEMM=0                # 关 DeepGEMM 路径（SM100-only）
export VLLM_FUSED_MOE_BACKEND=triton       # MoE 用 triton（不是 Marlin）
export VLLM_SPARSE_DECODE_FALLBACK=1       # patch 4 (sparse decode)
export VLLM_SPARSE_DECODE_FALLBACK_DEBUG=1
export VLLM_SPARSE_PREFILL_FALLBACK=1      # patch 5 (sparse prefill v4.1)
export VLLM_SPARSE_PREFILL_FALLBACK_DEBUG=1

# ── 可选性能调优 ───────────────────────────────────────────────────
# 降低 prefill fallback chunk size → 进一步压低单 chunk 峰值显存（默认 256）
# export VLLM_SPARSE_PREFILL_FALLBACK_CHUNK=128
# 缓解 PyTorch CUDA allocator 碎片化
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 启动 vllm serve ────────────────────────────────────────────────
vllm serve /workspace/models/DeepSeek-V4-Flash \
  --served-model-name deepseek-v4-flash \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-expert-parallel \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --port 8081 \
  --enable-prompt-tokens-details \
  --enforce-eager
```

**参数说明**：
- `--enforce-eager`：必须，CUDA Graph 在 fallback 路径下会捕获错的张量地址
- `--kv-cache-dtype fp8`：KV cache 用 fp8，省显存（fallback 会在 dequant 阶段处理）
- `--tensor-parallel-size 4`：跑满 4 卡
- `--max-model-len 32768`：初始保守；推到 262144 需要逐步验证（见 §6）

---

## 4. Phase-by-Phase 详细步骤

### Phase 0：环境预检（read-only，不改任何文件）

```bash
cd /vllm-workspace/vllm-deepseek-v4-flash-sm120
./scripts/apply_all.sh check
```

**预期输出**（关键行）：
```
==> repo root: /vllm-workspace/vllm-deepseek-v4-flash-sm120
==> checking presence of patch files in patches/:
  [OK]   indexer_mqa_logits_sm120_fallback.py   (15071 bytes)
  [OK]   cutlass_scaled_mm_ue8m0_fallback_v3.py   (11584 bytes)
  [OK]   fp8_einsum_sm120_fallback.py   (22549 bytes)
  [OK]   sparse_decode_fwd_sm120_fallback.py   (26140 bytes)
  [OK]   sparse_prefill_fwd_sm120_fallback.py   (26458 bytes)

==> target file presence:
  [OK]   /usr/local/lib/python3.12/dist-packages/vllm/utils/deep_gemm.py
  [OK]   /usr/local/lib/python3.12/dist-packages/vllm/_custom_ops.py
  [OK]   /usr/local/lib/python3.12/dist-packages/vllm/third_party/flashmla/flash_mla_interface.py
```

**失败动作**：
- `[MISS] patches/xxx.py` → 上传不完整，重新 scp 整个 `vllm-deepseek-v4-flash-sm120/` 目录
- `[MISS] /usr/local/.../vllm/...` → vLLM 装在别的位置，运行 `python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))"` 找真实路径，把 5 个 patch 文件顶部的 `TARGET = Path(...)` 改成真实路径

### Phase 1：本地 mock test（验证 patch 数学，不改 vllm）

```bash
./scripts/apply_all.sh selftest
```

**预期输出**：
```
==> mock test: sparse decode fallback (10 cases)
... 10 PASS ...

==> mock test: sparse prefill fallback v4.1 (12 cases)
[PASS] 01 basic_smoke
[PASS] 02 matches_reference
[PASS] 03 with_attn_sink
[PASS] 04 invalid_indices_negative
[PASS] 05 invalid_indices_overflow
[PASS] 06 topk_length
[PASS] 07 topk_length_zero
[PASS] 08 preallocated_out
[PASS] 09 attn_sink_with_invalid
[PASS] 10 shape_assertions
[PASS] 11 chunk_equivalence            ← v4.1 新增：chunk size ∈ {1,17,256,600,9999} bit-equivalent
[PASS] 12 chunk_with_invalid_and_sink  ← v4.1 新增：chunk 边界 + invalid + sink 联合正确

==== 12/12 PASS, 0 FAIL ====
```

**失败动作**：
- `chunk_equivalence FAIL` → patch 文件被改坏了，重新从 repo 拉一份
- `matches_reference FAIL` → torch 版本不兼容，确认是 torch 2.11.0+cu129

Phase 1 全过才能进 Phase 2。

### Phase 2：apply 全部 5 个 patch

```bash
./scripts/apply_all.sh apply
```

**预期输出**（每个 patch 都会打这种 3 段）：
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[1/5] indexer_mqa_logits_sm120_fallback.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[CHECK] patch NOT applied
[APPLY] backup -> .../deep_gemm.py.bak.indexer_sm120
[APPLY] injected SM120_HC_FALLBACK helpers + dispatch
[OK] self-check: ast.parse PASSED
[OK] self-check: dispatch at correct indent
[OK] self-check: live import + inspect.getsource verified
[CHECK] patch APPLIED

... [2/5] ... [3/5] ... [4/5] ... [5/5] ...

==> all 5 patches applied. Byte-compiling targets...
OK: all target files compile

==============================================================================
ADD THESE ENV VARS TO YOUR `vllm serve` COMMAND, then restart vllm:
==============================================================================
export VLLM_HC_FALLBACK=1
...
```

**失败动作**（取决于 fail 位置）：
- `[SKIP] patch already applied` → 之前已经 apply 过；要么继续，要么 `./scripts/apply_all.sh revert` 再重 apply
- `self-check ast.parse FAILED` → 极少见，patch 自己会自动 revert；查 stderr 看具体 SyntaxError，可能是 vllm 版本对不上（patch 用 regex 找 anchor，anchor 没了就插错位置）
- `live import check skipped: ImportError` → 不致命，vllm 模块还没 import 过，patch 实际已经写入磁盘；继续 phase 3

### Phase 3：启动 vllm

复制 phase 2 末尾打印的 env-var 块（或直接抄 §3 完整命令），然后启动：

```bash
# 把 §3 那一整段 export ... + vllm serve ... 跑起来
```

**预期 vllm 启动日志关键行**（应该看到 6 个 fallback 标记，证明 env var 生效）：
```
[SM120_HC_FALLBACK] engaged
[SM120_INDEXER_FALLBACK] engaged
[SM120_UE8M0_FALLBACK] engaged ...
[SM120_FP8_EINSUM_FALLBACK] engaged ...
[SM120_SPARSE_DECODE_FALLBACK] engaged ...
[SM120_SPARSE_PREFILL_FALLBACK] engaged: q=... chunk=256
```

**注意**：`[SM120_SPARSE_*_FALLBACK] engaged` 这两个标记在**第一次推理请求时**才打（lazy），不是 vllm 启动时；其他 4 个标记在 capture/profile 阶段就会出现。

**vllm 启动失败常见原因**：

| 错误 | 原因 | 修复 |
|---|---|---|
| `RuntimeError: Sparse Attention Decode Kernel is only supported on SM90a and SM100f` | `VLLM_SPARSE_DECODE_FALLBACK` 没 export | 重 export 完整 env 块 |
| `RuntimeError: ... DeepGEMM ...` | `VLLM_USE_DEEP_GEMM=0` 没设 | 同上 |
| `ImportError: cannot import _flashmla_C` | flash_mla c++ 没编译 / 装的是 wheel | 重装 vllm or 用 source 编译 |
| `CUDA out of memory` 启动阶段 | TP=4 显存不够装权重 | 检查 `nvidia-smi`，确认 4 卡都 free；模型 ~140GB / 4 = 35GB/卡，加 KV cache 应该 ~50GB/卡，72GB 足够 |

### Phase 4：smoke test（端到端验证）

```bash
./scripts/smoke_test.sh
```

7 步渐进测试：

| Step | 内容 | 预期 |
|---|---|---|
| 1 | `GET /health` | HTTP 200 |
| 2 | `GET /v1/models` | JSON 列出 `deepseek-v4-flash`, `max_model_len=32768` |
| 3 | `/v1/completions` 生成 1 token | "Hello" → 任意 1 个非空 token（验证 decode patch） |
| 4 | `/v1/chat/completions` 生成 16 tokens | 短答案如 "1+1 等于 2。"（首次走 prefill 路径） |
| 5 | `/v1/chat/completions` 生成 256 tokens | 200+ token 长回答；记录 decode tok/s |
| 6 | SSE 流式 chat completions | 持续 stream 输出（如果遇到 `curl (23) Failure writing output` 是 stdout 阻塞，不是服务端错） |
| 7 | 4k prompt 单次推理 | **v4.1 关键测试**：完整 prompt 处理 + 回答，无 OOM |

**Step 7 失败动作**（如果显存仍紧）：
1. `export VLLM_SPARSE_PREFILL_FALLBACK_CHUNK=128`（默认 256 → 128，再降一半）
2. `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
3. 重启 vllm
4. 如还 OOM，把 `--max-model-len 32768` 暂时降到 `16384` 看是否 KV cache 占太多

---

## 5. 故障排查（Troubleshooting）

### 5.1 patch 部署相关

#### 错误：`[SKIP] patch already (partially) applied`
**原因**：之前 apply 失败留下了部分 sentinel
**修复**：
```bash
# 单 patch revert
python3 patches/<patch_name>.py --revert
# 或一键 revert 全部
./scripts/apply_all.sh revert
# 然后重新 apply
./scripts/apply_all.sh apply
```

#### 错误：apply 后 vllm 启动撞 `SyntaxError` / `IndentationError`
**原因**：极少见，可能 vllm 版本不对，patch 的 regex anchor 找错位置
**修复**：
```bash
# 1. 立即 revert 所有 patch
./scripts/apply_all.sh revert
# 2. 找 backup 文件人肉对比
ls -la /usr/local/lib/python3.12/dist-packages/vllm/utils/deep_gemm.py*
# 3. 把 .bak.* 中最早的那个手动 cp 回去
# 4. 报 issue 附 vllm 版本 + python 版本
```

#### 错误：apply 成功但 vllm 行为没变（fallback 没生效）
**原因**：`__pycache__` 缓存了旧的 .pyc
**修复**：
```bash
# patch apply 时本应自动清理，再保险清一次
find /usr/local/lib/python3.12/dist-packages/vllm -name '*.pyc' -delete
# 然后重启 vllm
```

### 5.2 推理相关

#### 错误：`RuntimeError: ... SM90a and SM100f` 在 chat/completions 时
**根因**：对应的 env var 没 export 或 export 在 `vllm serve` 之后了
**修复**：
```bash
# 在同一个 shell session 里：先 export 再 vllm serve
# 验证 env 已被 vllm 看到：
python3 -c "import os; print({k:v for k,v in os.environ.items() if k.startswith('VLLM_')})"
```

#### 错误：Step 7 OOM
**根因**：sparse prefill fallback 默认 chunk=256 单层峰值 ~260 MiB，60 层 + KV cache 可能挤压
**修复（按优先级）**：
```bash
# 1. 降 chunk
export VLLM_SPARSE_PREFILL_FALLBACK_CHUNK=128
# 2. 开 expandable segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 3. 降 max-model-len
# 把启动命令的 --max-model-len 从 32768 改成 16384
# 4. 重启 vllm
```

#### 错误：decode 出来全是乱码 / 重复 token
**根因**：极少见，但如果 sparse decode fallback 的 fp8 dequant 路径有 bug
**诊断**：
```bash
# 看 worker 日志找 SM120_SPARSE_DECODE_FALLBACK debug line：
# 应该看到 "engaged: q=(B, S, H, D=512) cache=..."
grep SM120 /path/to/vllm.log | head -20
# 如果数值错，临时 revert decode patch 看看原 kernel 是否报错
python3 patches/sparse_decode_fwd_sm120_fallback.py --revert
# 重启 vllm，预期会撞 SM90a 错误（确认你的环境是 SM120 没救），
# 然后再 apply 回来，并附 worker 日志报 issue
```

### 5.3 性能相关

#### 现象：prefill 很慢（>5s for 1k tokens）
**预期**：fallback 比原生 SM100 kernel 慢 5-20×，prefill 1k token 可能 2-5 秒，4k 可能 10-30 秒。这是 trade-off，acceptable。

**轻度优化**：
- `VLLM_SPARSE_PREFILL_FALLBACK_CHUNK=512` 在显存充足时拉大 chunk → 减少 launch overhead
- 在 vllm 启动后第一次推理时会有 cudnn benchmark，第二次开始稳定

---

## 6. 进阶：把 max-model-len 推到 262144

默认手册用 `--max-model-len 32768` 是保守值。要推到 262144（DSV4-Flash 设计上限）：

```bash
# 1. 先确保 chunk=128 + expandable_segments 已开
export VLLM_SPARSE_PREFILL_FALLBACK_CHUNK=128
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. 改启动参数（注意：KV cache fp8 时每 token 占 ~256B/层 × 60 层 = 15KB；
#    262144 token 单序列 ~4GB KV cache，并发数会受限）
#    --max-model-len 262144
#    --max-num-seqs 1                      # 长上下文场景并发降到 1
#    --gpu-memory-utilization 0.92         # 给 KV cache 留空间

# 3. 重启后用 16k → 64k → 128k → 262144 渐进 prompt 验证
```

**已知风险**：MARLIN MoE 在长 prompt 下可能成为下个 SM120 撞点（高概率，未验证）。撞了按同一套路开 patch 6。

---

## 7. 单 patch 操作（细粒度控制）

如果只想 apply / revert 单个 patch（比如调试 v4 → v4.1 升级）：

```bash
# 直接调 patch 的 CLI
python3 patches/sparse_prefill_fwd_sm120_fallback.py --check
python3 patches/sparse_prefill_fwd_sm120_fallback.py --apply
python3 patches/sparse_prefill_fwd_sm120_fallback.py --revert

# 或用 deploy 包装脚本（带 byte-compile + decode v3 依赖检查）
./scripts/deploy_sparse_prefill.sh check
./scripts/deploy_sparse_prefill.sh selftest
./scripts/deploy_sparse_prefill.sh apply
./scripts/deploy_sparse_prefill.sh redeploy   # = revert + clear pycache + apply
```

每个 patch 文件是**自包含**的：包含完整 docstring（背景、错误现场、修复原理）、helpers 字符串、dispatch 注入逻辑、4 重 self-check、apply/revert/check CLI。可以单独 cp 到任何环境用。

---

## 8. 完整 rollback playbook（紧急）

如果 patched vllm 出问题，要回到原始状态：

```bash
# 一键 revert 全部 5 个 patch（按 5→4→3→2→1 反序）
./scripts/apply_all.sh revert

# 清字节码缓存
find /usr/local/lib/python3.12/dist-packages/vllm -name '*.pyc' -delete

# 验证所有 backup 已恢复
./scripts/apply_all.sh check
# 期望：所有 patch 显示 "NOT applied"

# 重启 vllm（不带 SM120 env vars，预期会撞原始 SM90/SM100 错误，证明已 revert 干净）
```

如果 revert 也失败（极端情况，backup 被误删），手动从干净的 vllm wheel 重装：
```bash
pip install --force-reinstall vllm==0.20.2rc1.dev246
```

---

## 9. 设计哲学 & 已知 trade-off

### 9.1 为什么是纯 PyTorch fallback 而不是 triton 重写？
- **正确性优先**：纯 PyTorch 算子是 reference implementation，单元测试容易写
- **可读性**：60 行 PyTorch ≪ 600 行 triton，patch 出 bug 时易调试
- **覆盖范围**：5 个 kernel 一致用同一套模板，统一维护

### 9.2 已知性能损失（粗估）
| 路径 | 原生 SM90/SM100 | SM120 fallback | 比例 |
|---|---|---|---|
| sparse decode | 0.5 ms | ~3-5 ms | 6-10× 慢 |
| sparse prefill | 2 ms / 1k tok | 20-50 ms / 1k tok | 10-25× 慢 |
| ue8m0 scaled mm | <1 ms | ~5 ms | 5× 慢 |
| fp8 einsum | 1 ms | 10-20 ms | 10-20× 慢 |

**结果**：在 SM120 上跑 DSV4-Flash 端到端 throughput 大约是 SM100 的 1/10，但能跑（这是用户的硬性约束：不能换模型）。

### 9.3 v4.1 vs v4（prefill patch 历史）
- **v4 问题**：fp32 `K_f` 升级 + 全 Sq 一次 bmm，单层峰值 4.2 GiB，4k prompt 撞 OOM
- **v4.1 修复**：bf16 native bmm（仅 logits 升 fp32 给数值稳定）+ Sq 维 streaming chunk loop（默认 256），单 chunk 峰值降到 ~260 MiB（16× 改善）
- **数学等价**：chunk 之间无依赖（per-query 独立），test_chunk_equivalence 用 `torch.equal` 验证 chunk size ∈ {1, 17, 256, 600, 9999} 输出 bit-equivalent

---

## 10. Quick reference cheat sheet

```bash
# repo
cd /vllm-workspace/vllm-deepseek-v4-flash-sm120

# 状态
./scripts/apply_all.sh check

# 数学验证
./scripts/apply_all.sh selftest

# 全套部署
./scripts/apply_all.sh apply
# 按提示 export env vars + vllm serve

# 验证
./scripts/smoke_test.sh

# 单 patch 操作
python3 patches/<name>.py --{check,apply,revert}

# Prefill v4 → v4.1 升级
./scripts/deploy_sparse_prefill.sh redeploy

# 紧急回滚
./scripts/apply_all.sh revert
find /usr/local/lib/python3.12/dist-packages/vllm -name '*.pyc' -delete
```

---

**版本信息**：本手册对应 patch 集合：
- indexer_mqa_logits v1
- cutlass_scaled_mm_ue8m0 v3
- fp8_einsum v2
- sparse_decode v3
- sparse_prefill **v4.1**（最新，含 OOM 修复）

**联系 / 反馈**：如发现新的 SM120 撞点（如 MARLIN MoE、kvcache routing 等），按同一套 patch 模板新增即可。模板：`patches/sparse_prefill_fwd_sm120_fallback.py` 是最完整的样板，可 cp 改名。
