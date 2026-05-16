#!/usr/bin/env bash
# smoke_test.sh
# ============================================================
# DeepSeek-V4-Flash @ vLLM 0.20.2rc1.dev246 on RTX PRO 5000 (SM120)
# 端口 8081, 模型名 deepseek-v4-flash
#
# 用法:
#   bash smoke_test.sh                  # 全跑
#   bash smoke_test.sh 1                # 只跑第 1 项
#   HOST=10.x.x.x bash smoke_test.sh    # 远端测试
# ============================================================
set -u

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8081}"
MODEL="${MODEL:-deepseek-v4-flash}"
BASE="http://${HOST}:${PORT}"

ONLY="${1:-all}"

step() { echo; echo "============================================================"; echo "[STEP $1] $2"; echo "============================================================"; }
want() { [[ "$ONLY" == "all" || "$ONLY" == "$1" ]]; }

# ---------- 1) 健康检查 ----------
if want 1; then
  step 1 "GET /health (无需推理, 验证 server alive)"
  curl -sS -m 5 -o /tmp/_health.txt -w "HTTP=%{http_code} time=%{time_total}s\n" "$BASE/health"
  cat /tmp/_health.txt; echo
fi

# ---------- 2) 模型列表 ----------
if want 2; then
  step 2 "GET /v1/models (验证模型已注册)"
  curl -sS -m 5 "$BASE/v1/models" | python3 -m json.tool
fi

# ---------- 3) 最小 completion (1 token, greedy, 不流式) ----------
# 这是最便宜的真推理探针, 任何 kernel 路径出问题都会立刻报 500.
if want 3; then
  step 3 "POST /v1/completions  (1 token, greedy, 探测推理路径)"
  curl -sS -m 60 "$BASE/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"prompt\": \"Hello\",
      \"max_tokens\": 1,
      \"temperature\": 0,
      \"stream\": false
    }" | python3 -m json.tool
fi

# ---------- 4) 短 chat (验证 chat template 和 tool/reason parser 不会爆) ----------
if want 4; then
  step 4 "POST /v1/chat/completions  (16 tokens, greedy)"
  curl -sS -m 120 "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"messages\": [
        {\"role\": \"user\", \"content\": \"用一句话回答: 1+1 等于几?\"}
      ],
      \"max_tokens\": 16,
      \"temperature\": 0,
      \"stream\": false
    }" | python3 -m json.tool
fi

# ---------- 5) 中等长度 chat (验证 prefill+decode 全路径, 看 token/s) ----------
if want 5; then
  step 5 "POST /v1/chat/completions  (256 tokens, 验证 decode 稳定性 + 速率)"
  T0=$(python3 -c "import time; print(time.time())")
  curl -sS -m 300 "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"messages\": [
        {\"role\": \"user\", \"content\": \"请用中文写一段大约 200 字的介绍, 主题: 大语言模型推理引擎.\"}
      ],
      \"max_tokens\": 256,
      \"temperature\": 0,
      \"stream\": false
    }" -o /tmp/_chat5.json
  T1=$(python3 -c "import time; print(time.time())")
  python3 -c "
import json, sys
d = json.load(open('/tmp/_chat5.json'))
print(json.dumps(d, ensure_ascii=False, indent=2))
print()
u = d.get('usage', {})
dt = $T1 - $T0
print(f'>>> wall_time={dt:.2f}s  prompt={u.get(\"prompt_tokens\")}  completion={u.get(\"completion_tokens\")}  total={u.get(\"total_tokens\")}')
if u.get('completion_tokens'):
    print(f'>>> decode_tok/s ≈ {u[\"completion_tokens\"]/dt:.2f}')
"
fi

# ---------- 6) 流式 (验证 SSE 路径) ----------
if want 6; then
  step 6 "POST /v1/chat/completions  stream=true  (验证 SSE)"
  curl -sS -N -m 60 "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"messages\": [{\"role\": \"user\", \"content\": \"数 1 到 10, 用逗号分隔.\"}],
      \"max_tokens\": 48,
      \"temperature\": 0,
      \"stream\": true
    }" | head -c 4000
  echo
fi

# ---------- 7) 长上下文小试 (验证 max-model-len=32768 中段) ----------
if want 7; then
  step 7 "POST /v1/completions  ~4k prompt (验证 prefill 不崩)"
  PROMPT=$(python3 -c "print(('量子力学 ' * 800)[:4000])")
  curl -sS -m 180 "$BASE/v1/completions" \
    -H "Content-Type: application/json" \
    --data-binary "$(python3 -c "
import json,sys
print(json.dumps({
  'model': '$MODEL',
  'prompt': '${PROMPT}\n请用一句话总结上文主题:',
  'max_tokens': 32,
  'temperature': 0,
  'stream': False
}, ensure_ascii=False))
")" | python3 -m json.tool
fi

echo
echo "============================================================"
echo "DONE. 通过判定:"
echo "  - step 1 HTTP=200"
echo "  - step 3 返回 'choices'[0]['text'] 非空 (任意字符)"
echo "  - step 4/5 返回的 message.content 不为空且无 finish_reason='error'"
echo "  - 没有 500 / Internal Server Error"
echo "============================================================"
