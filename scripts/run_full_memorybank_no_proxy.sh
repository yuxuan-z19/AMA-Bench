#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_DIR="${RUN_DIR:-results/full_memorybank_paratera_thinking}"
LOG_FILE="${LOG_FILE:-$RUN_DIR/run_nohup.log}"
PID_FILE="${PID_FILE:-$RUN_DIR/run.pid}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

LLM_CONFIG="${LLM_CONFIG:-configs/qwen3_api.yaml}"
METHOD_CONFIG="${METHOD_CONFIG:-configs/method_configs/memorybank_config.json}"
TEST_DIR="${TEST_DIR:-dataset/test}"
SUBSET="${SUBSET:-openend}"
MAX_CONCURRENCY_EPISODES="${MAX_CONCURRENCY_EPISODES:-5}"
MAX_CONCURRENCY_QUESTIONS_PER_EPISODE="${MAX_CONCURRENCY_QUESTIONS_PER_EPISODE:-1}"
EVALUATE="${EVALUATE:-False}"
DETACH="${DETACH:-1}"
EPISODE_IDS="${EPISODE_IDS:-}"

mkdir -p "$RUN_DIR"

cmd=(
  env
  -u http_proxy -u https_proxy -u all_proxy
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY
  NO_PROXY="*" no_proxy="*"
  PYTHONUNBUFFERED=1
  PYTHONPATH=.
  "$PYTHON_BIN" src/run.py
  --llm-server api
  --llm-config "$LLM_CONFIG"
  --subset "$SUBSET"
  --method memorybank
  --method-config "$METHOD_CONFIG"
  --test-dir "$TEST_DIR"
  --evaluate "$EVALUATE"
  --max-concurrency-episodes "$MAX_CONCURRENCY_EPISODES"
  --max-concurrency-questions-per-episode "$MAX_CONCURRENCY_QUESTIONS_PER_EPISODE"
  --output-dir "$RUN_DIR"
)

if [[ -n "$EPISODE_IDS" ]]; then
  cmd+=(--episode-ids "$EPISODE_IDS")
fi

echo "MemoryBank generation"
echo "  output_dir: $RUN_DIR"
echo "  log_file: $LOG_FILE"
echo "  llm_config: $LLM_CONFIG"
echo "  method_config: $METHOD_CONFIG"
echo "  max_concurrency_episodes: $MAX_CONCURRENCY_EPISODES"
echo "  max_concurrency_questions_per_episode: $MAX_CONCURRENCY_QUESTIONS_PER_EPISODE"
echo "  no_proxy: enabled"

if [[ "$DETACH" == "1" ]]; then
  : > "$LOG_FILE"
  setsid nohup "${cmd[@]}" >> "$LOG_FILE" 2>&1 < /dev/null &
  pid=$!
  printf "%s\n" "$pid" > "$PID_FILE"
  echo "  pid: $pid"
  echo "  pid_file: $PID_FILE"
else
  exec "${cmd[@]}"
fi
