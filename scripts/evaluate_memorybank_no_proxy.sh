#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_DIR="${RUN_DIR:-results/full_memorybank_paratera_thinking}"
ANSWERS_FILE="${ANSWERS_FILE:-$RUN_DIR/answers_Qwen3-32B_openend_memorybank_merged.jsonl}"
TEST_FILE="${TEST_FILE:-dataset/test/open_end_qa_set.jsonl}"
JUDGE_CONFIG="${JUDGE_CONFIG:-configs/qwen3_api.yaml}"
JUDGE_SERVER="${JUDGE_SERVER:-api}"
OUTPUT_FILE="${OUTPUT_FILE:-$RUN_DIR/evaluation_Qwen3-32B_openend_memorybank_merged_qwen3_judge_timeout2x.json}"
LOG_FILE="${LOG_FILE:-$RUN_DIR/eval_nohup.log}"
PID_FILE="${PID_FILE:-$RUN_DIR/eval.pid}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
DETACH="${DETACH:-1}"

mkdir -p "$RUN_DIR"

cmd=(
  env
  -u http_proxy -u https_proxy -u all_proxy
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY
  NO_PROXY="*" no_proxy="*"
  PYTHONUNBUFFERED=1
  PYTHONPATH=.
  "$PYTHON_BIN" src/evaluate.py
  --answers-file "$ANSWERS_FILE"
  --test-file "$TEST_FILE"
  --judge-config "$JUDGE_CONFIG"
  --judge-server "$JUDGE_SERVER"
  --output-file "$OUTPUT_FILE"
)

echo "MemoryBank evaluation"
echo "  answers_file: $ANSWERS_FILE"
echo "  output_file: $OUTPUT_FILE"
echo "  log_file: $LOG_FILE"
echo "  judge_config: $JUDGE_CONFIG"
echo "  judge_server: $JUDGE_SERVER"
echo "  judge_workers: evaluate.py default"
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
