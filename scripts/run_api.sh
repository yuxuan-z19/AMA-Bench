LLM_SERVER="api"    # api or vllm
# Available API configs: configs/gpt-5.2.yaml, configs/gemini-2.5-pro.yaml, configs/llm_judge_api.yaml
LLM_CONFIG="${LLM_CONFIG:-configs/gpt-5.2.yaml}"
SUBSET="openend"
TEST_DIR="${TEST_DIR:-dataset/test}"
OUTPUT_DIR="${OUTPUT_DIR:-results/openend}"

MAX_CONCURRENCY_EPISODES="${MAX_CONCURRENCY_EPISODES:-50}"  # Limit concurrency
MAX_CONCURRENCY_QUESTIONS_PER_EPISODE="${MAX_CONCURRENCY_QUESTIONS_PER_EPISODE:-12}"  # Limit concurrency for questions within an episode
METHOD="${METHOD:-longcontext}"  # Available methods: longcontext (default), bm25, embedding

# LLM-as-Judge configuration
JUDGE_CONFIG="${JUDGE_CONFIG:-configs/llm_judge_api.yaml}"
JUDGE_SERVER="${JUDGE_SERVER:-api}"
EVALUATE="${EVALUATE:-True}"  # Whether to evaluate answers

# Method-specific configuration (optional)
METHOD_CONFIG="${METHOD_CONFIG:-}"

# Build arguments
ARGS=(
  --llm-server "$LLM_SERVER"
  --llm-config "$LLM_CONFIG"
  --subset "$SUBSET"
  --method "$METHOD"
  --test-dir "$TEST_DIR"
  --output-dir "$OUTPUT_DIR"
  --max-concurrency-episodes "$MAX_CONCURRENCY_EPISODES"
  --max-concurrency-questions-per-episode "$MAX_CONCURRENCY_QUESTIONS_PER_EPISODE"
  --judge-config "$JUDGE_CONFIG"
  --judge-server "$JUDGE_SERVER"
  --evaluate "$EVALUATE"
)

# Add method config if provided
if [ -n "$METHOD_CONFIG" ]; then
  ARGS+=(--method-config "$METHOD_CONFIG")
fi

# Run evaluation with LLM-as-Judge
echo "Running OpenEnd evaluation with method: $METHOD"
echo "LLM config: $LLM_CONFIG (server: $LLM_SERVER)"
echo "LLM-as-Judge: $JUDGE_SERVER (config: $JUDGE_CONFIG)"
echo "Evaluate: $EVALUATE"
python src/run.py "${ARGS[@]}"
