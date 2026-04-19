LLM_SERVER="vllm"    # api or vllm
LLM_CONFIG="${LLM_CONFIG:-configs/qwen3-32B.yaml}"
SUBSET="openend"
TEST_DIR="${TEST_DIR:-dataset/test}"
OUTPUT_DIR="${OUTPUT_DIR:-results/openend}"

# Launch vLLM server
bash scripts/launch_vllm_32B.sh "$LLM_CONFIG"
echo ""
MAX_CONCURRENCY_EPISODES="${MAX_CONCURRENCY_EPISODES:-30}"  # Limit concurrency
MAX_CONCURRENCY_QUESTIONS_PER_EPISODE="${MAX_CONCURRENCY_QUESTIONS_PER_EPISODE:-12}"  # Limit concurrency for questions within an episode
METHOD="${METHOD:-ama_agent}"  # Available methods: longcontext (default), bm25, embedding

# LLM-as-Judge configuration
JUDGE_CONFIG="${JUDGE_CONFIG:-configs/qwen3-32B.yaml}"
JUDGE_SERVER="${JUDGE_SERVER:-vllm}"
EVALUATE="${EVALUATE:-True}"  # Whether to evaluate answers
JUDGE_MAX_CONCURRENCY="${JUDGE_MAX_CONCURRENCY:-$((MAX_CONCURRENCY_EPISODES * MAX_CONCURRENCY_QUESTIONS_PER_EPISODE))}"

# Method-specific configuration (optional)
METHOD_CONFIG="${METHOD_CONFIG:-configs/ama_agent.yaml}"  # Only needed for certain methods like ama_agent

# Sampling / filtering (mutually exclusive)
SAMPLES="${SAMPLES:-150}"        # e.g. SAMPLES=50 to randomly sample 50 episodes
DOMAINS="${DOMAINS:-}"        # e.g. DOMAINS="embodied_ai,software_engineer"

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
  --judge-max-concurrency "$JUDGE_MAX_CONCURRENCY"
  --evaluate "$EVALUATE"
)

# Add method config if provided
if [ -n "$METHOD_CONFIG" ]; then
  ARGS+=(--method-config "$METHOD_CONFIG")
fi

# Add sampling / domain filtering (mutually exclusive)
if [ -n "$SAMPLES" ] && [ -n "$DOMAINS" ]; then
  echo "Error: SAMPLES and DOMAINS cannot be used at the same time." >&2
  exit 1
fi
if [ -n "$SAMPLES" ]; then
  ARGS+=(--samples "$SAMPLES")
fi
if [ -n "$DOMAINS" ]; then
  ARGS+=(--domains "$DOMAINS")
fi

# Run evaluation with LLM-as-Judge
echo "Running OpenEnd evaluation with method: $METHOD"
echo "LLM-as-Judge: $JUDGE_SERVER (config: $JUDGE_CONFIG)"
echo "Evaluate: $EVALUATE"
python src/run.py "${ARGS[@]}"
