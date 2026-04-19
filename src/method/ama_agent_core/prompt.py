CODE_GENERATION_PROMPT_TEMPLATE = """You are helping to extract relevant information from a trajectory to answer a question by writing Python code.

**Question:** {query}

**Task:** {task}

**Trajectory Format Reference (first 2 turns):**
{trajectory_sample}

**Trajectory Data (JSON format):**
Available as variable `trajectory_json` with structure:
{{
  "trajectory": [
    {{
      "turn_idx": 0,      // Turn number (int)
      "action": "...",    // Action taken at this turn (string)
      "observation": "..." // Environment observation after action (string)
    }},
    ...
  ],
  "task": "...",        // Task description (string)
  "episode_id": "..."   // Episode identifier (string)
}}

**Your Task:**
Write Python code that processes the trajectory JSON and extracts the relevant information to answer the question.

**Available Examples:**

Example 1: Finding specific actions
```python
import json

# Parse trajectory
trajectory_data = json.loads(trajectory_json)
trajectory = trajectory_data['trajectory']

# Find turns where a specific action was taken
relevant_turns = []
for turn in trajectory:
    if 'pick up' in turn.get('action', '').lower():
        relevant_turns.append({{
            'turn': turn['turn_idx'],
            'action': turn['action'],
            'observation': turn.get('observation', '')[:200]
        }})

result = {{
    'relevant_turns': relevant_turns,
    'count': len(relevant_turns)
}}
```



Example 2: Finding when something happened (until turn X)
```python
import json

trajectory_data = json.loads(trajectory_json)
trajectory = trajectory_data['trajectory']

# Find first turn when door was opened
first_open = None
for turn in trajectory:
    if 'open door' in turn.get('action', '').lower():
        first_open = turn['turn_idx']
        break

# Get all turns until that point
turns_until_open = [t for t in trajectory if t['turn_idx'] <= first_open] if first_open else []

result = {{
    'event': 'door opened',
    'first_occurrence': first_open,
    'turns_until_event': len(turns_until_open)
}}
```

Example 3: Finding last occurrence of something
```python
import json

trajectory_data = json.loads(trajectory_json)
trajectory = trajectory_data['trajectory']

# Find last turn where agent picked up something
last_pickup = None
for turn in reversed(trajectory):
    if 'pick up' in turn.get('action', '').lower():
        last_pickup = {{
            'turn': turn['turn_idx'],
            'action': turn['action'],
            'observation': turn.get('observation', '')[:200]
        }}
        break

result = {{'last_pickup': last_pickup}}
```

Example 4: Causal relationship - what happened after X
```python
import json

trajectory_data = json.loads(trajectory_json)
trajectory = trajectory_data['trajectory']

# Find when key was picked up, then what happened next
key_pickup_turn = None
for turn in trajectory:
    if 'key' in turn.get('action', '').lower() and 'pick' in turn.get('action', '').lower():
        key_pickup_turn = turn['turn_idx']
        break

# Get next 3 turns after picking up key
subsequent_actions = []
if key_pickup_turn is not None:
    for turn in trajectory:
        if turn['turn_idx'] > key_pickup_turn and len(subsequent_actions) < 3:
            subsequent_actions.append({{
                'turn': turn['turn_idx'],
                'action': turn['action']
            }})

result = {{
    'trigger_event': 'picked up key',
    'trigger_turn': key_pickup_turn,
    'subsequent_actions': subsequent_actions
}}
```

**Instructions:**
1. Write Python code that processes the trajectory JSON (available as variable `trajectory_json`)
2. Extract information relevant to answering the question
3. The code should be self-contained and executable
4. Store the final result in a variable named `result`

**Output Format:**
You MUST format your response as follows:

**CODE**:
```python
# Your Python code here
```

Important: The code must be wrapped with **CODE**: marker followed by ```python code block.
<think><\think>
"""

COMPRESS_PROMPT_TEMPLATE = """You are presented with a section of agent trajectory (actions and observations). Read the provided section carefully and update the memory with new information that summarizes the agent's progress. Retain all relevant details from any previous memory while integrating new findings.

Task: {task}

Trajectory Section:
{trajectory_text}

{previous_state_text}

Identify KEY turns — turns where the environment or any object's state meaningfully changed.
For each key turn record the full env_state and object_states snapshots.
Also write a short Memory Summary describing the agent's overall progress in this section.
Output everything after the marker below.

**STATE_MEMORY**

memory_summary: <A concise summary of the agent's progress and key events in this section>
turn_id: <turn number>
env_state:
- <key>: <value>
- ...
object_states:
- name: <object_name>
  state:
  - <key>: <value>
  - ...

turn_id: <turn number>
env_state:
- <key>: <value>
- ...
object_states:
- name: <object_name>
  state:
  - <key>: <value>
  - ...

Constraints:
(1) Do not invent facts not supported by the provided trajectory.
(2) Only record turns where state meaningfully changed — skip no-op turns.
(3) Use consistent object names across all turns.
(4) When previous memory is provided, retain its relevant details and update with new findings.
"""

CHUNK_SUFFICIENCY_JUDGMENT_PROMPT_TEMPLATE = """
You have retrieved the top ranked most relevant turns from an agent trajectory.
Each turn has a UNIQUE TURN INDEX that you can reference.
Query: {query}
Retrieved Turns:
{retrieved_chunks}
Your Task
Carefully analyze the retrieved turns and determine ONE of the following.
1. SUFFICIENT
The retrieved turns contain enough information to answer the query completely.
If you choose this, you MUST provide the answer immediately in the same response.
Format:
SUFFICIENT
ANSWER: <your complete and accurate answer>
2. NEED_GRAPH
The query can likely be answered by looking at adjacent turns or specific ranges.
Use this when you found relevant information but need surrounding context.
You can specify retrieval in multiple ways.
A. Request adjacent turns
NEED_GRAPH: turn_5 before=2 after=1
NEED_GRAPH: turn_8 before=3 after=0, turn_15 before=0 after=2
B. Request turn ranges
NEED_GRAPH: turns 5 to 10
NEED_GRAPH: turns 3 to 8, turns 15 to 20
C. Request individual turns
NEED_GRAPH: turns 3, 7, 12, 18
NEED_GRAPH: turns 5, 8, 15
3. NEED_CODE
The query requires computational analysis, pattern finding, counting, or aggregation
across the full trajectory and cannot be answered from the retrieved turns alone.
Format:
NEED_CODE: <explain what computation or analysis is needed>
Guidelines
Choose SUFFICIENT only if you can answer completely right now.
Choose NEED_GRAPH when you need immediate context around retrieved turns.
Choose NEED_CODE for trajectory wide computation or when turns do not contain the answer.
Response:
"""

TOOL_USE_PROMPT_TEMPLATE = """You are helping retrieve relevant information from a trajectory to answer a question.

**Question:** {query}

**Available Tools:**

You have access to TWO powerful tools to search and retrieve information from the trajectory:

1. **traj_find** - Locates relevant turns
   - Purpose: Search for specific keywords/entities/actions in the trajectory
   - Parameters:
     * query (required): The search term (e.g., "open door", "key", "red box")
     * mode (optional): Search strategy
       - "keyword": Search anywhere in text (default)
       - "action": Search only in action field
       - "entity": Search for specific entity mentions
   - Returns: List of turn indices where the query was found
   - Example: traj_find(query="pick up", mode="action")

2. **traj_get** - Retrieves detailed information
   - Purpose: Get full details from specific turns
   - Parameters:
     * span (required): Which turns to get
       - {{"indices": [1, 2, 3]}} for specific turns
       - {{"start": 1, "end": 5}} for a range
     * fields (optional): What info to include ["action", "observation", "action_space"]
   - Returns: Formatted text with detailed turn information
   - Example: traj_get(span={{"indices": [5, 7, 9]}})

**Recommended Strategy:**
1. Use traj_find to locate turns related to the question
2. Use traj_get to retrieve detailed information from those turns
3. You can call tools multiple times to gather complete information

**Your Task:**
Use these tools strategically to find and retrieve ALL relevant information needed to answer the question thoroughly."""

ANSWER_WITH_RETRIEVAL_PROMPT_TEMPLATE = """Based on the compressed state memory and retrieved detailed information, provide a natural language answer to the query.

Query: {query}

State Memory (compressed):
{state_mem_str}

Retrieved Detailed Information:
{relevant_mem}

CRITICAL: You MUST format your response as follows:
ANSWER: [Your concise, accurate answer here]

Only include the answer after "ANSWER:", nothing else."""

ANSWER_WITHOUT_RETRIEVAL_PROMPT_TEMPLATE = """Based on the compressed state memory, provide a natural language answer to the query.

Query: {query}

State Memory:
{state_mem_str}

CRITICAL: You MUST format your response as follows:
ANSWER: [Your concise, accurate answer here]

Only include the answer after "ANSWER:", nothing else."""

CAUSAL_PROMPT_TEMPLATE = """You are analyzing a trajectory to extract causal relationships between events and state changes.

Task: {task}

Trajectory:
{trajectory_text}

{previous_state_text}

Your task is to identify and extract causal relationships from the trajectory.

For each causal relationship, identify:
1. The CAUSE: an action or event that triggers a change
2. The EFFECT: the resulting state change or consequence
3. The TURN(S): when this causal relationship occurs

Output your response after the markers below.

**CAUSAL_GRAPH**
[
  {{
    "cause": "description of triggering action/event",
    "effect": "description of resulting state change",
    "cause_turn": <turn number>,
    "effect_turn": <turn number>,
    "entities": ["entity1", "entity2"]
  }},
  ...
]

**STATE_MEMORY**
[Your state memory content here]
"""
