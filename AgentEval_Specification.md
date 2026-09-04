# AgentEval — AI Agent Evaluation Framework
## Technical Product & Implementation Specification

## 1. Purpose

Build a framework for evaluating full AI agentic systems.

The framework must evaluate an agent at multiple levels:

1. Final task outcome
2. Intermediate agent trajectory
3. LLM decisions
4. Tool selection
5. Tool arguments
6. Tool results and their use
7. State transitions
8. Error handling and recovery
9. Efficiency
10. Cost and latency
11. Safety/reliability where applicable

The framework is intended to work with arbitrary Python-based agents rather than being tightly coupled to one agent framework.

The central abstraction is:

> **An agent run is a trace of structured events, and evaluators operate over that trace.**

The system should distinguish between:
- evaluating whether the final answer is good
- evaluating whether the process was good
- diagnosing where a failure originated

The framework should be usable both as a Python SDK and through a CLI/API.

---

# 2. Core Design Principles

## 2.1 Framework agnostic

Do not assume LangChain, LlamaIndex, OpenAI Agents SDK, CrewAI, AutoGen, etc.

The framework should accept:
- custom Python agents
- agents using external frameworks
- synchronous execution
- asynchronous execution

Framework integrations can be added later through adapters.

## 2.2 Deterministic evaluation first

Use deterministic checks whenever possible.

Examples:
- exact match
- JSON/schema validation
- expected tool comparison
- argument validation
- latency
- token usage
- cost
- retrieval metrics
- repeated tool calls
- iteration limits

Use model-based or LLM-based evaluation only when semantic judgment is actually required.

## 2.3 Do not require a canonical trajectory

There is usually no single correct sequence of tool calls.

The evaluator should generally judge properties of a trajectory rather than requiring:

`actual trajectory == expected trajectory`

A good agent may reach the same result through different valid paths.

## 2.4 Preserve raw evidence

Every evaluation result must be traceable back to:
- the original input
- the relevant trace events
- the evaluator
- the evaluator configuration
- the produced score
- the reasoning/explanation, if applicable

Never overwrite raw traces with derived metrics.

## 2.5 Modular evaluators

Every evaluator should implement a common interface so new evaluators can be added without changing the runner.

---

# 3. High-Level Architecture

```text
                         ┌─────────────────────┐
                         │   Evaluation Set    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Experiment Runner │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    User's Agent     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       Tracer        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Trace Store     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Evaluation Engine   │
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
       Deterministic           Model-based             LLM-based
       Evaluators              Evaluators              Evaluators
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │   Result Aggregator │
                         └──────────┬──────────┘
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                  Experiment Results      Failure Analysis
                         │                     │
                         └──────────┬──────────┘
                                    ▼
                              API / CLI / UI
```

---

# 4. Core Data Model

## 4.1 Dataset

A dataset contains evaluation cases.

Example:

```json
{
  "id": "case_001",
  "input": {
    "task": "Find Apple's 2025 revenue and calculate year-over-year growth."
  },
  "expected": {
    "answer": "..."
  },
  "metadata": {
    "domain": "finance",
    "difficulty": "medium"
  }
}
```

The `expected` object is optional.

A test case may have:
- expected answer
- expected tools
- expected facts
- expected documents
- expected output schema
- constraints
- metadata

Do not require all of them.

## 4.2 Run

A run represents one execution of an agent on one dataset case.

```text
Run
- id
- experiment_id
- dataset_case_id
- input
- output
- status
- started_at
- finished_at
- duration
- token_usage
- estimated_cost
- metadata
```

## 4.3 Event

Every meaningful agent action becomes an event.

Required event types:

```text
AGENT_STARTED
AGENT_FINISHED
AGENT_FAILED

LLM_CALL
LLM_RESPONSE

TOOL_CALL
TOOL_RESULT
TOOL_ERROR

RETRIEVAL
EMBEDDING
RERANK

STATE_CHANGE

FINAL_OUTPUT
ERROR
```

Each event should contain:

```text
id
run_id
parent_id
type
timestamp_start
timestamp_end
input
output
metadata
status
error
```

Additional fields can be event-specific.

## 4.4 Trace

A trace is the ordered collection of events belonging to a run.

The trace must support:
- linear traversal
- parent/child relationships
- nested operations
- reconstruction of execution order
- duration calculation
- filtering by event type

Prefer a tree/DAG-compatible structure rather than assuming all execution is linear.

---

# 5. Tracing Requirements

The tracer is a core component.

It must be possible to instrument an agent with minimal changes.

Desired API:

```python
from agenteval import trace

@trace
def my_agent(task):
    ...
```

For async functions:

```python
@trace
async def my_agent(task):
    ...
```

The framework should also expose manual instrumentation:

```python
run = tracer.start_run(input=task)

with tracer.span("tool_call", name="search"):
    result = search(query)

run.finish(output=result)
```

The tracer must capture, where available:
- model name
- provider
- prompt/input
- output
- tool name
- tool arguments
- tool result
- timestamps
- duration
- token usage
- estimated cost
- errors
- parent-child relationships
- arbitrary metadata

Avoid storing secrets such as API keys.

---

# 6. Evaluator Architecture

Define a base evaluator interface.

Conceptually:

```python
class Evaluator:
    name: str

    def evaluate(self, case, run, trace) -> EvaluationResult:
        ...
```

For async evaluators, support:

```python
async def evaluate(...):
    ...
```

Result:

```python
class EvaluationResult:
    evaluator: str
    score: float | None
    passed: bool | None
    explanation: str | None
    metadata: dict
    evidence: list
```

`evidence` should identify the trace events or output fragments used for the evaluation.

---

# 7. Evaluator Categories

## 7.1 Outcome evaluators

Evaluate whether the task was successfully completed.

Initial evaluators:

### TaskSuccess

Question:

> Did the agent complete the requested task?

Possible implementation:
- deterministic if a structured expected result exists
- LLM judge for semantic tasks

Output:

```json
{
  "score": 0.0,
  "passed": false,
  "explanation": "The agent did not provide the requested comparison."
}
```

### AnswerCorrectness

Evaluate factual/semantic correctness against:
- reference answer
- reference facts
- structured expected output

### AnswerCompleteness

Determine whether all required parts of the task were addressed.

### OutputSchema

Validate structured output against a JSON schema/Pydantic model.

This should be deterministic.

---

# 8. Tool Evaluation

## 8.1 ToolSelection

Evaluate whether the selected tool was appropriate for the current task/state.

Input to evaluator:

```text
Task
Available tools
Current state
Previous relevant events
Selected tool
```

Do not require one exact tool if multiple tools are valid.

Return:
- score
- explanation
- evidence

## 8.2 ToolArgumentCorrectness

Check whether tool arguments were valid and appropriate.

Prefer deterministic validation where possible.

Examples:
- required fields
- types
- enum values
- schema validity
- expected values
- obvious mismatch with task

LLM judgment can supplement deterministic checks for semantic correctness.

## 8.3 ToolResultUsage

Evaluate whether the agent correctly interpreted and used the tool result.

Example:

Tool result:

```text
Revenue = $500M
```

Agent later claims:

```text
Revenue = $700M
```

This should be identified as a tool-result interpretation failure.

## 8.4 ToolErrorHandling

Evaluate:
- whether the agent recognized an error
- whether it retried appropriately
- whether it changed strategy
- whether it eventually recovered
- whether it repeatedly performed the same failed action

---

# 9. Trajectory Evaluation

Trajectory evaluation is one of the primary features.

## 9.1 PlanningQuality

Evaluate whether the sequence of actions represents a sensible approach to the task.

Do not require exact sequence matching.

Consider:
- relevance of actions
- logical progression
- unnecessary detours
- premature termination
- missing necessary steps

## 9.2 TrajectoryEfficiency

Measure:

```text
number of LLM calls
number of tool calls
number of repeated tool calls
number of failed calls
total tokens
latency
cost
```

Also identify:
- duplicate searches
- repeated identical tool calls
- no-progress loops
- unnecessary actions

## 9.3 LoopDetection

Detect:
- identical tool calls repeated
- near-identical tool calls repeated
- repeated LLM states
- cycles in agent state
- excessive iteration count

This evaluator should be deterministic where possible.

## 9.4 RecoveryQuality

Evaluate behavior after failures.

Good recovery:

```text
tool fails
→ recognize failure
→ alter strategy
→ retry differently
→ succeed
```

Bad recovery:

```text
tool fails
→ same call
→ same call
→ same call
→ timeout
```

---

# 10. State Evaluation

Agents may maintain:
- memory
- scratchpad
- task state
- variables
- retrieved information
- intermediate conclusions

The framework should support state-change events.

Evaluators can identify:
- state inconsistency
- forgotten constraints
- contradiction
- stale information
- invalid transitions

This does not require a full formal state machine in the MVP, but the trace model must support it.

---

# 11. Evidence and Grounding

For agents that retrieve information or use external sources, evaluate whether final claims are supported by evidence.

## EvidenceGrounding

Input:

```text
Question
Evidence/tool outputs
Final answer
```

Process:

```text
Final answer
     ↓
Claim extraction
     ↓
Claim → evidence matching
     ↓
Support classification
```

Possible output:

```json
{
  "claims": [
    {
      "claim": "Revenue was $500M",
      "supported": true,
      "evidence_event_id": "evt_17"
    },
    {
      "claim": "Revenue grew 30%",
      "supported": false,
      "evidence_event_id": null
    }
  ],
  "score": 0.5
}
```

This evaluator is especially important for RAG-enabled agents.

---

# 12. Citation Evaluation

If an agent outputs citations, evaluate:

1. Citation exists
2. Citation points to a valid source
3. Source supports the associated claim
4. Citation is attached to the correct claim
5. Source quality where quality metadata is available

Do not treat the mere presence of a URL/citation as evidence of correctness.

---

# 13. RAG + Agent Evaluation

The framework must support systems such as:

```text
Agent
  ↓
Search
  ↓
Retriever
  ↓
Reranker
  ↓
LLM
  ↓
Tool
  ↓
Final Answer
```

RAG-specific metrics should include:

### RetrievalRecall

```text
relevant retrieved documents / total relevant documents
```

### PrecisionAtK

```text
relevant documents in top K / K
```

### MRR

Mean reciprocal rank.

### NDCG

Normalized discounted cumulative gain.

### ContextRelevance

Semantic relevance of retrieved context to the query.

### Faithfulness

Whether generated claims are supported by retrieved context.

### CitationCorrectness

Whether citations support claims.

These should be available independently from agent-level evaluators.

---

# 14. LLM-as-a-Judge

LLM judges are supported but must be treated as one evaluator implementation, not as the entire framework.

A judge configuration should contain:

```text
model
provider
system_prompt
evaluation_prompt
score_scale
criteria
temperature
structured_output_schema
```

Example:

```text
Criteria:
- factual correctness
- relevance
- completeness

Scale:
0 = completely incorrect
1 = mostly incorrect
2 = partially correct
3 = mostly correct
4 = fully correct
```

Prefer structured JSON output.

Example:

```json
{
  "score": 4,
  "reason": "The answer addresses all requested fields and matches the reference facts."
}
```

Store the judge model and configuration with the evaluation result.

---

# 15. Judge Reliability

The framework should eventually support evaluating the evaluator.

Create a human-labeled benchmark:

```text
Sample
Human score
LLM judge score
```

Calculate:
- correlation
- agreement
- classification accuracy
- inter-rater agreement where applicable

The framework should make it possible to compare:

```text
Judge A
Judge B
Human evaluator
```

Do not assume an LLM judge is automatically correct.

---

# 16. Experiment System

An experiment is a collection of runs using:
- one dataset
- one agent/system configuration
- one evaluation suite

Support comparing multiple experiments.

Example:

```python
experiment(
    name="agent-prompt-comparison",
    dataset="research_tasks",
    variants={
        "baseline": agent_v1,
        "new_prompt": agent_v2,
        "new_model": agent_v3
    },
    evaluators=[
        TaskSuccess(),
        ToolSelection(),
        TrajectoryEfficiency(),
        AnswerCorrectness()
    ]
)
```

Results should support side-by-side comparison.

---

# 17. Aggregation

Do not only calculate one overall score.

Return:

```text
Outcome
- Task Success
- Correctness
- Completeness

Process
- Planning
- Tool Selection
- Tool Argument Correctness
- Recovery
- Efficiency

Evidence
- Grounding
- Citation Correctness

System
- Latency
- Token Usage
- Cost
- Error Rate
```

An optional weighted overall score may exist, but component scores must remain visible.

---

# 18. Baselines and Regression Detection

Support baseline experiments.

Example:

```text
Baseline:
TaskSuccess = 91%
Faithfulness = 94%

Current:
TaskSuccess = 94%
Faithfulness = 82%
```

Detect:

```text
REGRESSION:
Faithfulness dropped by 12 percentage points.
```

Allow thresholds:

```yaml
thresholds:
  task_success: 0.90
  faithfulness: 0.90
  tool_selection: 0.85
```

The CLI should exit with a non-zero status when configured thresholds are violated.

This enables CI/CD integration.

---

# 19. Failure Analysis

Failure analysis should be a first-class feature.

For a failed run:

```text
Task failure
    ↓
Trace
    ↓
Identify suspicious events
    ↓
Classify failure
    ↓
Estimate likely root cause
```

Initial failure taxonomy:

```text
RETRIEVAL
- irrelevant_context
- missing_context
- bad_ranking
- duplicate_context

GENERATION
- hallucination
- factual_error
- incomplete_answer
- reasoning_error
- citation_error

AGENT
- wrong_tool
- invalid_tool_arguments
- tool_misinterpretation
- bad_planning
- unnecessary_action
- loop
- state_inconsistency
- poor_recovery

SYSTEM
- timeout
- provider_error
- malformed_output
- token_limit
- tool_error
```

Every failure classification should link back to evidence events.

---

# 20. Root-Cause Analysis

A later-stage feature should attempt to find the earliest likely cause of a downstream failure.

Example:

```text
Final answer incorrect
        ↓
Reasoning used wrong number
        ↓
Tool output was correct
        ↓
Agent misinterpreted tool output
        ↓
ROOT CAUSE:
tool_result_misinterpretation
```

For RAG:

```text
Final answer incorrect
        ↓
Claim unsupported
        ↓
Retrieved context irrelevant
        ↓
Retriever ranked wrong document
        ↓
ROOT CAUSE:
retrieval_failure
```

The system should distinguish:

```text
symptom
```

from:

```text
likely root cause
```

Do not claim causal certainty unless the framework has sufficient evidence.

Use language such as:

```text
likely_root_cause
confidence
supporting_events
```

---

# 21. Counterfactual / Ablation Evaluation

A future advanced feature.

Given a trajectory:

```text
Search A
Search B
Search C
Calculator
Answer
```

Re-run or simulate variants with individual steps removed where possible.

Example:

```text
Original:
score = 0.95

Without Search A:
score = 0.61

Without Search B:
score = 0.94

Without Search C:
score = 0.93
```

This estimates contribution.

The framework should not initially implement this automatically. Design the data model so it can be added later.

---

# 22. CLI

Initial CLI:

```bash
agenteval run dataset.json
agenteval run --dataset dataset.json --agent my_agent
agenteval compare experiment_1 experiment_2
agenteval inspect run_123
agenteval failures experiment_1
agenteval report experiment_1
```

Expected output:

```text
AgentEval
────────────────────────────

Dataset: research_tasks
Cases: 500

Task Success       91.4%
Answer Correctness 89.7%
Tool Selection     92.1%
Recovery           81.3%
Efficiency         76.4%

Latency            4.21s
Tokens             7,821
Cost               $0.082

Failed Runs        43
```

---

# 23. API

Use FastAPI.

Suggested endpoints:

```text
POST   /datasets
GET    /datasets
GET    /datasets/{id}

POST   /experiments
GET    /experiments
GET    /experiments/{id}

POST   /runs
GET    /runs/{id}
GET    /runs/{id}/trace

GET    /runs/{id}/evaluations

POST   /evaluations/run
GET    /experiments/{id}/results

GET    /experiments/{id}/failures
GET    /experiments/{id}/comparison
```

Keep API models separate from database models.

---

# 24. Storage

Use PostgreSQL.

Initial schema should contain at least:

```text
datasets
dataset_cases

experiments
experiment_variants

runs
events

evaluators
evaluation_results

failure_classifications
```

JSONB is appropriate for:
- event input/output
- metadata
- evaluator metadata
- structured judge output

Do not store everything as unstructured JSON. Core relationships and queryable fields should remain relational.

---

# 25. Suggested Project Structure

```text
agenteval/
│
├── pyproject.toml
│
├── src/
│   └── agenteval/
│       ├── core/
│       │   ├── models.py
│       │   ├── trace.py
│       │   ├── runner.py
│       │   ├── evaluator.py
│       │   ├── results.py
│       │   └── exceptions.py
│       │
│       ├── tracing/
│       │   ├── tracer.py
│       │   ├── context.py
│       │   └── decorators.py
│       │
│       ├── evaluators/
│       │   ├── base.py
│       │   ├── outcome/
│       │   │   ├── task_success.py
│       │   │   ├── correctness.py
│       │   │   └── completeness.py
│       │   │
│       │   ├── tools/
│       │   │   ├── selection.py
│       │   │   ├── arguments.py
│       │   │   ├── result_usage.py
│       │   │   └── error_handling.py
│       │   │
│       │   ├── trajectory/
│       │   │   ├── planning.py
│       │   │   ├── efficiency.py
│       │   │   ├── loops.py
│       │   │   └── recovery.py
│       │   │
│       │   ├── rag/
│       │   │   ├── recall.py
│       │   │   ├── precision.py
│       │   │   ├── mrr.py
│       │   │   ├── ndcg.py
│       │   │   ├── relevance.py
│       │   │   └── faithfulness.py
│       │   │
│       │   └── llm_judge/
│       │       ├── judge.py
│       │       ├── prompts.py
│       │       └── schemas.py
│       │
│       ├── experiments/
│       │   ├── experiment.py
│       │   ├── comparison.py
│       │   └── regression.py
│       │
│       ├── analysis/
│       │   ├── failures.py
│       │   ├── root_cause.py
│       │   └── aggregation.py
│       │
│       ├── storage/
│       │   ├── database.py
│       │   ├── repositories.py
│       │   └── migrations/
│       │
│       ├── api/
│       │   ├── app.py
│       │   └── routes/
│       │
│       └── cli/
│           └── main.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
└── examples/
    ├── basic_agent.py
    ├── rag_agent.py
    └── evaluation.py
```

---

# 26. MVP Scope

The first implementation should NOT attempt everything in this document.

Implement in this order.

## Phase 1 — Core

- Dataset model
- Run model
- Event/trace model
- Basic tracer
- Evaluator interface
- Evaluation results
- CLI
- SQLite or PostgreSQL persistence

## Phase 2 — Basic Evaluators

Implement:

1. TaskSuccess
2. ExactMatch
3. AnswerCorrectness
4. ToolSelection
5. ToolArgumentCorrectness
6. ToolErrorHandling
7. Latency
8. TokenUsage
9. Cost
10. LoopDetection

## Phase 3 — LLM Judge

Implement:
- generic LLM judge
- structured output
- configurable criteria
- judge evidence
- judge configuration storage

## Phase 4 — RAG

Implement:
- Recall@K
- Precision@K
- MRR
- NDCG
- ContextRelevance
- Faithfulness
- CitationCorrectness

## Phase 5 — Experiments

Implement:
- experiment runs
- variants
- baseline comparison
- regression thresholds
- report generation

## Phase 6 — Failure Analysis

Implement:
- failure taxonomy
- trace-linked failure evidence
- likely root-cause classification
- confidence

## Phase 7 — API/UI

Only after the core evaluation engine is stable.

---

# 27. Testing Requirements

The framework itself must be tested heavily.

Test:
- nested traces
- async agents
- tool errors
- missing tool results
- malformed LLM outputs
- repeated tool calls
- zero-event runs
- partial runs
- agent timeouts
- evaluator failures
- judge API failures
- database failures
- concurrent runs

An evaluator failure should not silently become an agent failure.

Represent:

```text
agent_status
evaluation_status
```

separately.

---

# 28. Important Edge Cases

The implementation must handle:

### Agent finishes without tools

Valid.

### Agent calls multiple tools in parallel

Trace model must support parent/child or DAG relationships.

### Agent retries a failed tool

Record each attempt separately.

### Tool returns huge output

Store appropriately and avoid blindly duplicating it into every evaluator prompt.

### Agent produces no final answer

TaskSuccess should fail cleanly.

### LLM judge fails

The run itself should not be marked as failed.

### Evaluator returns malformed JSON

Handle and report evaluator failure.

### Multiple valid answers

Do not rely exclusively on exact matching.

### Multiple valid trajectories

Do not require exact trajectory matching.

---

# 29. Security / Privacy

Do not log:
- API keys
- authentication headers
- passwords
- secrets
- sensitive credentials

Allow configurable redaction before persistence.

Provide:

```python
redact(event)
```

or configurable filters.

LLM judge prompts should only contain information necessary for evaluation.

---

# 30. Performance Requirements

Evaluation can become expensive.

Support:
- parallel evaluation of independent evaluators
- batch dataset execution
- configurable concurrency
- caching of evaluator results
- caching of repeated judge calls
- configurable sampling
- evaluator timeouts

The framework should not unnecessarily evaluate every metric for every run.

---

# 31. Example End-to-End Usage

User agent:

```python
def research_agent(task):
    ...
```

Evaluation:

```python
from agenteval import Dataset, EvaluationSuite, evaluate
from agenteval.evaluators import (
    TaskSuccess,
    AnswerCorrectness,
    ToolSelection,
    ToolArgumentCorrectness,
    TrajectoryEfficiency,
    LoopDetection,
)

dataset = Dataset.load("research_tasks.json")

suite = EvaluationSuite([
    TaskSuccess(),
    AnswerCorrectness(),
    ToolSelection(),
    ToolArgumentCorrectness(),
    TrajectoryEfficiency(),
    LoopDetection(),
])

result = evaluate(
    agent=research_agent,
    dataset=dataset,
    suite=suite,
)

print(result.summary())
```

Expected result:

```text
Evaluation Summary
────────────────────────────

Cases: 100

Task Success             91.0%
Answer Correctness       88.0%
Tool Selection           92.0%
Tool Arguments           95.0%
Trajectory Efficiency    76.0%
Loop Detection           97.0%

Average latency           4.3s
Average tokens          6,921
Estimated cost           $0.071/run

Failed cases: 9
```

---

# 32. Example Failure Inspection

```bash
agenteval inspect run_183
```

Output:

```text
Run: 183
Status: FAILED

Task:
Find Apple's 2025 revenue and calculate YoY growth.

Final Answer:
Revenue growth was 32%.

Expected:
Revenue growth was approximately 8%.

Failure:
ANSWER_INCORRECT

Likely Root Cause:
TOOL_RESULT_MISINTERPRETATION

Confidence:
0.87

Relevant Trace:

[07] TOOL_CALL
    database.query(...)

[08] TOOL_RESULT
    2025 revenue = 391.0B
    2024 revenue = 383.3B

[09] LLM_RESPONSE
    "Growth is 32%."

[10] FINAL_OUTPUT
    "Revenue growth was 32%."

Evaluation Evidence:
Step 08 contains the correct values.
Step 09 calculates them incorrectly.
```

---

# 33. Definition of Done

The MVP is complete when all of the following work:

### Core
- [ ] Agent can be instrumented without requiring a specific agent framework
- [ ] Every run produces a structured trace
- [ ] Nested/parent-child events work
- [ ] Async execution works
- [ ] Runs can be persisted and retrieved

### Evaluation
- [ ] Evaluators share a common interface
- [ ] Multiple evaluators can run on one trace
- [ ] Evaluation results preserve evidence
- [ ] Evaluator failures are isolated from agent failures
- [ ] Deterministic evaluators work
- [ ] LLM judges work with structured output

### Agent evaluation
- [ ] Task success
- [ ] Answer correctness
- [ ] Tool selection
- [ ] Tool argument correctness
- [ ] Tool result usage
- [ ] Loop detection
- [ ] Error recovery
- [ ] Efficiency
- [ ] Cost/latency/token metrics

### RAG-enabled agents
- [ ] Retrieval events can be represented
- [ ] Retrieval metrics work where references exist
- [ ] Faithfulness evaluation works
- [ ] Citation evaluation works
- [ ] Evidence can be linked to final claims

### Experiments
- [ ] Dataset can be run against multiple agent versions
- [ ] Metrics can be compared
- [ ] Baselines can be stored
- [ ] Regression thresholds can fail CI

### Debugging
- [ ] Failed runs can be inspected
- [ ] Failure categories can be assigned
- [ ] Failure classifications link to trace evidence
- [ ] Root-cause analysis reports confidence rather than pretending to have certainty

---

# 34. Non-Goals for MVP

Do NOT initially build:
- a full LangSmith clone
- dozens of framework integrations
- a sophisticated React dashboard
- automatic causal inference
- autonomous agent optimization
- automatic prompt optimization
- distributed tracing across every cloud provider
- complex human annotation workflows

Build the **trace + evaluator + experiment engine first**.

---

# 35. Final Product Concept

The final framework should answer four questions for every agent run:

## 1. Did the agent succeed?

```text
Task Success
Answer Correctness
Completeness
```

## 2. How did it succeed or fail?

```text
Trajectory
Planning
Tool Selection
Tool Usage
State
Recovery
```

## 3. What evidence supports that assessment?

```text
Trace Events
Tool Results
Retrieved Documents
Judge Evidence
```

## 4. What should be fixed?

```text
Failure Type
Likely Root Cause
Confidence
Affected Trace Steps
Regression History
```

The core differentiator should be **trace-aware, multi-level evaluation and failure diagnosis**, not simply a collection of LLM-as-a-judge metrics.
