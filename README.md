# AI-Agent-Evaluation-Framework

## Groq judge usage

```python
from agenteval import EvaluationSuite, JudgeConfig, LLMJudge

judge = LLMJudge(
    config=JudgeConfig(
        provider="groq",
        model="llama-3.3-70b-versatile",
        # optional: api_key/base_url override
    )
)

suite = EvaluationSuite([judge])
```

Environment variables:

- `GROQ_API_KEY`
- optional `GROQ_BASE_URL` (defaults to `https://api.groq.com/openai/v1`)