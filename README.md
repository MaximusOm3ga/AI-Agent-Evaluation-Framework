# AI-Agent-Evaluation-Framework

## Groq judge usage

```python
from agenteval import EvaluationSuite, JudgeConfig, LLMJudge

judge = LLMJudge(
    config=JudgeConfig(
        provider="groq",
        model="llama-3.1-8b-instant",
        # optional: api_key/base_url override
    )
)

suite = EvaluationSuite([judge])
```

Environment variables:

- `GROQ_API_KEY`
- optional `GROQ_BASE_URL` (defaults to `https://api.groq.com/openai/v1`)
- optional `GROQ_MODEL` (defaults to the stable Groq model `llama-3.1-8b-instant`)

Use a model enabled on your Groq account. For example:

```bash
export GROQ_API_KEY="..."
export GROQ_MODEL="llama-3.1-8b-instant"
```
## Simple UI

You can launch a minimal Streamlit dashboard for mirrored tickets:

```powershell
streamlit run ui\dashboard.py
```

Point it at `mirror_inbox.log` and `agenteval.sqlite3` from the UI fields.