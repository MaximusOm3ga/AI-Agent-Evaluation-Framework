from__future__importannotations

importos
frompathlibimportPath

fromagentevalimport(
Dataset,
DatasetCase,
EvaluationSuite,
EventType,
JudgeConfig,
LLMJudge,
TaskSuccess,
ToolSelection,
evaluate,
)


def_load_env_file(project_root:str|None=None)->Path|None:
    """Load the target project's .env without overriding already-set variables.

    Each project may keep its own environment file, so the evaluation runner should
    explicitly load the app repo's .env before invoking the agent.
    """
candidates:list[Path]=[]

ifproject_root:
        project_root_path=Path(project_root).expanduser().resolve()
candidates.append(project_root_path/".env")
candidates.append(project_root_path)

forenv_pathin(
os.getenv("TARGET_PROJECT_ENV"),
os.getenv("AGENT_PROJECT_ENV"),
):
        ifnotenv_path:
            continue
env_file=Path(env_path).expanduser().resolve()
candidates.append(env_fileifenv_file.name==".env"elseenv_file/".env")

repo_root=Path(__file__).resolve().parent
candidates.extend(
[
repo_root.parent/"data-analyst-agent"/".env",
repo_root/".env",
Path.cwd()/".env",
]
)

seen:set[Path]=set()
forcandidateincandidates:
        normalized=candidate.resolve(strict=False)
ifnormalizedinseen:
            continue
seen.add(normalized)

ifnotnormalized.exists():
            continue

try:
            fromdotenvimportload_dotenv

load_dotenv(normalized,override=False)
returnnormalized
exceptException:
            forlineinnormalized.read_text(encoding="utf-8").splitlines():
                stripped=line.strip()
ifnotstrippedorstripped.startswith("#")or"="notinstripped:
                    continue
key,value=stripped.split("=",1)
key=key.strip()
value=value.strip().strip('"').strip("'")
os.environ.setdefault(key,value)
returnnormalized

returnNone




_load_env_file(os.getenv("TARGET_PROJECT_ENV"))


defmy_agent(task:dict,tracer=None):
    """A small real agent that does a tool call, computes a result, and returns a final answer."""
iftracerisnotNone:
        withtracer.span(EventType.LLM_CALL,name='plan',input=task)asllm_event:
            llm_event.finish(output={'strategy':'lookup revenue then compute growth'},status='succeeded')

withtracer.span(EventType.TOOL_CALL,name='finance_lookup',input={'metric':'revenue'})astool_event:
            tool_event.finish(
output={'2024_revenue':390.0,'2023_revenue':360.0},
status='succeeded',
)

revenue_2024=390.0
revenue_2023=360.0
growth=((revenue_2024-revenue_2023)/revenue_2023)*100.0
answer=f"Revenue grew by approximately {growth:.1f}% from 2023 to 2024."
returnanswer


dataset=Dataset(
id='finance_demo',
name='finance_demo',
cases=[
DatasetCase(
id='case-1',
input={'task':'Find revenue growth from 2023 to 2024.'},
expected={
'answer':'Revenue grew by approximately 8.3% from 2023 to 2024.',
'tools':['finance_lookup'],
},
)
],
)

judge=LLMJudge(
config=JudgeConfig(
provider='groq',
model='openai/gpt-oss-20b',
evaluation_prompt='Judge whether the agent answer matches the task and uses the correct evidence.',
criteria=['correctness','completeness','relevance'],
)
)

suite=EvaluationSuite([
TaskSuccess(),
ToolSelection(),
judge,
])

result=evaluate(
agent=my_agent,
dataset=dataset,
suite=suite,
)

print(result.detailed_summary())
print()
print("Aggregate score summary:")
print(result.summary().summary())
