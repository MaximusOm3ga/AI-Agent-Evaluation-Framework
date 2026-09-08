from__future__importannotations

importjson
importos
fromdatetimeimportdatetime,timezone
frompathlibimportPath

importrequests

fromagentevalimportDataset,DatasetCase,EvaluationSuite,evaluate
fromagenteval.core.evaluatorimportEvaluator
fromagenteval.core.modelsimportDatasetCaseasCoreDatasetCase,Run,Trace
fromagenteval.core.resultsimportEvaluationResult


def_load_project_env(project_root:str|None=None)->None:
    """Load the target app's .env without overriding already-set variables."""
env_candidates:list[Path]=[]
ifproject_root:
        env_candidates.append(Path(project_root).expanduser().resolve()/".env")
env_candidates.extend(
[
Path(os.getenv("TARGET_PROJECT_ENV","")).expanduser().resolve()/".env",
Path.cwd()/".env",
Path(__file__).resolve().parent.parent/"data-analyst-agent"/".env",
]
)
seen:set[Path]=set()
forcandidateinenv_candidates:
        try:
            resolved=candidate.resolve(strict=False)
exceptException:
            continue
ifresolvedinseenornotresolved.exists():
            continue
seen.add(resolved)
try:
            fromdotenvimportload_dotenv

load_dotenv(resolved,override=False)
return
exceptException:
            forlineinresolved.read_text(encoding="utf-8").splitlines():
                stripped=line.strip()
ifnotstrippedorstripped.startswith("#")or"="notinstripped:
                    continue
key,value=stripped.split("=",1)
os.environ.setdefault(key.strip(),value.strip().strip('"').strip("'"))
return


classDataAnalystAgentEvaluator(Evaluator):
    name="DataAnalystAgentEvaluator"

defevaluate(self,case:CoreDatasetCase,run:Run,trace:Trace)->EvaluationResult:
        payload=run.outputifisinstance(run.output,dict)else{}
checks=[]
expected=case.expectedor{}

ifexpected.get("status")isnotNone:
            checks.append(("status",payload.get("status")==expected["status"]))
ifexpected.get("action")isnotNone:
            checks.append(("action",payload.get("action")==expected["action"]))
ifexpected.get("category")isnotNone:
            checks.append(("category",payload.get("decision",{}).get("category")==expected["category"]))
ifexpected.get("recommended_action")isnotNone:
            checks.append(
("recommended_action",payload.get("decision",{}).get("recommended_action")==expected["recommended_action"])
)
ifexpected.get("guardrail_triggered")isnotNone:
            checks.append(("guardrail_triggered",payload.get("guardrail_triggered")==expected["guardrail_triggered"]))
ifexpected.get("priority")isnotNone:
            checks.append(("priority",payload.get("decision",{}).get("priority")==expected["priority"]))
ifexpected.get("queue")isnotNone:
            checks.append(("queue",payload.get("decision",{}).get("queue")==expected["queue"]))

passed=sum(1for_,okinchecksifok)
total=len(checks)
score=(passed/total)iftotalelse1.0

returnEvaluationResult(
evaluator=self.name,
score=score,
passed=score==1.0,
explanation=(
f"Checked {passed}/{total} expected properties against the real data-analyst-agent response."
iftotal
else"No checks were configured for this case."
),
metadata={
"case_id":case.id,
"expected":expected,
"actual":payload,
"checks":{name:okforname,okinchecks},
},
evidence=[event.idforeventintrace.ordered_events()[-10:]],
)


deftriage_agent(case:dict,tracer=None):
    """Call the real data-analyst-agent FastAPI service."""
url=os.getenv("DATA_ANALYST_AGENT_URL","http://127.0.0.1:8000/ingest/web_form")
payload={
"ticket_id_source":case["ticket_id_source"],
"source_channel":case.get("source_channel","web_form"),
"requester_identifier":case.get("requester_identifier","user@example.com"),
"subject":case.get("subject"),
"body_raw":case["body_raw"],
"body_cleaned":case.get("body_cleaned"),
"attachments":case.get("attachments",[]),
"timestamp_received":datetime.now(timezone.utc).isoformat(),
"channel_metadata":case.get("channel_metadata",{}),
}
response=requests.post(url,json=payload,timeout=60)
response.raise_for_status()
body=response.json()
iftracerisnotNone:
        withtracer.span("HTTP_CALL",name="triage_agent_post",input=payload)asevent:
            event.finish(output=body,status="succeeded")
returnbody


defbuild_dataset()->Dataset:
    returnDataset(
id="data-analyst-agent-eval",
name="data-analyst-agent-eval",
cases=[
DatasetCase(
id="password-reset",
input={
"ticket_id_source":"eval-web-1",
"source_channel":"web_form",
"requester_identifier":"user@example.com",
"subject":"Need password reset",
"body_raw":"I forgot my password and need a reset",
},
expected={
"status":"completed",
"action":"auto_resolve",
"category":"Access Request",
"recommended_action":"auto_resolve",
},
),
DatasetCase(
id="phishing-security",
input={
"ticket_id_source":"eval-web-2",
"source_channel":"web_form",
"requester_identifier":"user2@example.com",
"subject":"Urgent suspicious email",
"body_raw":"I think this is phishing and maybe a data breach.",
},
expected={
"status":"awaiting_approval",
"action":"force_security_route",
"category":"Security Incident",
"recommended_action":"auto_route",
"guardrail_triggered":True,
"priority":"P1-Critical",
"queue":"Security",
},
),
DatasetCase(
id="unknown-device-issue",
input={
"ticket_id_source":"eval-web-3",
"source_channel":"web_form",
"requester_identifier":"user3@example.com",
"subject":"My screen flickers",
"body_raw":"Sometimes the display flickers when I open the browser",
},
expected={
"status":"awaiting_approval",
"action":"auto_route_spotcheck",
"recommended_action":"auto_route",
"guardrail_triggered":False,
},
),
],
)


defmain()->int:
    _load_project_env(os.getenv("TARGET_PROJECT_ENV"))
result=evaluate(
agent=triage_agent,
dataset=build_dataset(),
suite=EvaluationSuite([DataAnalystAgentEvaluator()]),
)
print(result.detailed_summary())
return0


if__name__=="__main__":
    raiseSystemExit(main())
