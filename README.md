# hello-langchain-serving

A minimal, end-to-end demo: a **no-tools LangChain agent** deployed to **Databricks
Model Serving**, with **MLflow tracing written to Unity Catalog OpenTelemetry tables**
so every inference is queryable in SQL within seconds.

Built for learning — small on purpose. It uses only current MLflow 3.16 / LangChain 1.4
APIs (no deprecated calls).

## What it demonstrates

- `langchain.agents.create_agent(...)` with **no tools** (just a chat model + system prompt).
- The MLflow **agent-as-code** pattern (`mlflow.models.set_model`), served through a thin
  `ResponsesAgent` wrapper.
- **Automatic tracing** via `mlflow.langchain.autolog()`.
- Traces stored in **UC OTel tables** by binding the experiment to a `UnityCatalog` trace
  location — `mlflow.set_experiment(trace_location=UnityCatalog(...))` (the current
  Databricks-recommended approach; the older `set_destination(UCSchemaLocation)` is
  deprecated). Binding provisions four Delta tables:
  `hello_langchain_otel_{spans,logs,metrics,annotations}`.

## Layout

| File | Purpose |
|------|---------|
| `config.py` | All names in one place (workspace profile, model, endpoint, catalog/schema). |
| `agent.py` | The agent-as-code entrypoint that runs on the endpoint. |
| `deploy.py` | Log → register to UC → deploy to Model Serving (also provisions the trace tables). |
| `validate.py` | Live inference against the endpoint + immediate OTel trace read-back. |

## Configuration

Everything is set in `config.py`. Defaults target the FEVM workspace:

- Profile `fevm`, chat endpoint `databricks-claude-sonnet-5`
- Model `retail_consumer_goods.default.hello_langchain_agent`
- OTel tables in `retail_consumer_goods.agent_ops_traces`

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Log, register, and deploy (a few minutes to become READY)
DATABRICKS_CONFIG_PROFILE=fevm python deploy.py

# 2. Validate: UC-binding check + live inference + confirm the trace lands in the OTel table
DATABRICKS_CONFIG_PROFILE=fevm python validate.py
```

## Deployment note

`deploy.py` uses **plain model serving** (`serving_endpoints.create_and_wait`) rather than
`databricks.agents.deploy`. On the FEVM demo account, `agents.deploy` fails because it
auto-provisions several service principals (review app, feedback model, on-behalf-of auth)
and the account is at its user/SP cap (`RESOURCE_EXHAUSTED`). Instead the endpoint runs as an
**existing** service principal, whose OAuth credentials are injected from the
`retail_consumer_goods` secret scope via `environment_vars` — so no new SP is created. That
SP authenticates both the LLM call and the trace writes to the OTel tables.

## Query the traces

`attributes` is a VARIANT column, so render it with `to_json(...)` (or use `attributes:field`):

```sql
SELECT trace_id, name, status.code AS status,
       to_json(attributes) AS attributes, time
FROM retail_consumer_goods.agent_ops_traces.hello_langchain_otel_spans
ORDER BY time DESC
LIMIT 20;
```
