# hello-langchain-serving

A minimal, end-to-end demo: a **no-tools LangChain agent** deployed to **Databricks
Model Serving**, with **MLflow tracing written to Unity Catalog OpenTelemetry tables**
so every inference is queryable in SQL within seconds.

Built for learning — small on purpose. It uses only current MLflow 3.10 / LangChain 1.x
APIs (no deprecated calls).

## What it demonstrates

- `langchain.agents.create_agent(...)` with **no tools** (just a chat model + system prompt).
- The MLflow **agent-as-code** pattern (`mlflow.models.set_model`), served through a thin
  `ResponsesAgent` wrapper.
- **Automatic tracing** via `mlflow.langchain.autolog()`.
- Traces routed to **UC OTel tables** with `mlflow.tracing.set_destination(UCSchemaLocation(...))`,
  provisioned/linked once with `mlflow.tracing.set_experiment_trace_location(...)`.

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
pip install -r requirements.txt

# 1. Log, register, and deploy (a few minutes to become READY)
DATABRICKS_CONFIG_PROFILE=fevm python deploy.py

# 2. Validate: live inference + confirm the trace lands in the OTel table
DATABRICKS_CONFIG_PROFILE=fevm python validate.py
```

## Query the traces

```sql
SELECT trace_id, name, status.code AS status, attributes,
       CAST(start_time_unix_nano / 1e9 AS TIMESTAMP) AS start_time
FROM retail_consumer_goods.agent_ops_traces.mlflow_experiment_trace_otel_spans
ORDER BY start_time DESC
LIMIT 20;
```
