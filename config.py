"""Single source of truth for names used across the demo.

Kept intentionally tiny — this is a hello-world reference, not a framework.
"""

from __future__ import annotations

# Databricks CLI profile that maps to the FEVM workspace.
PROFILE: str = "fevm"

# Foundation-model chat endpoint the agent calls (Claude Sonnet 5 on FEVM).
LLM_ENDPOINT: str = "databricks-claude-sonnet-5"

# Unity Catalog model: <catalog>.<schema>.<name>.
UC_MODEL: str = "retail_consumer_goods.default.hello_langchain_agent"

# Serving endpoint name created by databricks.agents.deploy.
SERVING_ENDPOINT: str = "hello_langchain_agent"

# Unity Catalog schema that backs the OpenTelemetry trace tables. Binding the
# experiment to a UnityCatalog trace location (see deploy.py) provisions four Delta
# tables named <TABLE_PREFIX>_otel_{spans,logs,metrics,annotations} in this schema.
TRACE_CATALOG: str = "retail_consumer_goods"
TRACE_SCHEMA: str = "agent_ops_traces"
TABLE_PREFIX: str = "hello_langchain"
OTEL_SPANS_TABLE: str = f"{TRACE_CATALOG}.{TRACE_SCHEMA}.{TABLE_PREFIX}_otel_spans"

# MLflow experiment that owns the runs/model and is bound to the UC trace location.
# A UC trace location is permanent, so this is a fresh experiment (the earlier demo
# experiment is bound to the older table layout). /Shared keeps it reachable by the
# service principal the serving endpoint runs as.
EXPERIMENT_PATH: str = "/Shared/hello-langchain-serving-uc"

# SQL warehouse used to provision the trace tables and run read-back queries.
WAREHOUSE_ID: str = "d58e5fb998498840"

# Existing service principal (in this secret scope) that the serving endpoint runs
# as. Reused deliberately so deployment does not create a new SP — the FEVM account
# is at its user/SP cap, which is why databricks.agents.deploy's auto-provisioned SPs
# fail. This SP authenticates the LLM call and the trace writes to the OTel tables.
SECRET_SCOPE: str = "retail_consumer_goods"
SP_HOST_KEY: str = "RETAIL_AI_DATABRICKS_HOST"
SP_CLIENT_ID_KEY: str = "RETAIL_AI_DATABRICKS_CLIENT_ID"
SP_CLIENT_SECRET_KEY: str = "RETAIL_AI_DATABRICKS_CLIENT_SECRET"

# System prompt for the no-tools agent.
SYSTEM_PROMPT: str = "You are a helpful assistant. Answer concisely."
