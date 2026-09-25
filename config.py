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

# Unity Catalog schema that backs the OpenTelemetry trace tables. MLflow writes
# <TRACE_CATALOG>.<TRACE_SCHEMA>.mlflow_experiment_trace_otel_spans (and _otel_logs).
TRACE_CATALOG: str = "retail_consumer_goods"
TRACE_SCHEMA: str = "agent_ops_traces"
OTEL_SPANS_TABLE: str = f"{TRACE_CATALOG}.{TRACE_SCHEMA}.mlflow_experiment_trace_otel_spans"

# MLflow experiment that owns the runs/model logged by deploy.py.
EXPERIMENT_PATH: str = "/Users/nate.fleming@databricks.com/hello-langchain-serving"

# SQL warehouse used to provision the trace tables and run read-back queries.
WAREHOUSE_ID: str = "d58e5fb998498840"

# System prompt for the no-tools agent.
SYSTEM_PROMPT: str = "You are a helpful assistant. Answer concisely."
