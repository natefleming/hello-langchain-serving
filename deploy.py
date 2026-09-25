"""Log the agent, register it to Unity Catalog, and deploy it to Model Serving.

Run against the FEVM workspace:

    DATABRICKS_CONFIG_PROFILE=fevm python deploy.py

Steps:
  1. Point MLflow at the Databricks workspace + UC registry, and select the experiment.
  2. Provision the UC OpenTelemetry trace tables and link them to the experiment
     (idempotent — a re-run just reuses the existing link).
  3. Log ``agent.py`` with the models-from-code pattern.
  4. Register the logged model to Unity Catalog.
  5. Deploy it to a Model Serving endpoint via ``databricks.agents.deploy``.
"""

from __future__ import annotations

import mlflow
from databricks import agents
from mlflow.entities.trace_location import UCSchemaLocation
from mlflow.exceptions import MlflowException
from mlflow.models.resources import DatabricksServingEndpoint
from mlflow.types.responses import RESPONSES_AGENT_INPUT_EXAMPLE

import config

PIP_REQUIREMENTS: list[str] = [
    "mlflow==3.10.1",
    "databricks-langchain==0.19.0",
    "langchain==1.2.15",
    "langgraph==1.1.6",
]


def link_trace_tables(experiment_id: str) -> None:
    """Provision the UC OTel tables and link them to the experiment (idempotent)."""
    location = UCSchemaLocation(
        catalog_name=config.TRACE_CATALOG, schema_name=config.TRACE_SCHEMA
    )
    try:
        result = mlflow.tracing.set_experiment_trace_location(
            location=location,
            experiment_id=experiment_id,
            sql_warehouse_id=config.WAREHOUSE_ID,
        )
        print(f"Linked trace tables: {result.full_otel_spans_table_name}")
    except MlflowException as exc:
        # Re-running deploy.py hits "already linked" — that is the desired state.
        print(f"Trace location already configured ({exc}); continuing.")


def main() -> None:
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    experiment = mlflow.set_experiment(config.EXPERIMENT_PATH)

    link_trace_tables(experiment.experiment_id)

    resources: list[DatabricksServingEndpoint] = [
        DatabricksServingEndpoint(endpoint_name=config.LLM_ENDPOINT)
    ]

    with mlflow.start_run(run_name="log-hello-agent"):
        model_info = mlflow.pyfunc.log_model(
            name="agent",
            python_model="agent.py",
            code_paths=["config.py"],
            resources=resources,
            input_example=RESPONSES_AGENT_INPUT_EXAMPLE,
            pip_requirements=PIP_REQUIREMENTS,
        )
    print(f"Logged model: {model_info.model_uri}")

    registered = mlflow.register_model(model_uri=model_info.model_uri, name=config.UC_MODEL)
    print(f"Registered {config.UC_MODEL} version {registered.version}")

    deployment = agents.deploy(
        config.UC_MODEL,
        int(registered.version),
        scale_to_zero=True,
        endpoint_name=config.SERVING_ENDPOINT,
        tags={"demo": "hello-langchain"},
    )
    print(f"Deploying to endpoint '{deployment.endpoint_name}' — this takes a few minutes.")


if __name__ == "__main__":
    main()
