"""Log the agent, register it to Unity Catalog, and deploy it to Model Serving.

Run against the FEVM workspace:

    DATABRICKS_CONFIG_PROFILE=fevm python deploy.py

Steps:
  1. Point MLflow at the Databricks workspace + UC registry, and select the experiment.
  2. Provision the UC OpenTelemetry trace tables and link them to the experiment
     (idempotent — a re-run just reuses the existing link).
  3. Log ``agent.py`` with the models-from-code pattern.
  4. Register the logged model to Unity Catalog.
  5. Deploy it to a Model Serving endpoint with plain serving, running as an existing
     service principal (see note in config.py) so no new SP is created.
"""

from __future__ import annotations

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from mlflow.entities.trace_location import UCSchemaLocation
from mlflow.exceptions import MlflowException
from mlflow.types.responses import RESPONSES_AGENT_INPUT_EXAMPLE

import config

PIP_REQUIREMENTS: list[str] = [
    "mlflow==3.10.1",
    "databricks-langchain==0.19.0",
    "langchain==1.2.15",
    "langgraph==1.1.6",
]


def _secret(key: str) -> str:
    """Databricks secret reference resolved inside the serving container."""
    return f"{{{{secrets/{config.SECRET_SCOPE}/{key}}}}}"


# Service-principal OAuth creds the endpoint uses to call the LLM and write traces.
SERVING_ENV_VARS: dict[str, str] = {
    "DATABRICKS_HOST": _secret(config.SP_HOST_KEY),
    "DATABRICKS_CLIENT_ID": _secret(config.SP_CLIENT_ID_KEY),
    "DATABRICKS_CLIENT_SECRET": _secret(config.SP_CLIENT_SECRET_KEY),
}


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


def log_and_register() -> str:
    """Log the agent-as-code model and register it to UC. Returns the new version."""
    with mlflow.start_run(run_name="log-hello-agent"):
        model_info = mlflow.pyfunc.log_model(
            name="agent",
            python_model="agent.py",
            code_paths=["config.py"],
            input_example=RESPONSES_AGENT_INPUT_EXAMPLE,
            pip_requirements=PIP_REQUIREMENTS,
        )
    print(f"Logged model: {model_info.model_uri}")
    registered = mlflow.register_model(model_uri=model_info.model_uri, name=config.UC_MODEL)
    print(f"Registered {config.UC_MODEL} version {registered.version}")
    return registered.version


def deploy(version: str) -> None:
    """Create or update the serving endpoint, running as the existing SP."""
    client = WorkspaceClient(profile=config.PROFILE)
    served_entity = ServedEntityInput(
        entity_name=config.UC_MODEL,
        entity_version=version,
        workload_size="Small",
        scale_to_zero_enabled=True,
        environment_vars=SERVING_ENV_VARS,
    )
    core_config = EndpointCoreConfigInput(
        name=config.SERVING_ENDPOINT, served_entities=[served_entity]
    )

    existing = [e.name for e in client.serving_endpoints.list()]
    if config.SERVING_ENDPOINT in existing:
        print(f"Updating existing endpoint '{config.SERVING_ENDPOINT}' to version {version}.")
        client.serving_endpoints.update_config_and_wait(
            name=config.SERVING_ENDPOINT, served_entities=core_config.served_entities
        )
    else:
        print(f"Creating endpoint '{config.SERVING_ENDPOINT}' at version {version}.")
        client.serving_endpoints.create_and_wait(
            name=config.SERVING_ENDPOINT, config=core_config
        )
    print(f"Endpoint '{config.SERVING_ENDPOINT}' is READY.")


def main() -> None:
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    experiment = mlflow.set_experiment(config.EXPERIMENT_PATH)

    link_trace_tables(experiment.experiment_id)
    version = log_and_register()
    deploy(version)


if __name__ == "__main__":
    main()
