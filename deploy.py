"""Log the agent, register it to Unity Catalog, and deploy it to Model Serving.

Run against the FEVM workspace:

    DATABRICKS_CONFIG_PROFILE=fevm python deploy.py

Steps:
  1. Point MLflow at the Databricks workspace + UC registry.
  2. Create (or reuse) the experiment bound to a Unity Catalog trace location — this
     provisions the OTel Delta tables. A UC trace location is permanent, so the
     experiment is verified rather than reassigned.
  3. Log ``agent.py`` with the models-from-code pattern.
  4. Register the logged model to Unity Catalog.
  5. Deploy it to a Model Serving endpoint with plain serving, running as an existing
     service principal (see note in config.py) so no new SP is created.
"""

from __future__ import annotations

import os

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from mlflow.entities.trace_location import UnityCatalog
from mlflow.types.responses import RESPONSES_AGENT_INPUT_EXAMPLE

import config

PIP_REQUIREMENTS: list[str] = [
    "mlflow==3.16.1",
    "databricks-langchain==0.20.0",
    "langchain==1.4.2",
    "langgraph==1.2.12",
]

# Required so MLflow can provision/query the UC trace tables.
os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = config.WAREHOUSE_ID


def _secret(key: str) -> str:
    """Databricks secret reference resolved inside the serving container."""
    return f"{{{{secrets/{config.SECRET_SCOPE}/{key}}}}}"


# Service-principal OAuth creds the endpoint uses to call the LLM and write traces.
SERVING_ENV_VARS: dict[str, str] = {
    "DATABRICKS_HOST": _secret(config.SP_HOST_KEY),
    "DATABRICKS_CLIENT_ID": _secret(config.SP_CLIENT_ID_KEY),
    "DATABRICKS_CLIENT_SECRET": _secret(config.SP_CLIENT_SECRET_KEY),
}


def ensure_uc_experiment() -> None:
    """Create or reuse the experiment bound to the UC trace location, then select it.

    A UC trace location is permanent, so an existing experiment is verified to match
    rather than reassigned (a mismatch is a hard error, per MLflow guidance).
    """
    location = UnityCatalog(
        catalog_name=config.TRACE_CATALOG,
        schema_name=config.TRACE_SCHEMA,
        table_prefix=config.TABLE_PREFIX,
    )
    experiment = mlflow.get_experiment_by_name(config.EXPERIMENT_PATH)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            config.EXPERIMENT_PATH, trace_location=location
        )
        experiment = mlflow.get_experiment(experiment_id)
        print(f"Created experiment bound to {experiment.trace_location.full_otel_spans_table_name}")
    else:
        bound = experiment.trace_location
        if not isinstance(bound, UnityCatalog) or (
            bound.catalog_name,
            bound.schema_name,
            bound.table_prefix,
        ) != (config.TRACE_CATALOG, config.TRACE_SCHEMA, config.TABLE_PREFIX):
            raise RuntimeError(
                f"Experiment {config.EXPERIMENT_PATH} is bound to {bound}, not the "
                f"expected UC location {config.TRACE_CATALOG}.{config.TRACE_SCHEMA} "
                f"with prefix {config.TABLE_PREFIX!r}."
            )
        print(f"Reusing experiment bound to {bound.full_otel_spans_table_name}")

    mlflow.set_experiment(experiment_id=experiment.experiment_id)


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

    ensure_uc_experiment()
    version = log_and_register()
    deploy(version)


if __name__ == "__main__":
    main()
