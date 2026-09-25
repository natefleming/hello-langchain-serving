"""Validate the deployed agent: live inference + immediate OTel trace read-back.

    DATABRICKS_CONFIG_PROFILE=fevm python validate.py

Fails loudly (non-zero exit) if the endpoint does not answer or if the trace
does not appear in the Unity Catalog OTel table within the polling window.
"""

from __future__ import annotations

import sys
import time
import uuid
from typing import Any

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.deployments import get_deploy_client
from mlflow.entities.trace_location import UnityCatalog

import config

POLL_ATTEMPTS: int = 12
POLL_INTERVAL_S: int = 10


def resolve_spans_table() -> str:
    """Confirm the experiment stores traces in UC and return the real spans table.

    The table name is taken from the backend-populated trace location rather than
    assumed, falling back to the config default only if the backend omits it.
    """
    mlflow.set_tracking_uri("databricks")
    experiment = mlflow.get_experiment_by_name(config.EXPERIMENT_PATH)
    if experiment is None:
        sys.exit(f"FAIL: experiment {config.EXPERIMENT_PATH} does not exist.")
    location = experiment.trace_location
    if not isinstance(location, UnityCatalog):
        sys.exit(f"FAIL: experiment is not bound to UC: {location}")
    spans_table = location.full_otel_spans_table_name or config.OTEL_SPANS_TABLE
    print(f"UC-bound spans table: {spans_table}")
    return spans_table


def live_inference(marker: str) -> str:
    """Send one request to the serving endpoint and return the answer text."""
    client = get_deploy_client("databricks")
    response: dict[str, Any] = client.predict(
        endpoint=config.SERVING_ENDPOINT,
        inputs={"input": [{"role": "user", "content": f"Reply with exactly: {marker}"}]},
    )
    texts: list[str] = [
        content["text"]
        for item in response["output"]
        for content in item["content"]
        if content.get("type") == "output_text"
    ]
    return "".join(texts)


def spans_matching(client: WorkspaceClient, spans_table: str, marker: str) -> int:
    """Count spans whose recorded inputs contain the unique marker for this call.

    ``attributes`` is a VARIANT column, so match against its JSON rendering.
    """
    sql = (
        f"SELECT COUNT(*) FROM {spans_table} "
        f"WHERE to_json(attributes) LIKE '%{marker}%'"
    )
    result = client.statement_execution.execute_statement(
        warehouse_id=config.WAREHOUSE_ID, statement=sql, wait_timeout="30s"
    )
    if result.status.state.value != "SUCCEEDED":
        raise RuntimeError(f"trace query failed: {result.status.error}")
    return int(result.result.data_array[0][0])


def main() -> None:
    spans_table = resolve_spans_table()

    marker = f"OTEL_TEST_{uuid.uuid4().hex[:8].upper()}"
    answer = live_inference(marker)
    print(f"Endpoint answered: {answer!r}")
    if not answer.strip():
        sys.exit("FAIL: endpoint returned an empty answer.")

    client = WorkspaceClient(profile=config.PROFILE)
    for attempt in range(1, POLL_ATTEMPTS + 1):
        time.sleep(POLL_INTERVAL_S)
        count = spans_matching(client, spans_table, marker)
        print(f"[{attempt * POLL_INTERVAL_S:>3}s] spans matching {marker}: {count}")
        if count > 0:
            print(f"PASS: this call's trace is queryable in {spans_table}")
            return

    sys.exit("FAIL: this call's trace did not appear in the OTel table in time.")


if __name__ == "__main__":
    main()
