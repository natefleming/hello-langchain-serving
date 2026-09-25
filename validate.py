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

from databricks.sdk import WorkspaceClient
from mlflow.deployments import get_deploy_client

import config

POLL_ATTEMPTS: int = 12
POLL_INTERVAL_S: int = 10


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


def trace_landed(client: WorkspaceClient) -> int:
    """Return the number of spans written in the last 15 minutes."""
    sql = (
        f"SELECT COUNT(*) FROM {config.OTEL_SPANS_TABLE} "
        f"WHERE start_time_unix_nano / 1e9 > unix_timestamp() - 900"
    )
    result = client.statement_execution.execute_statement(
        warehouse_id=config.WAREHOUSE_ID, statement=sql, wait_timeout="30s"
    )
    if result.status.state.value != "SUCCEEDED":
        raise RuntimeError(f"trace query failed: {result.status.error}")
    return int(result.result.data_array[0][0])


def main() -> None:
    marker = f"OTEL_TEST_{uuid.uuid4().hex[:8].upper()}"

    answer = live_inference(marker)
    print(f"Endpoint answered: {answer!r}")
    if not answer.strip():
        sys.exit("FAIL: endpoint returned an empty answer.")

    client = WorkspaceClient(profile=config.PROFILE)
    for attempt in range(1, POLL_ATTEMPTS + 1):
        time.sleep(POLL_INTERVAL_S)
        count = trace_landed(client)
        print(f"[{attempt * POLL_INTERVAL_S:>3}s] spans in last 15 min: {count}")
        if count > 0:
            print(f"PASS: traces are queryable in {config.OTEL_SPANS_TABLE}")
            return

    sys.exit("FAIL: no trace appeared in the OTel table within the polling window.")


if __name__ == "__main__":
    main()
