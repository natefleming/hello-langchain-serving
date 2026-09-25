"""Hello-world LangChain agent, packaged with the MLflow "agent-as-code" pattern.

This module is what gets logged (via ``mlflow.models.set_model``) and what runs
inside the Databricks Model Serving endpoint. Two things happen at import time:

1. ``mlflow.langchain.autolog()`` turns on automatic MLflow tracing for the
   LangChain / LangGraph call graph.
2. ``mlflow.tracing.set_destination(UCSchemaLocation(...))`` routes the spans to
   the Unity Catalog OpenTelemetry tables, both locally and on the endpoint, so
   they are queryable in SQL within seconds of each inference. ``deploy.py``
   provisions those tables and links the experiment to them once, up front.

The agent itself is a no-tools ``create_agent`` graph wrapped in a thin
``ResponsesAgent`` so it presents the interface Databricks serving expects.
"""

from __future__ import annotations

import uuid
from typing import Any

import mlflow
from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage
from mlflow.entities.trace_location import UCSchemaLocation
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

import config

# --- Tracing: autolog captures the graph; send spans to the UC OTel tables. ---
mlflow.langchain.autolog()
mlflow.tracing.set_destination(
    UCSchemaLocation(catalog_name=config.TRACE_CATALOG, schema_name=config.TRACE_SCHEMA)
)

# --- The agent: a Databricks foundation model, no tools. ----------------------
_LLM = ChatDatabricks(endpoint=config.LLM_ENDPOINT)
_GRAPH = create_agent(model=_LLM, tools=None, system_prompt=config.SYSTEM_PROMPT)


def _message_text(message: BaseMessage) -> str:
    """Flatten a LangChain message's content to plain text.

    Chat models may return either a plain string or a list of content blocks
    (e.g. ``[{"type": "text", "text": "..."}]``); handle both explicitly.
    """
    content: str | list[Any] = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = [
        block["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(parts)


class HelloAgent(ResponsesAgent):
    """Minimal ResponsesAgent adapter over the LangGraph agent."""

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        messages: list[dict[str, Any]] = self.prep_msgs_for_cc_llm(request.input)
        result: dict[str, Any] = _GRAPH.invoke({"messages": messages})
        final: BaseMessage = result["messages"][-1]
        output_item = self.create_text_output_item(
            text=_message_text(final), id=str(uuid.uuid4())
        )
        return ResponsesAgentResponse(output=[output_item])


mlflow.models.set_model(HelloAgent())
