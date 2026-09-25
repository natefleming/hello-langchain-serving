"""Hello-world LangChain agent, packaged with the MLflow "agent-as-code" pattern.

This module is what gets logged (via ``mlflow.models.set_model``) and what runs
inside the Databricks Model Serving endpoint. Two things happen at import time:

The experiment ``config.EXPERIMENT_PATH`` is bound once (in ``deploy.py``) to a
Unity Catalog trace location. Here we simply select that experiment and turn on
``mlflow.langchain.autolog()`` — every trace for the experiment then lands in the
UC OpenTelemetry tables, both locally and on the serving endpoint. No deprecated
``set_destination`` call is needed.

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
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

import config

# --- Tracing: select the UC-bound experiment, then autolog the graph. ---------
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(config.EXPERIMENT_PATH)
mlflow.langchain.autolog()

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
