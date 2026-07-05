"""Tests for app.agent.ProductionAgent. ChatOpenAI is mocked - no network calls."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent import ProductionAgent


@pytest.fixture
def agent(monkeypatch):
    fake_model = MagicMock()
    monkeypatch.setattr("app.agent.ChatOpenAI", lambda **kwargs: fake_model)
    instance = ProductionAgent()
    return instance, fake_model


def _initial_state(message: str) -> dict:
    return {
        "messages": [HumanMessage(content=message)],
        "err": None,
        "retry_count": 0,
        "model_used": "",
    }


def test_agent_initializes_and_compiles_graph(agent):
    instance, _ = agent
    assert instance.graph is not None
    assert instance.max_retries == instance.settings.max_retries


def test_graph_returns_ai_message_on_success(agent):
    instance, fake_model = agent
    fake_model.invoke.return_value = AIMessage(content="I'm here to listen.")

    final_state = instance.graph.invoke(_initial_state("I'm feeling anxious today."))

    assert final_state["err"] is None
    assert final_state["messages"][-1].content == "I'm here to listen."
    assert final_state["model_used"] == instance.settings.primary_model


def test_graph_routes_to_error_handler_after_max_retries(agent):
    instance, fake_model = agent
    fake_model.invoke.side_effect = RuntimeError("upstream API down")

    final_state = instance.graph.invoke(_initial_state("hello"))

    assert final_state["model_used"] == "error_handler"
    assert final_state["messages"][-1].content == "I am Sorry!"


def test_invoke_wrapper_returns_response_text(agent):
    instance, fake_model = agent
    fake_model.invoke.return_value = AIMessage(content="Tell me more about that.")

    result = instance.invoke("I had a rough day at work.")

    assert result == "Tell me more about that."
