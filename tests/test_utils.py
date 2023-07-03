from cli import (
    redirect_to_human_assistant,
    HumanAgent,
    AIAgent,
    AIAgentConfig,
)
import pytest


class MockAgent:
    pass


def test_redirect_to_human_assistant_returns_human_agent():
    config = AIAgentConfig.from_yaml("tests/configs/test_config.yaml")
    agent = AIAgent(config)
    human_agent = redirect_to_human_assistant(agent)
    assert isinstance(human_agent, HumanAgent)


def test_redirect_to_human_assistant_returns_same_agent():
    agent = HumanAgent(chat_history=[])
    human_agent = redirect_to_human_assistant(agent)
    assert human_agent is agent


def test_redirect_to_human_assistant_raises_type_error():
    agent = MockAgent()
    with pytest.raises(TypeError):
        redirect_to_human_assistant(agent)
