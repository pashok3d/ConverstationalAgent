from cli import AIAgent, AIAgentConfig
import pytest


@pytest.fixture(scope="module")
def agent():
    config = AIAgentConfig.from_yaml("tests/configs/test_config.yaml")
    agent = AIAgent(config)
    yield agent


@pytest.mark.parametrize(
    "query, proper_tool",
    [
        ["Hello, how are you?", None],
        ["Hi, what do you know about aqua robots?", "Knowledge Base"],
        ["What do know about PacificTech Robotics?", "Knowledge Base"],
        ["Where is the headquarter of PacificTech Robotics located?", "Knowledge Base"],
        ["What is the current population of Tokyo, Japan?", "Search"],
        ["What is the current weather forecast in Warsaw?", "Search"],
        ["What movies are currently playing in theaters?", "Search"],
        ["now please connect me with a real agent", "Connect to real human agent"],
        ["I need to talk to a human", "Connect to real human agent"],
        ["redirect to a real agent", "Connect to real human agent"],
    ],
)
def test_agent_react(agent, query, proper_tool):
    agent.reset()
    output = agent(query)
    if agent.intermediate_steps:
        used_tool = agent.intermediate_steps[-1][0].tool
    else:
        used_tool = None
    assert used_tool == proper_tool


@pytest.mark.parametrize(
    "first_query, second_query, first_proper_tool, second_proper_tool",
    [
        [
            "Hello, how are you?",
            "Hi, what do you know about aqua robots?",
            None,
            "Knowledge Base",
        ],
        [
            "Hi, what do you know about aqua robots?",
            "What do know about PacificTech Robotics?",
            "Knowledge Base",
            "Knowledge Base",
        ],
        [
            "What is the weather forecast in Warsaw?",
            "What is the current population of Tokyo, Japan?",
            "Search",
            "Search",
        ],
        [
            "Where is the headquarters of PacificTech Robotics?",
            "Where is the headquarters of United Nations Organization?",
            "Knowledge Base",
            "Search",
        ],
        [
            "What movies are currently playing in theaters?",
            "What is the current weather forecast in Warsaw?",
            "Search",
            "Search",
        ],
        [
            "Where is the headquarters of PacificTech Robotics?",
            "What is the weather like there today?",
            "Knowledge Base",
            "Search",
        ],
    ],
)
def test_agent_react_multi_turn(
    agent, first_query, second_query, first_proper_tool, second_proper_tool
):
    agent.reset()
    output = agent(first_query)
    if agent.intermediate_steps:
        first_used_tool = agent.intermediate_steps[-1][0].tool
    else:
        first_used_tool = None

    output = agent(second_query)
    if agent.intermediate_steps:
        second_used_tool = agent.intermediate_steps[-1][0].tool
    else:
        second_used_tool = None

    assert first_used_tool == first_proper_tool
    assert second_used_tool == second_proper_tool
