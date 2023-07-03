from cli import redirect_to_human
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models.fake import FakeListChatModel
from langchain.memory import ChatMessageHistory
import pytest


def test_redirect_to_human_type():
    # Initialize custom tool with empty string as input
    tool = redirect_to_human
    tool_output = tool("")
    assert isinstance(tool_output, str)


def test_redirect_to_human_output_empty():
    # Initialize custom tool with empty string as input
    tool = redirect_to_human
    tool_output = tool("")
    assert tool_output == "{Redirect to human agent}"


def test_redirect_to_human_output_random():
    # Initialize custom tool with random string as input
    tool = redirect_to_human
    tool_output = tool("aaabbbcccddd 1234567890")
    assert tool_output == "{Redirect to human agent}"


def test_tool_with_chat_model_without_history():
    responses = [
        """{\n    "action": "Connect to real human agent",\n    "action_input": "Connect me to a real human agent, please."\n}""",
        "Final Answer: {Redirect to human agent}",
    ]

    fake_agent = FakeListChatModel(responses=responses)
    tools = [redirect_to_human]
    agent = initialize_agent(
        tools=tools,
        llm=fake_agent,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    )

    result = agent.run(input="Connect to the human agent", chat_history=[])

    assert result == "{Redirect to human agent}"


def test_tool_with_chat_model_with_history():
    responses = [
        """{\n    "action": "Connect to real human agent",\n    "action_input": "Connect me to a real human agent, please."\n}""",
        "Final Answer: {Redirect to human agent}",
    ]

    fake_agent = FakeListChatModel(responses=responses)
    tools = [redirect_to_human]
    agent = initialize_agent(
        tools=tools,
        llm=fake_agent,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    )

    history = ChatMessageHistory()
    history.add_user_message("Hi!")
    history.add_ai_message("Hello, how are you?")
    history.add_user_message("I'm fine, thanks. Connect me to the human agent, please.")
    history.add_ai_message("Please, give me a chance to help you.")

    result = agent.run(
        input="Connect to the human agent", chat_history=history.messages
    )

    assert result == "{Redirect to human agent}"


def test_tool_with_chat_model_with_no_action():
    responses = [
        """{\n    "action": "",\n    "action_input": "Connect me to a real human agent, please."\n}""",
    ]

    fake_agent = FakeListChatModel(responses=responses)
    tools = [redirect_to_human]
    agent = initialize_agent(
        tools=tools,
        llm=fake_agent,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    )

    with pytest.raises(IndexError):
        result = agent.run(input="I want to talk with a human", chat_history=[])


def test_tool_with_chat_model_with_invalid_action():
    responses = [
        """{\n    "action": "Random tool",\n    "action_input": "Connect me to a real human agent, please."\n}"""
    ]

    fake_agent = FakeListChatModel(responses=responses)
    tools = [redirect_to_human]
    agent = initialize_agent(
        tools=tools,
        llm=fake_agent,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    )

    with pytest.raises(IndexError):
        result = agent.run(input="I want to talk with a human", chat_history=[])


def test_tool_with_chat_model_with_invalid_answer():
    responses = [
        """{\n    "action": "Connect to real human agent",\n    "action_input": "Connect me to a real human agent, please."\n}""",
        "Final Answer: Please, give me a chance to help you.",
    ]

    fake_agent = FakeListChatModel(responses=responses)
    tools = [redirect_to_human]
    agent = initialize_agent(
        tools=tools,
        llm=fake_agent,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    )

    result = agent.run(input="I want to talk with a human", chat_history=[])
    assert result == "{Redirect to human agent}"
