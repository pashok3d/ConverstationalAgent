from cli import redirect_to_human


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
