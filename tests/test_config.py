import pytest

from cli import AIAgentConfig


def test_invalid_temperature():
    # Test with an invalid temperature value (outside the range 0.0 to 1.0)
    invalid_config_data = {
        "knowledge_base_file_path": "data/knowledge_base.txt",
        "temperature": 1.5,
    }
    with pytest.raises(ValueError):
        AIAgentConfig(**invalid_config_data)


def test_invalid_vector_store_dim():
    # Test with an invalid vector_store_dim (less than 1)
    invalid_config_data = {
        "knowledge_base_file_path": "data/knowledge_base.txt",
        "vector_store_dim": 0,
    }
    with pytest.raises(ValueError):
        AIAgentConfig(**invalid_config_data)


def test_invalid_max_react_iterations():
    # Test with an invalid max_react_iterations (less than 1)
    invalid_config_data = {
        "knowledge_base_file_path": "data/knowledge_base.txt",
        "max_react_iterations": -5,
    }
    with pytest.raises(ValueError):
        AIAgentConfig(**invalid_config_data)


def test_invalid_k_memory_interactions():
    # Test with an invalid k_memory_interactions (less than 1)
    invalid_config_data = {
        "knowledge_base_file_path": "data/knowledge_base.txt",
        "k_memory_interactions": -1,
    }
    with pytest.raises(ValueError):
        AIAgentConfig(**invalid_config_data)
