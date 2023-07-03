"""Cli script to run Conversational Agent."""

import argparse
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Union

import pinecone
import yaml
from dotenv import load_dotenv
from langchain import SerpAPIWrapper
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.vectorstores import Pinecone
from pydantic import BaseModel, Field
from langchain.schema import (
    AIMessage,
    HumanMessage,
)

logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(levelname)s in %(name)s: %(message)s",
    level=logging.ERROR,
)

load_dotenv()


class RequiredEnvVars(str, Enum):
    """Enum for required env variables."""

    OPENAI_API_KEY = "OPENAI_API_KEY"
    SERPAPI_API_KEY = "SERPAPI_API_KEY"
    PINECONE_API_KEY = "PINECONE_API_KEY"
    PINECONE_ENV = "PINECONE_ENV"
    PINECONE_INDEX = "PINECONE_INDEX"


def print_usage():
    print()
    print("This is Conversational Agent. Hello.\n")
    print("Type a query and press Enter to send a message to the Agent.")
    print()
    print("  /h     - redirect conversation to human assistant.")
    print("  /q     - quit.")
    print()


def load_docs_from_txt(path, chunk_size=500, chunk_overlap=100) -> List[Document]:
    """
    Loads text documents from a txt file and splits them into chunks.

    Parameters:
        path (str): The path to the txt file.
        chunk_size (int, optional): The size of each chunk.
        chunk_overlap (int, optional): The overlap size between consecutive chunks.

    Returns:
        List[Document]: A list of documents.

    Raises:
        FileNotFoundError: If the specified file path is not found.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    loader = TextLoader(path)
    texts = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    return text_splitter.split_documents(texts)


class HumanAgent:
    def __init__(self, chat_history: List[Union[AIMessage, HumanMessage]]):
        """
        Initializes the HumanAgent instance.

        Parameters:
            chat_history (List[Union[AIMessage, HumanMessage]]): Conversation history before redirection to HumanAgent.
        """
        self.chat_history = chat_history

    def __call__(self, query: str) -> str:
        return "No human assistant is available at the moment. Sorry."


@dataclass
class AIAgentConfig:
    """
    Configuration class for Conversational Agent.

    Args:
        knowledge_base_file_path (str): Path to the knowledge base file in txt.
        openai_model_name (str): Name of the OpenAI model to use.
        temperature (float): Temperature parameter for OpenAI model.
        vector_store_dim (int): Dimension of the vector store.
        vector_store_metric (str): Metric to use for vector store.
        max_react_iterations (int): The maximum number of steps to take before ending the execution loop.
        k_memory_interactions (int): The number of interactions to keep in memory.
        knowledge_base_tool_notation (Dict[str, str]): The notation for the knowledge base tool.
        search_tool_notation (Dict[str, str]): The notation for the search tool.
        redirect_tool_notation (Dict[str, str]): The notation for the redirect tool.
        redirect_to_human_msg (str): System message to recognize when parsing agent response during redirection.


    Methods:
        from_yaml(cls, file_path: str) -> AIAgentConfig:
            Creates an instance of AIAgentConfig from a YAML file.

    """

    knowledge_base_file_path: str
    openai_model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.0
    vector_store_dim: int = 1536
    vector_store_metric: str = "cosine"
    max_react_iterations: int = 5
    k_memory_interactions: int = 5
    knowledge_base_tool_notation: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "Knowledge Base",
            "description": "use when need to answer questions about follwoing topics: \
                PacificTech Robotics, Innovative IT company, Aquatic robots.",
        }
    )
    search_tool_notation: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "Search",
            "description": "use when you need to answer questions about current events \
                or the current state of the world.",
        }
    )
    redirect_tool_notation: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "Redirect to human",
            "description": "use when you are asked to redirect to human assistant or you can't answer the question",
        }
    )
    redirect_to_human_msg: str = "{Redirect to human agent}"

    @classmethod
    def from_yaml(cls, file_path: str) -> "AIAgentConfig":
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)

        return cls(**config_data)

    def __post_init__(self):
        if not self.knowledge_base_file_path.endswith(".txt"):
            raise ValueError("knowledge_base_file_path must be a .txt file.")
        if self.temperature < 0.0 or self.temperature > 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0.")
        if self.vector_store_dim < 1:
            raise ValueError("vector_store_dim must be greater than 0.")
        if self.max_react_iterations < 1:
            raise ValueError("max_react_iterations must be greater than 0.")
        if self.k_memory_interactions < 1:
            raise ValueError("k_memory_interactions must be greater than 0.")
        if self.vector_store_metric not in ["cosine", "euclidean"]:
            raise ValueError(
                "vector_store_metric must be either 'cosine' or 'euclidean'."
            )
        if self.openai_model_name not in ["gpt-3.5-turbo", "gpt-4"]:
            raise ValueError(
                "openai_model_name must be either 'gpt-3.5-turbo' or 'gpt-4'."
            )


class RedirectToHumanInput(BaseModel):
    query: str = Field(
        description="Request to connect with human (e.g. customer support) or an empty query"
    )


@tool(
    "Connect to real human agent", return_direct=True, args_schema=RedirectToHumanInput
)
def redirect_to_human(query: str) -> str:
    """use when human asks to connect or transfer to real agent / assistant / consultant."""
    return "{Redirect to human agent}"


class AIAgent:
    def __init__(self, config: AIAgentConfig):
        """
        Initializes the AI agent.

        Parameters:
            config (AIAgentConfig): Configuration object for the AI agent.
        """

        # Initialize vector store and index
        pinecone.init(environment=os.environ[RequiredEnvVars["PINECONE_ENV"].value])
        pinecone_index = os.environ[RequiredEnvVars["PINECONE_INDEX"].value]

        if pinecone_index not in pinecone.list_indexes():
            pinecone.create_index(
                name=pinecone_index,
                metric=config.vector_store_metric,
                dimension=config.vector_store_dim,
            )

        embeddings = OpenAIEmbeddings()
        docs = load_docs_from_txt(config.knowledge_base_file_path)
        docsearch = Pinecone.from_documents(docs, embeddings, index_name=pinecone_index)

        llm = ChatOpenAI(
            model_name=config.openai_model_name, temperature=config.temperature
        )

        # Initialize memory
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=config.k_memory_interactions,
            return_messages=True,
        )

        # Initialize knowledge base tool
        knowledge_base_tool = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
        )

        # Initialize search tool
        search_tool = SerpAPIWrapper()

        # Initialize tools
        tools = [
            Tool(
                name=config.search_tool_notation["name"],
                func=search_tool.run,
                description=config.search_tool_notation["description"],
            ),
            Tool(
                name=config.knowledge_base_tool_notation["name"],
                func=knowledge_base_tool.run,
                description=config.knowledge_base_tool_notation["description"],
            ),
            redirect_to_human,
        ]

        # Initialize agent
        self.conversational_agent = initialize_agent(
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            tools=tools,
            llm=llm,
            max_iterations=config.max_react_iterations,
            early_stopping_method="generate",
            memory=memory,
            verbose=False,
            handle_parsing_errors=True,
        )

    def __call__(self, query: str) -> str:
        return self.conversational_agent(query)["output"]


def redirect_to_human_assistant(agent: Union[HumanAgent, AIAgent]) -> "HumanAgent":
    """
    Redirects the conversation to a human assistant.

    Parameters:
        agent (Agent): The conversational agent.

    Returns:
        HumanAgent: The human assistant.
    """
    if isinstance(agent, HumanAgent):
        print("Already talking to human assistant.\n")
        return agent
    elif isinstance(agent, AIAgent):
        print("Redirecting to human assistant...\n")
        return HumanAgent(agent.conversational_agent.memory.chat_memory.messages)
    else:
        raise TypeError("agent must be either a HumanAgent or AIAgent.")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    # Check all required environment variables are set
    for env_var in set(RequiredEnvVars):
        if env_var not in os.environ:
            raise ValueError(f"Environment variable {env_var} is not set.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", "-c", type=str, required=True)
    args = parser.parse_args()
    config_path = args.config_path

    if not config_path.endswith(".yaml") or not os.path.isfile(config_path):
        raise ValueError(
            f"Config file {config_path} does not exist or is not a YAML file."
        )

    agent_config = AIAgentConfig.from_yaml(config_path)
    active_agent = AIAgent(agent_config)

    print_usage()

    while True:
        try:
            query = input(">> ")
            if query.startswith("/q"):
                break
            elif query.startswith("/h"):
                active_agent = redirect_to_human_assistant(active_agent)
            else:
                try:
                    response = active_agent(query)
                    if agent_config.redirect_to_human_msg.lower() in response.lower():
                        active_agent = redirect_to_human_assistant(active_agent)
                    else:
                        print(response)
                except Exception as e:
                    logging.error(e)
                    print("Sorry, something went wrong.\n")
                    active_agent = redirect_to_human_assistant(active_agent)
        except KeyboardInterrupt:
            print("Exiting the conversation...")
            break
