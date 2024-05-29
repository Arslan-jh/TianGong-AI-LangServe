from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.memory import XataChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from src.config.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    XATA_API_KEY,
    XATA_MEMORY_DB_URL,
    XATA_MEMORY_TABLE_NAME,
)

def init_chat_history(session_id: str) -> BaseChatMessageHistory:
    return XataChatMessageHistory(
        session_id=session_id,
        api_key=XATA_API_KEY,
        db_url=XATA_MEMORY_DB_URL,
        table_name=XATA_MEMORY_TABLE_NAME,
    )

def openai_agent_without_tools():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        temperature=0,
        model=OPENAI_MODEL,
    )

    agent = (
        {
            "input": lambda x: x["input"],
            "history": lambda x: x["history"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm
        | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(
        agent=agent, tools=[], verbose=True, handle_parsing_errors=True
    )

    agent_executor_with_history = RunnableWithMessageHistory(
        runnable=agent_executor,
        get_session_history=init_chat_history,
        history_messages_key="history",
    )

    return agent_executor_with_history
