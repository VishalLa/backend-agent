from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from agent.graphs.backend_graph import BackendAgent
from agent.graphs.base_graph import _requests_code_changes
from config import Config
from schema.agent_schema import AgentState


class SequenceModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.inputs = []

    def invoke(self, messages):
        self.inputs.append(messages)
        return self.responses.pop(0)


def _bare_backend_agent(tmp_path, responses):
    agent = object.__new__(BackendAgent)
    agent.config = Config(log_file=str(tmp_path / "agent_events.log"))
    agent.system_prompt = "Test coding agent."
    agent.tools_by_name = {"write_file": object()}
    agent.llm_with_tools = SequenceModel(responses)
    return agent


def test_mixed_text_tool_call_gets_one_retry_and_real_call_is_returned(tmp_path):
    leaked_response = AIMessage(
        content=(
            "I will create the module.\n"
            '```json\n{"name":"write_file","arguments":'
            '{"path":"src/tasks.py","content":"TASKS = []"}}\n```'
        )
    )
    actual_tool_call = AIMessage(
        content="",
        tool_calls=[{
            "name": "write_file",
            "args": {"path": "src/tasks.py", "content": "TASKS = []"},
            "id": "write-1",
        }],
    )
    agent = _bare_backend_agent(tmp_path, [leaked_response, actual_tool_call])

    result = agent.agent_node(AgentState(messages=[HumanMessage(content="Create tasks.py")]))

    assert len(agent.llm_with_tools.inputs) == 2
    assert isinstance(agent.llm_with_tools.inputs[1][-1], HumanMessage)
    assert "Invoke the provided tool directly" in agent.llm_with_tools.inputs[1][-1].content
    assert result["messages"][0].tool_calls[0]["name"] == "write_file"
    assert "status" not in result


def test_failed_tool_call_retry_returns_explicit_error(tmp_path):
    leaked_response = AIMessage(
        content=(
            "I will create the module.\n"
            '```json\n{"name":"write_file","arguments":'
            '{"path":"src/tasks.py","content":"TASKS = []"}}\n```'
        )
    )
    repeated_prose = AIMessage(content="Here is the code you can put in src/tasks.py.")
    agent = _bare_backend_agent(tmp_path, [leaked_response, repeated_prose])

    result = agent.agent_node(AgentState(messages=[HumanMessage(content="Create tasks.py")]))

    assert len(agent.llm_with_tools.inputs) == 2
    assert result["status"] == "error"
    assert "no tool was executed" in result["error"]
    assert "No change was made." in result["messages"][0].content


def test_plain_plan_for_change_request_gets_one_tool_retry(tmp_path):
    plan_response = AIMessage(content="Let's start by creating the requested files.")
    actual_tool_call = AIMessage(
        content="",
        tool_calls=[{
            "name": "write_file",
            "args": {"path": "src/tasks.py", "content": "TASKS = []"},
            "id": "write-2",
        }],
    )
    agent = _bare_backend_agent(tmp_path, [plan_response, actual_tool_call])

    result = agent.agent_node(AgentState(messages=[HumanMessage(content="Build the task API.")]))

    assert len(agent.llm_with_tools.inputs) == 2
    assert "implementation request" in agent.llm_with_tools.inputs[1][-1].content.lower()
    assert result["messages"][0].tool_calls[0]["name"] == "write_file"

def test_empty_response_to_change_request_gets_direct_tool_retry(tmp_path):
    actual_tool_call = AIMessage(
        content="",
        tool_calls=[{
            "name": "write_file",
            "args": {"path": "src/tasks.py", "content": "TASKS = []"},
            "id": "write-3",
        }],
    )
    agent = _bare_backend_agent(tmp_path, [AIMessage(content=""), actual_tool_call])

    result = agent.agent_node(AgentState(messages=[HumanMessage(content="Create the task module.")]))

    assert len(agent.llm_with_tools.inputs) == 2
    assert "invoke the appropriate provided tool" in agent.llm_with_tools.inputs[1][-1].content.lower()
    assert result["messages"][0].tool_calls[0]["name"] == "write_file"


def test_change_request_detection_does_not_force_explanatory_questions():
    assert _requests_code_changes("Build the task API.")
    assert _requests_code_changes("Please add API tests and fix failures.")
    assert not _requests_code_changes("How do I create a FastAPI endpoint?")
    assert not _requests_code_changes("Explain how to fix this bug.")
