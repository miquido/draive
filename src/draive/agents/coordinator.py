from typing import Protocol
from uuid import UUID

from draive.agents.agent import Agent
from draive.agents.errors import AgentException
from draive.agents.state import AgentsChat, AgentsData
from draive.generation import generate_model
from draive.parameters import Field, ParametrizedData
from draive.scope import ctx
from draive.tools import Toolbox
from draive.types import Model, MultimodalContent, State

__all__ = [
    "AgentDisposition",
    "AgentsCoordinator",
    "basic_agent_coordinator",
]


class AgentDisposition[Data: ParametrizedData](State):
    recipient: Agent[Data]
    message: MultimodalContent


class AgentsCoordinator[Data: ParametrizedData](Protocol):
    async def __call__(
        self,
        chat: AgentsChat,
        data: AgentsData[Data],
        agents: frozenset[Agent[Data]],
    ) -> AgentDisposition[Data]: ...


class Disposition(Model):
    recipient: UUID = Field(
        description="Exact ID of the employee which should take over the task",
    )
    message: str = Field(
        description="Observation about current progress and"
        " proposal of task for the chosen employee",
    )


INSTRUCTION: str = """\
You are a Coordinator managing work of a group of employees trying to achieve a common goal.
Your task is to verify current progress and propose a next step in order \
to complete the goal using available employees. Examine the conversation and current progress \
to prepare the most suitable next step. You have to choose ID associated with given employee \
to ask that particular employee for continuation but you are fully responsible for the final result. \
The instructions you propose should be step by step, small and easily achievable tasks to avoid \
misunderstanding and to allow continuous tracking of the progress until it is fully done.

Available employees:
---
{agents}
---

Provide only a single disposition without any additional comments or elements.
"""  # noqa: E501


async def basic_agent_coordinator[Data: ParametrizedData](
    chat: AgentsChat,
    data: AgentsData[Data],
    agents: frozenset[Agent[Data]],
) -> AgentDisposition[Data]:
    disposition: Disposition = await generate_model(
        Disposition,
        instruction=INSTRUCTION.format(
            agents="\n---\n".join(agent.description for agent in agents)
        ),
        input=(
            f"CONVERSATION:\n{chat.as_str()}",
            "PROGRESS:\n---\n"
            + "\n---\n".join(
                f"{key}:\n{value}" for key, value in (await data.current_contents).items()
            ),
        ),
        tools=Toolbox(data.read_tool()),
        # TODO: add examples
    )
    ctx.log_debug("Agent %s disposition: %s", disposition.recipient, disposition.message)
    selected_agent: Agent[Data]
    for agent in agents:
        if agent.identifier == disposition.recipient:
            selected_agent = agent
            break
        else:
            continue
    else:
        raise AgentException("Selected invalid agent")

    return AgentDisposition(
        recipient=selected_agent,
        message=disposition.message,
    )
