from typing import Protocol, Self, final, runtime_checkable
from uuid import UUID, uuid4

from draive.agents.errors import AgentException
from draive.parameters import ParametrizedData
from draive.types import MultimodalContent, MultimodalContentConvertible, frozenlist
from draive.utils import MISSING, Missing, freeze, is_missing, not_missing

__all__ = [
    "Agent",
    "AgentError",
    "AgentInitializer",
    "AgentMessage",
    "AgentNode",
    "AgentOutput",
]


class AgentNode:
    def __init__(
        self,
        identifier: UUID | None = None,
        /,
        *,
        name: str,
        description: str,
    ) -> None:
        self._identifier: UUID = identifier or uuid4()
        self._name: str = name
        self._description: str = description
        self._initializer: AgentInitializer | Missing = MISSING
        self._concurrent: bool = False

    def __hash__(self) -> int:
        return hash(self._identifier)

    def __str__(self) -> str:
        return f"{self.name}|{self._identifier}"

    def __eq__(
        self,
        other: object,
    ) -> bool:
        if isinstance(other, AgentNode):
            return self._identifier == other._identifier

        else:
            return False

    @property
    def identifier(self) -> UUID:
        return self._identifier

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def concurrent(self) -> bool:
        return self._concurrent

    def address(
        self,
        content: MultimodalContent | MultimodalContentConvertible,
        /,
        *_content: MultimodalContent | MultimodalContentConvertible,
        addressee: "AgentNode | None" = None,
    ) -> "AgentMessage":
        return AgentMessage(
            identifier=uuid4(),
            sender=MISSING,
            recipient=self,
            addressee=addressee,
            content=MultimodalContent.of(content, *_content),
            responding=None,
        )

    def _associate(
        self,
        initializer: "AgentInitializer",
        /,
        concurrent: bool,
    ) -> None:
        assert is_missing(self._initializer), "AgentNode can be associated with only one agent"  # nosec: B101

        self._initializer = initializer
        self._concurrent = concurrent

        freeze(self)

    def initialize(self) -> "Agent":
        assert not_missing(self._initializer), (  # nosec: B101
            "AgentNode has to be associated with an agent to be used!"
            " Make sure you have defined an agent referring to it."
        )

        return self._initializer()


@final
class AgentMessage(ParametrizedData):
    identifier: UUID
    sender: AgentNode | Missing
    recipient: AgentNode
    addressee: AgentNode | None
    content: MultimodalContent
    responding: "AgentMessage | None"

    def __hash__(self) -> int:
        return hash(self.identifier)

    def __eq__(
        self,
        other: object,
    ) -> bool:
        if isinstance(other, AgentMessage):
            return self.identifier == other.identifier

        else:
            return False

    @property
    def should_respond(self) -> bool:
        return self.addressee is not None

    def respond(
        self,
        content: MultimodalContent | MultimodalContentConvertible,
        /,
        *_content: MultimodalContent | MultimodalContentConvertible,
        addressee: AgentNode | None = None,
    ) -> Self:
        assert not_missing(self.sender), "Missing message sender, can't respond to message drafts!"  # nosec: B101

        return self.__class__(
            identifier=uuid4(),
            sender=MISSING,
            recipient=self.addressee or self.sender,
            addressee=addressee,
            content=MultimodalContent.of(content, *_content),
            responding=self,
        )

    def forward(
        self,
        recipient: AgentNode,
        addressee: AgentNode | None = None,
    ) -> Self:
        assert not_missing(self.sender), "Missing message sender, can't forward message drafts!"  # nosec: B101

        return self.__class__(
            identifier=uuid4(),
            sender=MISSING,
            recipient=recipient,
            addressee=addressee or self.addressee,
            content=self.content,
            responding=None,
        )


type AgentOutput = frozenlist[AgentMessage] | AgentMessage | None


class AgentError(AgentException):
    def __init__(
        self,
        agent: AgentNode,
        message: AgentMessage,
        cause: BaseException,
        *args: object,
    ) -> None:
        self.agent: AgentNode = agent
        self.message: AgentMessage = message
        self.__cause__ = cause
        super().__init__(*args)


@runtime_checkable
class Agent(Protocol):
    async def __call__(
        self,
        message: AgentMessage,
    ) -> AgentOutput: ...


@runtime_checkable
class AgentInitializer(Protocol):
    def __call__(self) -> Agent: ...
