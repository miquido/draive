import asyncio

from pytest import mark, raises

from draive import (
    META_EMPTY,
    Meta,
    ModelToolRequest,
    MultimodalContent,
    Toolbox,
    ctx,
    tool,
)
from draive.models import ModelToolSpecification


@mark.asyncio
async def test_of_empty_defaults_and_declaration_none_selection() -> None:
    async with ctx.scope("test"):
        tb: Toolbox = Toolbox.of(None)

        assert tb.tools == {}
        assert tb.tool_turns_limit == 0

        decl = tb.available_tools_declaration(tools_turn=0)
        assert decl.selection == "none"
        assert decl.specifications == ()


@mark.asyncio
async def test_of_single_tool_defaults_and_turn_limit_behavior() -> None:
    async with ctx.scope("test"):

        @tool
        async def compute(value: int) -> int:
            return value

        tb: Toolbox = Toolbox.of(compute)
        assert set(tb.tools.keys()) == {compute.name}
        assert tb.tool_turns_limit == 3

        # before turns limit -> no suggestion => auto
        decl0 = tb.available_tools_declaration(tools_turn=0)
        assert decl0.selection == "auto"
        assert [spec.name for spec in decl0.specifications] == [compute.name]

        # at limit -> none
        decl3 = tb.available_tools_declaration(tools_turn=3)
        assert decl3.selection == "none"


@mark.asyncio
async def test_of_iterable_plus_additional_and_suggest_any() -> None:
    async with ctx.scope("test"):

        @tool
        async def alpha() -> str:
            return "A"

        @tool
        async def beta() -> str:
            return "B"

        tb: Toolbox = Toolbox.of([alpha], beta, suggesting=True)

        # first turn suggested -> required
        decl0 = tb.available_tools_declaration(tools_turn=0)
        assert decl0.selection == "required"
        assert {spec.name for spec in decl0.specifications} == {alpha.name, beta.name}

        # next turn suggestion expires -> auto
        decl1 = tb.available_tools_declaration(tools_turn=1)
        assert decl1.selection == "auto"


@mark.asyncio
async def test_of_suggest_specific_tool_and_missing_suggestion() -> None:
    async with ctx.scope("test"):
        # Use a minimal Tool implementation with a typed specification
        from draive import ModelToolFunctionSpecification

        class SimpleTool:
            def __init__(self, name: str) -> None:
                self._name = name
                self._spec = ModelToolFunctionSpecification(
                    name=name,
                    description=None,
                    parameters=None,
                    meta=META_EMPTY,
                )

            @property
            def name(self) -> str:
                return self._name

            @property
            def description(self) -> str | None:
                return None

            @property
            def parameters(self):  # type: ignore[override]
                return None

            @property
            def specification(self):  # type: ignore[override]
                return self._spec

            @property
            def meta(self) -> Meta:  # type: ignore[override]
                return META_EMPTY

            @property
            def handling(self):  # type: ignore[override]
                return "response"

            def available(self, *, tools_turn: int) -> bool:  # type: ignore[override]
                return True

            async def call(self, call_id: str, /, **arguments):  # type: ignore[override]
                return MultimodalContent.of(self._name)

        first = SimpleTool("first")
        second = SimpleTool("second")

        # Suggest existing tool -> selection equals tool name
        tb1: Toolbox = Toolbox.of([first, second], suggesting=second)
        decl0 = tb1.available_tools_declaration(tools_turn=0)
        assert decl0.selection == second.name

        # Next turn suggestion expires -> auto
        decl1 = tb1.available_tools_declaration(tools_turn=1)
        assert decl1.selection == "auto"

        # Suggest a tool that's not included -> no suggestion -> auto
        tb2: Toolbox = Toolbox.of([first], suggesting=second)
        decl_missing = tb2.available_tools_declaration(tools_turn=0)
        assert decl_missing.selection == "auto"


@mark.asyncio
async def test_available_tools_filtered_by_availability() -> None:
    async with ctx.scope("test"):

        def even_turns_only(
            tools_turn: int,
            specification: ModelToolSpecification,
        ) -> bool:
            return tools_turn % 2 == 0

        @tool(availability=even_turns_only)
        async def even() -> str:
            return "even"

        @tool
        async def always() -> str:
            return "always"

        tb: Toolbox = Toolbox.of([even, always])

        decl_odd = tb.available_tools_declaration(tools_turn=1)
        assert decl_odd.selection == "auto"
        assert [spec.name for spec in decl_odd.specifications] == [always.name]

        decl_even = tb.available_tools_declaration(tools_turn=2)
        assert {spec.name for spec in decl_even.specifications} == {even.name, always.name}


@mark.asyncio
async def test_call_tool_success_and_missing() -> None:
    async with ctx.scope("test"):

        @tool
        async def echo(value: int) -> int:
            return value

        tb: Toolbox = Toolbox.of(echo)

        result = await tb.call_tool("echo", call_id="c1", arguments={"value": 42})
        assert result == MultimodalContent.of("42")

        with raises(Exception) as err:
            await tb.call_tool("missing", call_id="c2", arguments={})
        # ToolError is expected and carries empty formatted content
        assert getattr(err.value, "content", None) == MultimodalContent.empty


@mark.asyncio
async def test_respond_success_and_error_handling() -> None:
    async with ctx.scope("test"):

        class Boom(Exception):
            pass

        @tool
        async def ok(value: int) -> int:
            return value

        @tool
        async def fails() -> str:
            raise Boom("boom")

        tb: Toolbox = Toolbox.of([ok, fails])

        req_ok = ModelToolRequest.of("r1", tool="ok", arguments={"value": 7})
        resp_ok = await tb.respond(req_ok)
        assert resp_ok.identifier == "r1"
        assert resp_ok.tool == "ok"
        assert resp_ok.handling == "response"
        assert resp_ok.content == MultimodalContent.of("7")

        req_fail = ModelToolRequest.of("r2", tool="fails")
        resp_fail = await tb.respond(req_fail)
        assert resp_fail.identifier == "r2"
        assert resp_fail.tool == "fails"
        assert resp_fail.handling == "error"
        # default error formatter includes the exception class name
        assert any("ERROR: Tool execution failed" in t.text for t in resp_fail.content.texts())


@mark.asyncio
async def test_respond_unknown_tool_fallback_and_logging() -> None:
    async with ctx.scope("test"):
        tb: Toolbox = Toolbox.of(None)

        req = ModelToolRequest.of("rx", tool="unknown")
        resp = await tb.respond(req)
        assert resp.identifier == "rx"
        assert resp.tool == "unknown"
        assert resp.handling == "error"
        assert resp.content == MultimodalContent.of("ERROR: Unavailable tool unknown")


@mark.asyncio
async def test_respond_detached_spawns_and_returns_immediate_message() -> None:
    async with ctx.scope("test"):
        marker: list[str] = []

        @tool(handling="detached")
        async def bg(task: str) -> str:
            marker.append(task)
            return task

        tb: Toolbox = Toolbox.of(bg)
        req = ModelToolRequest.of("d1", tool="bg", arguments={"task": "work"})

        resp = await tb.respond(req)
        assert resp.identifier == "d1"
        assert resp.tool == "bg"
        assert resp.handling == "detached"
        assert resp.content == MultimodalContent.of("bg tool execution has been requested")

        # allow the spawned task to run
        await asyncio.sleep(0)
        assert marker == ["work"]


@mark.asyncio
async def test_with_tools_adds_new_and_is_immutable() -> None:
    async with ctx.scope("test"):

        @tool
        async def t1() -> str:
            return "1"

        @tool
        async def t2() -> str:
            return "2"

        base = Toolbox.of(t1)
        extended = base.with_tools(t2)

        assert set(base.tools.keys()) == {t1.name}
        assert set(extended.tools.keys()) == {t1.name, t2.name}
        # policy and limits preserved
        assert extended.tool_turns_limit == base.tool_turns_limit == 3


@mark.asyncio
async def test_with_suggestion_true_false_and_specific() -> None:
    async with ctx.scope("test"):
        from draive import ModelToolFunctionSpecification

        class SimpleTool:
            def __init__(self, name: str) -> None:
                self._name = name
                self._spec = ModelToolFunctionSpecification(
                    name=name,
                    description=None,
                    parameters=None,
                    meta=META_EMPTY,
                )

            @property
            def name(self) -> str:
                return self._name

            @property
            def description(self) -> str | None:
                return None

            @property
            def parameters(self):  # type: ignore[override]
                return None

            @property
            def specification(self):  # type: ignore[override]
                return self._spec

            @property
            def meta(self) -> Meta:  # type: ignore[override]
                return META_EMPTY

            @property
            def handling(self):  # type: ignore[override]
                return "response"

            def available(self, *, tools_turn: int) -> bool:  # type: ignore[override]
                return True

            async def call(self, call_id: str, /, **arguments):  # type: ignore[override]
                return MultimodalContent.of(self._name)

        main = SimpleTool("main")

        tb = Toolbox.of(main)

        # suggest any for two turns
        tb_any = tb.with_suggestion(True, turns=2)
        assert tb_any.available_tools_declaration(tools_turn=0).selection == "required"
        assert tb_any.available_tools_declaration(tools_turn=1).selection == "required"
        assert tb_any.available_tools_declaration(tools_turn=2).selection == "auto"

        # disable suggestions
        tb_none = tb.with_suggestion(False)
        assert tb_none.available_tools_declaration(tools_turn=0).selection == "auto"

        # suggest a specific tool
        tb_specific = tb.with_suggestion(main, turns=1)
        assert tb_specific.available_tools_declaration(tools_turn=0).selection == main.name


@mark.asyncio
async def test_filtered_by_tool_names_and_tags() -> None:
    async with ctx.scope("test"):

        @tool(meta=Meta.of({"tags": ("x", "y")}))
        async def tagged_xy() -> str:
            return "xy"

        @tool(meta=Meta.of({"tags": ("y",)}))
        async def tagged_y() -> str:
            return "y"

        tb = Toolbox.of([tagged_xy, tagged_y])

        # filter by names
        by_name = tb.filtered(tools={tagged_y.name})
        assert set(by_name.tools.keys()) == {tagged_y.name}

        # filter by tags (subset match)
        by_tags = tb.filtered(tags=("x",))
        assert set(by_tags.tools.keys()) == {tagged_xy.name}
