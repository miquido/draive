from typing import Any, Literal

from draive.agents.node import AgentMessage
from draive.parameters import State
from draive.scope import ctx

__all__ = [
    "AgentWorkflowVisualization",
]

try:
    from networkx import DiGraph  # pyright: ignore
    from plotly.graph_objects import (  # pyright: ignore
        Figure,
        Layout,
        Scatter,
    )

    def _prepare_message_graph(  # pyright: ignore
        messages: list[AgentMessage],
        /,
    ) -> DiGraph:  # pyright: ignore
        graph: DiGraph = DiGraph()  # pyright: ignore

        for message in messages:
            graph.add_node(  # pyright: ignore
                message.identifier.hex,
                message=message.content.as_string(),
                sender=str(message.sender),
                recipient=str(message.recipient),
                y=message.timestamp.timestamp(),
            )
            if message.responding:
                graph.add_edge(  # pyright: ignore
                    message.responding.identifier.hex,
                    message.identifier.hex,
                )

        return graph  # pyright: ignore

    def _draw_message_graph(
        graph: DiGraph,  # pyright: ignore
        /,
    ) -> None:
        node_indices = list(graph.nodes())  # pyright: ignore

        x: list[Any] = [0] * len(node_indices)  # pyright: ignore
        y: list[Any] = [
            -graph.nodes[node]["y"]  # pyright: ignore
            for node in node_indices  # pyright: ignore
        ]  # Negative to flip the order
        messages: list[Any] = [graph.nodes[node]["message"] for node in node_indices]  # pyright: ignore
        senders: list[Any] = [graph.nodes[node]["sender"] for node in node_indices]  # pyright: ignore
        recipients: list[Any] = [graph.nodes[node]["recipient"] for node in node_indices]  # pyright: ignore

        # Create nodes
        node_trace = Scatter(
            x=x,
            y=y,
            mode="markers+text",
            marker={
                "size": 20,
                "color": "lightblue",
                "line": {
                    "width": 2,
                    "color": "black",
                },
            },
            text=messages,
            textposition="middle right",
            hoverinfo="text",
            hovertext=[
                f"ID: {id}<br>Message: {msg}<br>Sender: {sender}<br>Recipient: {recipient}"
                for id, msg, sender, recipient in zip(  # pyright: ignore  # noqa: A001
                    node_indices,  # pyright: ignore
                    messages,
                    senders,
                    recipients,
                    strict=False,
                )
            ],
        )

        # Create edges
        edge_traces: list[Any] = []
        for edge in graph.edges():  # pyright: ignore
            start_y: Any = -graph.nodes[edge[0]]["y"]  # pyright: ignore
            end_y: Any = -graph.nodes[edge[1]]["y"]  # pyright: ignore
            edge_trace = Scatter(
                x=[0, 0, None],
                y=[start_y, end_y, None],
                mode="lines",
                line={"width": 1, "color": "black"},
                hoverinfo="none",
            )
            edge_traces.append(edge_trace)

        # Create layout
        layout = Layout(
            title="Message Flow",
            showlegend=False,
            hovermode="closest",
            margin={
                "b": 20,
                "l": 5,
                "r": 5,
                "t": 40,
            },
            xaxis={
                "showgrid": False,
                "zeroline": False,
                "showticklabels": False,
            },
            yaxis={
                "showgrid": False,
                "zeroline": False,
                "showticklabels": False,
            },
            height=40 * len(node_indices),  # pyright: ignore
            width=800,
        )

        # Create figure and add traces
        fig: Figure = Figure(layout=layout)
        for edge_trace in edge_traces:
            fig.add_trace(edge_trace)  # pyright: ignore
        fig.add_trace(node_trace)  # pyright: ignore

        # Add arrow annotations
        for edge in graph.edges():  # pyright: ignore
            start_y = -graph.nodes[edge[0]]["y"]  # pyright: ignore
            end_y = -graph.nodes[edge[1]]["y"]  # pyright: ignore
            fig.add_annotation(  # pyright: ignore
                x=0,
                y=end_y,
                ax=0,
                ay=start_y,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="black",
            )

        fig.show()  # pyright: ignore

    class AgentWorkflowVisualization(State):  # pyright: ignore
        output: Literal["messages"] | None = None

        def visualize(
            self,
            messages: list[AgentMessage],
            /,
        ) -> None:
            if self.output is None:
                return  # skip visualization

            try:
                message_graph: DiGraph = _prepare_message_graph(messages)  # pyright: ignore

                match self.output:
                    case "messages":
                        _draw_message_graph(message_graph)

            except BaseException:
                pass  # nosec: B110 prevent visualization from breaking workflow


except ImportError:  # fallback implementation

    class AgentWorkflowVisualization(State):
        output: Literal["messages"] | None = None

        def visualize(
            self,
            messages: list[AgentMessage],
            /,
        ) -> None:
            if self.output is None:
                return  # skip visualization

            ctx.log_debug(
                "Workflow visualization is disabled due to missing imports."
                "Please install draive[visualization] optional dependency to use it."
            )
