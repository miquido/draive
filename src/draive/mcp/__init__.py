try:
    import mcp  # pyright: ignore[reportUnusedImport]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "draive.mcp requires the 'mcp' extra. Install via `pip install draive[mcp]`."
    ) from exc

from draive.mcp.client import MCPClient, MCPClients
from draive.mcp.server import MCPServer

__all__ = (
    "MCPClient",
    "MCPClients",
    "MCPServer",
)
