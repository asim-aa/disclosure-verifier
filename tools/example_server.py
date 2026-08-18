"""Boilerplate MCP server — the pattern each real tool server (filing retriever,
transcript retriever, numerical reconciler) follows starting in Phase 1.

Run directly for a stdio smoke test:
    python -m tools.example_server
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("disclosure-verifier-example")


@mcp.tool()
def ping(message: str = "pong") -> str:
    """Echo a message back. Exists only to prove the server boots and a tool call round-trips."""
    return message


if __name__ == "__main__":
    mcp.run()
