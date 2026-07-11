"""Per-framework adapters for the generic MCP client.

Each adapter is a thin format translator. None of them contain per-MCP-server
logic - they speak to ``MCPClient.call`` for every execution.
"""
