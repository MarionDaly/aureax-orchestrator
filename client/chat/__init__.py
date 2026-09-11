"""Chat session persistence — schema for chat sessions and messages.

The orchestrator dashboard is the only side that constructs ``SessionStore``
today; the module lives here so an edge-side agent loop can reuse the same
schema later, each side against its own SQLite file.
"""
