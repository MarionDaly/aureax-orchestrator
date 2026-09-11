"""Persistent system-log store. Shared shape across the orchestrator
dashboard and the edge client (autonomous runs, local observability).
Each side points ``LogStore`` at its own SQLite file.
"""
