"""Strategy-evolution system: an AI agent that proposes, validates, and
shadow-tests strategy changes, with human-approved promotion.

Trades stay deterministic in signals/; this package only manipulates
strategy configurations as data. See evolve/guardrails.py for the
AI-immutable safety limits.
"""
