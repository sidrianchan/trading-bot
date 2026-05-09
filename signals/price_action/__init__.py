"""Price-action detectors (S/R levels, candlestick patterns, breakouts).

Each detector is a pure function: input is an OHLCV ``pd.DataFrame`` (and
optional level / config arguments), output is a structured dataclass
describing what was found. No global state, no I/O, no mutation.
"""
