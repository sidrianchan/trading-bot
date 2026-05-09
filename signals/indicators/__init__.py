"""Indicator computations (trend / momentum / volatility / volume).

Each module exposes pure functions that take a ``pd.DataFrame`` of OHLCV bars
and return ``pd.Series`` aligned to the input index. No state, no I/O.
"""
