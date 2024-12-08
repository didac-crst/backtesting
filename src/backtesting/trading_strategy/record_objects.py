from dataclasses import dataclass, field


@dataclass
class Signal:
    """
    Dataclass to store the signals in the ledger.

    """
    timestamp: int
    symbol: str
    trade_signal: str
    value_signal: float
