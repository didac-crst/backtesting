from dataclasses import dataclass, field


@dataclass
class Capital:
    """
    Dataclass to store the capital movements (Investment and Disbursement) in the ledger.

    """

    timestamp: int
    action: str
    amount: float


@dataclass
class CurrencyPrice:
    """
    Dataclass to store the currency prices in the ledger.

    """

    timestamp: int
    symbol: str
    price: float


@dataclass
class Transaction:
    """
    Dataclass to store the transactions in the ledger.

    """
    id: int
    action: str
    symbol: str
    amount: float
    price: float
    traded: float
    commission: float
    balance_pre: float

@dataclass
class MetaTransaction:
    """
    Dataclass to store the meta transactions in the ledger.

    """
    id: int
    timestamp: int
    trade: bool
    ack: bool
    messages: list[str] = field(default_factory=list)
