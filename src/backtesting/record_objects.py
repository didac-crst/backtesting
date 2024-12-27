from dataclasses import dataclass, field
from typing import Literal


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

LITERAL_TRANSACTION_ACTION = Literal["BUY", "SELL"]
LITERAL_TRANSACTION_REASON = Literal["SIGNAL", "LIQUIDITY", "VOLATILITY", "INVESTMENT", "DISBURSEMENT"]

@dataclass
class Transaction:
    """
    Dataclass to store the transactions in the ledger.

    """
    id: int
    action: LITERAL_TRANSACTION_ACTION
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
    reason: LITERAL_TRANSACTION_REASON
    purchase_price_avg: float
    deducted_commissions: float
    ack: bool
    messages: list[str] = field(default_factory=list)
