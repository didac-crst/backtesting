from dataclasses import dataclass


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

    timestamp: int
    action: str
    symbol: str
    amount: float
    price: float
    traded: float
    commission: float
