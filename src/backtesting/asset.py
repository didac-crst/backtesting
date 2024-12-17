from dataclasses import dataclass

from .support import display_price


@dataclass
class Asset:
    """
    Dataclass defining the basic structure of an asset.

    """

    symbol: str
    name: str # This is the portfolio name

    def __post_init__(self) -> None:
        self.balance: float = 0.0
        self.commissions_sum: float = 0.0

    def check_amount(self, amount) -> None:
        """
        Method to check if the amount is less than the balance.

        """
        if amount > self.balance:
            raise ValueError(
                f"[{self.name}] {self.symbol}: Insufficient funds: balance is {self.balance} and you want to spend {amount}."
            )