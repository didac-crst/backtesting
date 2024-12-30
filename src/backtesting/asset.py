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
        balance = self.balance
        if amount > balance:
            if hasattr(self, "portfolio_symbol") and hasattr(self, "price"): 
                price = self.price
                balance_quote = balance * price
                amount_quote = amount * price
                raise ValueError(
                    f"[{self.name}] {self.symbol}: Insufficient funds: balance is {balance} ({balance_quote} {self.portfolio_symbol}) and you want to spend {amount} ({amount_quote} {self.portfolio_symbol})."
                )
            else:
                raise ValueError(f"[{self.name}] {self.symbol}: Insufficient funds: balance is {balance} and you want to spend {amount}.")