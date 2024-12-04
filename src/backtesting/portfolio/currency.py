from dataclasses import dataclass

from .asset import Asset

from .support import check_positive, display_price


@dataclass
class Currency(Asset):
    """
    Dataclass keeping track the value and the balance of a Currency.

    """

    portfolio_symbol: str
    commission: float

    def __post_init__(self) -> None:
        super().__post_init__()
        self.price = 0.0

    @check_positive
    def _update_price(self, price: float) -> None:
        """
        Method to update the price of the currency.

        """
        self.price = price

    @check_positive
    def _deposit(self, amount: float) -> None:
        """
        Method to deposit an amount into the asset balance.
        """
        self.balance += amount

    @check_positive
    def _withdraw(self, amount: float) -> None:
        """
        Method to withdraw an amount from the asset balance.

        """
        self.check_amount(amount)
        self.balance -= amount
    
    @property
    def value(self) -> float:
        """
        Property to get the value of the asset.

        """
        return self.balance * self.price

    @property
    def info(self) -> tuple[str, str, str]:
        """
        Property to get the balance, value and price of the asset.

        """
        return (
            display_price(self.balance),
            display_price(self.balance * self.price, self.portfolio_symbol),
            display_price(self.price, self.portfolio_symbol),
        )
