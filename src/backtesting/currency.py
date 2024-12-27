from dataclasses import dataclass

import numpy as np

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
        self.purchase_price_avg = np.nan
        self.commissions_amortization = 0.0

    @check_positive
    def _update_price(self, price: float) -> None:
        """
        Method to update the price of the currency.

        """
        self.price = float(price)
    
    @check_positive
    def _update_purchase_price_avg(self, purchase_price: float) -> None:
        """
        Method to update the average purchase price of the currency.

        """
        self.purchase_price_avg = float(purchase_price)
    
    @check_positive
    def _add_commissions_in_commissions_amortization(self, commissions: float) -> None:
        """
        Method to add commissions to the commissions amortization.
        
        This is needed to calculate the real profit when selling the asset.

        """
        self.commissions_amortization += commissions
    
    @check_positive
    def _deduct_commissions_from_comissions_amortization(self, amount_sold_quote: float) -> float:
        """
        Method to deduct commissions from the commissions amortization.
        
        This is needed to calculate the real profit when selling the asset.
        It returns the deducted commissions to take into account in the profit calculation.

        """
        amount_sold_base = amount_sold_quote / self.price
        balance = self.balance
        proportion_sold = amount_sold_base / balance
        deducted_commissions = self.commissions_amortization * proportion_sold
        self.commissions_amortization -= deducted_commissions
        return deducted_commissions

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
            display_price(self.purchase_price_avg, self.portfolio_symbol),
        )
