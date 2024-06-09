from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .record_objects import Capital, CurrencyPrice, Transaction


@dataclass
class Ledger:
    """
    Dataclass to store the ledger of a portfolio.

    """

    portfolio_symbol: str
    capital: list[Capital] = field(default_factory=list)
    prices: list[CurrencyPrice] = field(default_factory=list)
    transactions: list[Transaction] = field(default_factory=list)

    # Ledger actions methods ------------------------------------------------

    def record_capital_movement(
        self, timestamp: int, action: str, amount: float
    ) -> None:
        """
        Method to record a capital movement in the ledger.

        """
        capital = Capital(timestamp=timestamp, action=action, amount=amount)
        self.capital.append(capital)

    def invest_capital(self, timestamp: int, amount: float) -> None:
        """
        Method to record an investment in the ledger.

        """
        self.record_capital_movement(
            timestamp=timestamp, action="INVEST", amount=amount
        )

    def disburse_capital(self, timestamp: int, amount: float) -> None:
        """
        Method to record a disbursement in the ledger.

        """
        self.record_capital_movement(
            timestamp=timestamp, action="DISBURSE", amount=amount
        )

    def record_price(self, timestamp: int, symbol: str, price: float) -> None:
        """
        Method to record a currency price in the ledger.

        """
        currency_price = CurrencyPrice(timestamp=timestamp, symbol=symbol, price=price)
        self.prices.append(currency_price)

    def record_transaction(
        self,
        timestamp: int,
        action: str,
        symbol: str,
        amount: float,
        price: float,
        commission: float,
    ) -> None:
        """
        Method to record a transaction in the ledger.

        """
        traded = amount * price
        transaction = Transaction(
            timestamp=timestamp,
            action=action,
            symbol=symbol,
            amount=amount,
            price=price,
            traded=traded,
            commission=commission,
        )
        self.transactions.append(transaction)

    def buy(
        self,
        timestamp: int,
        symbol: str,
        amount: float,
        price: float,
        commission: float,
    ) -> None:
        """
        Method to record a buy transaction in the ledger.

        """
        self.record_transaction(timestamp, "BUY", symbol, amount, price, commission)

    def sell(
        self,
        timestamp: int,
        symbol: str,
        amount: float,
        price: float,
        commission: float,
    ) -> None:
        """
        Method to record a sell transaction in the ledger.

        """
        self.record_transaction(timestamp, "SELL", symbol, amount, price, commission)

    # Ledger reporting methods ------------------------------------------------

    @property
    def total_commissions(self) -> float:
        """
        Property to get the total commissions paid by the portfolio.

        """
        ledger_df = self.transactions_df
        return ledger_df["Commission"].sum()

    def transactions_count(self, symbol: Optional[str] = None) -> int:
        """
        Method to count the number of transactions in the ledger.

        If a symbol is provided, it returns the number of transactions for the asset.

        The transactions for the portfolio currency are not counted. As they would be the same as the total number of transactions.

        """
        ledger_df = self.transactions_df
        if symbol is not None:
            return len(ledger_df[ledger_df["Symbol"] == symbol])
        else:
            return len(ledger_df[ledger_df["Symbol"] != self.portfolio_symbol])

    def transactions_sum(self, symbol: Optional[str] = None) -> float:
        """
        Method to get the total traded amount in the ledger.

        If a symbol is provided, it returns the total traded amount for the asset.

        The transactions for the portfolio currency are not counted. As they would be the same as the total traded amount.

        """
        ledger_df = self.transactions_df
        if symbol is not None:
            return ledger_df[ledger_df["Symbol"] == symbol]["Traded"].sum()
        else:
            return ledger_df[ledger_df["Symbol"] != self.portfolio_symbol][
                "Traded"
            ].sum()

    @property
    def capital_df(self) -> pd.DataFrame:
        """
        Property to get the capital movements' list as a DataFrame.

        """
        data = []
        for capital in self.capital:
            data.append(
                (
                    capital.timestamp,
                    capital.action,
                    capital.amount,
                )
            )
        columns = (
            "Timestamp",
            "Action",
            "Amount",
        )
        df = pd.DataFrame(data, columns=columns)
        return df

    @property
    def capital_summary(self) -> dict[str, float]:
        """
        Property to get a summary of the capital movements.

        This returns a dictionary with the total investment and disbursement.

        """
        capital_dict = dict()
        capital_df = self.capital_df
        investment = capital_df[capital_df["Action"] == "INVEST"]["Amount"].sum()
        disbursement = capital_df[capital_df["Action"] == "DISBURSE"]["Amount"].sum()
        capital_dict["Investment"] = investment
        capital_dict["Disbursement"] = disbursement
        return capital_dict

    @property
    def prices_df(self) -> pd.DataFrame:
        """
        Property to get the currency prices' list as a DataFrame.

        """
        data = []
        for currency_price in self.prices:
            data.append(
                (
                    currency_price.timestamp,
                    currency_price.symbol,
                    currency_price.price,
                )
            )
        columns = (
            "Timestamp",
            "Symbol",
            "Price",
        )
        df = pd.DataFrame(data, columns=columns)
        return df

    @property
    def transactions_df(self) -> pd.DataFrame:
        """
        Property to get the transactions' list as a DataFrame.

        """
        data = []
        for transaction in self.transactions:
            data.append(
                (
                    transaction.timestamp,
                    transaction.action,
                    transaction.symbol,
                    transaction.amount,
                    transaction.price,
                    transaction.traded,
                    transaction.commission,
                )
            )
        columns = (
            "Timestamp",
            "Action",
            "Symbol",
            "Amount",
            "Price",
            "Traded",
            "Commission",
        )
        df = pd.DataFrame(data, columns=columns)
        return df

    @property
    def equity_df(self) -> pd.DataFrame:
        """
        Property to calculate the equity of the portfolio as a DataFrame.

        The time granularity depends on the prices' and transactions' timestamps.

        """
        merging_cols = ["Timestamp", "Symbol"]
        transactions_df = self.transactions_df
        prices_df = self.prices_df
        # Sells need to be marked as the negative of the amount
        transactions_df.loc[transactions_df["Action"] == "SELL", "Amount"] = (
            transactions_df["Amount"] * -1
        )
        # Commissions are burned from the traded amount
        transactions_df["Net_Amount"] = (
            transactions_df["Amount"] - transactions_df["Commission"]
        )
        # We need to cumulate each transaction to get the balance
        transactions_df["Balance"] = (
            transactions_df["Net_Amount"].groupby(transactions_df["Symbol"]).cumsum()
        )
        # Get rid of the unnecessary columns
        transactions_df.drop(
            columns=["Price", "Traded", "Action", "Amount", "Commission"],
            inplace=True,
        )
        # It's important to only keep the last values when grouping.
        # Depending on the moment we could have more than one transaction per symbol and timestamp.
        transactions_df = (
            transactions_df.groupby(["Symbol", "Timestamp"]).last().reset_index()
        )
        # Merge the transactions and prices
        # It's important to use an outer join to keep all the timestamps
        preequity_df = transactions_df.merge(
            prices_df, how="outer", left_on=merging_cols, right_on=merging_cols
        )
        # The portfolio currency price is always 1
        preequity_df.loc[preequity_df["Symbol"] == self.portfolio_symbol, "Price"] = 1.0
        # Fill the missing prices with the last known price
        preequity_df["Price"] = preequity_df.groupby("Symbol")["Price"].ffill()
        # Fill the missing balance with the last known equity
        preequity_df["Balance"] = preequity_df.groupby("Symbol")["Balance"].ffill()
        # The first balances before any action are NaN
        preequity_df["Balance"] = preequity_df["Balance"].fillna(0)
        # Calculate the equity in Quote currency
        preequity_df["Equity"] = preequity_df["Balance"] * preequity_df["Price"]
        # Pivot the table to have the equity of each symbol in columns
        equity_df = preequity_df.pivot(
            index="Timestamp", columns="Symbol", values="Equity"
        )
        # Fill the missing values with the last known equity
        equity_df = equity_df.ffill()
        # Fill the NaN values with 0
        equity_df = equity_df.fillna(0)
        # Calculate the total equity
        equity_df["Total"] = equity_df.sum(axis=1)
        return equity_df
