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
    active_assets: list[str] = field(default_factory=list)
    timestamp_low_resolution: int = 500 # samples

    def __post_init__(self):
        self._evolution_id = self.evolution_id

    # Ledger actions methods ------------------------------------------------
    
    def record_active_asset(self, symbol: str) -> None:
        """
        Method to record an active asset in the ledger.

        """
        if symbol not in self.active_assets:
            self.active_assets.append(symbol)

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
        self.record_active_asset(symbol)

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
    def timestamp_reporting_list(self) -> list:
        """
        Reduce the timestamp resolution of a list of objects.

        """
        # We only calculate the reporting list if the ledger has been updated
        if self.check_update:
            prices_list = self.prices
            def get_exact_timejump(timespan: int, multiple: int, resolution: int) -> int:
                # Function to get a timejump that is a multiple of multiple and that fits the resolution.
                raw_timejump = timespan / resolution
                return round(raw_timejump / multiple) * multiple

            unique_timestamps = set()
            for obj in prices_list:
                value = getattr(obj, 'timestamp', None)
                if isinstance(value, int):
                    unique_timestamps.add(value)
            timestamp_list = list(unique_timestamps)
            timestamp_list.sort()
            timestamp_series = pd.Series(timestamp_list)
            # Get the minimal value of the timestamp
            timestamp_min = timestamp_series.min()
            # Get the maximal value of the timestamp
            timestamp_max = timestamp_series.max()
            # Get the timespan between the minimal and maximal timestamp
            timespan = timestamp_max - timestamp_min
            # Get the most common frequency between timestamps
            if len(timestamp_series) > 1:
                timestamp_multiple = int(timestamp_series.diff().median())
            else:
                timestamp_multiple = 1
            timejumps = get_exact_timejump(timespan=timespan, multiple=timestamp_multiple, resolution=self.timestamp_low_resolution)
            # If the timejumps is 0, we return the minimal timestamp
            # range does not accept a step of 0 (timejumps)
            if timejumps == 0:
                timestamps_list = [timestamp_min]
            else:
                timestamps_list = list(range(timestamp_min, timestamp_max + 1, timejumps))
            self._timestamp_reporting_list = timestamps_list
        return self._timestamp_reporting_list

    @property
    def total_commissions(self) -> float:
        """
        Property to get the total commissions paid by the portfolio.

        """
        # We only calculate the total commissions if the ledger has been updated
        if self.check_update:
            ledger_df = self.transactions_df
            total_commissions = ledger_df["Commission"].sum()
            self._total_commissions = total_commissions
        return self._total_commissions

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
        # We only calculate the capital movements if the ledger has been updated
        if self.check_update:
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
            self._capital_df = df
        return self._capital_df

    @property
    def capital_summary(self) -> dict[str, float]:
        """
        Property to get a summary of the capital movements.

        This returns a dictionary with the total investment and disbursement.

        """
        # We only calculate the capital summary if the ledger has been updated
        if self.check_update:
            capital_dict = dict()
            capital_df = self.capital_df
            investment = capital_df[capital_df["Action"] == "INVEST"]["Amount"].sum()
            disbursement = capital_df[capital_df["Action"] == "DISBURSE"]["Amount"].sum()
            capital_dict["Investment"] = investment
            capital_dict["Disbursement"] = disbursement
            self._capital_summary = capital_dict
        return self._capital_summary

    @property
    def prices_df(self) -> pd.DataFrame:
        """
        Property to get the currency prices' list as a DataFrame.

        """
        # We only calculate the prices if the ledger has been updated
        if self.check_update:
            data = []
            for currency_price in self.prices:
                data.append(
                    (
                        currency_price.timestamp,
                        currency_price.symbol,
                        currency_price.price,
                    )
                )
        
            columns = [
                "Timestamp",
                "Symbol",
                "Price",
            ]
            df = pd.DataFrame(data, columns=columns)
            # We only keep the prices of the active assets
            df = df[df["Symbol"].isin(self.active_assets)]
            # We only keep the prices that are in the reporting list
            # The resolution of the reporting list is lower than the prices' timestamps
            df = df[df["Timestamp"].isin(self.timestamp_reporting_list)]
            df.reset_index(drop=True, inplace=True)
            self._prices_df = df
        return self._prices_df

    @property
    def transactions_df(self) -> pd.DataFrame:
        """
        Property to get the transactions' list as a DataFrame.

        """
        # We only calculate the transactions if the ledger has been updated
        if self.check_update:
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
            self._transactions_df = df
        return self._transactions_df

    @property
    def equity_df(self) -> pd.DataFrame:
        """
        Property to calculate the equity of the portfolio as a DataFrame.

        The time granularity depends on the prices' and transactions' timestamps.

        """
        # We only calculate the equity if the ledger has been updated
        if self.check_update:
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
            # # We only keep the prices of the active assets
            # equity_df = equity_df[self.active_assets]
            # # We only keep the prices that are in the reporting list
            # # The resolution of the reporting list is lower than the prices' timestamps
            # equity_df = equity_df[equity_df.index.isin(self.timestamp_reporting_list)]
            # Calculate the total equity
            equity_df["Total"] = equity_df.sum(axis=1)
            self._equity_df = equity_df
        return self._equity_df

    @property
    def traded_assets_values(self) -> pd.DataFrame:
        """
        Property to get the total traded amount for each asset and segregated by action [BUY, SELL].

        """
        # We only calculate the traded assets values if the ledger has been updated
        if self.check_update:
            transactions = self.transactions_df
            assets_transactions=transactions[transactions['Symbol']!='USDT']
            traded_assets_values = assets_transactions.pivot_table(index='Symbol', columns='Action', values='Traded', aggfunc='sum').fillna(0)
            if 'BUY' not in traded_assets_values.columns:
                traded_assets_values['BUY']=0
            if 'SELL' not in traded_assets_values.columns:
                traded_assets_values['SELL']=0
            self._traded_assets_values = traded_assets_values
        return self._traded_assets_values
    
    @property
    def evolution_id(self) -> str:
        """
        Property to identify the number of transactions and prices in the ledger.
        
        This will be used to know if the ledger has been updated and therefore know if the properties need to be recalculated.

        """
        prices_count = len(self.prices)
        transactions_count = len(self.transactions)
        id = f"p{prices_count}_t{transactions_count}"
        return id
    
    @property
    def check_update(self) -> bool:
        """
        Property to check if the ledger has been updated.
        
        If there are new transactions or prices, the ledger has been updated.
        
        """
        return self.evolution_id != self._evolution_id