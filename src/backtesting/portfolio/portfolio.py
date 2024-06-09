from dataclasses import dataclass
from typing import Optional, Literal

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
import matplotlib.dates as mdates

from .asset import Asset
from .ledger import Ledger
from .currency import Currency
from .support import (
    check_positive,
    display_price,
    display_percentage,
    display_integer,
    now_ms,
    thousands,
    to_percent,
    display_pretty_table,
)

VerboseType = Literal["silent", "action", "status", "verbose"]


@dataclass
class Portfolio(Asset):
    """
    Dataclass containing the structure of a portfolio.

    """

    commission_trade: float = 0.0
    commission_transfer: float = 0.0
    frequency_displayed: str = "1h"

    # Portfolio internal methods ------------------------------------------------

    def __post_init__(self) -> None:
        super().__post_init__()
        self.set_verbosity("verbose")
        self.assets = dict()
        self.Ledger = Ledger(portfolio_symbol=self.symbol)

    def set_verbosity(self, type: VerboseType) -> None:
        """
        Method to set the verbose status and action flags.

        """
        if type == "status":
            self.verbose_status = True
            self.verbose_action = False
        elif type == "action":
            self.verbose_status = False
            self.verbose_action = True
        elif type == "verbose":
            self.verbose_status = True
            self.verbose_action = True
        elif type == "silent":
            self.verbose_status = False
            self.verbose_action = False

    @property
    def _assets_table(self) -> str:
        """
        Property to create a pretty table with the assets' information.

        """
        data = [
            [
                "Symbol",
                "Balance",
                "Value",
                "Price",
                "Transactions",
                "Total traded",
                "Currency growth",
            ]
        ]
        for asset_symbol, Currency in self.assets.items():
            data.append(
                [
                    asset_symbol,
                    *Currency.values,
                    display_integer(self.transactions_count(asset_symbol)),
                    display_price(self.transactions_sum(asset_symbol), self.symbol),
                    display_percentage(self.get_asset_growth(asset_symbol)),
                ]
            )
        return display_pretty_table(data, padding=6)

    def __repr__(self) -> str:
        """
        Dunder method to display the portfolio information as a string.

        """
        text = (
            f"Portfolio:\n"
            f"  -> Symbol = {self.symbol}\n"
            f"  -> Transfer commission = {display_percentage(self.commission_transfer)}\n"
            f"  -> Trade commission = {display_percentage(self.commission_trade)}\n"
            f"  -> Transactions = {display_integer(self.transactions_count())}\n"
            f"  -> Total traded = {display_price(self.transactions_sum(), self.symbol)}\n"
            f"  -> Total commissions = {display_price(self.total_commissions, self.symbol)}\n"
            f"  -> Commission gains ratio = {self.commission_gains_ratio_str}\n"
            f"  -> Invested capital = {display_price(self.invested_capital, self.symbol)}\n"
            f"  -> Disbursed capital = {display_price(self.disbursed_capital, self.symbol)}\n"
            f"  -> Quote balance = {display_price(self.balance, self.symbol)}\n"
            f"  -> Assets value = {display_price(self.assets_value, self.symbol)}\n"
            f"  -> Equity value = {display_price(self.equity_value, self.symbol)}\n"
            f"  -> Gains = {display_price(self.gains, self.symbol)}\n"
            f"  -> ROI = {display_percentage(self.roi)}\n"
            f"  -> Assets:"
        )
        if len(self.assets) > 0:
            text += f"\n{self._assets_table}\n"
        else:
            text += " None\n"
        return text

    def print_portfolio(self) -> None:
        """
        Method to print the portfolio information if the verbose_status flag is set to True.

        """
        if self.verbose_status:
            print(self.__repr__())

    # Portfolio action methods ------------------------------------------------

    def add_Currency(self, symbol: str) -> None:
        """
        Method to create a currency into the portfolio.

        """
        self.assets[symbol] = Currency(
            symbol=symbol,
            portfolio_symbol=self.symbol,
            commission=self.commission_trade,
        )

    def add_Currency_list(self, symbols_list: list[str]) -> None:
        """
        Method to create a list of currencies into the portfolio.

        """
        for symbol in symbols_list:
            self.add_Currency(symbol=symbol)
        self.print_portfolio()

    def is_asset_in_portfolio(self, symbol: str) -> bool:
        """
        Method to check if an asset exists in the portfolio.

        """
        return symbol in self.assets.keys()

    @check_positive
    def update_single_price(
        self, symbol: str, price: float, timestamp: Optional[int] = None
    ) -> None:
        """
        Method to update a single price of an asset in the portfolio.

        """
        if timestamp is None:
            timestamp = now_ms()
        try:
            self.assets[symbol]._update_price(price)
            self.Ledger.record_price(timestamp=timestamp, symbol=symbol, price=price)
        except KeyError:
            if not self.is_asset_in_portfolio(symbol):
                self.add_Currency(symbol)
                self.update_single_price(
                    symbol=symbol, price=price, timestamp=timestamp
                )
        self.print_portfolio()

    def update_prices(
        self, prices: dict[str, float], timestamp: Optional[int] = None
    ) -> None:
        """
        Method to update the prices of the assets in the portfolio.

        """
        verbose_status_flag = self.verbose_status
        if timestamp is None:
            timestamp = now_ms()
        # Temporarily disable the verbose_status flag to avoid printing the portfolio multiple times
        self.verbose_status = False
        for symbol, price in prices.items():
            self.update_single_price(symbol=symbol, price=price, timestamp=timestamp)
        # Record the price of the portfolio currency
        # This is necessary for the equity calculation
        # However, we don't want to create a new asset for the portfolio currency
        self.Ledger.record_price(timestamp=timestamp, symbol=self.symbol, price=1)
        # Re-enable the verbose_status flag if it was enabled before
        if verbose_status_flag:
            self.verbose_status = True
        self.print_portfolio()

    @check_positive
    def deposit(self, amount: float, timestamp: Optional[int] = None) -> None:
        """
        Method to deposit an amount into the portfolio.

        """
        if timestamp is None:
            timestamp = now_ms()
        commission = amount * self.commission_transfer
        net_amount = amount - commission
        self.commissions_sum += commission
        self.balance += net_amount
        # Record the transaction in the ledger for the portfolio currency
        self.Ledger.buy(
            timestamp=timestamp,
            symbol=self.symbol,
            amount=amount,
            price=1,
            commission=commission,
        )
        # Record the capital movement in the ledger
        self.Ledger.invest_capital(timestamp=timestamp, amount=amount)
        self.print_portfolio()

    @check_positive
    def withdraw(self, amount: float, timestamp: Optional[int] = None) -> None:
        """
        Method to withdraw an amount from the portfolio.

        """
        if timestamp is None:
            timestamp = now_ms()
        commission = amount * self.commission_transfer
        gross_amount = amount + commission
        self.check_amount(gross_amount)
        self.commissions_sum += commission
        self.balance -= gross_amount
        # Record the transaction in the ledger for the portfolio currency
        self.Ledger.sell(
            timestamp=timestamp,
            symbol=self.symbol,
            amount=amount,
            price=1,
            commission=commission,
        )
        # Record the capital movement in the ledger
        self.Ledger.disburse_capital(timestamp=timestamp, amount=amount)
        self.print_portfolio()

    @check_positive
    def buy(
        self, symbol: str, amount_base: float, timestamp: Optional[int] = None
    ) -> None:
        """
        Method to buy an amount of an asset in the portfolio.

        """
        if timestamp is None:
            timestamp = now_ms()
        amount_quote = amount_base * self.assets[symbol].price
        commission_quote = amount_quote * self.commission_trade
        gross_amount_quote = amount_quote + commission_quote
        self.check_amount(gross_amount_quote)
        self.balance -= gross_amount_quote
        self.assets[symbol]._deposit(amount_base)
        if self.verbose_action:
            msg = (
                f"Buying {display_price(amount_base, symbol)} for {display_price(amount_quote, self.symbol)} "
                f"at {display_price(self.assets[symbol].price, self.symbol)}/{symbol} "
                f"(Commission: {display_price(commission_quote, self.symbol)})"
            )
            print(msg)
        # Record the transaction in the ledger for the asset
        self.Ledger.buy(
            timestamp=timestamp,
            symbol=symbol,
            amount=amount_base,
            price=self.assets[symbol].price,
            commission=0,
        )
        # Record the transaction in the ledger for the portfolio currency
        self.Ledger.sell(
            timestamp=timestamp,
            symbol=self.symbol,
            amount=amount_quote,
            price=1,
            commission=commission_quote,
        )
        self.print_portfolio()

    @check_positive
    def sell(
        self, symbol: str, amount_base: float, timestamp: Optional[int] = None
    ) -> None:
        """
        Method to sell an amount of an asset in the portfolio.

        """
        if timestamp is None:
            timestamp = now_ms()
        amount_quote = amount_base * self.assets[symbol].price
        commission_quote = amount_quote * self.commission_trade
        self.check_amount(commission_quote)
        net_amount_quote = amount_quote - commission_quote
        self.assets[symbol]._withdraw(amount_base)
        self.balance += net_amount_quote
        if self.verbose_action:
            msg = (
                f"Selling {display_price(amount_base, symbol)} for {display_price(amount_quote, self.symbol)} "
                f"at {display_price(self.assets[symbol].price, self.symbol)}/{symbol} "
                f"(Commission: {display_price(commission_quote, self.symbol)})"
            )
            print(msg)
        # Record the transaction in the ledger for the asset
        self.Ledger.sell(
            timestamp=timestamp,
            symbol=symbol,
            amount=amount_base,
            price=self.assets[symbol].price,
            commission=0,
        )
        # Record the transaction in the ledger for the portfolio currency
        self.Ledger.buy(
            timestamp=timestamp,
            symbol=self.symbol,
            amount=amount_quote,
            price=1,
            commission=commission_quote,
        )
        self.print_portfolio()

    # Portfolio reporting methods ------------------------------------------------

    @property
    def assets_list(self) -> list[str]:
        """
        Property to get the list of assets in the portfolio.

        """
        return list(self.assets.keys())

    @property
    def assets_value(self) -> float:
        """
        Property to get the value of the assets in the portfolio.

        """
        return sum(
            [Currency.balance * Currency.price for Currency in self.assets.values()]
        )

    @property
    def equity_value(self) -> float:
        """
        Property to get the total or equity value of the portfolio.

        """
        return self.balance + self.assets_value

    @property
    def total_commissions(self) -> float:
        """
        Property to get the total commissions paid by the portfolio.

        """
        return self.Ledger.total_commissions

    def transactions_count(self, symbol: Optional[str] = None) -> int:
        """
        Method to count the number of transactions in the ledger.

        If a symbol is provided, it returns the number of transactions for the asset.

        """
        return self.Ledger.transactions_count(symbol)

    def transactions_sum(self, symbol: Optional[str] = None) -> float:
        """
        Method to get the total traded amount in the ledger.

        If a symbol is provided, it returns the total traded amount for the asset.

        """
        return self.Ledger.transactions_sum(symbol)

    def get_balance(self, symbol: Optional[str] = None) -> float:
        """
        Method to get the balance of an asset in the portfolio.

        If no symbol is provided, it returns the balance of the portfolio currency.

        """
        if symbol is None:
            return self.balance
        else:
            return self.assets[symbol].balance

    def get_value(
        self, symbol: Optional[str] = None, quote: Optional[str] = None
    ) -> float:
        """
        Method to get the value of an asset in the portfolio.

        If no symbol is provided, it returns the value of the portfolio currency.

        If a quote is provided, it returns the value of the asset in the specified currency.

        """
        if quote is None or quote == self.symbol:
            price = 1
        else:
            try:
                price = self.assets[quote].price
            except KeyError:
                raise ValueError(f"Currency {quote} not found in the portfolio")
        if symbol is None:
            return self.balance / price
        else:
            return (self.assets[symbol].balance * self.assets[symbol].price) / price

    @property
    def historical_capital(self) -> pd.DataFrame:
        """
        Property to get the capital movements as a DataFrame.

        """
        return self.Ledger.capital_df

    @property
    def ledger_capital(self) -> pd.DataFrame:
        """
        Property to get the capital movements as a DataFrame with readable timestamps.

        """
        capital_df = self.historical_capital
        capital_df["Timestamp"] = pd.to_datetime(capital_df["Timestamp"], unit="ms")
        return capital_df.set_index("Timestamp")

    @property
    def invested_capital(self) -> float:
        """
        Property to get the total investment in the portfolio.

        """
        return self.Ledger.capital_summary["Investment"]

    @property
    def disbursed_capital(self) -> float:
        """
        Property to get the total disbursement in the portfolio.

        """
        return self.Ledger.capital_summary["Disbursement"]

    @property
    def historical_prices(self) -> pd.DataFrame:
        """
        Property to get the historical currency prices as a DataFrame.

        """
        return self.Ledger.prices_df

    @property
    def historical_prices_pivot(self) -> pd.DataFrame:
        """
        Property to pivot the historical currency prices DataFrame.

        This is useful to have the prices of each currency in columns.

        """
        return self.historical_prices.pivot_table(
            index="Timestamp", columns="Symbol", values="Price"
        )

    @property
    def ledger_prices(self) -> pd.DataFrame:
        """
        Property to get the historical currency prices as a DataFrame with readable timestamps.

        """
        prices_df = self.historical_prices_pivot
        prices_df.reset_index(inplace=True)
        prices_df["Timestamp"] = pd.to_datetime(prices_df["Timestamp"], unit="ms")
        prices_df.drop(columns=self.symbol, inplace=True)
        return prices_df.set_index("Timestamp")

    @property
    def historical_transactions(self) -> pd.DataFrame:
        """
        Property to get the historical transactions as a DataFrame.

        The transactions for the portfolio currency are not included as they would be the same as the total traded amount.
        However, these are not filtered in the Ledger because they are needed for the equity calculation.

        """
        transactions_df = self.Ledger.transactions_df
        transactions_df.drop(["Commission"], axis=1, inplace=True)
        return pd.DataFrame(transactions_df[transactions_df.Symbol != self.symbol])

    @property
    def ledger_transactions(self) -> pd.DataFrame:
        """
        Property to get the historical transactions as a DataFrame with readable timestamps.

        """
        transactions_df = self.historical_transactions
        transactions_df["Timestamp"] = pd.to_datetime(
            transactions_df["Timestamp"], unit="ms"
        )
        return transactions_df.set_index("Timestamp")

    @property
    def historical_equity(self) -> pd.DataFrame:
        """
        Property to get the historical equity of the portfolio as a DataFrame.

        """
        return self.Ledger.equity_df

    @property
    def ledger_equity(self) -> pd.DataFrame:
        """
        Property to get the historical equity of the portfolio as a DataFrame with readable timestamps.

        """
        equity_df = self.historical_equity
        equity_df.reset_index(inplace=True)
        equity_df["Timestamp"] = pd.to_datetime(equity_df["Timestamp"], unit="ms")
        return equity_df.set_index("Timestamp")

    @property
    def gains(self) -> float:
        """
        Property to get the gains of the portfolio.

        """
        investment = self.invested_capital
        disbursement = self.disbursed_capital
        return self.equity_value + disbursement - investment

    @property
    def commission_gains_ratio_str(self) -> str:
        gains = self.gains
        if gains > 0:
            ratio = self.total_commissions / gains
            return display_percentage(ratio)
        else:
            return "N/A"

    @property
    def roi(self) -> float:
        """
        Property to get the Return on Investment (ROI) of the portfolio.

        """
        investment = self.invested_capital
        if investment > 0:
            return self.gains / investment
        else:
            return 0.0

    def get_asset_growth(self, symbol: str) -> float:
        """
        Method to get the growth of a currency in the portfolio.

        """
        prices = self.historical_prices_pivot[symbol]
        first_price = prices.iloc[0]
        last_price = prices.iloc[-1]
        return (last_price / first_price) - 1

    def normalize_to_growth(self, series: pd.Series) -> pd.Series:
        """
        Method to normalize a series to the growth of the first non-zero value.

        """
        series_positive = series[series > 0].copy()
        reference = series_positive.iloc[0]
        return (series / reference) - 1

    def resample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method to resample the data to the frequency displayed.

        This is useful to have a more readable plot and less computational cost.

        """
        freq = self.frequency_displayed
        # df.index = pd.to_datetime(df.index, unit="ms")
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index, unit="ms"))
        return df.resample(freq).last()

    def plot_portfolio(self) -> None:
        """
        Method to plot the portfolio equity and the assets' prices over time.

        """
        # Create a figure with multiple subplots
        symbol = self.symbol
        num_plots = len(self.assets_list) + 1
        h_size = num_plots * 5
        fig, ax = plt.subplots(
            nrows=num_plots, ncols=1, figsize=(10, h_size), sharex=True
        )

        # Plot the equity on the first subplot
        historical_equity = self.historical_equity
        resampled_historical_equity = self.resample_data(historical_equity)
        datetime = resampled_historical_equity.index
        total_equity = resampled_historical_equity["Total"]
        label = f"Equity (Gains: {display_price(self.gains, symbol)})"
        ax[0].plot(datetime, total_equity, label=label)
        ax[0].set_ylabel(f"Equity ({symbol})")
        ax[0].set_title("Portfolio Value Over Time")
        ax[0].grid(True)

        # Plot the asset prices on the next ones
        historical_prices = self.historical_prices_pivot
        resampled_historical_prices = self.resample_data(historical_prices)
        ax_counter = 1
        for asset in self.assets_list:
            datetimes = resampled_historical_prices.index
            prices = resampled_historical_prices[asset]
            label = f"{asset} Price (Change: {display_percentage(self.get_asset_growth(asset))})"
            ax[ax_counter].plot(datetimes, prices, label=label)
            ax[ax_counter].set_ylabel(f"Price ({symbol})")
            ax[ax_counter].set_title(f"{asset} Price Over Time")
            ax_counter += 1

        for i in range(num_plots):
            ax[i].set_xlabel("Time")
            ax[i].grid(True)
            # To enable the grid for minor ticks
            ax[i].xaxis.set_minor_locator(AutoMinorLocator())
            ax[i].yaxis.set_minor_locator(AutoMinorLocator())
            ax[i].xaxis.set_major_locator(mdates.AutoDateLocator())
            ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax[i].yaxis.set_major_formatter(FuncFormatter(thousands))
            plt.setp(ax[i].get_xticklabels(), rotation=45, visible=True)
            ax[i].grid(which="both")
            ax[i].grid(which="minor", alpha=0.3)
            ax[i].grid(which="major", alpha=0.5)
            ax[i].legend()

        # Show the plot
        plt.show()

    def plot_benchmark(self) -> None:
        """
        Method to plot the portfolio equity and the assets' change over time.

        """
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(10, 5))

        # Create an Axes object for the figure
        ax = fig.add_subplot(111)

        # Plot the equity on the first subplot
        historical_equity = self.historical_equity
        resampled_historical_equity = self.resample_data(historical_equity)
        equity = self.normalize_to_growth(resampled_historical_equity["Total"])
        datetime = equity.index
        label = f"Equity Change ({display_percentage(self.roi)})"
        ax.plot(datetime, equity, label=label, color="black", linewidth=1)
        # Fill areas where y < 0 with red and y > 0 with green
        ax.fill_between(datetime, equity, where=(equity > 0), color="green", alpha=0.2)  # type: ignore
        ax.fill_between(datetime, equity, where=(equity < 0), color="red", alpha=0.2)  # type: ignore

        # Plot the asset prices on the next ones
        historical_prices = self.historical_prices_pivot
        resampled_historical_prices = self.resample_data(historical_prices)
        for asset in self.assets_list:
            asset_prices = self.normalize_to_growth(resampled_historical_prices[asset])
            datetime = asset_prices.index
            label = (
                f"{asset} Change ({display_percentage(self.get_asset_growth(asset))})"
            )
            ax.plot(datetime, asset_prices, label=label, linewidth=1, alpha=0.8)
        ax.set_xlabel("Time")
        ax.set_ylabel("Growth")
        ax.grid(True)
        # To enable the grid for minor ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
        plt.setp(ax.get_xticklabels(), rotation=45, visible=True)
        ax.grid(which="both")
        ax.grid(which="minor", alpha=0.3)
        ax.grid(which="major", alpha=0.5)
        ax.legend()

        # Show the plot
        plt.show()
