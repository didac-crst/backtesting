from dataclasses import dataclass
from typing import Optional, Literal, Union

import pandas as pd
import numpy as np
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
    get_random_name
)

VerboseType = Literal["silent", "action", "status", "verbose"]


@dataclass
class Portfolio(Asset):
    """
    Dataclass containing the structure of a portfolio.

    """

    name: str = ""
    commission_trade: float = 0.0
    commission_transfer: float = 0.0
    frequency_displayed: str = "1h"

    # Portfolio internal methods ------------------------------------------------

    def __post_init__(self) -> None:
        super().__post_init__()
        self.set_verbosity("verbose")
        self.set_portfolio_name()
        self.assets = dict()
        self.Ledger = Ledger(portfolio_symbol=self.symbol)
    
    def set_portfolio_name(self) -> None:
        """
        Method to fill the name of the portfolio if it is empty.

        """
        if self.name == "":
            self.name = get_random_name()

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
                "Hold gains",
            ]
        ]
        hold_gains = self._hold_gains_assets
        for asset_symbol in self.assets_list:
            Currency = self.assets[asset_symbol]
            data.append(
                [
                    asset_symbol,
                    *Currency.values,
                    display_integer(self.transactions_count(asset_symbol)),
                    display_price(self.transactions_sum(asset_symbol), self.symbol),
                    display_percentage(self.get_asset_growth(asset_symbol)),
                    display_price(hold_gains[asset_symbol], self.symbol),
                ]
            )
        return display_pretty_table(data, padding=6)

    def __repr__(self) -> str:
        """
        Dunder method to display the portfolio information as a string.

        """
        self.calculate_hold_gains_assets()
        text = (
            f"Portfolio ({self.name}):\n"
            f"  -> Symbol = {self.symbol}\n"
            f"  -> Transfer commission = {display_percentage(self.commission_transfer)}\n"
            f"  -> Trade commission = {display_percentage(self.commission_trade)}\n"
            f"  -> Invested capital = {display_price(self.invested_capital, self.symbol)}\n"
            f"  -> Disbursed capital = {display_price(self.disbursed_capital, self.symbol)}\n"
            f"  -> Quote balance = {display_price(self.balance, self.symbol)}\n"
            f"  -> Assets value = {display_price(self.assets_value, self.symbol)}\n"
            f"  -> Equity value = {display_price(self.equity_value, self.symbol)}\n"
            f"  -> Transactions = {display_integer(self.transactions_count())}\n"
            f"  -> Total traded = {display_price(self.transactions_sum(), self.symbol)}\n"
            f"  -> Gains = {display_price(self.gains, self.symbol)}\n"
            f"  -> Total commissions = {display_price(self.total_commissions, self.symbol)}\n"
            f"  -> Commission gains ratio = {self.commission_gains_ratio_str}\n"
            f"  -> ROI = {display_percentage(self.roi)}\n"
            f"  -> Hold Gains (Theoretical) = {display_price(self.hold_gains, self.symbol)}\n"
            f"  -> Hold ROI (Theoretical) = {display_percentage(self.hold_roi)}\n"
            f"  -> ROI Performance (vs Hold) = {display_percentage(self.roi_vs_hold_roi)}\n"
            f"  -> Assets:"
        )
        if len(self.assets) > 0:
            text += f"\n{self._assets_table}\n\n"
        else:
            text += " None\n\n"
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
        self.create_consistent_colors_labels()

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
    def withdraw(self, amount: float, relative_amount: bool= False, timestamp: Optional[int] = None) -> None:
        """
        Method to withdraw an amount from the portfolio.

        """
        if relative_amount:
            gross_amount = self.balance * amount
            amount = gross_amount / (1 + self.commission_transfer)
        if timestamp is None:
            timestamp = now_ms()
        commission = amount * self.commission_transfer
        gross_amount = amount + commission
        try:
            self.check_amount(gross_amount)
        except ValueError as e:
            # Due to some calculations it could happen that the amount is slightly higher than the balance
            print('Exception')
            if gross_amount < self.balance * 1.00001:
                gross_amount = self.balance
            # If the amount is still higher, raise the error
            else:
                raise e
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
        self, symbol: str, amount_quote: float, timestamp: Optional[int] = None
    ) -> None:
        """
        Method to buy an amount of an asset in the portfolio.

        """
        price = self.assets[symbol].price
        if timestamp is None:
            timestamp = now_ms()
        amount_base = amount_quote / price
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
        self, symbol: str, amount_quote: float, timestamp: Optional[int] = None
    ) -> None:
        """
        Method to sell an amount of an asset in the portfolio.

        """
        price = self.assets[symbol].price
        # if relative_amount:
        #     amount = self.get_balance(symbol=symbol) * amount
        if timestamp is None:
            timestamp = now_ms()
        amount_base = amount_quote / price
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
        assets_list = list(self.assets.keys())
        assets_list.sort()
        return assets_list
    
    @property
    def all_symbols(self) -> list[str]:
        """
        Property to get the list of all symbols in the portfolio.

        """
        return [self.symbol, *self.assets_list]

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
    def liquidity_ratio(self) -> float:
        """
        Property to get the liquidity ratio of the portfolio.

        """
        return self.balance / self.equity_value

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
    def get_last_prices(self) -> dict[str, float]:
        """
        Method to get the last price of an asset in the portfolio.

        """
        return {symbol: self.assets[symbol].price for symbol in self.assets_list}

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
        transactions_df=self.historical_transactions
        transactions_df["Timestamp"] = pd.to_datetime(
            transactions_df["Timestamp"], unit="ms"
        )
        transactions_df.loc[transactions_df['Action']=='SELL', 'Traded'] = -transactions_df['Traded']
        transactions_pivot = transactions_df.pivot_table(index='Timestamp', columns='Symbol', values='Traded', aggfunc='sum')
        transactions_pivot.fillna(0, inplace=True)
        return transactions_pivot

    def calculate_historical_equity(self) -> None:
        """
        Method to calculate the historical equity of the portfolio.

        """
        self._historical_equity = self.Ledger.equity_df

    @property
    def historical_equity(self) -> pd.DataFrame:
        """
        Property to get the historical equity of the portfolio as a DataFrame.

        """
        self.calculate_historical_equity()
        return self._historical_equity
    
    def calculate_ledger_equity(self) -> None:
        """
        Method to calculate the historical equity of the portfolio.

        """
        self.calculate_historical_equity()
        equity_df = self._historical_equity.copy()
        equity_df.reset_index(inplace=True)
        equity_df["Timestamp"] = pd.to_datetime(equity_df["Timestamp"], unit="ms")
        self._ledger_equity = equity_df.set_index("Timestamp")

    @property
    def ledger_equity(self) -> pd.DataFrame:
        """
        Property to get the historical equity of the portfolio as a DataFrame with readable timestamps.

        """
        self.calculate_ledger_equity()
        return self._ledger_equity
    
    @property
    def ledger_equity_share(self) -> pd.DataFrame:
        """
        Property to get the historical equity of the portfolio as a DataFrame with the share of each asset.

        """
        equity_df = self._ledger_equity
        equity_df= equity_df.div(equity_df['Total'], axis=0)
        equity_df.drop(columns=["Total"], inplace=True)
        return equity_df

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
            return (self.gains / investment).round(5)
        else:
            return 0.0

    def calculate_historical_theoretical_hold_equity(self) -> None:
        """
        Method to calculate the theoretical gains of holding the assets in the portfolio.

        """
        prices = self.ledger_prices
        # Need to recalculate the equity to have the correct values
        self.calculate_ledger_equity()
        equity = self._ledger_equity.copy()
        # No price for portfolio currency, set it to 1
        prices[self.symbol] = 1
        equity.drop(columns=["Total"], inplace=True, errors="ignore")
        initial_price = prices.iloc[0]
        initial_assets_quote = equity.iloc[0]
        initial_assets_base = initial_assets_quote / initial_price
        # As it is an equity calculation, we don't need to consider the commissions
        # They are already debited after the first transaction
        historical_assets_quote = initial_assets_base * prices
        self._historical_theoretical_hold_equity = historical_assets_quote.sum(axis=1)
    
    @property
    def historical_theoretical_hold_equity(self) -> pd.Series:
        """
        Property to provide the theoretical gains of holding the assets in the portfolio.

        """
        self.calculate_historical_theoretical_hold_equity()
        return self._historical_theoretical_hold_equity

    def calculate_hold_gains_assets(self) -> pd.Series:
        """
        Method to calculate the theoretical gains of holding the assets in the portfolio.

        """
        prices = self.ledger_prices
        # Need to recalculate the equity to have the correct values
        self.calculate_ledger_equity()
        equity = self._ledger_equity.copy()
        if (len(prices) > 0) & (len(equity) > 0):
            # No price for portfolio currency, set it to 1
            prices[self.symbol] = 1
            equity.drop(columns=["Total"], inplace=True, errors="ignore")
            initial_assets_quote = equity.iloc[0]
            initial_prices = prices.iloc[0]
            final_prices = prices.iloc[-1]
            initial_assets_base = (initial_assets_quote / initial_prices)
            # Initial commissions can't be extracted from total_commissions
            # When trading, total commissions increases
            commissions = self.invested_capital - initial_assets_quote.sum()
            # Commissions are only applied to the portfolio currency at the initial point
            initial_assets_quote[self.symbol] = initial_assets_quote[self.symbol] + commissions
            # Final commissions are not part of the final assets
            final_assets_quote = initial_assets_base * final_prices
            hold_gains = final_assets_quote - initial_assets_quote
        else:
            assets_list = [self.symbol, *self.assets_list]
            hold_gains = pd.Series([0.0] * len(assets_list), index=assets_list)
        self._hold_gains_assets = hold_gains
    
    @property
    def hold_gains_assets(self) -> pd.Series:
        """
        Property to provide the theoretical gains of holding the assets in the portfolio.

        """
        self.calculate_hold_gains_assets()
        return self._hold_gains_assets
    
    @property
    def hold_gains(self) -> float:
        """
        Property to provide the sum of the theoretical gains of holding the assets in the portfolio.

        """
        return self._hold_gains_assets.sum()

    @property
    def hold_roi(self) -> float:
        """
        Property to provide the Return on Investment (ROI) of theoretical holding the assets in the portfolio.

        """
        return (self.hold_gains / self.invested_capital).round(5)
    
    @property
    def roi_vs_hold_roi(self) -> float:
        """
        Property to provide the difference between the ROI and the theoretical holding ROI.

        """
        return self.roi - self.hold_roi


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

    def resample_data(self, df: pd.DataFrame, type: Literal['last', 'mean', 'sum'], freq: Optional[str] = None) -> pd.DataFrame:
        """
        Method to resample the data to the frequency displayed.

        This is useful to have a more readable plot and less computational cost.

        """
        if freq is None:
            freq = self.frequency_displayed
        # df.index = pd.to_datetime(df.index, unit="ms")
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index, unit="ms"))
        if type == 'mean':
            return df.resample(freq).mean()
        elif type == 'last':
            return df.resample(freq).last()
        elif type == 'sum':
            return df.resample(freq).sum()

    def plot_portfolio(self) -> None:
        """
        Method to plot the portfolio equity and the assets' prices over time.

        """
        # Create a figure with multiple subplots
        self.calculate_historical_equity()
        symbol = self.symbol
        num_plots = len(self.assets_list) + 1
        h_size = num_plots * 5
        fig, ax = plt.subplots(
            nrows=num_plots, ncols=1, figsize=(15, h_size), sharex=True
        )

        # Plot the equity on the first subplot
        historical_equity = self._historical_equity
        resampled_historical_equity = self.resample_data(historical_equity, type='last')
        datetime = resampled_historical_equity.index
        total_equity = resampled_historical_equity["Total"]
        label = f"Equity (Gains: {display_price(self.gains, symbol)})"
        ax[0].plot(datetime, total_equity, label=label)
        ax[0].set_ylabel(f"Equity ({symbol})")
        ax[0].set_title("Portfolio Value Over Time")
        ax[0].grid(True)

        # Plot the asset prices on the next ones
        historical_prices = self.historical_prices_pivot
        resampled_historical_prices = self.resample_data(historical_prices, type='last')
        ax_counter = 1
        for asset in self.assets_list:
            datetimes = resampled_historical_prices.index
            prices = resampled_historical_prices[asset]
            label = f"{asset} Price ({display_percentage(self.get_asset_growth(asset))})"
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
            ax[i].legend(fontsize='small', loc='upper left', bbox_to_anchor=(1, 1))

        # Show the plot
        plt.show()

    def plot_benchmark(self, fig_ax: Optional[tuple] = None) -> None:
        """
        Method to plot the portfolio equity and the assets' change over time.

        """
        if fig_ax is not None:
            fig, ax = fig_ax
        else:
            # Create a figure with multiple subplots
            fig = plt.figure(figsize=(15, 5))
            # Create an Axes object for the figure
            ax = fig.add_subplot(111)
            # Need to recalculate the equity to have the correct values
            self.calculate_historical_equity()
        
        label_colors = self.label_colors
        ax.axhline(y=0, color='black', linewidth=1)

        # EQUITY
        historical_equity = self._historical_equity
        resampled_historical_equity = self.resample_data(historical_equity, type='mean')
        equity = self.normalize_to_growth(resampled_historical_equity["Total"])
        datetime = equity.index
        label = f"Equity ({display_percentage(self.roi)})"
        ax.plot(datetime, equity, label=label, color="black", linewidth=1)
        # Fill areas where y < 0 with red and y > 0 with green
        ax.fill_between(datetime, equity, where=(equity > 0), color="green", alpha=0.1)  # type: ignore
        ax.fill_between(datetime, equity, where=(equity < 0), color="red", alpha=0.1)  # type: ignore

        # THEORETICAL HOLD EQUITY
        # Need to recalculate the theoretical hold equity to have the correct values
        self.calculate_historical_theoretical_hold_equity()
        theoretical_hold_equity = self._historical_theoretical_hold_equity
        resampled_theoretical_hold_equity = self.resample_data(theoretical_hold_equity, type='mean')
        theoretical_equity = self.normalize_to_growth(resampled_theoretical_hold_equity)
        datetime = theoretical_equity.index
        label = f"Hold ({display_percentage(self.hold_roi)})"
        hold_color = label_colors['hold']
        # Plot line
        ax.plot(datetime, theoretical_equity, label=label, linewidth=2, alpha=1, color=hold_color)
        # Plot shadow
        ax.plot(datetime, theoretical_equity, linewidth=6, alpha=0.3, color=hold_color)

        # CURRENCY PRICES
        historical_prices = self.historical_prices_pivot
        resampled_historical_prices = self.resample_data(historical_prices, type='mean')
        for asset in self.assets_list:
            label = (
                f"{asset} ({display_percentage(self.get_asset_growth(asset))})"
            )
            asset_prices = self.normalize_to_growth(resampled_historical_prices[asset])
            datetime = asset_prices.index
            color = label_colors[asset]
            # Plot line
            ax.plot(datetime, asset_prices, label=label, linewidth=1, alpha=1, color=color)
            # Plot shadow
            ax.plot(datetime, asset_prices, linewidth=4, alpha=0.3, color=color)
        ax.set_title("Equity and Currency Change Over Time")
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
        ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1, 1))

        if fig_ax is None:
            # Show the plot
            plt.show()
    
    def plot_assets_share(self, fig_ax: Optional[tuple] = None) -> None:
        """
        Method to plot the portfolio equity and the assets' share over time.

        """
        if fig_ax is not None:
            fig, ax = fig_ax
        else:
            # Create a figure with multiple subplots
            fig = plt.figure(figsize=(15, 5))
            # Create an Axes object for the figure
            ax = fig.add_subplot(111)
            # Need to recalculate the equity to have the correct values
            self.calculate_ledger_equity()

        label_colors = self.label_colors
        equity_share_df = self.ledger_equity_share
        resampled_equity_share_df = self.resample_data(equity_share_df, type='mean')
        # Reverse the columns to do the cumsum in the correct order
        # Reorder the columns to have the portfolio currency at the end
        columns = self.all_symbols
        columns.reverse()
        resampled_equity_share_df = resampled_equity_share_df[columns]
        resampled_equity_share_df_cumsum = resampled_equity_share_df.cumsum(axis=1)
        # Reverse the columns to have a proper display
        columns.reverse()
        # columns = resampled_equity_share_df_cumsum.columns
        for column in columns:
            color = label_colors[column]
            ax.fill_between(resampled_equity_share_df_cumsum.index, resampled_equity_share_df_cumsum[column], label=column, color=color, alpha=1.0, edgecolor='black')
        ax.set_title("Equity Share Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Equity Share")
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
        ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1, 1))

        if fig_ax is None:
            # Show the plot
            plt.show()
    
    def plot_transactions(self, fig_ax: Optional[tuple] = None) -> None:
        """
        Method to plot the historical transactions of the portfolio.

        """
        if fig_ax is not None:
            fig, ax = fig_ax
        else:
            # Create a figure with multiple subplots
            fig = plt.figure(figsize=(15, 5))
            # Create an Axes object for the figure
            ax = fig.add_subplot(111)   
            # Need to recalculate the equity to have the correct values
            self.calculate_ledger_equity()

        label_colors = self.label_colors

        ax.axhline(y=0, color='black', linewidth=1)

        color_hold = self.label_colors['hold']
        color_symbol = self.label_colors[self.symbol]

        # Non liquid assets
        equity = self._ledger_equity.copy()
        non_liquid = equity['Total'] - equity[self.symbol]
        non_liquid_df = pd.DataFrame(non_liquid, columns=['Total'])
        non_liquid_df = self.resample_data(non_liquid_df, type='last')
        datetime = non_liquid_df.index
        assets_value = non_liquid_df["Total"]
        ax.plot(datetime, assets_value, color=color_hold, alpha=1.0, linewidth=1)

        # Equity
        historical_equity = self._historical_equity
        resampled_historical_equity = self.resample_data(historical_equity, type='last')
        datetime = resampled_historical_equity.index
        total_equity = resampled_historical_equity["Total"]
        ax.plot(datetime, total_equity, color=color_symbol, alpha=1.0, linewidth=1)

        # Fill where the differences
        ax.fill_between(datetime, assets_value, color=color_hold, alpha=0.2, label='Assets')
        ax.fill_between(datetime, assets_value, total_equity, color=color_symbol, alpha=0.2, label='Liquid')

        # Need to recalculate the equity to have the correct values
        transactions_df = self.ledger_transactions
        # Resample buys and sells to have a better visualization
        # It is needed to split buys and sells to have a better visualization
        # When resampling buys and sells they would cancel each other in the same sampling window
        resample_type = 'sum'
        resample_freq = '6h'
        buys_df = transactions_df[transactions_df > 0].fillna(0)
        resampled_buys_df = self.resample_data(buys_df, type=resample_type, freq=resample_freq)
        sells_df = transactions_df[transactions_df < 0].fillna(0)
        resampled_sells_df = self.resample_data(sells_df, type=resample_type, freq=resample_freq)
        # Initialize a variable to keep track of the bottom position for buys
        bottoms_buy = np.zeros(len(resampled_buys_df))
        # Initialize a variable to keep track of the bottom position for sells
        bottoms_sell = np.zeros(len(resampled_sells_df))
        # Set the width of the bars
        bars_width = 0.23
        for column in self.assets_list:
            color = label_colors[column]
            # Plot the bar chart buys
            ax.bar(resampled_buys_df.index, resampled_buys_df[column], label=column, bottom=bottoms_buy, width=bars_width, color=color, alpha=1.0, edgecolor='black')
            # Plot the bar chart sells
            ax.bar(resampled_sells_df.index, resampled_sells_df[column], bottom=bottoms_sell, width=bars_width, color=color, alpha=1.0, edgecolor='black')
            
            # Update bottoms for buys and sells separately if needed
            bottoms_buy += resampled_buys_df[column]
            bottoms_sell += resampled_sells_df[column]

        ax.set_title("Transactions Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel(f"Value / Amount ({self.symbol})")
        ax.grid(True)
        # To enable the grid for minor ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.yaxis.set_major_formatter(FuncFormatter(thousands))
        plt.setp(ax.get_xticklabels(), rotation=45, visible=True)
        ax.grid(which="both")
        ax.grid(which="minor", alpha=0.3)
        ax.grid(which="major", alpha=0.5)
        ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1, 1))

        if fig_ax is None:
            # Show the plot
            plt.show()
    
    def create_consistent_colors_labels(self) -> None:
        hold = 'hold'
        labels = [self.symbol, hold, *self.assets_list]
        cmap = plt.get_cmap("Set2")
        colors = {label: cmap(i) for i, label in enumerate(labels)}
        self.label_colors = colors

    def plot_summary(self) -> None:
        """
        Method to plot the portfolio summary.

        """
        # Create an Axes object for the figure
        fig, ax = plt.subplots(
            nrows=3, ncols=1, figsize=(15, 15), sharex=True
        )

        # Need to recalculate the equity to have the correct values
        self.calculate_ledger_equity()

        self.plot_benchmark((fig, ax[0]))
        self.plot_assets_share((fig, ax[1]))
        self.plot_transactions((fig, ax[2]))

        plt.show()