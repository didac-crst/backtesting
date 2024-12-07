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
    get_random_name,
    check_property_update,
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
    transaction_id: int = 0
    displayed_assets: int = 9
    time_chart_resolution: int = 400

    # Portfolio internal methods ------------------------------------------------

    def __post_init__(self) -> None:
        super().__post_init__()
        # EVERYTHING SMALLER THAN 0.1 IS CONSIDERED 0
        # This is useful to avoid displaying assets with a very small balance and trying to sell assets with a very small balance
        self.MINIMAL_BALANCE = 0.1 # QUOTE
        self.set_verbosity("verbose")
        self.set_portfolio_name()
        self.assets = dict()
        self.Ledger = Ledger(portfolio_symbol=self.symbol)
        # We need to keep track of the properties that have been calculated
        self._properties_evolution_id = dict() # This dictionary will store the evolution_id of each property
        self._properties_cached = dict() # This dictionary will store the cached value of each property
    
    def new_transaction_id(self) -> int:
        """
        Method to get a new transaction ID.

        """
        self.transaction_id += 1
        return self.transaction_id
    
    @property
    def evolution_id(self) -> int:
        """
        Property to get the evolution ID of the Ledger.
        
        This is useful to check if the Ledger has new transactions or prices and
        force the recalculation of the properties.

        """
        return self.Ledger.evolution_id
    
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
    @check_property_update
    def assets_table(self) -> str:
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
                "Performance",
                "Gains",
                "Currency growth",
                "Hold gains",
            ]
        ]
        hold_gains = self.hold_gains_assets
        performance_assets = self.performance_assets
        gains_assets = self.gains_assets
        # for asset_symbol in self.assets_list:
        for asset_symbol in self.assets_traded_list:
            Currency = self.assets[asset_symbol]
            # Display only the assets with transactions
            # if self.transactions_count(asset_symbol) > 0:
            data.append(
                [
                    asset_symbol,
                    *Currency.info,
                    display_integer(self.transactions_count(asset_symbol)),
                    display_price(self.transactions_sum(asset_symbol), self.symbol),
                    display_percentage(performance_assets[asset_symbol]),
                    display_price(gains_assets[asset_symbol], self.symbol),
                    display_percentage(self.get_asset_growth(asset_symbol)),
                    display_price(hold_gains[asset_symbol], self.symbol),
                    gains_assets[asset_symbol], # Only for sorting #1
                ]
            )
        return display_pretty_table(data, quote_currency= self.symbol, padding=6, sorting_columns=1)
    
    @property
    @check_property_update
    def text_repr(self) -> str:
        """
        Property to display the portfolio information as a string.
        
        """
        self.empty_negligible_assets()
        text = (
                f"Portfolio ({self.name}):\n"
                f"  -> Symbol = {self.symbol}\n"
                f"  -> Transfer commission = {display_percentage(self.commission_transfer)}\n"
                f"  -> Trade commission = {display_percentage(self.commission_trade)}\n"
                f"  -> Invested capital = {display_price(self.invested_capital, self.symbol)}\n"
                f"  -> Disbursed capital = {display_price(self.disbursed_capital, self.symbol)}\n"
                f"  -> Quote balance = {display_price(self.balance, self.symbol)}\n"
        )
        # If there are no historical prices, it can't display any assets information.
        if self.historical_prices.empty:
            text = (
                "!!! NO DATA ASSETS' PRICES AVAILABLE YET !!!\n"
                "    -> Please update assets' prices.\n\n"
                f"{text}"
            )
        else:
            text = (
                f"{text}"
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
            if len(self.assets_traded_list) > 0:
                text += f"\n{self.assets_table}\n\n"
            # This applies only in case of having updated the prices but not having any transaction yet.
            else:
                text += " None\n\n"
        return text

    def __repr__(self) -> str:
        """
        Dunder method to display the portfolio information as a string.

        """
        return self.text_repr

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
        self.Ledger.record_price(timestamp=timestamp, symbol=self.symbol, price=1.0)
        # Re-enable the verbose_status flag if it was enabled before
        if verbose_status_flag:
            self.verbose_status = True
        self.print_portfolio()
        
    @check_positive
    def deposit(self, amount: float, timestamp: Optional[int] = None) -> None:
        """
        Method to deposit an amount into the portfolio.

        """
        # If the timestamp is not provided, it will be the current time
        # This is only if we needed to test the class without having any data.
        if timestamp is None:
            timestamp = now_ms()
        transaction_id = self.new_transaction_id()
        balance_pre = self.balance
        commission = amount * self.commission_transfer
        # Record the transaction in the ledger for the portfolio currency
        # The transaction is not acknowledged yet
        self.Ledger.buy(
            id=transaction_id,
            timestamp=timestamp,
            trade=False,
            symbol=self.symbol,
            amount=amount,
            price=1,
            commission=commission,
            balance_pre=balance_pre,
        )
        net_amount = amount - commission
        # Here starts the real transaction
        self.commissions_sum += float(commission)
        self.balance += float(net_amount)
        # Record the capital movement in the ledger
        self.Ledger.invest_capital(timestamp=timestamp, amount=amount)
        # Acknowledge the transaction
        self.Ledger.confirm_transaction(transaction_id)
        self.print_portfolio()

    @check_positive
    def withdraw(self, amount: float, relative_amount: bool= False, timestamp: Optional[int] = None) -> None:
        """
        Method to withdraw an amount from the portfolio.

        """
        transaction_id = self.new_transaction_id()
        balance_pre = self.balance
        if relative_amount:
            gross_amount = self.balance * amount
            amount = gross_amount / (1 + self.commission_transfer)
        if timestamp is None:
            timestamp = now_ms()
        commission = amount * self.commission_transfer
        # Record the transaction in the ledger for the portfolio currency
        # The transaction is not acknowledged yet
        self.Ledger.sell(
            id=transaction_id,
            timestamp=timestamp,
            trade=False,
            symbol=self.symbol,
            amount=amount,
            price=1,
            commission=commission,
            balance_pre=balance_pre,
        )
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
        # Here starts the real transaction
        self.commissions_sum += float(commission)
        self.balance -= float(gross_amount)
        # Record the capital movement in the ledger
        self.Ledger.disburse_capital(timestamp=timestamp, amount=amount)
        # Acknowledge the transaction
        self.Ledger.confirm_transaction(transaction_id)
        self.print_portfolio()

    @check_positive
    def buy(
        self, symbol: str, amount_quote: float, timestamp: Optional[int] = None, msg: Optional[str] = None
    ) -> None:
        """
        Method to buy an amount of an asset in the portfolio.

        """
        transaction_id = self.new_transaction_id()
        balance_liquid_pre = self.balance
        balance_asset_pre = self.get_value(symbol=symbol, quote=symbol)
        price = self.assets[symbol].price
        if timestamp is None:
            timestamp = now_ms()
        amount_base = amount_quote / price
        commission_quote = amount_quote * self.commission_trade
        # Record the transaction in the ledger for the asset
        # The transaction is not acknowledged yet
        self.Ledger.buy(
            id=transaction_id,
            timestamp=timestamp,
            trade=True,
            symbol=symbol,
            amount=amount_base,
            price=self.assets[symbol].price,
            commission=0,
            balance_pre=balance_asset_pre
        )
        # Record the transaction in the ledger for the portfolio currency
        # The transaction is not acknowledged yet
        self.Ledger.sell(
            id=transaction_id,
            timestamp=timestamp,
            trade=True,
            symbol=self.symbol,
            amount=amount_quote,
            price=1,
            commission=commission_quote,
            balance_pre=balance_liquid_pre
        )
        # Add message to the transaction
        self.Ledger.add_message_to_transaction(transaction_id, msg)
        gross_amount_quote = amount_quote + commission_quote
        self.check_amount(gross_amount_quote)
        # Here starts the real transaction
        self.balance -= float(gross_amount_quote)
        self.assets[symbol]._deposit(amount_base)
        # Acknowledge the transaction
        self.Ledger.confirm_transaction(transaction_id)
        self.print_portfolio()

    @check_positive
    def sell(
        self, symbol: str, amount_quote: float, timestamp: Optional[int] = None, msg: Optional[str] = None
    ) -> None:
        """
        Method to sell an amount of an asset in the portfolio.

        """
        transaction_id = self.new_transaction_id()
        balance_liquid_pre = self.balance
        balance_asset_pre = self.get_value(symbol=symbol, quote=symbol)
        price = self.assets[symbol].price
        # if relative_amount:
        #     amount = self.get_balance(symbol=symbol) * amount
        if timestamp is None:
            timestamp = now_ms()
        # In case of the amount is higher than the balance, sell the maximum possible
        asset_value_quote = self.get_value(symbol=symbol, quote=self.symbol)
        if amount_quote > asset_value_quote:
            amount_quote = asset_value_quote * 0.9999 # To avoid rounding errors
        amount_base = amount_quote / price
        commission_quote = amount_quote * self.commission_trade
        # Record the transaction in the ledger for the asset
        # The transaction is not acknowledged yet
        self.Ledger.sell(
            id=transaction_id,
            timestamp=timestamp,
            trade=True,
            symbol=symbol,
            amount=amount_base,
            price=self.assets[symbol].price,
            commission=0,
            balance_pre=balance_asset_pre
        )
        # Record the transaction in the ledger for the portfolio currency
        # The transaction is not acknowledged yet
        self.Ledger.buy(
            id=transaction_id,
            timestamp=timestamp,
            trade=True,
            symbol=self.symbol,
            amount=amount_quote,
            price=1,
            commission=commission_quote,
            balance_pre=balance_liquid_pre
        )
        # Add message to the transaction
        self.Ledger.add_message_to_transaction(transaction_id, msg)
        self.check_amount(commission_quote)
        net_amount_quote = amount_quote - commission_quote
        # Here starts the real transaction
        self.assets[symbol]._withdraw(amount_base)
        self.balance += float(net_amount_quote)
        # Acknowledge the transaction
        self.Ledger.confirm_transaction(transaction_id)
        self.print_portfolio()
    
    def empty_negligible_assets(self) -> None:
        """
        Method to empty the assets with a balance smaller than a tenth of MINIMAL_BALANCE.
        
        This is useful to avoid displaying assets with a very small balance and trying to sell assets with a very small balance.

        """
        for symbol in self.assets_list:
            if self.get_value(symbol=symbol, quote=self.symbol) < self.MINIMAL_BALANCE/10:
                self.assets[symbol].balance = 0

    # Portfolio reporting methods ------------------------------------------------
    
    @property
    @check_property_update
    def timerange(self) -> int:
        """
        Property to get the timespan of the portfolio.

        """
        timestamps = self.Ledger.prices_df.Timestamp
        start = timestamps.min()
        end = timestamps.max()
        return (start, end)
    
    @property
    def timespan(self) -> int:
        """
        Property to get the timespan of the portfolio in seconds.

        """
        start, end = self.timerange
        return end - start
    
    @property
    def time_bar_width(self) -> int:
        """
        Property to get the width of the time bars in the chart.

        """
        return int(self.timespan / self.time_chart_resolution) / 25000
    
    @property
    def frequency_resampling(self) -> int:
        """
        Property to get the frequency for resampling the data.

        """
        frequency = int(self.timespan / self.time_chart_resolution)
        return frequency

    @property
    @check_property_update
    def assets_list(self) -> list[str]:
        """
        Property to get the list of assets in the portfolio.
        
        Even if the balance is zero but there are prices, the asset is considered in the list.

        """
        assets_list = list(self.assets.keys())
        assets_list.sort()
        return assets_list
    
    @property
    @check_property_update
    def assets_traded_list(self) -> list[str]:
        """
        Property to get the list of assets in the portfolio with transactions.

        """
        return [symbol for symbol in self.assets_list if self.transactions_count(symbol) > 0]
    
    @property
    @check_property_update
    def positive_balance_assets_list(self) -> list[str]:
        """
        Property to get the list of assets in the portfolio with a positive balance.

        """
        return [symbol for symbol in self.assets_list if self.get_value(symbol=symbol, quote=self.symbol) > self.MINIMAL_BALANCE]
    
    @property
    @check_property_update
    def all_symbols(self) -> list[str]:
        """
        Property to get the list of all symbols in the portfolio.

        """
        return [self.symbol, *self.assets_to_display_list]

    @property
    @check_property_update
    def assets_value(self) -> float:
        """
        Property to get the value of the assets in the portfolio.

        """
        return sum(
                [Currency.value for Currency in self.assets.values()]
            )

    @property
    @check_property_update
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
    def export_assets(self) -> dict[str, float]:
        """
        Property to export the assets of the portfolio in a dictionary.

        """
        self.empty_negligible_assets()
        export_assets = {asset:self.get_value(symbol=asset, quote=self.symbol) \
                        for asset in self.assets if self.assets[asset].balance > 0}
        export_assets[self.symbol] = float(self.balance)
        return export_assets
    
    @property
    @check_property_update
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
    @check_property_update
    def ledger_capital(self) -> pd.DataFrame:
        """
        Property to get the capital movements as a DataFrame with readable timestamps.

        """
        capital_df = self.historical_capital
        capital_df["Timestamp"] = pd.to_datetime(capital_df["Timestamp"], unit="s")
        ledger_capital = capital_df.set_index("Timestamp")
        return ledger_capital

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
    @check_property_update
    def historical_prices_pivot(self) -> pd.DataFrame:
        """
        Property to pivot the historical currency prices DataFrame.

        This is useful to have the prices of each currency in columns.

        """
        return self.historical_prices.pivot_table(
            index="Timestamp", columns="Symbol", values="Price"
        )
    
    @property
    @check_property_update
    def historical_prices_pivot_displayable(self) -> pd.DataFrame:
        """
        Property to get the historical prices pivot of the portfolio as a DataFrame with the share for relevant assets.

        """
        assets_list = self.assets_to_display_list
        assets_list.append(self.symbol)
        assets_list.remove(self.other_assets)
        historical_prices_pivot = self.historical_prices_pivot[assets_list].copy()
        return historical_prices_pivot

    @property
    @check_property_update
    def ledger_prices(self) -> pd.DataFrame:
        """
        Property to get the historical currency prices as a DataFrame with readable timestamps.

        """
        prices_df = self.historical_prices_pivot.copy()
        prices_df.reset_index(inplace=True)
        prices_df["Timestamp"] = pd.to_datetime(prices_df["Timestamp"], unit="s")
        prices_df.drop(columns=self.symbol, inplace=True)
        ledger_prices = prices_df.set_index("Timestamp")
        return ledger_prices
    
    @property
    @check_property_update
    def get_last_prices(self) -> dict[str, float]:
        """
        Method to get the last price of an asset in the portfolio.

        """
        return {symbol: self.assets[symbol].price for symbol in self.assets_list}

    @property
    @check_property_update
    def historical_transactions(self) -> pd.DataFrame:
        """
        Property to get the historical transactions as a DataFrame.

        The transactions for the portfolio currency are not included as they would be the same as the total traded amount.
        However, these are not filtered in the Ledger because they are needed for the equity calculation.

        """
        transactions_df = self.Ledger.transactions_df.copy()
        transactions_df.drop(["Commission"], axis=1, inplace=True)
        return transactions_df[transactions_df.Symbol != self.symbol]

    @property
    @check_property_update
    def ledger_transactions(self) -> pd.DataFrame:
        """
        Property to get the historical transactions as a DataFrame with timestamps.

        """
        transactions_df=self.historical_transactions.copy()
        transactions_df.loc[transactions_df['Action']=='SELL', 'Traded'] = -transactions_df['Traded']
        transactions_pivot = transactions_df.pivot_table(index='Timestamp', columns='Symbol', values='Traded', aggfunc='sum')
        transactions_pivot.fillna(0, inplace=True)
        return transactions_pivot

    @property
    @check_property_update
    def ledger_transactions_datetime(self) -> pd.DataFrame:
        """
        Property to get the historical transactions as a DataFrame with readable timestamps.

        """
        transactions_df = self.ledger_transactions.copy()
        transactions_df.reset_index(inplace=True)
        transactions_df["DateTime"] = pd.to_datetime(transactions_df["Timestamp"], unit="s")
        transactions_df.drop(columns=["Timestamp"], inplace=True)
        return transactions_df.set_index("DateTime")


    @property
    @check_property_update
    def ledger_equity(self) -> pd.DataFrame:
        """
        Property to get the historical equity of the portfolio as a DataFrame with timestamps.

        """
        equity_df = self.Ledger.equity_df.copy()
        equity_df.reset_index(inplace=True)
        return equity_df.set_index("Timestamp")
    
    @property
    @check_property_update
    def ledger_equity_datetime(self) -> pd.DataFrame:
        """
        Property to get the historical equity of the portfolio as a DataFrame with readable timestamps.

        """
        equity_df = self.ledger_equity.copy()
        equity_df.reset_index(inplace=True)
        equity_df["DateTime"] = pd.to_datetime(equity_df["Timestamp"], unit="s")
        equity_df.drop(columns=["Timestamp"], inplace=True)
        return equity_df.set_index("DateTime")
    
    @property
    @check_property_update
    def ledger_equity_share(self) -> pd.DataFrame:
        """
        Property to get the historical equity of the portfolio as a DataFrame with the share of each asset.

        """
        equity_df = self.ledger_equity
        equity_df= equity_df.div(equity_df['Total'], axis=0)
        equity_df.drop(columns=["Total"], inplace=True)
        return equity_df
    
    @property
    @check_property_update
    def ledger_equity_share_displayable(self) -> pd.DataFrame:
        """
        Property to get the historical equity of the portfolio as a DataFrame with the share for relevant assets.

        """
        equity_share = self.ledger_equity_share
        cols = equity_share.columns
        cols_displayable_assets = [c for c in cols if c in self.all_symbols]
        cols_aggregated_assets = [c for c in cols if c not in self.all_symbols]
        displayable_assets = equity_share[cols_displayable_assets].copy()
        aggreted_assets = equity_share[cols_aggregated_assets].sum(axis=1)
        displayable_assets[self.other_assets] = aggreted_assets.copy()
        return displayable_assets

    @property
    @check_property_update
    def gains(self) -> float:
        """
        Property to get the gains of the portfolio.

        """
        return self.equity_value + self.disbursed_capital - self.invested_capital

    @property
    @check_property_update
    def commission_gains_ratio_str(self) -> str:
        """
        Property to get the ratio of the total commissions to the gains of the portfolio.
        
        """
        gains = self.gains
        if gains > 0:
            ratio = self.total_commissions / gains
            return display_percentage(ratio)
        else:
            return "N/A"

    @property
    @check_property_update
    def roi(self) -> float:
        """
        Property to get the Return on Investment (ROI) of the portfolio.

        """
        investment = self.invested_capital
        if investment > 0:
            return (self.gains / investment).round(5)
        else:
            return 0.0
    
    @property
    @check_property_update
    def performance_assets_raw_info(self) -> pd.DataFrame:
        """
        Property to get the information to calculate the performance and the gains for each asset in the portfolio.

        """
        # We get the quote quote of the assets traded (bought and sold) segregated by the symbol
        traded_assets_value = self.Ledger.traded_assets_values
        traded_assets_list = list(traded_assets_value.index)
        # We get the remaining values of the assets in the portfolio
        assets_values = list()
        for symbol in traded_assets_list:
            assets_values.append(self.assets[symbol].value)
        assets_values_df = pd.DataFrame({'asset':traded_assets_list, 'value':assets_values})
        assets_values_df.set_index('asset', inplace=True)
        # We merge the traded assets and the remaining assets
        performance_assets_raw_info = pd.merge(traded_assets_value, assets_values_df, left_index=True, right_index=True, how='outer')
        # We need to count the commissions payed for each transaction
        performance_assets_raw_info['commissions'] = (performance_assets_raw_info['BUY'] + performance_assets_raw_info['SELL']) * self.commission_trade
        return performance_assets_raw_info
    
    @property
    @check_property_update
    def performance_assets(self) -> pd.Series:
        """
        Property to get the performance of the assets in the portfolio.

        """
        assets_raw_info = self.performance_assets_raw_info
        performance_assets = ((assets_raw_info['value'] + assets_raw_info['SELL'] - assets_raw_info['commissions']) / assets_raw_info['BUY'])-1
        return performance_assets
    
    @property
    @check_property_update
    def gains_assets(self) -> pd.Series:
        """
        Property to get the gains of the assets in the portfolio.

        """
        assets_raw_info = self.performance_assets_raw_info
        gains_assets = assets_raw_info['value'] + assets_raw_info['SELL'] - assets_raw_info['commissions'] - assets_raw_info['BUY']
        return gains_assets
    
    def top_performers(self, n: int) -> pd.DataFrame:
        """
        Method to get the top n performers in the portfolio.
        
        """
        assets = self.gains_assets
        return assets.nlargest(n)
    
    def bottom_performers(self, n: int) -> pd.DataFrame:
        """
        Method to get the bottom n performers in the portfolio.
        
        """
        assets = self.gains_assets
        return assets.nsmallest(n)
    
    @property
    @check_property_update
    def historical_theoretical_hold_equity(self) -> pd.Series:
        """
        Property to provide the theoretical gains of holding the assets in the portfolio.

        """
        prices = self.ledger_prices
        # Need to recalculate the equity to have the correct values
        # self.calculate_ledger_equity()
        equity = self.ledger_equity.copy()
        # No price for portfolio currency, set it to 1
        prices[self.symbol] = 1
        equity.drop(columns=["Total"], inplace=True, errors="ignore")
        initial_price = prices.iloc[0]
        initial_assets_quote = equity.iloc[0]
        initial_assets_base = initial_assets_quote / initial_price
        # As it is an equity calculation, we don't need to consider the commissions
        # They are already debited after the first transaction
        historical_assets_quote = initial_assets_base * prices
        historical_theoretical_hold_equity = historical_assets_quote.sum(axis=1)
        return historical_theoretical_hold_equity
    
    @property
    @check_property_update
    def hold_gains_assets(self) -> pd.Series:
        """
        Property to provide the theoretical gains of holding the assets in the portfolio.

        """
        prices = self.ledger_prices
        # Need to recalculate the equity to have the correct values
        # self.calculate_ledger_equity()
        equity = self.ledger_equity.copy()
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
        return hold_gains
    
    @property
    @check_property_update
    def hold_gains(self) -> float:
        """
        Property to provide the sum of the theoretical gains of holding the assets in the portfolio.

        """
        return self.hold_gains_assets.sum()

    @property
    @check_property_update
    def hold_roi(self) -> float:
        """
        Property to provide the Return on Investment (ROI) of theoretical holding the assets in the portfolio.

        """
        return (self.hold_gains / self.invested_capital).round(5)
    
    @property
    @check_property_update
    def roi_vs_hold_roi(self) -> float:
        """
        Property to provide the difference between the ROI and the theoretical holding ROI.

        """
        return self.roi - self.hold_roi


    def get_asset_growth(self, symbol: str) -> float:
        """
        Method to get the growth of a currency in the portfolio.

        """
        prices = self.historical_prices_pivot[symbol].copy()
        first_price = prices.iloc[0]
        last_price = prices.iloc[-1]
        return (last_price / first_price) - 1

    def normalize_to_growth(self, series: pd.Series) -> pd.Series:
        """
        Method to normalize a series to the growth of the first non-zero value.

        """
        series_positive = series[series > 0].copy()
        reference = series_positive.iloc[0]
        return (series / reference) -1
    
    # def log_returns(self, series: pd.Series) -> pd.Series:
    #     """
    #     Method to calculate the log returns of a series.

    #     """
    #     series_positive = series[series > 0].copy()
    #     reference = series_positive.iloc[0]
    #     return np.log(series) - np.log(reference)

    def resample_data(self, df: pd.DataFrame, type: Literal['last', 'mean', 'sum'], factor: int = 1) -> pd.DataFrame:
        """
        Method to resample the data to the frequency displayed.

        This is useful to have a more readable plot and less computational cost.
        
        The factor is used mainly when displaying bars in the chart.
        In that case we want a smaller resolution to see the bars.

        """
        # COMMENTED BECAUSE I WANT TO TEST PLOTTING WITH THE ORIGINAL DATA
        freq = int(self.frequency_resampling * factor)
        freq_str = f"{freq}s"
        # df.index = pd.to_datetime(df.index, unit="s")
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index, unit="s"))
        if type == 'mean':
            return df.resample(freq_str).mean()
        elif type == 'last':
            return df.resample(freq_str).last()
        elif type == 'sum':
            return df.resample(freq_str).sum()
        return df

    # Portfolio logging methods ------------------------------------------------
    
    def print_logs(self, ids: Optional[list[int]]) -> None:
        """
        Method to print the log entries for a list of ids.

        """
        self.Ledger.print_logs(ids)
        
    
    def print_logs_timestamp(self, timestamp: int) -> None:
        """
        Method to print all the log entries for a specific timestamp.
        
        """
        self.Ledger.print_logs_timestamp(timestamp)
    
    def print_logs_symbol(self, symbol: str) -> None:
        """
        Method to print all the log entries for a specific symbol.
            
        """
        self.Ledger.print_logs_symbol(symbol)
    
    def print_last_log(self) -> None:
        """
        Method to print the last log entry.
            
        """
        self.Ledger.print_last_log()
    
    def print_logs_all(self) -> None:
        """
        Method to print all the log entries.
            
        """
        self.Ledger.print_logs_all()

    
    # Portfolio ploting methods ------------------------------------------------

    def plot_portfolio(self) -> None:
        """
        Method to plot the portfolio equity and the assets' prices over time.

        """
        # Create a figure with multiple subplots
        symbol = self.symbol
        assets_list = ["Total"] + self.assets_traded_list
        num_plots = len(assets_list)
        h_size = num_plots * 5
        fig, ax = plt.subplots(
            nrows=num_plots, ncols=1, figsize=(15, h_size), sharex=True
        )
        ax2 = dict()
        # Plot the equity on the first subplot
        historical_equity = self.ledger_equity_datetime
        resampled_historical_equity = self.resample_data(historical_equity, type='last')
        # datetime = resampled_historical_equity.index
        # total_equity = resampled_historical_equity["Total"]
        # label = f"Equity (Gains: {display_price(self.gains, symbol)})"
        # ax[0].plot(datetime, total_equity, label=label)
        # ax[0].set_ylabel(f"Equity ({symbol})")
        # ax[0].set_title("Portfolio Value Over Time")
        # ax[0].grid(True)
        # ax_counter_value_magnitude[0] = np.max(total_equity)
        # # Plot the asset prices on the next ones
        historical_prices = self.historical_prices_pivot.copy()
        resampled_historical_prices = self.resample_data(historical_prices, type='last')
        ax_counter = 0
        for asset in assets_list:
            # Ploting the equity of the asset
            historical_equity = self.ledger_equity_datetime
            resampled_historical_equity = self.resample_data(historical_equity, type='last')
            datetime = resampled_historical_equity.index
            total_equity = resampled_historical_equity[asset]
            # Different settings for the total and the assets
            if asset == "Total":
                alpha = 1
            else:
                alpha = 0.1
                ax[ax_counter].fill_between(datetime, total_equity, where=(total_equity > 0), color="green", alpha=0.2)  # type: ignore
            ax[ax_counter].plot(datetime, total_equity, color="green", alpha=alpha)
            ax[ax_counter].set_ylabel(f"Equity ({symbol})")
            ax[ax_counter].set_xlabel("Time")
            ax[ax_counter].grid(True)
            # To enable the grid for minor ticks
            ax[ax_counter].xaxis.set_minor_locator(AutoMinorLocator())
            ax[ax_counter].yaxis.set_minor_locator(AutoMinorLocator())
            ax[ax_counter].xaxis.set_major_locator(mdates.AutoDateLocator())
            ax[ax_counter].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
            # Format to thousands the y-axis only if the maximal value is higher than 1000
            if total_equity.max() > 1000:
                ax[ax_counter].yaxis.set_major_formatter(FuncFormatter(thousands))
            # Total doesn't have a price
            if asset != "Total":
                # Create a secondary y-axis
                ax2[ax_counter] = ax[ax_counter].twinx()
                # Ploting the price of the asset
                datetimes = resampled_historical_prices.index
                prices = resampled_historical_prices[asset]
                ax2[ax_counter].plot(datetimes, prices, color="red")
                ax2[ax_counter].set_ylabel(f"Price ({symbol})")
                # Format to thousands the y-axis only if the maximal value is higher than 1000
                if prices.max() > 1000:
                    ax[ax_counter].yaxis.set_major_formatter(FuncFormatter(thousands))
            # Display different title for the total and the assets
            if asset == "Total":
                info = f"(Gains: {display_price(self.gains, symbol)} | ROI: {display_percentage(self.roi)})"
                ax[ax_counter].set_title(f"Portfolio Value Over Time {info}")
            else:
                info = f"(Price Growth: {display_percentage(self.get_asset_growth(asset))})"
                ax[ax_counter].set_title(f"Price Over Time [{asset}] {info}")
            ax[ax_counter].grid(which="both")
            ax[ax_counter].grid(which="minor", alpha=0.3)
            ax[ax_counter].grid(which="major", alpha=0.5)
            # ax[ax_counter].legend(fontsize='small', loc='upper left', bbox_to_anchor=(1, 1))
            plt.setp(ax[ax_counter].get_xticklabels(), rotation=45, visible=True)
            ax_counter += 1

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
        
        label_colors = self.label_colors.copy()
        ax.axhline(y=0, color='black', linewidth=1)

        # EQUITY
        historical_equity = self.Ledger.equity_df
        resampled_historical_equity = self.resample_data(historical_equity, type='mean')
        equity = self.normalize_to_growth(resampled_historical_equity["Total"])
        timestamp = equity.index
        datetime = pd.to_datetime(timestamp, unit="s")
        label = f"Equity ({display_percentage(self.roi)})"
        ax.plot(datetime, equity, label=label, color="black", linewidth=1)
        # Fill areas where y < 0 with red and y > 0 with green
        ax.fill_between(datetime, equity, where=(equity > 0), color="green", alpha=0.1)  # type: ignore
        ax.fill_between(datetime, equity, where=(equity < 0), color="red", alpha=0.1)  # type: ignore

        # THEORETICAL HOLD EQUITY
        theoretical_hold_equity = self.historical_theoretical_hold_equity
        resampled_theoretical_hold_equity = self.resample_data(theoretical_hold_equity, type='mean')
        theoretical_equity = self.normalize_to_growth(resampled_theoretical_hold_equity)
        timestamp = theoretical_equity.index
        datetime = pd.to_datetime(timestamp, unit="s")
        label = f"Hold ({display_percentage(self.hold_roi)})"
        hold_color = label_colors['hold']
        # Plot line
        ax.plot(datetime, theoretical_equity, label=label, linewidth=2, alpha=1, color=hold_color)
        # Plot shadow
        ax.plot(datetime, theoretical_equity, linewidth=6, alpha=0.3, color=hold_color)

        # CURRENCY PRICES
        historical_prices = self.historical_prices_pivot_displayable.copy()
        resampled_historical_prices = self.resample_data(historical_prices, type='mean')
        for asset in resampled_historical_prices.columns:
            label = (
                f"{asset} ({display_percentage(self.get_asset_growth(asset))})"
            )
            asset_prices = self.normalize_to_growth(resampled_historical_prices[asset])
            timestamp = asset_prices.index
            datetime = pd.to_datetime(timestamp, unit="s")
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
        # ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
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
            # self.calculate_ledger_equity()

        label_colors = self.label_colors.copy()
        equity_share_df = self.ledger_equity_share_displayable.copy()
        resampled_equity_share_df = self.resample_data(equity_share_df, type='mean')
        # Reverse the columns to do the cumsum in the correct order
        # Reorder the columns to have the portfolio currency at the end
        columns = list(equity_share_df.columns)
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
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
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
            # self.calculate_ledger_equity()

        label_colors = self.label_colors.copy()

        ax.axhline(y=0, color='black', linewidth=1)

        color_hold = label_colors['hold']
        color_symbol = label_colors[self.symbol]

        # Non liquid assets
        equity = self.ledger_equity_datetime.copy()
        non_liquid = equity['Total'] - equity[self.symbol]
        non_liquid_df = pd.DataFrame(non_liquid, columns=['Total'])
        non_liquid_df = self.resample_data(non_liquid_df, type='last')
        datetime = non_liquid_df.index
        assets_value = non_liquid_df["Total"]
        ax.plot(datetime, assets_value, color=color_hold, alpha=1.0, linewidth=1)

        # Equity
        historical_equity = self.ledger_equity_datetime.copy()
        resampled_historical_equity = self.resample_data(historical_equity, type='last')
        datetime = resampled_historical_equity.index
        total_equity = resampled_historical_equity["Total"]
        ax.plot(datetime, total_equity, color=color_symbol, alpha=1.0, linewidth=1)

        # Fill where the differences
        ax.fill_between(datetime, assets_value, color=color_hold, alpha=0.2, label='Assets')
        ax.fill_between(datetime, assets_value, total_equity, color=color_symbol, alpha=0.2, label='Liquid')

        # Need to recalculate the equity to have the correct values
        transactions_df = self.ledger_transactions_datetime.copy()
        # Resample buys and sells to have a better visualization
        # It is needed to split buys and sells to have a better visualization
        # When resampling buys and sells they would cancel each other in the same sampling window
        resample_type = 'sum'
        factor = 5
        buys_df = transactions_df[transactions_df > 0].fillna(0)
        resampled_buys_df = self.resample_data(buys_df, type=resample_type, factor=factor)
        sells_df = transactions_df[transactions_df < 0].fillna(0)
        resampled_sells_df = self.resample_data(sells_df, type=resample_type, factor=factor)
        # Initialize a variable to keep track of the bottom position for buys
        bottoms_buy = np.zeros(len(resampled_buys_df))
        # Initialize a variable to keep track of the bottom position for sells
        bottoms_sell = np.zeros(len(resampled_sells_df))
        # We only want to display the top and bottom performers
        for column in self.assets_traded_list:
            if column in self.assets_to_display_list:
                color = label_colors[column]
                # Plot the bar chart buys
                ax.bar(resampled_buys_df.index, resampled_buys_df[column], label=column, bottom=bottoms_buy, width=self.time_bar_width, color=color, alpha=1.0, edgecolor='black')
                # Plot the bar chart sells
                ax.bar(resampled_sells_df.index, resampled_sells_df[column], bottom=bottoms_sell, width=self.time_bar_width, color=color, alpha=1.0, edgecolor='black')
                # Update bottoms for buys and sells separately if needed
                bottoms_buy += resampled_buys_df[column]
                bottoms_sell += resampled_sells_df[column]
        # For the other assets, we aggregate them
        assets_to_aggregate = [asset for asset in self.assets_traded_list if asset not in self.assets_to_display_list]
        resampled_buys_agg_df = resampled_buys_df[assets_to_aggregate].sum(axis=1)
        resampled_sells_agg_df = resampled_sells_df[assets_to_aggregate].sum(axis=1)
        color = label_colors[self.other_assets]
        # Plot the bar chart buys
        ax.bar(resampled_buys_agg_df.index, resampled_buys_agg_df, label=self.other_assets, bottom=bottoms_buy, width=self.time_bar_width, color=color, alpha=1.0, edgecolor='black')
        # Plot the bar chart sells
        ax.bar(resampled_sells_agg_df.index, resampled_sells_agg_df, bottom=bottoms_sell, width=self.time_bar_width, color=color, alpha=1.0, edgecolor='black')
        ax.set_title("Transactions Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel(f"Value / Amount ({self.symbol})")
        ax.grid(True)
        # To enable the grid for minor ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        ax.yaxis.set_major_formatter(FuncFormatter(thousands))
        plt.setp(ax.get_xticklabels(), rotation=45, visible=True)
        ax.grid(which="both")
        ax.grid(which="minor", alpha=0.3)
        ax.grid(which="major", alpha=0.5)
        ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1, 1))

        if fig_ax is None:
            # Show the plot
            plt.show()
    
    @property
    def assets_to_display_list(self) -> None:
        """
        Method to create a consistent list of assets to display in the plots.

        """
        top_performers = self.top_performers(n=self.displayed_assets).index.tolist()
        bottom_performers = self.bottom_performers(n=self.displayed_assets).index.tolist()
        self.other_assets = 'Other Assets'
        assets_to_display_list = top_performers + bottom_performers + [self.other_assets]
        return assets_to_display_list
    
    @property
    @check_property_update
    def label_colors(self) -> None:
        hold = 'hold'
        labels = [self.symbol, hold, *self.assets_to_display_list]
        cmap = plt.get_cmap("tab20")
        colors = {label: cmap(i) for i, label in enumerate(labels)}
        return colors

    def plot_summary(self) -> None:
        """
        Method to plot the portfolio summary.

        """
        # Create an Axes object for the figure
        fig, ax = plt.subplots(
            nrows=3, ncols=1, figsize=(15, 15), sharex=True
        )

        # Need to recalculate the equity to have the correct values
        # self.calculate_ledger_equity()

        self.plot_benchmark((fig, ax[0]))
        self.plot_assets_share((fig, ax[1]))
        self.plot_transactions((fig, ax[2]))

        plt.show()