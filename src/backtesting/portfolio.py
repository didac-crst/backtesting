from dataclasses import dataclass
import os
from typing import Optional, Literal, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, LogLocator, LogFormatter
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

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
    to_percent_log_growth,
    display_pretty_table,
    check_property_update,
    move_to_end,
    max_2_numbers,
    save_plot_pdf,
    export_info_to_pdf,
    export_table_to_pdf,
    merge_pdf_files,
    COLOR_PALETTE_DICT,
)
from .record_objects import LITERAL_TRANSACTION_REASON
from .pandas_units import SeriesUnits

VerboseType = Literal["silent", "action", "status", "verbose"]


@dataclass
class Portfolio(Asset):
    """
    Dataclass containing the structure of a portfolio.

    """

    commission_trade: float = 0.0
    commission_transfer: float = 0.0
    transaction_id: int = 0
    displayed_assets: int = 22
    time_chart_resolution: int = 400
    threshold_buy: float = 0.0 # This is the signal threshold to buy an asset - Only for display purposes
    threshold_sell: float = 0.0 # This is the signal threshold to sell an asset - Only for display purposes
    output_folder_path: str = "./tmp/"

    # Portfolio internal methods ------------------------------------------------

    def __post_init__(self) -> None:
        super().__post_init__()
        # EVERYTHING SMALLER THAN 0.1 IS CONSIDERED 0
        # This is useful to avoid displaying assets with a very small balance and trying to sell assets with a very small balance
        self.MINIMAL_BALANCE = 0.1 # QUOTE
        self.set_verbosity("verbose")
        self.assets = dict()
        self.Ledger = Ledger(portfolio_symbol=self.symbol)
        # We need to keep track of the properties that have been calculated
        self._properties_evolution_id = dict() # This dictionary will store the evolution_id of each property
        self._properties_cached = dict() # This dictionary will store the cached value of each property
        self.signals_df: Optional[pd.DataFrame] = None # This will be automatically provided by the strategy after running the backtest
        self.pdf_export_files = []
    
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

    def set_verbosity(self, verbosity_type: VerboseType) -> None:
        """
        Method to set the verbose status and action flags.

        """
        if verbosity_type == "status":
            self.verbose_status = True
            self.verbose_action = False
        elif verbosity_type == "action":
            self.verbose_status = False
            self.verbose_action = True
        elif verbosity_type == "verbose":
            self.verbose_status = True
            self.verbose_action = True
        elif verbosity_type == "silent":
            self.verbose_status = False
            self.verbose_action = False
        
    @property
    @check_property_update
    def assets_table(self) -> str:
        """
        Property to create a table with the assets' information.

        """
        data = [
            [
                "Currency",
                "Balance",
                "Value",
                "Last Price",
                "Avg. Purchase",
                "Transactions",
                "Total traded",
                "Commissions",
                "Performance",
                "T. Gains",
                "R. Gains",
                "P. Gains",
                "Currency growth",
                "Hold gains",
                "_sort_column",
            ]
        ]
        hold_gains = self.hold_gains_assets
        performance_assets = self.performance_assets
        gains_assets = self.gains_assets
        # for asset_symbol in self.assets_list:
        for asset_symbol in self.assets_traded_list:
            Currency = self.assets[asset_symbol]
            total_gains_asset = gains_assets[asset_symbol]
            realized_gains_asset = self.get_realized_gains_single_asset(asset_symbol)
            paper_gains = total_gains_asset - realized_gains_asset
            # Make paper_gains 0.0 if the value is really small for display purposes
            if abs(paper_gains) < 0.01:
                paper_gains = 0.0
            # Display only the assets with transactions
            # if self.transactions_count(asset_symbol) > 0:
            data.append(
                [
                    asset_symbol,
                    *Currency.info,
                    display_integer(self.transactions_count(asset_symbol)),
                    display_price(self.transactions_sum(asset_symbol), self.symbol),
                    display_price(Currency.commissions_sum, self.symbol),
                    display_percentage(performance_assets[asset_symbol]),
                    display_price(total_gains_asset, self.symbol),
                    display_price(realized_gains_asset, self.symbol),
                    display_price(paper_gains, self.symbol),
                    display_percentage(self.get_asset_growth(asset_symbol)),
                    display_price(hold_gains[asset_symbol], self.symbol),
                    gains_assets[asset_symbol], # Only for sorting #1
                ]
            )
        return data

    def assets_table_df(self, assets_list: Optional[list['str']] = None) -> pd.DataFrame:
        """
        Property to create a DataFrame with the assets' information.

        """
        data = self.assets_table
        df = pd.DataFrame(data[1:], columns=data[0])
        df.sort_values(by='_sort_column', ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.drop(columns='_sort_column', inplace=True)
        if assets_list is not None:
            df = df[df['Currency'].isin(assets_list)]
        return df
    
    @property
    @check_property_update
    def assets_pretty_table(self) -> str:
        """
        Property to display the assets' information as a pretty table.

        """
        data = self.assets_table
        return display_pretty_table(data, quote_currency= self.symbol, padding=6, sorting_columns=1)
    
    
    @property
    def period(self) -> str:
        """
        Property to get the period of the portfolio.

        """
        timerange = self.timerange
        start = pd.to_datetime(timerange[0], unit="s")
        end = pd.to_datetime(timerange[1], unit="s")
        return start, end
    
    @property
    @check_property_update
    def info(self) -> dict[str, str]:
        """
        Property to get the portfolio information as a dictionary.

        """
        info = dict()
        self.empty_negligible_assets()
        equity_value = self.equity_value
        quote_balance = self.balance
        quote_equity_ratio = quote_balance / equity_value
        info["Name"] = self.name.replace('_',' ').title()
        info["Transfer commission"] = self.commission_transfer
        info["Trade commission"] = self.commission_trade
        info["Start time"] = self.period[0]
        info["End time"] = self.period[1]
        info["Time span"] = self.timespan
        info["Invested capital"] = self.invested_capital
        info["Disbursed capital"] = self.disbursed_capital
        info["Cash balance"] = self.balance
        info["Cash balance ratio"] = quote_equity_ratio
        # If there are no historical prices, it can't display any assets information.
        if not self.historical_prices.empty:
            assets_value = self.assets_value
            assets_equity_ratio = assets_value / equity_value
            total_gains = self.gains
            realized_gains = self.realized_gains_sum
            paper_gains = total_gains - realized_gains
            info["Assets value"] = self.assets_value
            info["Assets value ratio"] = assets_equity_ratio
            info["Equity value"] = self.equity_value
            info["Transactions"] = self.transactions_count()
            info["Amount traded"] = self.transactions_sum()
            info["ROI"] = self.roi
            info["Total Gains"] = total_gains
            info["Realized Gains"] = realized_gains
            info["Paper Gains"] = paper_gains
            info["Commissions"] = self.total_commissions
            info["Commissions/Gains ratio"] = self.commission_gains_ratio
            info["Hold ROI (Theoretical)"] = self.hold_roi
            info["Hold Gains (Theoretical)"] = self.hold_gains
            info["ROI Performance (vs Hold)"] = self.roi_vs_hold_roi
            info["Assets Traded"] = len(self.assets_traded_list)
        return info
    
    @property
    @check_property_update
    def info_pd(self) -> pd.DataFrame:
        """
        Property to get the portfolio information as a DataFrame.

        """
        info_s = pd.Series(self.info)
        units = pd.Series(
            {
                "Name": "",
                "Transfer commission": "%",
                "Trade commission": "%",
                "Start time": "",
                "End time": "",
                "Time span": "s",
                "Invested capital": self.symbol,
                "Disbursed capital": self.symbol,
                "Cash balance": self.symbol,
                "Cash balance ratio": "%",
                "Assets value": self.symbol,
                "Assets value ratio": "%",
                "Equity value": self.symbol,
                "Transactions": "",
                "Amount traded": self.symbol,
                "ROI": "%",
                "Total Gains": self.symbol,
                "Realized Gains": self.symbol,
                "Paper Gains": self.symbol,
                "Commissions": self.symbol,
                "Commissions/Gains ratio": "%",
                "Hold ROI (Theoretical)": "%",
                "Hold Gains (Theoretical)": self.symbol,
                "ROI Performance (vs Hold)": "%",
                "Assets Traded": "",
            }
        )
        info_units = SeriesUnits(data=info_s, units=units)
        return info_units

    def export_info_pdf(self) -> None:
        """
        Method to export the portfolio information as a PDF.

        """
        # Example DataFrame (replace this with your actual TS1().info_pd.D DataFrame)
        output_file="portfolio_info.pdf"
        pdf_file_path = os.path.join(self.output_folder_path, output_file)
        portfolio_df = pd.DataFrame(self.info_pd.D).reset_index()
        portfolio_df = portfolio_df.iloc[3:]
        export_info_to_pdf(df=portfolio_df,
                           pdf_file_path=pdf_file_path,
                           title="Portfolio Summary",
        )
        self.pdf_export_files.append(pdf_file_path)

    def export_assets_table_pdf(self) -> None:
        output_file="assets_table.pdf"
        pdf_file_path = os.path.join(self.output_folder_path, output_file)
        export_table_to_pdf(df=self.assets_table_df(),
                            pdf_file_path = pdf_file_path,
                            title= "Traded Assets Details"
        )
        self.pdf_export_files.append(pdf_file_path)
    
    def create_pdf_report(self) -> None:
        """
        Method to create the PDF files with the portfolio information.

        """
        self.pdf_export_files = []
        portfolio_name = self.name.replace(' ', '_')
        original_folder_path = self.output_folder_path
        self.output_folder_path = os.path.join(self.output_folder_path, portfolio_name)
        if not os.path.exists(self.output_folder_path):
            os.makedirs(self.output_folder_path)
        self.export_info_pdf()
        self.plot_summary(export_pdf=True)
        self.plot_portfolio(export_pdf=True)
        self.export_assets_table_pdf()
        merge_pdf_files(self.pdf_export_files, os.path.join(original_folder_path, f"{portfolio_name}_report.pdf"))
        # Delete the output folder and its files
        if os.path.exists(self.output_folder_path):
            for file in self.pdf_export_files:
                if os.path.isfile(file):
                    os.remove(file)
            os.rmdir(self.output_folder_path)
        # Restore the original output folder path
        self.output_folder_path = original_folder_path

    @property
    @check_property_update
    def text_repr(self) -> str:
        """
        Property to display the portfolio information as a string.
        
        """
        info = self.info_pd.D
        text = (
                f"Portfolio: {info['Name']}:\n"
                f"  -> Cash currency: {self.symbol}\n"
                f"  -> Transfer commission: {info['Transfer commission']}\n"
                f"  -> Trade commission: {info['Trade commission']}\n"
                f"  -> Time range: from {info['Start time']} to {info['End time']}\n"
                f"  -> Time span: {info['Time span']}\n"
                f"  -> Invested capital: {info['Invested capital']}\n"
                f"  -> Disbursed capital: {info['Disbursed capital']}\n"
                f"  -> Cash balance: {info['Cash balance']}"
                f" ({info['Cash balance ratio']})\n"
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
                f"  -> Assets value: {info['Assets value']}"
                f" ({info['Assets value ratio']})\n"
                f"  -> Equity value: {info['Equity value']}\n"
                f"  -> Transactions: {info['Transactions']}\n"
                f"  -> Amount traded: {info['Amount traded']}\n"
                f"  -> ROI: {info['ROI']}\n"
                f"  -> Total Gains: {info['Total Gains']}\n"
                f"  -> Realized Gains: {info['Realized Gains']}\n"
                f"  -> Paper Gains: {info['Paper Gains']}\n"
                f"  -> Commissions: {info['Commissions']}\n"
                f"  -> Commissions/Gains ratio: {info['Commissions/Gains ratio']}\n"
                f"  -> Hold ROI (Theoretical): {info['Hold ROI (Theoretical)']}\n"
                f"  -> Hold Gains (Theoretical): {info['Hold Gains (Theoretical)']}\n"
                f"  -> ROI Performance (vs Hold): {info['ROI Performance (vs Hold)']}\n"
            )
            if len(self.assets_traded_list) > 0:
                text += (
                    f"  -> Assets Traded: {info['Assets Traded']}\n"
                    f"{self.assets_pretty_table}\n\n"
                )
            # This applies only in case of having updated the prices but not having any transaction yet.
            else:
                text += f"  -> Assets Traded: None\n"
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
            name=self.name,
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
    
    def _update_purchase_price_avg(self, symbol: str, price: float, amount_base: float) -> None:
        """
        Method to update the average purchase price of an asset in the portfolio.

        """
        current_balance = self.assets[symbol].balance
        prev_purchase_price_avg = self.assets[symbol].purchase_price_avg
        if np.isnan(prev_purchase_price_avg):
            updated_purchase_price_avg = price
        else:
            updated_purchase_price_avg = (
                ( (prev_purchase_price_avg * current_balance) + (price * amount_base) ) / (current_balance + amount_base)
            )
        self.assets[symbol]._update_purchase_price_avg(updated_purchase_price_avg)
        
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
            reason="INVESTMENT",
            symbol=self.symbol,
            amount=amount,
            price=1,
            commission=commission,
            balance_pre=balance_pre,
        )
        net_amount = amount - commission
        # Here starts the real transaction
        ## Keep track of the commissions paid globally
        self.commissions_sum += float(commission)
        ## Keep track of the balance
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
            reason="DISBURSEMENT",
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
        ## Keep track of the commissions paid globally
        self.commissions_sum += float(commission)
        ## Keep track of the balance
        self.balance -= float(gross_amount)
        # Record the capital movement in the ledger
        self.Ledger.disburse_capital(timestamp=timestamp, amount=amount)
        # Acknowledge the transaction
        self.Ledger.confirm_transaction(transaction_id)
        self.print_portfolio()

    @check_positive
    def buy(
        self, symbol: str, amount_quote: float, reason: LITERAL_TRANSACTION_REASON, timestamp: Optional[int] = None, msg: Optional[str] = None
    ) -> None:
        """
        Method to buy an amount of an asset in the portfolio.

        """
        transaction_id = self.new_transaction_id()
        balance_liquid_pre = self.balance
        prev_purchase_price_avg = self.assets[symbol].purchase_price_avg
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
            reason=reason,
            symbol=symbol,
            amount=amount_base,
            price=self.assets[symbol].price,
            commission=0,
            balance_pre=balance_asset_pre,
            purchase_price_avg=prev_purchase_price_avg,
        )
        # Record the transaction in the ledger for the portfolio currency
        # The transaction is not acknowledged yet
        self.Ledger.sell(
            id=transaction_id,
            timestamp=timestamp,
            trade=True,
            reason=reason,
            symbol=self.symbol,
            amount=amount_quote,
            price=1,
            commission=commission_quote,
            balance_pre=balance_liquid_pre,
        )
        if np.isnan(prev_purchase_price_avg):
            self.Ledger.add_message_to_transaction(transaction_id, "Buying asset for the first time")
        # Add message to the transaction
        self.Ledger.add_message_to_transaction(transaction_id, msg)
        gross_amount_quote = amount_quote + commission_quote
        self.check_amount(gross_amount_quote)
        # Add the commissions to the commissions amortization
        self.assets[symbol]._add_commissions_in_commissions_to_deduct(commission_quote)
        # Update the average purchase price of the asset
        self._update_purchase_price_avg(symbol, price, amount_base)
        # Here starts the real transaction
        ## Keep track of the commissions impacted on each asset
        self.assets[symbol].commissions_sum += float(commission_quote)
        ## Keep track of the cash balance
        self.balance -= float(gross_amount_quote)
        ## Keep track of the asset balance
        self.assets[symbol]._deposit(amount_base)
        # Acknowledge the transaction
        self.Ledger.confirm_transaction(transaction_id)
        self.print_portfolio()

    @check_positive
    def sell(
        self, symbol: str, amount_quote: float, reason: LITERAL_TRANSACTION_REASON, timestamp: Optional[int] = None, msg: Optional[str] = None
    ) -> None:
        """
        Method to sell an amount of an asset in the portfolio.

        """
        transaction_id = self.new_transaction_id()
        balance_liquid_pre = self.balance
        purchase_price_avg = self.assets[symbol].purchase_price_avg
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
        # Deduct the commissions from the commissions amortization
        deducted_commissions = self.assets[symbol]._deduct_commissions(amount_quote)
        deducted_commissions += commission_quote
        # Record the transaction in the ledger for the asset
        # The transaction is not acknowledged yet
        self.Ledger.sell(
            id=transaction_id,
            timestamp=timestamp,
            trade=True,
            reason=reason,
            symbol=symbol,
            amount=amount_base,
            price=self.assets[symbol].price,
            commission=0,
            balance_pre=balance_asset_pre,
            purchase_price_avg=purchase_price_avg,
            deducted_commissions=deducted_commissions,
        )
        # Record the transaction in the ledger for the portfolio currency
        # The transaction is not acknowledged yet
        self.Ledger.buy(
            id=transaction_id,
            timestamp=timestamp,
            trade=True,
            reason=reason,
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
        ## Keep track of the commissions impacted on each asset
        self.assets[symbol].commissions_sum += float(commission_quote)
        ## Keep track of the asset balance
        self.assets[symbol]._withdraw(amount_base)
        ## Keep track of the cash balance
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
        return int(end - start)
    
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
    def realized_gains_displayable(self) -> pd.DataFrame:
        """
        Property to get the realized gains for each asset in the portfolio with a positive balance.

        """
        # We use the ledger index to have a complete DataFrame with all the timestamps
        # sales_profit_loss_df has only the timestamps when sales happen, meaning that not all the timestamps are present
        timestamp_indexes = pd.DataFrame(self.ledger_equity.index, columns=['Timestamp'])
        realized_gains=self.Ledger.sales_profit_loss_df.copy()
        realized_gains_complete = pd.merge(timestamp_indexes, realized_gains, on='Timestamp', how='left')
        realized_gains_complete['Profit_Netto'] = realized_gains_complete['Profit_Netto'].fillna(0.0)
        # In order to have a complete DataFrame with all the assets, we need to fill the NaN values with some value
        # After pivotting the DataFrame, we will fill delete the given column
        realized_gains_complete['Symbol'] = realized_gains_complete['Symbol'].fillna('_column_to_delete')
        realized_gains_complete['Datetime'] = pd.to_datetime(realized_gains_complete['Timestamp'], unit='s')
        realized_gains_pivot = realized_gains_complete.pivot_table(index='Datetime', columns='Symbol', values='Profit_Netto', aggfunc='sum', observed=True)
        realized_gains_pivot.fillna(0, inplace=True)
        realized_gains_pivot.drop(columns='_column_to_delete', inplace=True)
        return realized_gains_pivot
    
    @property
    @check_property_update
    def realized_gains_assets(self) -> dict[str, float]:
        """
        Property to get the realized gains for each asset in the portfolio.

        """
        return self.Ledger.sales_profit_loss_df.groupby('Symbol')['Profit_Netto'].sum()
    
    def get_realized_gains_single_asset(self, symbol: str) -> float:
        """
        Method to get the realized gains for a single asset in the portfolio.
        
        In case of not having any record registered yet, it returns 0 to avoid errors.

        """
        if symbol in self.realized_gains_assets:
            return float(self.realized_gains_assets[symbol])
        else:
            return 0.0
    
    @property
    @check_property_update
    def realized_gains_sum(self) -> float:
        """
        Property to get the total realized gains in the portfolio.

        """
        return float(self.realized_gains_assets.sum())

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
        return float(self.balance + self.assets_value)

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
            balance = self.balance
        else:
            balance = self.assets[symbol].balance
        return float(balance)

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
            value = self.balance / price
        else:
            value = (self.assets[symbol].balance * self.assets[symbol].price) / price
        return float(value)
        
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
        return float(self.balance / self.equity_value)

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

    def get_price_asset_on_timestamp(self, symbol: str, timestamp: int) -> float:
        """
        Method to get the price of an asset in the portfolio at a specific timestamp.

        """
        return self.Ledger.get_price_asset_on_timestamp(symbol, timestamp)
    
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
    def transactions_info(self) -> pd.DataFrame:
        """
        Property to get the historical transactions as a DataFrame with timestamps.

        """
        transactions_df = self.historical_transactions.copy()
        transactions_df = transactions_df[['Symbol', 'Timestamp', 'Action', 'Reason', 'Amount', 'Price', 'Traded']]
        return transactions_df

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
    def ledger_equity_average(self) -> float:
        """
        Property to get the average equity for each asset in the portfolio.

        """
        return self.ledger_equity.mean()
    
    def top_ledger_equity_assets_average(self, n: int) -> pd.Series:
        """
        Property to get the average equity for the top assets in the portfolio.

        """
        ledger_equity_average = self.ledger_equity_average
        entries_to_drop = [self.symbol, "Total"]
        ledger_equity_average.drop(entries_to_drop, inplace=True)
        return ledger_equity_average.nlargest(n)
    
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
    def signals_pivot_df(self) -> pd.DataFrame:
        """
        Property to get the signals DataFrame as a pivot table.

        """
        signals_df = self.signals_df.copy()
        if signals_df is not None:
            signals_pivot_df = signals_df.pivot_table(index='timestamp', columns='symbol', values='value_signal', observed=True)
            return signals_pivot_df

    @property
    @check_property_update
    def signals_displayable(self) -> pd.DataFrame:
        """
        Property to get the signals DataFrame as a pivot table.

        """
        signals_df = self.signals_pivot_df.copy()
        if signals_df is not None:
            # We display only the assets that are traded
            # columns_to_display = [c for c in signals_df.columns if c in self.assets_traded_list]
            columns_to_display = self.assets_traded_list
            signals_df = signals_df[columns_to_display].copy()
            signals_df["DateTime"] = pd.to_datetime(signals_df.index, unit="s")
            return signals_df.set_index("DateTime", drop=True)
        
    @property
    @check_property_update
    def volatility_pivot_df(self) -> pd.DataFrame:
        """
        Property to get the signals DataFrame as a pivot table.

        """
        volatility_df = self.volatility_df.copy()
        if volatility_df is not None:
            volatility_pivot_df = volatility_df.pivot_table(index='timestamp', columns='symbol', values='volatility', observed=True)
            return volatility_pivot_df
        
    @property
    @check_property_update
    def volatility_displayable(self) -> pd.DataFrame:
        """
        Property to get the volatility DataFrame as a pivot table.

        """
        volatility_df = self.volatility_pivot_df.copy()
        if volatility_df is not None:
            # We display only the assets that are traded
            # columns_to_display = [c for c in signals_df.columns if c in self.assets_traded_list]
            columns_to_display = self.assets_traded_list
            volatility_df = volatility_df[columns_to_display].copy()
            volatility_df["DateTime"] = pd.to_datetime(volatility_df.index, unit="s")
            return volatility_df.set_index("DateTime", drop=True)

    @property
    @check_property_update
    def gains(self) -> float:
        """
        Property to get the gains of the portfolio.

        """
        return self.equity_value + self.disbursed_capital - self.invested_capital

    @property
    @check_property_update
    def commission_gains_ratio(self) -> str:
        """
        Property to get the ratio of the total commissions to the gains of the portfolio.
        
        """
        gains = self.gains
        if gains > 0:
            ratio = self.total_commissions / gains
            return ratio
        else:
            return np.nan

    @property
    @check_property_update
    def roi(self) -> float:
        """
        Property to get the Return on Investment (ROI) of the portfolio.

        """
        investment = self.invested_capital
        if investment > 0:
            roi = np.round(self.gains / investment, 5)
        else:
            roi = 0.0
        return float(roi)
    
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
    
    def relevant_assets(self, n: int) -> pd.DataFrame:
        """
        Method to get the most relevant assets in the portfolio.
        
        It gets N top and N bottom performers and returns the list of the most relevant assets.
        
        """
        def remove_duplicates_keep_order(lst):
            seen = set()
            return [x for x in lst if not (x in seen or seen.add(x))]
        top_performers = self.top_performers(n=n)
        top_performers_list = top_performers.index.tolist()
        bottom_performers = self.bottom_performers(n=n)
        bottom_performers_list = bottom_performers.index.tolist()
        bottom_performers_list.reverse()
        edge_performers_list = top_performers_list + bottom_performers_list
        edge_performers_list = remove_duplicates_keep_order(edge_performers_list)
        return edge_performers_list
    
    @property
    @check_property_update
    def historical_theoretical_hold_equity(self) -> pd.Series:
        """
        Property to provide the theoretical gains of holding the assets in the portfolio.

        """
        prices = self.ledger_prices.copy()
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
        prices = self.ledger_prices.copy()
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
        return float(self.hold_gains_assets.sum())

    @property
    @check_property_update
    def hold_roi(self) -> float:
        """
        Property to provide the Return on Investment (ROI) of theoretical holding the assets in the portfolio.

        """
        return float(np.round(self.hold_gains / self.invested_capital, 5))
    
    @property
    @check_property_update
    def roi_vs_hold_roi(self) -> float:
        """
        Property to provide the difference between the ROI and the theoretical holding ROI.

        """
        return float(self.roi - self.hold_roi)


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

    def resample_data(self, df: pd.DataFrame, agg_type: Literal['last', 'mean', 'sum'], factor: int = 1) -> pd.DataFrame:
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
        if agg_type == 'mean':
            df = df.resample(freq_str).mean()
        elif agg_type == 'last':
            df = df.resample(freq_str).last()
        elif agg_type == 'sum':
            df = df.resample(freq_str).sum()
        # In order to have a better display, we center the index
        # By default it takes the first timestamp of the window
        df.index = df.index + pd.Timedelta(seconds=freq/2)
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

    def print_logs_nak(self) -> None:
        """
        Method to print the not acknowledged log entries.
            
        """
        self.Ledger.print_logs_nak()
    
    # Portfolio ploting methods ------------------------------------------------

    def plot_portfolio(self, assets: Union[int,str,list[str]] = 10, export_pdf: bool = False) -> None:
        """
        Method to plot the portfolio equity and the assets' prices over time.
        
        assets number is the number of assets to plot twice - once for the top performers and once for the bottom performers.
        assets = 10 means 10 top and 10 bottom performers.

        """
        # We need to know which assets to plot
        if isinstance(assets, int):
            assets_list = self.relevant_assets(assets)
        elif isinstance(assets, str):
            assets_list = [assets]
        elif isinstance(assets, list):
            assets_list = assets

        # Create a figure with multiple subplots
        symbol = self.symbol
        # assets_list = ["Total"] + self.assets_traded_list
        assets_list = ["Total"] + assets_list
        num_plots = len(assets_list)
        h_size = num_plots * 6
        fig, axs = plt.subplots(
            nrows=num_plots, ncols=1, figsize=(15, h_size), sharex=True
        )
        # Ensure axs is always an array
        axs = np.atleast_1d(axs)
        # For these we need equity and prices
        # Afterwards we will also need the transactions
        historical_equity = self.ledger_equity_datetime.copy()
        resampled_historical_equity = self.resample_data(historical_equity, agg_type='last')
        historical_prices = self.historical_prices_pivot.copy()
        resampled_historical_prices = self.resample_data(historical_prices, agg_type='last')
        signals_df = self.signals_displayable.copy()
        resampled_signals_df = self.resample_data(signals_df, agg_type='mean')
        # We normalize the signals among all the assets
        # Then all assets will have the same scale
        # First we get the absolute max value
        max_signal_ratio = resampled_signals_df.abs().max().max()
        # Then we normalize the signals
        resampled_signals_df = resampled_signals_df / max_signal_ratio
        threshold_buy = self.threshold_buy / max_signal_ratio
        threshold_sell = self.threshold_sell / max_signal_ratio
        # We get the volatility of the returns
        volatility_df = self.volatility_displayable.copy()
        resampled_volatility_df = self.resample_data(volatility_df, agg_type='mean')
        max_volatility = resampled_volatility_df.abs().max().max()
        # Retrieve the transactions info for the scatter plot
        transactions_info = self.transactions_info.copy()
        transactions_info['Datetime'] = pd.to_datetime(transactions_info['Timestamp'], unit="s")
        transactions_info['Size'] = transactions_info['Traded'].astype(int).abs()
        for i, asset in enumerate(assets_list):
            ax = axs[i]
            ax.tick_params(axis='y', which="both", right=False, labelright=False, left=False, labelleft=False)
            ax.grid(False)
            realized_gains = self.realized_gains_displayable.copy()
            realized_gains_assets = realized_gains.columns.tolist()
            i_gs = 0
            gs_plots_ratios = [3, 6, 2]
            if (asset == "Total") or (asset not in realized_gains_assets):
                # If there is no realized gains for the asset, we display 1 gs less
                # And we start from the second plot
                gs_plots_ratios = gs_plots_ratios[1:]
                gs_plots = len(gs_plots_ratios)
                gs = gridspec.GridSpecFromSubplotSpec(gs_plots, 1, subplot_spec=ax.get_subplotspec(), height_ratios=gs_plots_ratios, hspace=0.1)
            elif (asset != "Total") and (asset in realized_gains_assets):
                # Realized Gains Plot
                # Granular Realized Gains
                # Resample profits and losses to have a better visualization
                # It is needed to split profits and losses to have a better visualization
                # When resampling profits and losses they would cancel each other in the same sampling window
                gs_plots = len(gs_plots_ratios)
                gs = gridspec.GridSpecFromSubplotSpec(gs_plots, 1, subplot_spec=ax.get_subplotspec(), height_ratios=gs_plots_ratios, hspace=0.1)
                resample_agg_type = 'sum'
                factor = 20
                time_bar_width = self.time_bar_width * factor/5
                ax_top = fig.add_subplot(gs[i_gs], sharex=ax)
                i_gs += 1
                asset_realized_gains = realized_gains[asset]
                realized_profit = asset_realized_gains[asset_realized_gains > 0].fillna(0)
                resampled_realized_profit = self.resample_data(realized_profit, agg_type=resample_agg_type, factor=factor)
                resampled_realized_profit = resampled_realized_profit[resampled_realized_profit != 0]
                realized_loss = asset_realized_gains[asset_realized_gains < 0].fillna(0)
                resampled_realized_loss = self.resample_data(realized_loss, agg_type=resample_agg_type, factor=factor)
                resampled_realized_loss = resampled_realized_loss[resampled_realized_loss != 0]
                # Plot the bar chart profits
                # To avoid displaying not useful legends, we only plot the bars if there are values
                if len(resampled_realized_profit) > 0:
                    ax_top.bar(resampled_realized_profit.index, resampled_realized_profit.values, width=time_bar_width, color='green', alpha=0.5, edgecolor='black', label='Realized Profits')
                # Plot the bar chart losses
                # To avoid displaying not useful legends, we only plot the bars if there are values
                if len(resampled_realized_loss) > 0:
                    ax_top.bar(resampled_realized_loss.index, resampled_realized_loss.values, width=time_bar_width, color='red', alpha=0.5, edgecolor='black', label='Realized Losses')
                ax_top.grid(True)
                ax_top.axhline(y=0, color='black', linewidth=1)
                value_ylim = max_2_numbers(resampled_realized_profit.abs().max(), resampled_realized_loss.abs().max()) * 1.2
                ax_top.set_ylim(-value_ylim, value_ylim)
                ax_top.grid(True)
                ax_top.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax_top.xaxis.set_minor_locator(AutoMinorLocator())
                ax_top.yaxis.set_minor_locator(AutoMinorLocator())
                ax_top.set_ylabel(f"Gains ({self.symbol})")
                ax_top.grid(which="both")
                ax_top.grid(which="minor", alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
                ax_top.grid(which="major", alpha=0.5, linestyle='-', linewidth=1.0, color='black')
                ax_top.tick_params(axis='x', which="both", top=False, labeltop=False, bottom=False, labelbottom=False)
                ax_top.legend(fontsize=6)
                
            # Main Plot            
            ax_main1 = fig.add_subplot(gs[i_gs], sharex=ax)
            i_gs += 1
            # Ploting the equity of the asset
            datetime = resampled_historical_equity.index
            total_equity = resampled_historical_equity[asset]
            # Different settings for the total and the assets
            if asset == "Total":
                alpha = 1
                label_plot = "Equity"
            else:
                alpha = 0.1
                label_plot = None
                ax_main1.fill_between(datetime, total_equity, where=(total_equity > 0), color="green", alpha=0.2, label="Equity")
            ax_main1.plot(datetime, total_equity, color="green", alpha=alpha, label=label_plot)
            ax_main1.set_ylabel(f"Equity ({symbol})", color="green")
            # Color the y-axis tick labels            
            ax_main1.tick_params(axis='y', colors='green')
            ax_main1.grid(True)
            # To enable the grid for minor ticks
            # ax.set_xticks([])
            # ax.set_xticklabels([])
            ax_main1.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax_main1.xaxis.set_minor_locator(AutoMinorLocator())
            ax_main1.yaxis.set_minor_locator(AutoMinorLocator())
            # Hide x-axis labels and ticks on ax_main
            ax_main1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax_main1.grid(which="both")
            ax_main1.grid(which="minor", alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
            ax_main1.grid(which="major", alpha=0.5, linestyle='-', linewidth=1.0, color='black')
            ax_main1.tick_params(axis='x', which="both", top=False, labeltop=False, bottom=False, labelbottom=False)
            # Format to thousands the y-axis only if the maximal value is higher than 1000
            if total_equity.max() > 1000:
                ax_main1.yaxis.set_major_formatter(FuncFormatter(thousands))
            # Total doesn't have a price
            if asset != "Total":
                # Create a secondary y-axis for the asset price
                ax_main2 = ax_main1.twinx()
                # Ploting the price of the asset
                datetime = resampled_historical_prices.index
                prices = resampled_historical_prices[asset]
                # Avoid scientific notation on the y-axis
                ax_main2.ticklabel_format(axis='y', style='plain')
                # Color the y-axis tick labels
                ax_main2.plot(datetime, prices, color="purple", label="Price")
                ax_main2.set_ylabel(f"Price ({symbol})", color="purple")
                ax_main2.tick_params(axis='y', colors='purple')
                ax_main2.grid(False)
                ax_main2.tick_params(axis='x', which="both", top=False, labeltop=False, bottom=False, labelbottom=False)
                # Format to thousands the y-axis only if the maximal value is higher than 1000
                if prices.max() > 1000:
                    ax_main2.yaxis.set_major_formatter(FuncFormatter(thousands))
                # Plot the transactions
                # Define marker and color mappings
                marker_map = {
                    "BUY": {"INVESTMENT": 'o', "SIGNAL": '^'},
                    "SELL": {"SIGNAL": 'v', "LIQUIDITY": 'x', "VOLATILITY": 'h'}
                }
                color_map = {"BUY": 'blue', "SELL": 'red'}
                transactions_asset = transactions_info[transactions_info.Symbol == asset].copy()
                if not transactions_asset.empty:
                    for transaction_type, reasons in marker_map.items():
                        color = color_map[transaction_type]
                        for transaction_reason, marker in reasons.items():
                            transactions_asset_filtered = transactions_asset[
                                (transactions_asset['Action'] == transaction_type) & 
                                (transactions_asset['Reason'] == transaction_reason)
                            ]
                            if not transactions_asset_filtered.empty:
                                ax_main2.scatter(
                                    transactions_asset_filtered['Datetime'], 
                                    transactions_asset_filtered['Price'], 
                                    color=color, 
                                    alpha=0.1, 
                                    s=transactions_asset_filtered['Size'], 
                                    marker=marker, 
                                    label=None
                                )
            # Display different title for the total and the assets
            if asset == "Total":
                info = f"(Gains: {display_price(self.gains, symbol)} | ROI: {display_percentage(self.roi)})"
                ax.set_title(f"Portfolio Value over Time {info}")
                ax_main1.legend(fontsize='small', loc='upper left')
            else:
                scatter_signals = ['Investment', 'Buy Signal', 'Sell Signal', 'Sell Liquidity', 'Sell Volatility']
                info = f"(Price Growth: {display_percentage(self.get_asset_growth(asset))})"
                ax.set_title(f"Asset [{asset}] Evolution over Time {info}")
                # Combine legends from both axes
                lines, labels = ax_main1.get_legend_handles_labels()
                lines2, labels2 = ax_main2.get_legend_handles_labels()
                # Create custom legend handles for scatter plots with fixed size
                marker_size_legend = 7
                custom_lines = [Line2D([0], [0], color='blue', alpha=0.3, marker='o', linestyle='None', markersize=marker_size_legend, label=scatter_signals[0]),
                                Line2D([0], [0], color='blue', alpha=0.3, marker='^', linestyle='None', markersize=marker_size_legend, label=scatter_signals[1]),
                                Line2D([0], [0], color='red', alpha=0.3, marker='v', linestyle='None', markersize=marker_size_legend, label=scatter_signals[2]),
                                Line2D([0], [0], color='red', alpha=0.3, marker='x', linestyle='None', markersize=marker_size_legend, label=scatter_signals[3]),
                                Line2D([0], [0], color='red', alpha=0.3, marker='h', linestyle='None', markersize=marker_size_legend, label=scatter_signals[4])]
                ax_main2.legend(lines + lines2 + custom_lines, labels + labels2 + scatter_signals, fontsize=6)
            # Create the narrow chart (bottom)
            # But we only display it if there are signals for the asset
            ax_narrow1 = fig.add_subplot(gs[i_gs], sharex=ax)
            i_gs += 1
            if (asset in resampled_signals_df.columns) or (asset == "Total"):
                datetime = resampled_signals_df.index
                if asset != "Total":
                    signals = resampled_signals_df[asset]
                    # The next step is pointless if regarded alone.
                    # However, when assessing Total, we need 2 different "signals".
                    # In order to pass 2 different signals for each fill_between, we need 2 different variables.         
                    positive_signals = signals.where(signals >= threshold_buy, np.nan)
                    negative_signals = signals.where(signals <= threshold_sell, np.nan)
                    hold_signals = signals.where((signals < threshold_buy) & (signals > threshold_sell), np.nan)
                    yellow = "gold"
                    ax_narrow1.fill_between(datetime, hold_signals, where=hold_signals>0, color=yellow, alpha=0.3)
                    ax_narrow1.fill_between(datetime, hold_signals, where=hold_signals<0, color=yellow, alpha=0.3)
                    threshold_positive = threshold_buy
                    threshold_negative = threshold_sell
                    # Display the volatility
                    ax_narrow2 = ax_narrow1.twinx()
                    volatilities = resampled_volatility_df[asset]
                    ax_narrow2.plot(datetime, volatilities, color="black", label="Volatility", alpha=0.5)
                    ax_narrow2.set_ylim(0, max_volatility)
                    ax_narrow2.set_ylabel("Volatility")
                    ax_narrow2.tick_params(axis='x', which="both", top=False, labeltop=False, bottom=False, labelbottom=False)
                else:
                    # This allows to aggregate the sell and buy signals without cancelling each other.
                    positive_signals = resampled_signals_df[resampled_signals_df >= self.threshold_buy].sum(axis=1)
                    negative_signals = resampled_signals_df[resampled_signals_df <= self.threshold_sell].sum(axis=1)
                    # To have a better visualization we also normalize the total signals
                    # The Total signals will have a different scale than the assets,
                    # but if the assets had the same scale as totals, it would be very difficult to see them.
                    max_signal = max(positive_signals.abs().max(), negative_signals.abs().max())
                    positive_signals = positive_signals / max_signal
                    negative_signals = negative_signals / max_signal
                    threshold_positive = 0
                    threshold_negative = 0
                lim = 1.1 
                ax_narrow1.set_ylim(-lim, lim)
                ax_narrow1.yaxis.set_minor_locator(AutoMinorLocator())
                ax_narrow1.tick_params(axis='y', which="both", left=False, labelleft=False, right=False, labelright=False)
                ax_narrow1.fill_between(datetime, positive_signals, where=positive_signals>threshold_positive, color="blue", alpha=0.2, label="Buy Signals")
                ax_narrow1.fill_between(datetime, negative_signals, where=negative_signals<threshold_negative, color="red", alpha=0.2, label="Sell Signals")
                ax_narrow1.legend(fontsize=6)
                ax_narrow1.grid(True)
                ax_narrow1.grid(which="both")
                ax_narrow1.grid(which="minor", alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
                ax_narrow1.grid(which="major", alpha=0.5, linestyle='-', linewidth=1.0, color='black')
                ax_narrow1.tick_params(axis='x', which="both", top=False, labeltop=False, bottom=False, labelbottom=False)
            # Adjust layout to remove spaces between subplots
            plt.subplots_adjust(hspace=0)  # hspace=0 removes horizontal space between subplots
            # Rotate x-axis tick labels by 20 degrees
            ax.set_xlabel("Time")
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d %Hh"))
            ax.tick_params(axis='x', which="both", top=False, labeltop=False, bottom=False, labelbottom=True)
            plt.setp(ax.get_xticklabels(), rotation=-20, ha='left')
            
        # Adjust layout to add space between main subplots
        plt.subplots_adjust(hspace=0.4)
        
        # Adjust layout to minimize blank space
        fig.tight_layout()
        
        if export_pdf:
            # Save the plot to a PDF file
            output_file = "portfolio_assets.pdf"
            pdf_file_path = os.path.join(self.output_folder_path, output_file)
            save_plot_pdf(fig=fig, pdf_file_path=pdf_file_path)
            plt.close()
            self.pdf_export_files.append(pdf_file_path)
        else:
            plt.show()

    def plot_benchmark(self, fig_ax: Optional[tuple] = None, log_scale: bool = True) -> None:
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
        resampled_historical_equity = self.resample_data(historical_equity, agg_type='mean')
        equity = self.normalize_to_growth(resampled_historical_equity["Total"])
        # To make it logarithmic, we add 1 to the equity
        baseline = 0
        if log_scale:
            equity = equity + 1
            baseline = 1
        timestamp = equity.index
        datetime = pd.to_datetime(timestamp, unit="s")
        label = f"Equity ({display_percentage(self.roi)})"
        ax.plot(datetime, equity, label=label, color="black", linewidth=1)
        # Fill areas where y < 0 with red and y > 0 with green
        # If log_scale is True, we need to fill the areas where y < 1 with red and y > 1 with green
        ax.fill_between(datetime, equity, baseline, where=(equity > baseline), color="green", alpha=0.1)  # type: ignore
        ax.fill_between(datetime, equity, baseline, where=(equity < baseline), color="red", alpha=0.1)  # type: ignore

        # CURRENCY PRICES
        historical_prices = self.historical_prices_pivot_displayable.copy()
        resampled_historical_prices = self.resample_data(historical_prices, agg_type='mean')
        for asset in resampled_historical_prices.columns:
            # We don't want to plot the portfolio currency
            # There is no growth for the portfolio currency
            if asset != self.symbol:
                label = (
                    f"{asset} ({display_percentage(self.get_asset_growth(asset))})"
                )
                asset_prices = self.normalize_to_growth(resampled_historical_prices[asset])
                # To make it logarithmic, we add 1 to the asset prices
                if log_scale:
                    asset_prices = asset_prices + 1
                timestamp = asset_prices.index
                datetime = pd.to_datetime(timestamp, unit="s")
                color = label_colors[asset]
                # Plot line
                ax.plot(datetime, asset_prices, label=label, linewidth=1, alpha=1, color=color)
                # Plot shadow
                ax.plot(datetime, asset_prices, linewidth=4, alpha=0.3, color=color)
                
        # THEORETICAL HOLD EQUITY
        theoretical_hold_equity = self.historical_theoretical_hold_equity
        resampled_theoretical_hold_equity = self.resample_data(theoretical_hold_equity, agg_type='mean')
        theoretical_equity = self.normalize_to_growth(resampled_theoretical_hold_equity)
        # To make it logarithmic, we add 1 to the theoretical equity
        if log_scale:
            theoretical_equity = theoretical_equity + 1
        timestamp = theoretical_equity.index
        datetime = pd.to_datetime(timestamp, unit="s")
        label = f"Hold ({display_percentage(self.hold_roi)})"
        # Plot line
        ax.plot(datetime, theoretical_equity, label=label, linewidth=0.7, alpha=1, color="black", linestyle='--')
        # Plot shadow
        ax.plot(datetime, theoretical_equity, linewidth=3.5, alpha=0.3, color='black')
        
        
        ax.set_title("Equity and Currency Change Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Growth")
        if log_scale:
            number_minor_ticks = 10
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=2.0, numticks=10))
            ax.yaxis.set_minor_locator(LogLocator(base=2.0, subs=np.arange(1, number_minor_ticks) * (1/number_minor_ticks), numticks=number_minor_ticks))  # Minor ticks
            ax.yaxis.set_major_formatter(FuncFormatter(to_percent_log_growth))
            ax.yaxis.set_minor_formatter(FuncFormatter(to_percent_log_growth))
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True)
        # To enable the grid for minor ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d %Hh"))
        plt.setp(ax.get_xticklabels(), rotation=-20, ha='left')
        ax.grid(which="both")
        ax.grid(which="minor", alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
        ax.grid(which="major", alpha=0.5, linestyle='-', linewidth=1.0, color='black')
        ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))

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
        label_colors = self.label_colors.copy()
        equity_share_df = self.ledger_equity_share_displayable.copy()
        resampled_equity_share_df = self.resample_data(equity_share_df, agg_type='mean')
        columns = list(equity_share_df.columns)
        # Force the portfolio currency to be the last one
        columns = move_to_end(columns, self.symbol)
        # Reverse the columns to do the cumsum in the correct order
        columns.reverse()
        resampled_equity_share_df = resampled_equity_share_df[columns]
        resampled_equity_share_df_cumsum = resampled_equity_share_df.cumsum(axis=1)
        # Reverse the columns to have a proper display
        columns.reverse()
        # columns = resampled_equity_share_df_cumsum.columns
        for column in columns:
            color = label_colors[column]
            if column == self.symbol:
                label = "Cash"
            else:
                label = column
            ax.fill_between(resampled_equity_share_df_cumsum.index, resampled_equity_share_df_cumsum[column], label=label, color=color, alpha=1.0, edgecolor='black', linewidth=0.5)
        ax.set_title("Equity Share Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Equity Share")
        ax.grid(True)
        # To enable the grid for minor ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d %Hh"))
        ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
        plt.setp(ax.get_xticklabels(), rotation=-20, ha='left')
        ax.grid(which="both")
        ax.grid(which="minor", alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
        ax.grid(which="major", alpha=0.5, linestyle='-', linewidth=1.0, color='black')
        ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_ylim(0, 1)

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
            
        gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=ax.get_subplotspec(), height_ratios=[1, 1], hspace=0.1)
        ax.grid(False)
        ax1 = fig.add_subplot(gs[0], sharex=ax)
        ax2 = fig.add_subplot(gs[1], sharex=ax)

        label_colors = self.label_colors.copy()

        ax1.axhline(y=0, color='black', linewidth=1)

        color_other = label_colors['other']
        color_symbol = label_colors[self.symbol]

        # Non liquid assets
        equity = self.ledger_equity_datetime.copy()
        non_liquid = equity['Total'] - equity[self.symbol]
        non_liquid_df = pd.DataFrame(non_liquid, columns=['Total'])
        non_liquid_df = self.resample_data(non_liquid_df, agg_type='last')
        datetime = non_liquid_df.index
        assets_value = non_liquid_df["Total"]
        ax2.plot(datetime, assets_value, color=color_other, alpha=0.4, linewidth=1)

        # Equity
        historical_equity = self.ledger_equity_datetime.copy()
        resampled_historical_equity = self.resample_data(historical_equity, agg_type='last')
        datetime = resampled_historical_equity.index
        total_equity = resampled_historical_equity["Total"]
        ax2.plot(datetime, total_equity, color=color_symbol, alpha=1.0, linewidth=1)

        # Fill where the differences
        ax2.fill_between(datetime, assets_value, color=color_other, alpha=0.4, label='Assets')
        ax2.fill_between(datetime, assets_value, total_equity, color=color_symbol, alpha=0.4, label='Cash')

        # Need to recalculate the equity to have the correct values
        transactions_df = self.ledger_transactions_datetime.copy()
        # Resample buys and sells to have a better visualization
        # It is needed to split buys and sells to have a better visualization
        # When resampling buys and sells they would cancel each other in the same sampling window
        resample_agg_type = 'sum'
        factor = 5
        buys_df = transactions_df[transactions_df > 0].fillna(0)
        resampled_buys_df = self.resample_data(buys_df, agg_type=resample_agg_type, factor=factor)
        sells_df = transactions_df[transactions_df < 0].fillna(0)
        resampled_sells_df = self.resample_data(sells_df, agg_type=resample_agg_type, factor=factor)
        # Initialize a variable to keep track of the bottom position for buys
        bottoms_buy = np.zeros(len(resampled_buys_df))
        # Initialize a variable to keep track of the bottom position for sells
        bottoms_sell = np.zeros(len(resampled_sells_df))
        # We only want to display the top and bottom performers
        for column in self.assets_traded_list:
            if column in self.assets_to_display_list:
                color = label_colors[column]
                # Plot the bar chart buys
                ax1.bar(resampled_buys_df.index, resampled_buys_df[column], label=column, bottom=bottoms_buy, width=self.time_bar_width, color=color, alpha=1.0, edgecolor='black', linewidth=0.5)
                # Plot the bar chart sells
                ax1.bar(resampled_sells_df.index, resampled_sells_df[column], bottom=bottoms_sell, width=self.time_bar_width, color=color, alpha=1.0, edgecolor='black', linewidth=0.5)
                # Update bottoms for buys and sells separately if needed
                bottoms_buy += resampled_buys_df[column]
                bottoms_sell += resampled_sells_df[column]
        # For the other assets, we aggregate them
        assets_to_aggregate = [asset for asset in self.assets_traded_list if asset not in self.assets_to_display_list]
        resampled_buys_agg_df = resampled_buys_df[assets_to_aggregate].sum(axis=1)
        resampled_sells_agg_df = resampled_sells_df[assets_to_aggregate].sum(axis=1)
        color = label_colors[self.other_assets]
        # Plot the bar chart buys
        ax1.bar(resampled_buys_agg_df.index, resampled_buys_agg_df, label=self.other_assets, bottom=bottoms_buy, width=self.time_bar_width, color=color, alpha=1.0, edgecolor='black')
        # Plot the bar chart sells
        ax1.bar(resampled_sells_agg_df.index, resampled_sells_agg_df, bottom=bottoms_sell, width=self.time_bar_width, color=color, alpha=1.0, edgecolor='black')
        ax.set_title("Transactions Over Time")
        ax.set_xlabel("Time")
        ax1.set_ylabel(f"Amount ({self.symbol})")
        ax2.set_ylabel(f"Value ({self.symbol})")
        ax1.grid(True)
        ax2.grid(True)
        # To enable the grid for minor ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d %Hh"))
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_major_formatter(FuncFormatter(thousands))
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_major_formatter(FuncFormatter(thousands))
        ax.tick_params(axis='y', which="both", right=False, labelright=False, left=False, labelleft=False)
        ax1.tick_params(axis='x', which="both", top=False, labeltop=False, bottom=False, labelbottom=False)
        ax2.tick_params(axis='x', which="both", top=False, labeltop=False, bottom=False, labelbottom=False)
        plt.setp(ax.get_xticklabels(), rotation=-20, ha='left')
        ax1.grid(which="both")
        ax1.grid(which="minor", alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
        ax1.grid(which="major", alpha=0.5, linestyle='-', linewidth=1.0, color='black')
        ax2.grid(which="both")
        ax2.grid(which="minor", alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
        ax2.grid(which="major", alpha=0.5, linestyle='-', linewidth=1.0, color='black')
        ax1.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
        ax2.legend(fontsize=7)
        upper_ylimit = ax2.get_ylim()[1]
        ax2.set_ylim(bottom=0, top=upper_ylimit)

        if fig_ax is None:
            # Show the plot
            plt.show()
    
    def plot_realized_gains(self, fig_ax: Optional[tuple] = None) -> None:
        """
        Method to plot the realized gains of the portfolio.

        """
        if fig_ax is not None:
            fig, ax = fig_ax
        else:
            # Create a figure with multiple subplots
            fig = plt.figure(figsize=(15, 5))
            # Create an Axes object for the figure
            ax = fig.add_subplot(111)
        gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=ax.get_subplotspec(), height_ratios=[1, 1], hspace=0.1)
        ax.grid(False)
        ax1 = fig.add_subplot(gs[0], sharex=ax)
        ax2 = fig.add_subplot(gs[1], sharex=ax)

        label_colors = self.label_colors.copy()

        ax1.axhline(y=0, color='black', linewidth=1)
        ax2.axhline(y=0, color='black', linewidth=1)

        # color_other = label_colors['other']
        realized_gains = self.realized_gains_displayable.copy()
        realized_gains_assets = realized_gains.columns.tolist()
        
        # Cumulative Realized Gains
        realized_gains_cum = realized_gains.sum(axis=1).cumsum()
        realized_gains_cum = self.resample_data(realized_gains_cum, agg_type='last')
        datetime = realized_gains_cum.index
        cum_gains = realized_gains_cum.values
        ax2.plot(datetime, cum_gains, color='black', alpha=1.0, linewidth=1)
        # Fill where the differences
        ax2.fill_between(datetime, cum_gains, where=(cum_gains > 0), color="green", alpha=0.1)
        ax2.fill_between(datetime, cum_gains, where=(cum_gains < 0), color="red", alpha=0.1)
        
        # Granular Realized Gains
        # Resample profits and losses to have a better visualization
        # It is needed to split profits and losses to have a better visualization
        # When resampling profits and losses they would cancel each other in the same sampling window
        resample_agg_type = 'sum'
        factor = 5
        profits_df = realized_gains[realized_gains > 0].fillna(0)
        resampled_profits_df = self.resample_data(profits_df, agg_type=resample_agg_type, factor=factor)
        losses_df = realized_gains[realized_gains < 0].fillna(0)
        resampled_losses_df = self.resample_data(losses_df, agg_type=resample_agg_type, factor=factor)
        # Initialize a variable to keep track of the bottom position for profits
        bottoms_profits = np.zeros(len(resampled_profits_df))
        # Initialize a variable to keep track of the bottom position for losses
        bottoms_losses = np.zeros(len(resampled_losses_df))
        # We only want to display the top and bottom performers
        for column in self.assets_traded_list:
            # We only want to display the top and bottom performers
            # We need to make sure that the asset has realized gains
            if (column in self.assets_to_display_list) and (column in realized_gains_assets):
                color = label_colors[column]
                # Plot the bar chart profits
                ax1.bar(resampled_profits_df.index, resampled_profits_df[column], label=column, bottom=bottoms_profits, width=self.time_bar_width, color=color, alpha=1.0, edgecolor='black', linewidth=0.5)
                # Plot the bar chart losses
                ax1.bar(resampled_losses_df.index, resampled_losses_df[column], bottom=bottoms_losses, width=self.time_bar_width, color=color, alpha=1.0, edgecolor='black', linewidth=0.5)
                # Update bottoms for buys and sells separately if needed
                bottoms_profits += resampled_profits_df[column]
                bottoms_losses += resampled_losses_df[column]
        # For the other assets, we aggregate them
        assets_to_aggregate = [asset for asset in self.assets_traded_list if (asset not in self.assets_to_display_list) and (asset in realized_gains_assets)]
        resampled_profits_agg_df = resampled_profits_df[assets_to_aggregate].sum(axis=1)
        resampled_losses_agg_df = resampled_losses_df[assets_to_aggregate].sum(axis=1)
        color = label_colors[self.other_assets]
        # Plot the bar chart buys
        ax1.bar(resampled_profits_agg_df.index, resampled_profits_agg_df, label=self.other_assets, bottom=bottoms_profits, width=self.time_bar_width, color=color, alpha=1.0, edgecolor='black')
        # Plot the bar chart sells
        ax1.bar(resampled_losses_agg_df.index, resampled_losses_agg_df, bottom=bottoms_losses, width=self.time_bar_width, color=color, alpha=1.0, edgecolor='black')
        ax.set_title("Realized Gains Over Time")
        ax.set_xlabel("Time")
        ax1.set_ylabel(f"Gains ({self.symbol})")
        ax2.set_ylabel(f"Cumulated Gains ({self.symbol})")
        ax1.grid(True)
        ax2.grid(True)
        # To enable the grid for minor ticks
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_major_formatter(FuncFormatter(thousands))
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_major_formatter(FuncFormatter(thousands))
        ax1.grid(which="both")
        ax1.grid(which="minor", alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
        ax1.grid(which="major", alpha=0.5, linestyle='-', linewidth=1.0, color='black')
        ax1.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
        ax2.grid(which="both")
        ax2.grid(which="minor", alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
        ax2.grid(which="major", alpha=0.5, linestyle='-', linewidth=1.0, color='black')
        ax.tick_params(axis='y', which="both", right=False, labelright=False, left=False, labelleft=False)
        ax1.tick_params(axis='x', which="both", top=False, labeltop=False, bottom=False, labelbottom=False)
        ax2.tick_params(axis='x', which="both", top=False, labeltop=False, bottom=False, labelbottom=False)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d %Hh"))
        plt.setp(ax.get_xticklabels(), rotation=-20, ha='left')
        if fig_ax is None:
            # Show the plot
            plt.show()
    
    @property
    def assets_to_display_list(self) -> None:
        """
        Method to create a consistent list of assets to display in the plots.

        """
        top_equity_assets = self.top_ledger_equity_assets_average(n=self.displayed_assets).index.tolist()
        self.other_assets = 'Other Assets'
        assets_to_display_list = top_equity_assets + [self.other_assets]
        return assets_to_display_list
    
    @property
    @check_property_update
    def label_colors(self) -> dict:
        other = 'other'
        labels = [self.symbol, other, *self.assets_to_display_list]
        # Use the predefined color dictionary
        predefined_colors = list(COLOR_PALETTE_DICT.values())  # Extract HEX codes from the color dictionary
        colors = {}
        for i, label in enumerate(labels):
            # Assign colors cyclically if there are more labels than predefined colors
            color_index = i % len(predefined_colors)
            colors[label] = predefined_colors[color_index]
        return colors

    def plot_summary(self, log_scale: bool = True, export_pdf: bool = False) -> None:
        """
        Method to plot the portfolio summary.

        """
        # Create an Axes object for the figure
        fig, ax = plt.subplots(
            nrows=4, ncols=1, figsize=(15, 20), sharex=True
        )

        # Need to recalculate the equity to have the correct values
        # self.calculate_ledger_equity()

        self.plot_benchmark((fig, ax[0]), log_scale)
        self.plot_assets_share((fig, ax[1]))
        self.plot_transactions((fig, ax[2]))
        self.plot_realized_gains((fig, ax[3]))
        
        # Adjust layout to minimize blank space
        fig.tight_layout()
        if export_pdf:
            # Save the plot to a PDF file
            output_file = "portfolio_summary.pdf"
            pdf_file_path = os.path.join(self.output_folder_path, output_file)
            save_plot_pdf(fig=fig, pdf_file_path=pdf_file_path)
            plt.close()
            self.pdf_export_files.append(pdf_file_path)
        else:
            plt.show()