from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

from .record_objects import Capital, CurrencyPrice, Transaction, MetaTransaction
from .support import check_property_update, display_price


@dataclass
class Ledger:
    """
    Dataclass to store the ledger of a portfolio.

    """

    portfolio_symbol: str
    capital: list[Capital] = field(default_factory=list)
    prices: list[CurrencyPrice] = field(default_factory=list)
    transactions: list[Transaction] = field(default_factory=list)
    meta_transactions: list[MetaTransaction] = field(default_factory=list)
    active_assets: list[str] = field(default_factory=list)
    timestamp_low_resolution: int = 500 # samples

    def __post_init__(self):
        # We need to keep track of the properties that have been calculated
        self._properties_evolution_id = dict() # This dictionary will store the evolution_id of each property
        self._properties_cached = dict() # This dictionary will store the cached value of each property

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
        capital = Capital(timestamp=timestamp, action=action, amount=float(amount))
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
        currency_price = CurrencyPrice(timestamp=timestamp, symbol=symbol, price=float(price))
        self.prices.append(currency_price)
    
    def record_meta_transaction(
        self,
        id: int,
        timestamp: int,
        trade: bool,
    ) -> None:
        """
        Method to record a meta transaction in the ledger.

        """
        meta_transaction = MetaTransaction(
            id=id,
            timestamp=timestamp,
            trade=trade,
            ack=False,
        )
        # We only record the meta transaction if the id is not already in the list
        if id not in [mt.id for mt in self.meta_transactions]:
            self.meta_transactions.append(meta_transaction)

    def record_transaction(
        self,
        id: int,
        timestamp: int,
        trade: bool,
        action: str,
        symbol: str,
        amount: float,
        price: float,
        commission: float,
        balance_pre: float,
    ) -> None:
        """
        Method to record a transaction in the ledger.

        """
        traded = amount * price
        transaction = Transaction(
            id=id,
            action=action,
            symbol=symbol,
            amount=float(amount),
            price=float(price),
            traded=float(traded),
            commission=float(commission),
            balance_pre=float(balance_pre),
        )
        self.transactions.append(transaction)
        # We record the meta transaction
        self.record_meta_transaction(id, timestamp, trade)
        # We keep track of the assets that have been traded
        self.record_active_asset(symbol)
    
    def confirm_transaction(self, id: int) -> None:
        """
        Method to confirm a transaction in the ledger.

        """
        transaction = next((t for t in self.meta_transactions if t.id == id), None)
        if transaction is not None:
            transaction.ack = True
    
    def add_message_to_transaction(self, id: int, message: str) -> None:
        """
        Method to add a message to a transaction in the ledger.

        """
        if message:
            transaction = next((t for t in self.meta_transactions if t.id == id), None)
            if transaction is not None:
                transaction.messages.append(message)

    def buy(
        self,
        id: int,
        timestamp: int,
        trade: bool,
        symbol: str,
        amount: float,
        price: float,
        commission: float,
        balance_pre: float,
    ) -> None:
        """
        Method to record a buy transaction in the ledger.

        """
        self.record_transaction(id, timestamp, trade, "BUY", symbol, amount, price, commission, balance_pre)

    def sell(
        self,
        id: int,
        timestamp: int,
        trade: bool,
        symbol: str,
        amount: float,
        price: float,
        commission: float,
        balance_pre: float,
    ) -> None:
        """
        Method to record a sell transaction in the ledger.

        """
        self.record_transaction(id, timestamp, trade, "SELL", symbol, amount, price, commission, balance_pre)

    # Ledger reporting methods ------------------------------------------------
        
    @property
    @check_property_update
    def timestamp_reporting_list(self) -> list:
        """
        Reduce the timestamp resolution of a list of objects.

        """
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
        return timestamps_list

    @property
    @check_property_update
    def total_commissions(self) -> float:
        """
        Property to get the total commissions paid by the portfolio.

        """
        ledger_df = self.transactions_df.copy()
        total_commissions = ledger_df["Commission"].sum()
        return total_commissions

    def transactions_count(self, symbol: Optional[str] = None) -> int:
        """
        Method to count the number of transactions in the ledger.

        If a symbol is provided, it returns the number of transactions for the asset.

        The transactions for the portfolio currency are not counted. As they would be the same as the total number of transactions.

        """
        ledger_df = self.transactions_df.copy()
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
        ledger_df = self.transactions_df.copy()
        if symbol is not None:
            return ledger_df[ledger_df["Symbol"] == symbol]["Traded"].sum()
        else:
            return ledger_df[ledger_df["Symbol"] != self.portfolio_symbol][
                "Traded"
            ].sum()

    @property
    @check_property_update
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
    @check_property_update
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
    @check_property_update
    def prices_raw_df(self) -> pd.DataFrame:
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
    
        columns = [
            "Timestamp",
            "Symbol",
            "Price",
        ]
        df = pd.DataFrame(data, columns=columns)
        return df

    @property
    @check_property_update
    def prices_df(self) -> pd.DataFrame:
        """
        Property to get the currency prices' list as a DataFrame.
        
        It keeps a low resolution of the prices to match the reporting list.
        As well as only keeping the prices of the active assets.

        """
        df = self.prices_raw_df.copy()
        # We only keep the prices of the active assets
        df = df[df["Symbol"].isin(self.active_assets)]
        # We only keep the prices that are in the reporting list
        # The resolution of the reporting list is lower than the prices' timestamps
        df = df[df["Timestamp"].isin(self.timestamp_reporting_list)]
        df.reset_index(drop=True, inplace=True)
        return df

    def get_price_asset_on_timestamp(self, symbol: str, timestamp: int) -> float:
        """
        Method to get the price of an asset in the portfolio at a specific timestamp.

        """
        try:
            prices = self.prices_raw_df
            price = prices[(prices["Symbol"] == symbol) & (prices["Timestamp"] == timestamp)].iloc[0]["Price"]
            return float(price)
        except IndexError:
            raise ValueError(f"Price for {symbol} not found at timestamp {timestamp}")

    @property
    @check_property_update
    def meta_transactions_df(self) -> pd.DataFrame:
        """
        Property to get the meta transactions' list as a DataFrame.

        """
        data = []
        for meta_transaction in self.meta_transactions:
            data.append(
                (
                    meta_transaction.id,
                    meta_transaction.timestamp,
                    meta_transaction.trade,
                    meta_transaction.ack,
                    meta_transaction.messages,
                )
            )
        columns = (
            "Id",
            "Timestamp",
            "Trade",
            "Ack",
            "Messages",
        )
        df = pd.DataFrame(data, columns=columns)
        df['Number_Messages'] = df['Messages'].apply(len)
        return df

    @property
    @check_property_update
    def transactions_df(self) -> pd.DataFrame:
        """
        Property to get the transactions' list as a DataFrame.

        """
        data = []
        for transaction in self.transactions:
            data.append(
                (
                    transaction.id,
                    transaction.action,
                    transaction.symbol,
                    transaction.amount,
                    transaction.price,
                    transaction.traded,
                    transaction.commission,
                    transaction.balance_pre,
                )
            )
        columns = (
            "Id",
            "Action",
            "Symbol",
            "Amount",
            "Price",
            "Traded",
            "Commission",
            "Balance_Pre",
        )
        df = pd.DataFrame(data, columns=columns)
        df = df.merge(self.meta_transactions_df, on="Id")
        return df

    @property
    @check_property_update
    def equity_df(self) -> pd.DataFrame:
        """
        Property to calculate the equity of the portfolio as a DataFrame.

        The time granularity depends on the prices' and transactions' timestamps.

        """
        merging_cols = ["Timestamp", "Symbol"]
        transactions_df = self.transactions_df.copy()
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
        # We only keep the prices that are in the reporting list
        equity_df = equity_df[equity_df.index.isin(self.timestamp_reporting_list)]
        return equity_df

    @property
    @check_property_update
    def traded_assets_values(self) -> pd.DataFrame:
        """
        Property to get the total traded amount for each asset and segregated by action [BUY, SELL].

        """
        transactions = self.transactions_df.copy()
        assets_transactions=transactions[transactions['Symbol']!='USDT']
        traded_assets_values = assets_transactions.pivot_table(index='Symbol', columns='Action', values='Traded', aggfunc='sum').fillna(0)
        if 'BUY' not in traded_assets_values.columns:
            traded_assets_values['BUY']=0
        if 'SELL' not in traded_assets_values.columns:
            traded_assets_values['SELL']=0
        return traded_assets_values
    
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

    # Portfolio logging methods ------------------------------------------------
    
    @property
    @check_property_update
    def log_df(self) -> pd.DataFrame:
        transaction_df=self.transactions_df
        # We only keep the trades, not the capital movements
        trade_df=transaction_df[transaction_df['Trade']].copy()
        trade_df.drop(columns=['Trade'], inplace=True)
        # We segregate the liquid side from the asset side of the trades
        # Afterwards we will merge them thanks to the Transaction id
        asset_trade_df = trade_df[trade_df.Symbol != self.portfolio_symbol].copy()
        # Commissions are burned on the liquid side
        # All assets have 0 commission
        asset_trade_df.drop(columns=['Commission'], inplace=True)
        # We change the name of the columns to avoid confusion
        asset_trade_df.rename(columns={'Balance_Pre':'Balance_Asset_Pre'}, inplace=True)
        asset_trade_df.set_index('Id', inplace=True)
        liquid_trade_df = trade_df[trade_df.Symbol == self.portfolio_symbol].copy()
        # We keep only the necessary columns - Everything else is already in the asset_trade_df - no need to duplicate
        liquid_trade_df=liquid_trade_df[['Id','Commission', 'Balance_Pre']]
        liquid_trade_df.rename(columns={'Balance_Pre':'Balance_Liquid_Pre'}, inplace=True)
        liquid_trade_df.set_index('Id', inplace=True)
        # We merge the asset and liquid trades to have a complete log
        trade_log = asset_trade_df.merge(liquid_trade_df, on='Id')
        # Depending on the action we will add or subtract the traded amount to the balance
        trade_log['Trade_Sign'] = np.where(trade_log['Action'] == 'BUY', 1, -1)
        # We calculate the post trade balances
        trade_log['Balance_Asset_Post'] = trade_log['Balance_Asset_Pre'] + (trade_log['Amount'] * trade_log['Trade_Sign'])
        trade_log['Balance_Liquid_Post'] = trade_log['Balance_Liquid_Pre'] - (trade_log['Traded'] * trade_log['Trade_Sign']) - trade_log['Commission']
        return trade_log
    
    def print_log(self, id: int) -> None:
        """
        Method to print a log entry.
            
        """
        log_entry = self.log_df.loc[id].to_dict()
        trade_type = f"{log_entry["Action"]}ING"
        timestamp = log_entry["Timestamp"]
        symbol = log_entry["Symbol"]
        price = log_entry["Price"]
        amount_base = log_entry["Amount"]
        amount_quote = amount_base * price
        balance_asset_pre = log_entry["Balance_Asset_Pre"]
        balance_asset_post = log_entry["Balance_Asset_Post"]
        balance_liquid_pre = log_entry["Balance_Liquid_Pre"]
        balance_liquid_post = log_entry["Balance_Liquid_Post"]
        ack = ["(NAK)", "(ACK)"][log_entry["Ack"]]
        messages = log_entry["Messages"]
        display_msg = f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {symbol.upper()} - {trade_type.upper()} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        transaction_msg = (
            f"{trade_type.capitalize()} {display_price(amount_base, symbol)} for {display_price(amount_quote, self.portfolio_symbol)} "
            f"at {display_price(price, self.portfolio_symbol)}/{symbol} "
            f"(Commission: {display_price(log_entry['Commission'], self.portfolio_symbol)})"
        )   
        display_msg += f"\n[{id}] {ack}- Timestamp: {timestamp}"
        display_msg += f"\n-> BALANCE PRE-TRADE >> ASSET: {display_price(balance_asset_pre, symbol)} ({display_price(balance_asset_pre * price, self.portfolio_symbol)}) - LIQUIDITY: {display_price(balance_liquid_pre, self.portfolio_symbol)}"
        display_msg += f"\n-> {transaction_msg}"
        display_msg += f"\n-> BALANCE POST-TRADE >> ASSET: {display_price(balance_asset_post, symbol)} ({display_price(balance_asset_post * price, self.portfolio_symbol)}) - LIQUIDITY: {display_price(balance_liquid_post, self.portfolio_symbol)}"
        if messages:
            display_msg += f"\n----- Messages: -----------------------------------------------------------------------"
            count = 0
            for msg in messages:
                count += 1
                display_msg += f"\n-> {count}: {msg}"
        display_msg += f"\n"
        print(display_msg)
    
    def print_logs(self, ids: Optional[list]) -> None:
        """
        Method to print the log entries for a list of ids.
            
        """
        if isinstance(ids, int):
            ids = [ids]
        for id in ids:
            self.print_log(id)
    
    def print_logs_timestamp(self, timestamp: int) -> None:
        """
        Method to print all the log entries for a specific timestamp.
        
        """
        logs_ids = self.log_df[self.log_df['Timestamp'] == timestamp].index
        self.print_logs(logs_ids)
    
    def print_logs_symbol(self, symbol: str) -> None:
        """
        Method to print all the log entries for a specific symbol.
            
        """
        logs_ids = self.log_df[self.log_df['Symbol'] == symbol].index
        self.print_logs(logs_ids)
    
    def print_last_log(self) -> None:
        """
        Method to print the last log entry.
            
        """
        last_log_id = self.log_df.index[-1]
        self.print_log(last_log_id)
    
    def print_logs_all(self) -> None:
        """
        Method to print all the log entries.
            
        """
        self.print_logs(self.log_df.index)
        