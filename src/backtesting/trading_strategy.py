from dataclasses import dataclass, field
import multiprocessing
from multiprocessing import Queue, get_context, Manager
import os
import random
from typing import Optional, Literal, Union, Callable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator, FixedLocator, LogLocator, FuncFormatter
import seaborn as sns
from tqdm import tqdm

from .portfolio import Portfolio
from .support import get_random_name, get_coloured_markers, to_percent_log_growth
from .pandas_units import DataFrameUnits

DEFAULT_INITIAL_ASSETS_LIST = 10

TIME_GRANULARITY_MAP = {
    'D': {
        'label': 'Daily',
        'value': 1,
    },
    'W': {
        'label': 'Weekly',
        'value': 7,
    },
    'M': {
        'label': 'Monthly',
        'value': 30,
    },
    'Q': {
        'label': 'Quarterly',
        'value': 91,
    },
    'Y': {
        'label': 'Yearly',
        'value': 365,
    },
}

@dataclass
class TradingStrategy:
    """
    Dataclass defining the basic structure of a strategy.

    """

    historical_prices: pd.DataFrame
    triggering_feature: str
    threshold_buy: float
    threshold_sell: float
    initial_equity: Optional[float] = None
    quote_ticket_amount: float = 100.0
    maximal_assets_to_buy: int = 5
    commission_trade: float = 0.00075
    commission_transfer: float = 0.0
    portfolio_symbol: str = "USDT"
    description: Optional['str'] = None
    initial_assets_list: Union[list,dict,int] = DEFAULT_INITIAL_ASSETS_LIST # Number of random initial assets
    minimal_liquidity_ratio: float = 0.05
    maximal_equity_per_asset_ratio: float = 0.1
    number_of_portfolios: int = 1
    max_volatility_to_buy: Optional[float] = None
    max_volatility_to_hold: Optional[float] = None
    noise_factor: Optional[float] = None

    def __post_init__(self):
        self.Portfolios = []
        self.Signals = []
        self._prepare_historical_prices()
        self.get_timestamps_list()
        self.current_timestamp = self.initial_timestamp
        self.assets_symbols_list = self.historical_prices['base'].unique()
        self.create_portfolios()

    # def portfolios_overview(self, agg: bool = False) -> pd.DataFrame:
    #     """
    #     Get an overview of the Portfolios.

    #     """
    #     portfolios_overview_list = []
    #     for PF in self.Portfolios:
    #         portfolios_overview_list.append(PF.info_pd.D)
    #     overview_df = pd.DataFrame(portfolios_overview_list)
    #     overview_df.reset_index(inplace=True, drop=False)
    #     overview_df.rename(columns={'index': 'Portfolio index'}, inplace=True)
    #     overview_df.set_index('Name', inplace=True)
    #     return overview_df.T
    
    @property
    def portfolios_dfu(self) -> pd.DataFrame:
        """
        Get an overview of the Portfolios.

        """
        portfolios_dict = dict()
        for PF in self.Portfolios:
            name = PF.name
            portfolios_dict[name] = PF.info_pd
            overview_dfu = DataFrameUnits(portfolios_dict)
        return overview_dfu
    
    @property
    def describe(self) -> pd.DataFrame:
        """
        Describe statistically the strategy.

        """
        drop_columns = ['Name', 'Time start', 'Time end', 'Commissions/Gains ratio']
        described_data = self.portfolios_dfu.describe(drop_columns=drop_columns)
        return described_data
            
    @property
    def portfolios(self) -> pd.DataFrame:
        """
        Property to get the details of the each Portfolio from the strategy.

        """
        return self.portfolios_dfu.D

    def __call__(self, portfolio: Optional[int] = None) -> Union[pd.DataFrame,Portfolio]:
        """
        Display a Portfolio object.

        """
        # Be careful with portfolio 0, it could be assessed as False.
        if portfolio is None:
            return self.describe
        elif portfolio >= len(self.Portfolios):
            msg = f"Portfolio {portfolio} doesn't exist - Number of Portfolios: {len(self.Portfolios)}"
            raise ValueError(msg)
        else:
            return self.Portfolios[portfolio]
    
    def _add_noise_into_signal(self) -> None:
        """
        Add noise into the signal.

        """
        noise_factor = self.noise_factor
        if noise_factor and noise_factor > 0:
            column_name = self.triggering_feature
            historical_prices = self.historical_prices
            std_dev = np.std(historical_prices[column_name])
            std_dev_factor = std_dev * noise_factor
            historical_prices[column_name] = historical_prices[column_name] + np.random.normal(0, std_dev_factor, len(historical_prices))
            self.historical_prices = historical_prices
        elif noise_factor and noise_factor < 0:
            raise ValueError("The noise_factor should be positive.")
    
    def create_new_signal_noise(self, noise_factor: Optional[float] = None) -> pd.Series:
        """
        Get signal series for a given portfolio.
        
        The idea is to get different signals with different noise for each portfolio.

        """
        trigger_feature = self.triggering_feature
        columns = ['base', trigger_feature]
        historical_prices = self.historical_prices
        signal = historical_prices[columns].copy()
        if not noise_factor:
            noise_factor = self.noise_factor
        if not noise_factor:
            # If there is no noise factor, we don't add noise.
            self.Signals.append(signal)
        elif noise_factor >= 0:
            std_dev = np.std(signal[trigger_feature])
            std_dev_factor = std_dev * noise_factor
            signal[trigger_feature] = signal[trigger_feature] + np.random.normal(0, std_dev_factor, len(signal))
            self.Signals.append(signal)
        else:
            raise ValueError("The noise_factor shouldn't be negative.")
    
    def _prepare_historical_prices(self) -> None:
        """
        Prepare the historical prices DataFrame.

        """
        self._add_noise_into_signal()
        # Get rid of the missing values in the historical prices.
        columns = ['price', 'volatility', self.triggering_feature]
        self.historical_prices.dropna(subset=columns, inplace=True)
        # Reset the index of the historical
        self.historical_prices.set_index('timestamp_id', inplace=True)
        
    def get_timestamps_list(self) -> None:
        """
        Get the timestamps of the historical prices.
        
        """
        timestamps_list = self.historical_prices.index.unique().astype(int).tolist()
        timestamps_list.sort()
        self.timestamps_list = timestamps_list
        self.timestamps_list_length = len(timestamps_list)
    
    def get_timestamps_list_length(self) -> int:
        """
        Get the length of the timestamps list.

        """
        return len(self.timestamps_list)
    
    @property
    def initial_timestamp(self) -> pd.Timestamp:
        """
        Get the initial timestamp of the strategy.

        """
        return int(np.min(self.timestamps_list))
    
    @property
    def random_asset_distribution_weights(self) -> tuple[np.ndarray, list]:
        """
        Generate random weights for the assets in the strategy.

        """
        decimals = 4
        # To be sure that we have prices for the initial timestamp, we get the assets available at this timestamp.
        historical_prices_first_timestamp = self.historical_prices.loc[self.initial_timestamp]
        available_initial_assets_list = list(historical_prices_first_timestamp['base'].unique())
        random.shuffle(available_initial_assets_list)
        # We create a list with the assets that form the initial assets list of the portfolio.
        # If there is no initial assets list, we use all the assets in the historical prices.
        assets_symbols_list = []
        if isinstance(self.initial_assets_list, int):
            initial_assets = random.sample(list(available_initial_assets_list), self.initial_assets_list)
            assets_symbols_list.extend(initial_assets)
        # if not self.initial_assets_list:
        #     assets_symbols_list.extend(list(available_initial_assets_list))
        # If there is an initial assets list, we check if the assets are part of the historical prices.
        else:
            for asset in self.initial_assets_list:
                if asset in available_initial_assets_list:
                    assets_symbols_list.append(asset)
                else:
                    print(f"{asset} is not part of the historical prices.")
        # We create random weights for each asset + the cash weight.
        weights = np.random.random(len(assets_symbols_list) + 1)
        # We normalize the weights to make sure they sum up to 1.
        weights_norm = weights / np.sum(weights)
        # We need to make sure, that the last value (cash), has at least the minimal_liquidity_ratio.
        min_ratio = self.minimal_liquidity_ratio
        weights_norm = weights_norm * (1 - min_ratio)
        weights_norm[-1] = weights_norm[-1] + min_ratio
        # The weights_norm's sum should be 1.
        # The last value is the cash weight, so we don't include it.
        weights_assets = weights_norm[:-1]
        weights_assets_rounded_list = list(np.around(weights_assets, decimals))
        # Now we need to provide the weight in the right position matching the assets list.
        # We create a dictionary with the assets and their weights.
        weights_all_assets_symbol_list = []
        for asset in self.assets_symbols_list:
            if asset in assets_symbols_list:
                weight = weights_assets_rounded_list.pop(0)
            else:
                weight = 0
            weights_all_assets_symbol_list.append(weight)
        weights_np = np.array(weights_all_assets_symbol_list)
        return weights_np, assets_symbols_list

    @property
    def next_timestamp(self) -> pd.Timestamp:
        """
        Get the next timestamp of the strategy.

        """
        current_timestamp = self.current_timestamp
        timestamp_list = self.historical_prices.index.unique()
        next_timestamp = timestamp_list[timestamp_list > current_timestamp].min()
        if pd.isna(next_timestamp):
            return None
        else:
            return int(next_timestamp)
    
    def increase_timestamp(self) -> None:
        """
        Increase the timestamp pointer to the next timestamp.

        """
        self.current_timestamp = self.next_timestamp

    
    def get_prices_on_timestamp(self, timestamp: int) -> dict[str, float]:
        """
        Get the prices of the assets in the strategy on a specific timestamp.

        """
        return self.historical_prices.loc[timestamp].set_index('base')['price'].to_dict()
    
    def buy_random_asset(self, portfolio: Portfolio) -> None:
        """
        Buy a random asset in the portfolio.

        """
        buy_msg = "Buying random asset based on the random weights."
        random_weights ,_= self.random_asset_distribution_weights
        initial_balance = portfolio.balance
        initial_timestamp = self.initial_timestamp
        prices = self.get_prices_on_timestamp(initial_timestamp)
        portfolio.update_prices(prices=prices, timestamp=initial_timestamp)
        initial_weights_dict = dict(zip(self.assets_symbols_list,random_weights))
        initial_weights = pd.Series(initial_weights_dict)
        initial_weights = initial_weights[initial_weights > 0]
        initial_assets_dict = dict()
        # We buy the assets based on the random weights.
        for asset, weight in initial_weights.items():
            amount_quote = initial_balance * float(weight)
            # We buy the asset if the amount is higher than 0, if not we skip it.
            if amount_quote > 0:
                portfolio.buy(symbol=asset, amount_quote=amount_quote, reason = "INVESTMENT", timestamp=initial_timestamp, msg=buy_msg)
                # Keep track of the initial assets list.
                initial_assets_dict[asset] = amount_quote
        initial_assets = pd.Series(initial_assets_dict)
        portfolio.initial_assets = initial_assets
    
    def buy_defined_asset(self, portfolio: Portfolio) -> None:
        """
        Buy the assets defined in the initial assets list as a dictionary.
        
        The dictionary contains the assets and their amounts in quote currency.

        """
        buy_msg = "Buying defined asset based on the initial assets list."
        initial_timestamp = self.initial_timestamp
        prices = self.get_prices_on_timestamp(initial_timestamp)
        portfolio.update_prices(prices=prices, timestamp=initial_timestamp)
        portfolio.initial_assets_list = self.initial_assets_list
        initial_assets_dict = dict()
        for asset, amount_quote in self.initial_assets_list.items():
            # We need to skip the portfolio symbol as we can't buy it.
            if asset != self.portfolio_symbol:
                portfolio.buy(symbol=asset, amount_quote=amount_quote, reason = "INVESTMENT", timestamp=initial_timestamp, msg=buy_msg)
                # Keep track of the initial assets list.
                initial_assets_dict[asset] = amount_quote
        initial_assets = pd.Series(initial_assets_dict)
        portfolio.initial_assets = initial_assets

    def create_single_portfolio(self) -> None:
        """
        Create a Portfolio object based on the strategy's attributes.

        Returns:
            Portfolio: Portfolio object.

        """
        portfolio_name = get_random_name()
        # We do a copy if it's not an integer.
        initial_assets_list = self.initial_assets_list.copy() if not isinstance(self.initial_assets_list, int) else self.initial_assets_list
        if isinstance(initial_assets_list, dict):
            initial_equity = None
        else:
            initial_equity = self.initial_equity
        PF = Portfolio(
            name=portfolio_name,
            symbol=self.portfolio_symbol,
            commission_trade=self.commission_trade,
            commission_transfer=self.commission_transfer,
            threshold_buy=self.threshold_buy,
            threshold_sell=self.threshold_sell,
        )
        PF.set_verbosity(verbosity_type='silent')
        # If we provide a dictionary, we use the assets and their amounts.
        if isinstance(self.initial_assets_list, dict):
            if initial_equity:
                raise ValueError("You cannot provide an initial equity with initial assets list as a dictionary.")
            # We calculate the initial equity based on the assets and their amounts.
            initial_equity = pd.Series(initial_assets_list).sum()
            PF.deposit(amount=initial_equity, timestamp=self.initial_timestamp)
            self.buy_defined_asset(PF)
        # If we provide a list or nothing, we use the assets and their weights.
        else:
            PF.deposit(amount=initial_equity, timestamp=self.initial_timestamp)
            self.buy_random_asset(PF)
        self.Portfolios.append(PF)
        self.create_new_signal_noise()

    
    def create_portfolios(self) -> None:
        """
        Create multiple Portfolio objects based on the strategy's attributes.

        """
        for _ in range(self.number_of_portfolios):
            self.create_single_portfolio()
    
    def update_prices(self) -> None:
        """
        Update the prices of the assets in the strategy until the current timestamp.

        """
        prices = self.get_prices_on_timestamp(self.current_timestamp)
        for PF in self.Portfolios:
            PF.update_prices(prices=prices, timestamp=self.current_timestamp)
    
    @property
    def current_volatilities(self) -> pd.Series:
        """
        Calculate the volatility of the prices.

        """
        volatility = self.historical_prices.loc[self.current_timestamp].set_index('base')['volatility']
        return volatility
    
    @property
    def volatility_df(self) -> pd.Series:
        """
        Calculate the volatility of the prices.

        """
        volatility_df = self.historical_prices[['base', 'volatility']].copy()
        volatility_df.reset_index(inplace=True)
        volatility_df.rename(columns={'timestamp_id': 'timestamp', 'base': 'symbol'}, inplace=True)
        return volatility_df

    # @property
    # def current_triggers(self) -> pd.Series:
    #     """
    #     Property to get the current triggers of the strategy for all assets on the current timestamp.

    #     """
    #     historical_prices = self.historical_prices
    #     current_prices = historical_prices[historical_prices.index == self.current_timestamp].copy()
    #     current_prices.set_index('base', inplace=True)
    #     return current_prices[self.triggering_feature]
    
    @property
    def current_triggers(self) -> list[pd.DataFrame]:
        """
        Property to get the current triggers (corresponding to each portfolio) of the strategy for all assets on the current timestamp.

        """
        current_signals = []
        for signal in self.Signals:
            signal_current = signal[signal.index == self.current_timestamp].copy()
            signal_current.set_index('base', inplace=True)
            current_signals.append(signal_current[self.triggering_feature])
        return current_signals

    # @property
    # def signals_df(self) -> pd.DataFrame:
    #     """
    #     Convert the signals list to a DataFrame.

    #     """
    #     signals_df = self.historical_prices[['base', 'label_returns']].copy()
    #     signals_df.reset_index(inplace=True)
    #     signals_df.rename(columns={'timestamp_id': 'timestamp', 'base': 'symbol', 'label_returns': 'value_signal'}, inplace=True)
    #     signals_df['trade_signal'] = 'HOLD'
    #     signals_df.loc[signals_df['value_signal'] >= self.threshold_buy, 'trade_signal'] = 'BUY'
    #     signals_df.loc[signals_df['value_signal'] <= self.threshold_sell, 'trade_signal'] = 'SELL'
    #     signals_df['value_signal'] = signals_df['value_signal'].astype(np.float32)
    #     return signals_df
    
    def prepare_signals_df(self, signal: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the signals list to a DataFrame.

        """
        trigger_feature = self.triggering_feature
        signals_df = signal[['base', trigger_feature]].copy()
        signals_df.reset_index(inplace=True)
        signals_df.rename(columns={'timestamp_id': 'timestamp', 'base': 'symbol', trigger_feature: 'value_signal'}, inplace=True)
        signals_df['trade_signal'] = 'HOLD'
        signals_df.loc[signals_df['value_signal'] >= self.threshold_buy, 'trade_signal'] = 'BUY'
        signals_df.loc[signals_df['value_signal'] <= self.threshold_sell, 'trade_signal'] = 'SELL'
        signals_df['value_signal'] = signals_df['value_signal'].astype(np.float32)
        return signals_df
    
    # @property
    # def current_assets_to_buy(self) -> list:
    #     """
    #     Get the assets to buy.
        
    #     When buying we have to consider a maximal number of assets to buy as there is a limited amount of cash.

    #     """
    #     current_triggers = self.current_triggers
    #     candidates = current_triggers[current_triggers > self.threshold_buy]
    #     # As we want to buy the assets with the highest values, we sort the candidates in descending order.
    #     candidates.sort_values(ascending=False, inplace=True)
    #     return candidates[:self.maximal_assets_to_buy].index.tolist()
    
    # @property
    # def current_assets_to_sell(self) -> list:
    #     """
    #     Get the assets to sell.

    #     """
    #     current_triggers = self.current_triggers
    #     candidates = current_triggers[current_triggers < self.threshold_sell]
    #     # No need to sort the candidates as we are returning all the assets under the threshold.
    #     return candidates.index.tolist()
    
    def current_assets_to_buy(self, i_PF: int) -> list:
        """
        Get the assets to buy.
        
        When buying we have to consider a maximal number of assets to buy as there is a limited amount of cash.

        """
        current_triggers = self.current_triggers[i_PF].copy()
        candidates = current_triggers[current_triggers > self.threshold_buy]
        # As we want to buy the assets with the highest values, we sort the candidates in descending order.
        candidates.sort_values(ascending=False, inplace=True)
        return candidates[:self.maximal_assets_to_buy].index.tolist()
    
    def current_assets_to_sell(self, i_PF: int) -> list:
        """
        Get the assets to sell.

        """
        current_triggers = self.current_triggers[i_PF].copy()
        candidates = current_triggers[current_triggers < self.threshold_sell]
        # No need to sort the candidates as we are returning all the assets under the threshold.
        return candidates.index.tolist()
    
    # def get_current_triggers_on_positive_balance_assets(self, asset_list: list) -> pd.Series:
    #     """
    #     Get the triggers of the assets with a positive balance.
        
    #     """
    #     triggers = self.current_triggers
    #     return triggers[triggers.index.isin(asset_list)]
        
    def ensure_liquidity(self) -> None:
        """
        Ensure liquidity in the Portfolios.

        """
        sell_msg = "Selling asset to ensure liquidity."
        for i_PF, PF in enumerate(self.Portfolios):
            min_liquidity = self.minimal_liquidity_ratio * PF.equity_value
            while PF.balance < min_liquidity:
                # We find out the owned assets with low performance.
                triggers = self.current_triggers[i_PF]
                # We get the assets that we own.
                owned_assets = PF.positive_balance_assets_list
                # We get the triggers of the assets that we own.
                triggers_owned_assets = triggers[triggers.index.isin(owned_assets)]
                # We get the asset with the lowest performance.
                lowest_asset = triggers_owned_assets.idxmin()
                try:
                    # We sell the asset with the lowest performance.
                    PF.sell(symbol=lowest_asset, amount_quote=self.quote_ticket_amount, reason = "LIQUIDITY", timestamp=self.current_timestamp, msg=sell_msg)
                except Exception as e:
                    print(f"[{PF.name}] [{self.current_timestamp}] [{lowest_asset}] [SELL - LIQUIDITY] - Error: {e}")
    
    def trade_on_signals(self) -> None:
        """
        Apply the signals to the Portfolios.

        """
        self.ensure_liquidity()
        volatilities = self.current_volatilities
        for i_PF, PF in enumerate(self.Portfolios):
            # ASSETS TO BUY >>>>>>>>>>>
            current_assets_to_buy = self.current_assets_to_buy(i_PF)
            for asset in current_assets_to_buy:
                try:
                    volatility = volatilities[volatilities.index == asset].values[0]
                    # We buy the asset only if the volatility is below the threshold.
                    if (volatility) and ((volatility < self.max_volatility_to_buy) or (self.max_volatility_to_buy is None)):
                        # We buy the asset only if we have enough cash plus we still have enough liquidity.
                        # The minimal threshold liquidity ensures to still have some cash after buying the asset.
                        mininmal_threshold_liquidity = self.minimal_liquidity_ratio * PF.equity_value / 100
                        if PF.balance > (self.quote_ticket_amount + mininmal_threshold_liquidity):
                            # We buy the asset only if the maximal equity per asset ratio is not reached.
                            asset_equity_ratio = PF.get_value(symbol=asset, quote='USDT') / PF.equity_value
                            if asset_equity_ratio < self.maximal_equity_per_asset_ratio:
                                PF.buy(symbol=asset, amount_quote=self.quote_ticket_amount, reason = "SIGNAL", timestamp=self.current_timestamp)
                except Exception as e:
                    print(f"[{PF.name}] [{self.current_timestamp}] [{asset}] [BUY] - Error: {e}")
            # ASSETS TO SELL IF SIGNALS >>>>>>>>>>>
            current_assets_to_sell = self.current_assets_to_sell(i_PF)
            for asset in current_assets_to_sell:
                try:
                    # We sell the asset only if we own the asset.
                    if asset in PF.positive_balance_assets_list:
                        msg = f"Selling asset based on signal."
                        PF.sell(symbol=asset, amount_quote=self.quote_ticket_amount, reason = "SIGNAL", timestamp=self.current_timestamp, msg=msg)
                except Exception as e:
                    print(f"[{PF.name}] [{self.current_timestamp}] [{asset}] [SELL - SIGNAL] - Error: {e}")
            # ASSETS TO SELL IF VOLATILITY TOO HIGH >>>>>>>>>>>
            for asset in PF.positive_balance_assets_list:
                try:
                    volatility = volatilities[volatilities.index == asset].values[0]
                    # We sell the asset only if the volatility is above the threshold.
                    if (volatility) and (volatility > self.max_volatility_to_hold):
                        msg = f"Selling asset as the volatility is too high: {volatility:.4f}."
                        PF.sell(symbol=asset, amount_quote=self.quote_ticket_amount, reason = "VOLATILITY", timestamp=self.current_timestamp, msg=msg)
                except Exception as e:
                    print(f"[{PF.name}] [{self.current_timestamp}] [{asset}] [SELL - VOLATILITY] - Error: {e}")
    
    def dispatch_signals_in_portfolios(self) -> None:
        """
        In order to make the signals available in the Portfolios, we dispatch the signals_df and volatility_df in the Portfolios.

        """
        for PF, signal in zip(self.Portfolios, self.Signals):
            PF.signals_df = self.prepare_signals_df(signal = signal)
            PF.volatility_df = self.volatility_df

    def run_strategy(self) -> None:
        """
        Run the strategy.

        """
        # The first timestamp is the initial timestamp when we buy all the assets.
        # Therefore we increase the timestamp before starting to trade.
        self.increase_timestamp()
        with tqdm(total=self.timestamps_list_length-1, position=0, leave=True) as pbar:
            while self.current_timestamp is not None:
                self.update_prices()
                self.trade_on_signals()
                self.increase_timestamp()
                pbar.update(1)
        self.dispatch_signals_in_portfolios()
    
    @property
    def performance(self) -> list[dict]:
        """
        Get the performance of the strategy for all the Portfolios.

        """
        performance_list = []
        for PF in self.Portfolios:
            perfo_dict = dict(
                name = PF.name,
                timerange = PF.timerange,
                investment = PF.invested_capital,
                transactions = PF.transactions_count(),
                traded = PF.transactions_sum(),
                gains = PF.gains,
                roi = PF.roi,
                commissions = PF.total_commissions,
                hold_gains = PF.hold_gains,
                hold_roi = PF.hold_roi,
            )
            performance_list.append(perfo_dict)
        return performance_list

    @property
    def performance_df(self) -> pd.DataFrame:
        """
        Get the performance of the strategy for all the Portfolios as a DataFrame.

        """
        performace_df = pd.DataFrame(self.performance)
        return performace_df

@dataclass
class MultiPeriodBacktest:
    """
    Dataclass that runs a multi-period backtest on a given strategy.
    
    This Class runs on multiprocessing to speed up the backtest.
    However, it doesn't keep track of the objects created in the TradingStrategy class, but only the performance.

    """

    data_path: str
    triggering_feature: str
    threshold_buy: float
    threshold_sell: float
    number_timeperiods: Optional[int] = None
    time_granularity: Optional[int] = None
    # signals: list[str] = field(default_factory=list)
    initial_equity: Optional[float] = None
    quote_ticket_amount: float = 100.0
    maximal_assets_to_buy: int = 5
    commission_trade: float = 0.00075
    commission_transfer: float = 0.0
    portfolio_symbol: str = "USDT"
    description: Optional['str'] = None
    initial_assets_list: Union[list,dict,int] = DEFAULT_INITIAL_ASSETS_LIST # Number of random initial assets
    minimal_liquidity_ratio: float = 0.05
    maximal_equity_per_asset_ratio: float = 0.1
    number_of_portfolios: int = 1
    max_volatility_to_buy: Optional[float] = None
    max_volatility_to_hold: Optional[float] = None
    noise_factor: Optional[float] = None
    
    def __post_init__(self):
        self.backtest_performed = False
        self.get_files_list()
        self.coloured_markers = get_coloured_markers()
        self.coloured_markers_number = len(self.coloured_markers)
    
    def get_files_list(self) -> None:
        """
        Get the list of files in the data path.

        """
        files_list = os.listdir(self.data_path)
        files_list.sort()
        self.files_list = files_list
        
    def read_historical_prices(self, file: str) -> pd.DataFrame:
        """
        Read the historical prices from a file.

        """
        historical_prices = pd.read_feather(os.path.join(self.data_path, file))
        # We fill the missing labels with 0 - This should only happen on the first data files
        historical_prices['label_returns'] = historical_prices['label_returns'].fillna(0)
        # Reduce the granularity of the historical prices.
        if self.time_granularity:
            historical_prices = historical_prices[historical_prices['timestamp_id'] % self.time_granularity == 0]
        return historical_prices
        
    def TradingStrategy(self, historical_prices: pd.DataFrame) -> TradingStrategy:
        """
        Create the TradingStrategy object.

        """
        TradingStrategy_obj = TradingStrategy(
            historical_prices=historical_prices,
            triggering_feature=self.triggering_feature,
            threshold_buy=self.threshold_buy,
            threshold_sell=self.threshold_sell,
            # signals=self.signals,
            initial_equity=self.initial_equity,
            quote_ticket_amount=self.quote_ticket_amount,
            maximal_assets_to_buy=self.maximal_assets_to_buy,
            commission_trade=self.commission_trade,
            commission_transfer=self.commission_transfer,
            portfolio_symbol=self.portfolio_symbol,
            description=self.description,
            initial_assets_list=self.initial_assets_list,
            minimal_liquidity_ratio=self.minimal_liquidity_ratio,
            maximal_equity_per_asset_ratio=self.maximal_equity_per_asset_ratio,
            number_of_portfolios=self.number_of_portfolios,
            max_volatility_to_buy=self.max_volatility_to_buy,
            max_volatility_to_hold=self.max_volatility_to_hold,
            noise_factor=self.noise_factor,
        )
        return TradingStrategy_obj
        
    def _launch_single_backtest(self, file: str, queue: Queue) -> None:
        """
        Launch a single backtest.
        
        This is used in the multiprocessing pool.
        However, we don't collect any TradingStrategy objects, but only the performance.
        
        """
        historical_prices = self.read_historical_prices(file)
        root_file = file.split('.')[0]
        TS = self.TradingStrategy(historical_prices=historical_prices)
        TS.run_strategy()
        performance_df = TS.performance_df
        queue.put((root_file, performance_df))
    
    def _run_backtest(self) -> None:
        """
        Run the backtest.
        """
        files_list = self.files_list
        if self.number_timeperiods:
            files_list = files_list[:self.number_timeperiods]
        if not self.backtest_performed:
            print("Launching the backtest in parallel...")
            ctx = get_context("spawn")
            manager = Manager()
            queue = manager.Queue()

            with ctx.Pool() as pool:
                pool.starmap(self._launch_single_backtest, [(file, queue) for file in files_list])

            print("Backtest finished and waiting for the results...")
            # Collect results from the queue
            performance_dict = dict()
            while not queue.empty():
                file, performance = queue.get()
                print(f"Getting the results for {file}")
                performance_dict[file] = performance

            performance_df = pd.DataFrame()
            for file, performance_TS in performance_dict.items():
                performance_TS['file'] = file
                performance_df = pd.concat([performance_df, performance_TS], ignore_index=True)
            performance_df.sort_values(by='file', inplace=True)
            performance_df.reset_index(drop=True, inplace=True)
            self._performance = performance_df
            self.backtest_performed = True
            print("Backtest completed.")
        else:
            print("The backtest has already been performed - No need to run it again.")

    @property
    def performance(self) -> pd.DataFrame:
        """
        Get the performance of the backtest.

        """
        if not self.backtest_performed:
            self._run_backtest()
        return self._performance
    
    @property
    def performance_overview(self) -> pd.DataFrame:
        """
        Get the performance overview of the backtest.

        """
        values_columns = ['transactions', 'traded', 'gains', 'roi', 'hold_gains', 'hold_roi']
        agg_columns = ['mean', 'std']
        performance_df = self.performance
        performace_pivot = performance_df.pivot_table(index='file', values=values_columns, aggfunc=agg_columns)
        return performace_pivot
    
    @property
    def roi_performance_df(self) -> pd.DataFrame:
        """
        Property that provides the performance of the backtest based on the ROI.

        """
        cols_roi = ['roi', 'hold_roi']
        col_timerange = 'timerange'
        cols_needed = cols_roi + [col_timerange]
        roi_perfo = self.performance[cols_needed].copy()
        roi_perfo['timestart'] = roi_perfo[col_timerange].apply(lambda x: x[0])
        return roi_perfo
    
    @property
    def roi_performance_daily_df(self) -> pd.DataFrame:
        """
        Property that provides the performance of the backtest based on the ROI on a daily basis.
        
        """
        def convert_returns_to_daily_returns(returns, timespan):
            day= 24*3600
            daily_returns = (1 + returns) ** (day/timespan) - 1
            return daily_returns
        roi_perfo = self.roi_performance_df.copy()
        roi_perfo['timespan'] = roi_perfo.timerange.apply(lambda x: x[1]-x[0])
        roi_perfo['roi'] = roi_perfo.apply(lambda x: convert_returns_to_daily_returns(x['roi'], x['timespan']), axis=1)
        roi_perfo['hold_roi'] = roi_perfo.apply(lambda x: convert_returns_to_daily_returns(x['hold_roi'], x['timespan']), axis=1)
        return roi_perfo
    
    def change_roi_performance_time_granularity(self, time_granularity: Literal['D','W','M','Q','Y']) -> pd.DataFrame:
        """
        Change the time granularity of the ROI performance.

        D: Daily
        W: Weekly
        M: Monthly
        Q: Quarterly
        Y: Yearly
        
        """
        roi_perfo = self.roi_performance_daily_df.copy()
        if time_granularity != 'D':
            for col in ['roi', 'hold_roi']:
                roi_perfo[col] = roi_perfo[col].apply(lambda x: (1 + x) ** TIME_GRANULARITY_MAP[time_granularity]['value'] - 1)
        return roi_perfo
    
    # SIMULATION RETURNS
    
    def _stochastic_compounded_single_performance(self, roi_performance: pd.DataFrame, days: int) -> pd.Series:
        """
        Get the random performance of the backtest based on the ROI for a single day and compounded over a number of days.
        
        """
        size = len(roi_performance)
        randnum = np.random.randint(0, high=size, size=days)
        strat_roi_list = []
        hold_roi_list = []
        for i in randnum:
            values = roi_performance.iloc[i]
            hold_roi_list.append(values['hold_roi'])
            strat_roi_list.append(values['roi'])
        values_df = pd.DataFrame([hold_roi_list,strat_roi_list], index=['Hold','Strategy']).T
        values_df = values_df + 1
        compound_df = values_df.prod()
        compound_df = compound_df - 1
        return compound_df
    
    def _run_multiprocess_probabilistic_performance_batch(self, roi_performance: pd.DataFrame, days: int, iterations: int, queue: Queue) -> None:
        """
        Run the stochastic compounded performance in a multiprocessing pool.

        """
        results_list = []
        for i in range(iterations):
            results_list.append(self._stochastic_compounded_single_performance(roi_performance = roi_performance, days=days))
        queue.put(results_list)

    def probabilistic_performance(self, days: int = 1, iterations: int = 20_000) -> pd.DataFrame:
        """
        Get the probabilistic performance of the backtest based on empirical ROI data.
        
        To get the performance, we compound the daily returns over a number of days.
        
        To speed up the process, we use multiprocessing.
        
        """
        results_list = []
        roi_performance = self.roi_performance_daily_df.copy()
        iterations_per_batch = 1_000
        batchs = iterations // iterations_per_batch
        ctx = get_context("spawn")
        manager = Manager()
        queue = manager.Queue()

        with ctx.Pool() as pool:
            pool.starmap(self._run_multiprocess_probabilistic_performance_batch, [(roi_performance, days, iterations_per_batch, queue) for _ in range(batchs)])
        while not queue.empty():
            batch_results_list = queue.get()
            results_list.extend(batch_results_list)
        results_df = pd.DataFrame(results_list)
        return results_df
    
    # PLOTTING
    
    def plot_roi_performance(self, time_granularity: Literal['D','W','M','Q','Y'] = 'D') -> None:
        """
        Plot the performance of the backtest based on the ROI.

        """
        def get_prob_tickers_labels(ticks_labels):
            """
            Get rid of the 0% on the y-axis and make sure that the labels are not too long.

            """
            tick_labels_format = []
            magnitude = int(np.ceil(np.abs(np.log10(ticks_labels[1]))) + 2)
            for tick in ticks_labels:
                if tick != 0:
                    tick = str(tick)
                    if len(tick) > magnitude:
                        tick = f'{tick[:magnitude + 1]}%'
                    else:
                        tick = f'{tick}%'
                    tick_labels_format.append(tick)
                    
                else:
                    tick_labels_format.append('')
            return tick_labels_format
        time_label = TIME_GRANULARITY_MAP[time_granularity]['label']
        time_value = TIME_GRANULARITY_MAP[time_granularity]['value']
        labels_size = 14
        tickers_size = 12
        roi_perfo = self.change_roi_performance_time_granularity(time_granularity)
        fig, ax = plt.subplots(figsize=(15, 15))
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        ax.tick_params(axis='x', which="both", top=False, labeltop=False, bottom=False, labelbottom=False)
        ax.tick_params(axis='y', which="both", left=False, labelleft=False, right=False, labelright=False)
        gs = gridspec.GridSpecFromSubplotSpec(2, 2,
                                              subplot_spec=ax.get_subplotspec(),
                                              height_ratios=[3, 6],
                                              width_ratios=[6, 3],
                                              hspace=0,
                                              wspace=0)
        # Create subplots within the GridSpec
        ax_scatter = fig.add_subplot(gs[1, 0])
        ax_hist_hold = fig.add_subplot(gs[0, 0])
        ax_hist_str = fig.add_subplot(gs[1, 1])
        ax_boxes = fig.add_subplot(gs[0, 1])
        # ax_hist_hold.set_xticklabels([])
        # ax_hist_str.set_yticklabels([])
        ax_hist_hold.tick_params(axis='x', which="both", top=False, labeltop=False, bottom=False, labelbottom=False)
        ax_hist_str.tick_params(axis='y', which="both", left=False, labelleft=False, right=False, labelright=False)
        cols_roi = ['roi', 'hold_roi']
        rois_values = roi_perfo[cols_roi].values
        min_roi = np.min(rois_values)
        max_roi = np.max(rois_values)
        roi_span = max_roi - min_roi
        max_displ_roi = max_roi + 0.2 * roi_span
        min_displ_roi = min_roi - 0.2 * roi_span
        counter = 0
        periods = roi_perfo['timestart'].unique()
        number_timeperiods = len(periods)
        number_portfolios = self.number_of_portfolios
        number_simulations = number_timeperiods * number_portfolios
        simulations_info = f"Number of time periods: {number_timeperiods}\nNumber of portfolios: {number_portfolios}\nNumber of simulations: {number_simulations}"
        ax_scatter.axhline(y=0, color='darkgoldenrod', linestyle='-', linewidth=1.0)
        ax_scatter.axvline(x=0, color='darkgoldenrod', linestyle='-', linewidth=1.0)
        # Diagonal line for reference Hold = Strategy
        ax_scatter.plot([min_displ_roi, max_displ_roi], [min_displ_roi, max_displ_roi], color='orange', linestyle='--', linewidth=0.5, alpha=0.6)
        ax_scatter.text(x=(min_displ_roi*0.99), y=(min_displ_roi*0.96), s='Same performance', color='orange', fontsize=9, ha='left', va='bottom', rotation=45)
        # Diagonal line twice better than Hold
        ax_scatter.plot([min_displ_roi, 0], [min_displ_roi/2, 0], color='green', linestyle='--', linewidth=0.5, alpha=0.6)
        ax_scatter.plot([0, max_displ_roi], [0, max_displ_roi*2], color='green', linestyle='--', linewidth=0.5, alpha=0.6)
        ax_scatter.text(x=(min_displ_roi*0.99), y=(min_displ_roi*0.49), s='2x Better', color='green', fontsize=9, ha='left', va='bottom', rotation=26.57)
        # Diagonal line 5x better than Hold
        ax_scatter.plot([min_displ_roi, 0], [min_displ_roi/5, 0], color='green', linestyle=':', linewidth=0.7, alpha=0.9)
        ax_scatter.plot([0, max_displ_roi], [0, max_displ_roi*5], color='green', linestyle=':', linewidth=0.7, alpha=0.9)
        ax_scatter.text(x=(min_displ_roi*0.99), y=(min_displ_roi*0.195), s='5x Better', color='green', fontsize=9, ha='left', va='bottom', rotation=11.31)
        # Diagonal line twice worse than Hold
        ax_scatter.plot([min_displ_roi, 0], [min_displ_roi*2, 0], color='red', linestyle='--', linewidth=0.5, alpha=0.6)
        ax_scatter.plot([0, max_displ_roi], [0, max_displ_roi/2], color='red', linestyle='--', linewidth=0.5, alpha=0.6)
        ax_scatter.text(x=(min_displ_roi*0.55), y=(min_displ_roi*0.99), s='2x Worse', color='red', fontsize=9, ha='left', va='bottom', rotation=63.43)
        # Diagonal line 5x worse than Hold
        ax_scatter.plot([min_displ_roi, 0], [min_displ_roi*5, 0], color='red', linestyle=':', linewidth=0.7, alpha=0.9)
        ax_scatter.plot([0, max_displ_roi], [0, max_displ_roi/5], color='red', linestyle=':', linewidth=0.7, alpha=0.9)
        ax_scatter.text(x=(min_displ_roi*0.26), y=(min_displ_roi*0.99), s='5x Worse', color='red', fontsize=9, ha='left', va='bottom', rotation=78.69)
        for timestart in periods:
            # As we have a limited number of markers, we need to recycle them.
            marker, color = self.coloured_markers[counter % self.coloured_markers_number]
            x = roi_perfo[roi_perfo['timestart'] == timestart]['hold_roi']
            y = roi_perfo[roi_perfo['timestart'] == timestart]['roi']
            label = timestart
            ax_scatter.scatter(x=x, y=y, label=label, color=color, marker=marker, alpha=0.1, s=120)
            counter += 1
        # SCATTER PLOT
        ax_scatter.set_xlabel(f'Hold {time_label} ROI (%)', fontsize=labels_size)
        ax_scatter.set_ylabel(f'Strategy {time_label} ROI (%)', fontsize=labels_size)
        if len(periods) <= 40:
            ax_scatter.legend(fontsize='small')
        ax_scatter.grid(True)
        ax_scatter.xaxis.set_minor_locator(AutoMinorLocator())
        ax_scatter.yaxis.set_minor_locator(AutoMinorLocator())
        ax_scatter.grid(which="both")
        ax_scatter.grid(which="minor", alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
        ax_scatter.grid(which="major", alpha=0.5, linestyle='-', linewidth=1.0, color='black')
        ax_scatter.set_xlim(min_displ_roi, max_displ_roi)
        ax_scatter.set_ylim(min_displ_roi, max_displ_roi)
        # # Convert the x and y axis values to percentage
        ax_scatter.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
        ax_scatter.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        # Use tick_params to set the font size of the tick labels
        ax_scatter.tick_params(axis='x', labelsize=tickers_size, rotation=-90)
        ax_scatter.tick_params(axis='y', labelsize=tickers_size)
        ## Print note
        ax_scatter.annotate(
            simulations_info,
            xy=(0.99, 0),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=8,
            alpha=1,
        )
        # HISTOGRAMS
        # Probabilites calculation
        bins = np.linspace(min_displ_roi, max_displ_roi, 100)
        ## Hold ROI
        hold_roi = roi_perfo.hold_roi.copy()
        hist_hold_roi, bins_hold_roi = np.histogram(hold_roi, bins=bins)
        prob_hold_roi = hist_hold_roi / hist_hold_roi.sum()
        cum_prob_hold_roi = np.cumsum(prob_hold_roi)
        ## Strategy ROI
        strat_roi = roi_perfo.roi.copy()
        hist_strat_roi, bins_strat_roi = np.histogram(strat_roi, bins=bins)
        prob_strat_roi = hist_strat_roi / hist_strat_roi.sum()
        cum_prob_strat_roi = np.cumsum(prob_strat_roi)
        # Display the histograms
        ## Hold ROI
        ax_hist_hold.axvline(x=0, color='darkgoldenrod', linestyle='-', linewidth=1.0)
        ax_hist_hold.bar(bins_hold_roi[:-1], prob_hold_roi, width=np.diff(bins_hold_roi), alpha=0.3, color='blue')
        ax_hist_hold.bar(bins_hold_roi[:-1], prob_hold_roi, width=np.diff(bins_hold_roi), alpha=0.3, color='blue')
        ax_hist_hold.set_xlim(min_displ_roi, max_displ_roi)
        ax_hist_hold.xaxis.set_minor_locator(AutoMinorLocator())
        ax_hist_hold.yaxis.set_minor_locator(AutoMinorLocator())
        ax_hist_hold.grid(which="both")
        ax_hist_hold.grid(which="minor", alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
        ax_hist_hold.grid(which="major", alpha=0.5, linestyle='-', linewidth=1.0, color='black')
        ax_hist_hold.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
        ax_hist_hold.tick_params(axis='y', labelsize=tickers_size)
        ax_hist_hold.set_ylabel('Hold Probabilitly (%)', fontsize=labels_size)
        ### Avoid displaying 0% on the y-axis
        yticks = ax_hist_hold.get_yticks()
        ytick_labels = get_prob_tickers_labels(yticks)
        ax_hist_hold.yaxis.set_major_locator(FixedLocator(yticks))
        ax_hist_hold.set_yticklabels(ytick_labels)
        ## Cum Hold ROI
        ax_hist_hold_cum = ax_hist_hold.twinx()
        ax_hist_hold_cum.plot(bins_hold_roi[:-1], cum_prob_hold_roi, color='purple', label='Cumulative Probability')
        ax_hist_hold_cum.set_ylim(0, 1)
        # ax_hist_hold_cum.set_yticks([])
        # ax_hist_hold_cum.set_yticklabels([])
        ax_hist_hold_cum.tick_params(axis='y', which="both", left=False, labelleft=False, right=False, labelright=False)
        ax_hist_hold_cum.legend(fontsize='small')
        ax_hist_hold_cum.axhline(y=0.25, color='red', linestyle='--', linewidth=0.8)
        ax_hist_hold_cum.text(x=max_roi, y=0.26, s='Cum. Prob. 25%', color='red', fontsize=9, ha='right', va='bottom')
        ax_hist_hold_cum.axhline(y=0.5, color='orange', linestyle='--', linewidth=0.8)
        ax_hist_hold_cum.text(x=max_roi, y=0.51, s='Cum. Prob. 50%', color='orange', fontsize=9, ha='right', va='bottom')
        ax_hist_hold_cum.axhline(y=0.75, color='green', linestyle='--', linewidth=0.8)
        ax_hist_hold_cum.text(x=max_roi, y=0.76, s='Cum. Prob. 75%', color='green', fontsize=9, ha='right', va='bottom')
        ## Strategy ROI
        ax_hist_str.axhline(y=0, color='darkgoldenrod', linestyle='-', linewidth=1.0)
        ax_hist_str.barh(bins_strat_roi[:-1], prob_strat_roi, height=np.diff(bins_strat_roi), alpha=0.3, color='blue')
        ax_hist_str.set_ylim(min_displ_roi, max_displ_roi)
        ax_hist_str.xaxis.set_minor_locator(AutoMinorLocator())
        ax_hist_str.yaxis.set_minor_locator(AutoMinorLocator())
        ax_hist_str.grid(which="both")
        ax_hist_str.grid(which="minor", alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
        ax_hist_str.grid(which="major", alpha=0.5, linestyle='-', linewidth=1.0, color='black')
        ax_hist_str.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
        ax_hist_str.tick_params(axis='x', labelsize=tickers_size, rotation=-90)
        ax_hist_str.set_xlabel('Strategy Probabilitly (%)', fontsize=labels_size)
        ### Avoid displaying 0% on the y-axis
        xticks = ax_hist_str.get_xticks()
        xtick_labels = get_prob_tickers_labels(xticks)
        ax_hist_str.xaxis.set_major_locator(FixedLocator(xticks))
        ax_hist_str.set_xticklabels(xtick_labels)
        ## Cum Strategy ROI
        ax_hist_str_cum = ax_hist_str.twiny()
        ax_hist_str_cum.plot(cum_prob_strat_roi, bins_strat_roi[:-1], color='purple', label='Cumulative Probability')
        ax_hist_str_cum.set_xlim(0, 1)
        # ax_hist_str_cum.set_xticks([])
        # ax_hist_str_cum.set_xticklabels([])
        ax_hist_str_cum.tick_params(axis='x', which="both", top=False, labeltop=False, bottom=False, labelbottom=False)
        ax_hist_str_cum.axvline(x=0.25, color='red', linestyle='--', linewidth=0.8)
        ax_hist_str_cum.text(x=0.26, y=max_roi, s='Cum. Prob. 25%', color='red', fontsize=9, ha='left', va='top', rotation=-90)
        ax_hist_str_cum.axvline(x=0.5, color='orange', linestyle='--', linewidth=0.8)
        ax_hist_str_cum.text(x=0.51, y=max_roi, s='Cum. Prob. 50%', color='orange', fontsize=9, ha='left', va='top', rotation=-90)
        ax_hist_str_cum.axvline(x=0.75, color='green', linestyle='--', linewidth=0.8)
        ax_hist_str_cum.text(x=0.76, y=max_roi, s='Cum. Prob. 75%', color='green', fontsize=9, ha='left', va='top', rotation=-90)
        # BOX PLOT
        # Simulate data (replace this with your actual data)
        simulated_compounded_returns = self.probabilistic_performance(days=time_value) + 1

        # Create the boxplot on the specific Axes object
        sns.boxplot(data=simulated_compounded_returns, palette="pastel", width=0.6, linewidth=2.5, ax=ax_boxes)
        ax_boxes.set_yscale('log')
        ax_boxes.axhline(y=1, color='darkgoldenrod', linestyle='-', linewidth=1.0)
        # Enhance the plot with a descriptive title and labels
        ax_boxes.set_ylabel(f'{time_label} Capital Growth', fontsize=labels_size)
        ax_boxes.yaxis.set_label_position("right")

        # Customize gridlines and layout
        ax_boxes.tick_params(axis='x', labelsize=labels_size, top=True, labeltop=True, bottom=False, labelbottom=False)
        ax_boxes.tick_params(axis='y', which='both', labelsize=tickers_size, left=False, labelleft=False, right=True, labelright=True)
        number_minor_ticks = 10
        ax_boxes.yaxis.set_major_locator(LogLocator(base=2.0, numticks=10))
        ax_boxes.yaxis.set_minor_locator(LogLocator(base=2.0, subs=np.arange(1, int(number_minor_ticks/2)) * (1/number_minor_ticks), numticks=number_minor_ticks))
        ax_boxes.yaxis.set_major_formatter(FuncFormatter(to_percent_log_growth))
        ax_boxes.yaxis.set_minor_formatter(FuncFormatter(to_percent_log_growth))
        ax_boxes.grid(which="minor", alpha=0.3, linestyle='--', linewidth=0.5, color='gray', axis='y')
        ax_boxes.grid(which="major", alpha=0.5, linestyle='-', linewidth=1.0, color='black', axis='y')
        # ax_boxes.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
        
        ax.set_title(f'{time_label} ROI', fontsize=22)
        plt.show()