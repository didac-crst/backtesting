from dataclasses import dataclass, field
from typing import Optional, Literal, Union

import pandas as pd
import numpy as np
from tqdm import tqdm

from backtesting import Portfolio

@dataclass
class TradingStrategies:
    """
    Dataclass defining the basic structure of a strategy.

    """

    historical_prices: pd.DataFrame
    triggering_feature: str
    threshold_buy: float
    threshold_sell: float
    signals: list[str] = field(default_factory=list)
    initial_equity: Optional[float] = None
    quote_ticket_amount: float = 100.0
    maximal_assets_to_buy: int = 5
    commission_trade: float = 0.00075
    commission_transfer: float = 0.0
    portfolio_symbol: str = "USDT"
    description: Optional['str'] = None
    initial_assets_list: Optional[Union[list,dict]] = None
    minimal_liquidity_ratio: float = 0.05
    maximal_equity_per_asset_ratio: float = 0.1
    number_of_portfolios: int = 1

    def __post_init__(self):
        self.Portfolios = []
        self.historical_prices.set_index('timestamp_id', inplace=True)
        self.get_timestamps_list()
        self.current_timestamp = self.initial_timestamp
        self.assets_symbols_list = self.historical_prices['base'].unique()
        self.create_portfolios()
        
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
    def random_asset_distribution_weights(self) -> np.ndarray:
        """
        Generate random weights for the assets in the strategy.

        """
        decimals = 4
        # To be sure that we have prices for the initial timestamp, we get the assets available at this timestamp.
        historical_prices_first_timestamp = self.historical_prices.loc[self.initial_timestamp]
        available_initial_assets_list = list(historical_prices_first_timestamp['base'].unique())
        # We create a list with the assets that form the initial assets list of the portfolio.
        # If there is no initial assets list, we use all the assets in the historical prices.
        assets_symbols_list = []
        if not self.initial_assets_list:
            assets_symbols_list.extend(list(available_initial_assets_list))
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
        return np.array(weights_all_assets_symbol_list)

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
        random_weights = self.random_asset_distribution_weights
        initial_balance = portfolio.balance
        initial_timestamp = self.initial_timestamp
        prices = self.get_prices_on_timestamp(initial_timestamp)
        portfolio.update_prices(prices=prices, timestamp=initial_timestamp)
        for asset, weight in zip(self.assets_symbols_list, random_weights):
            amount_quote = initial_balance * float(weight)
            # We buy the asset if the amount is higher than 0, if not we skip it.
            if amount_quote > 0:
                portfolio.buy(symbol=asset, amount_quote=amount_quote, timestamp=initial_timestamp, msg=buy_msg)
    
    def buy_defined_asset(self, portfolio: Portfolio) -> None:
        """
        Buy the assets defined in the initial assets list as a dictionary.
        
        The dictionary contains the assets and their amounts in quote currency.

        """
        buy_msg = "Buying defined asset based on the initial assets list."
        initial_timestamp = self.initial_timestamp
        prices = self.get_prices_on_timestamp(initial_timestamp)
        portfolio.update_prices(prices=prices, timestamp=initial_timestamp)
        for asset, amount_quote in self.initial_assets_list.items():
            # We need to skip the portfolio symbol as we can't buy it.
            if asset != self.portfolio_symbol:
                portfolio.buy(symbol=asset, amount_quote=amount_quote, timestamp=initial_timestamp, msg=buy_msg)

    def create_single_portfolio(self) -> None:
        """
        Create a Portfolio object based on the strategy's attributes.

        Returns:
            Portfolio: Portfolio object.

        """
        PF = Portfolio(
            symbol=self.portfolio_symbol,
            commission_trade=self.commission_trade,
            commission_transfer=self.commission_transfer,
            threshold_buy=self.threshold_buy,
            threshold_sell=self.threshold_sell,
        )
        PF.set_verbosity(verbosity_type='silent')
        # If we provide a dictionary, we use the assets and their amounts.
        if isinstance(self.initial_assets_list, dict):
            if self.initial_equity:
                raise ValueError("You cannot provide an initial equity with initial assets list as a dictionary.")
            # We calculate the initial equity based on the assets and their amounts.
            self.initial_equity = pd.Series(self.initial_assets_list).sum()
            PF.deposit(amount=self.initial_equity, timestamp=self.initial_timestamp)
            self.buy_defined_asset(PF)
        # If we provide a list or nothing, we use the assets and their weights.
        else:
            PF.deposit(amount=self.initial_equity, timestamp=self.initial_timestamp)
            self.buy_random_asset(PF)
        # previous_amount_factor = {asset: 0 for asset in PF.assets_list}
        # print(previous_amount_factor)
        # PF.previous_amount_factor = previous_amount_factor
        self.Portfolios.append(PF)

    
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
    def volatility_df(self) -> pd.Series:
        """
        Calculate the volatility of the prices.

        """
        volatility_df = self.historical_prices[['base', 'volatility']].copy()
        volatility_df.reset_index(inplace=True)
        volatility_df.rename(columns={'timestamp_id': 'timestamp', 'base': 'symbol'}, inplace=True)
        return volatility_df
    
    @property
    def current_triggers(self) -> pd.Series:
        """
        Property to get the current triggers of the strategy for all assets on the current timestamp.

        """
        historical_prices = self.historical_prices
        current_prices = historical_prices[historical_prices.index == self.current_timestamp].copy()
        current_prices.set_index('base', inplace=True)
        return current_prices[self.triggering_feature]
    
    @property
    def signals_df(self) -> pd.DataFrame:
        """
        Convert the signals list to a DataFrame.

        """
        signals_df = self.historical_prices[['base', 'label_returns']].copy()
        signals_df.reset_index(inplace=True)
        signals_df.rename(columns={'timestamp_id': 'timestamp', 'base': 'symbol', 'label_returns': 'value_signal'}, inplace=True)
        signals_df['trade_signal'] = 'HOLD'
        signals_df.loc[signals_df['value_signal'] >= self.threshold_buy, 'trade_signal'] = 'BUY'
        signals_df.loc[signals_df['value_signal'] <= self.threshold_sell, 'trade_signal'] = 'SELL'
        signals_df['value_signal'] = signals_df['value_signal'].astype(np.float32)
        return signals_df
    
    @property
    def current_assets_to_buy(self) -> list:
        """
        Get the assets to buy.
        
        When buying we have to consider a maximal number of assets to buy as there is a limited amount of cash.

        """
        current_triggers = self.current_triggers
        candidates = current_triggers[current_triggers > self.threshold_buy]
        # As we want to buy the assets with the highest values, we sort the candidates in descending order.
        candidates.sort_values(ascending=False, inplace=True)
        return candidates[:self.maximal_assets_to_buy].index.tolist()
    
    @property
    def current_assets_to_sell(self) -> list:
        """
        Get the assets to sell.

        """
        current_triggers = self.current_triggers
        candidates = current_triggers[current_triggers < self.threshold_sell]
        # No need to sort the candidates as we are returning all the assets under the threshold.
        return candidates.index.tolist()
    
    def get_current_triggers_on_positive_balance_assets(self, asset_list: list) -> pd.Series:
        """
        Get the triggers of the assets with a positive balance.
        
        """
        triggers = self.current_triggers
        return triggers[triggers.index.isin(asset_list)]
        
    def ensure_liquidity(self) -> None:
        """
        Ensure liquidity in the Portfolios.

        """
        sell_msg = "Selling asset to ensure liquidity."
        for PF in self.Portfolios:
            min_liquidity = self.minimal_liquidity_ratio * PF.equity_value
            while PF.balance < min_liquidity:
                # We find out the owned assets with low performance.
                triggers = self.current_triggers
                # We get the assets that we own.
                owned_assets = PF.positive_balance_assets_list
                # We get the triggers of the assets that we own.
                triggers_owned_assets = triggers[triggers.index.isin(owned_assets)]
                # We get the asset with the lowest performance.
                lowest_asset = triggers_owned_assets.idxmin()
                # We sell the asset with the lowest performance.
                PF.sell(symbol=lowest_asset, amount_quote=self.quote_ticket_amount, timestamp=self.current_timestamp, msg=sell_msg)
    
    def trade_on_signals(self) -> None:
        """
        Apply the signals to the Portfolios.

        """
        self.ensure_liquidity()
        for PF in self.Portfolios:
            # ASSETS TO BUY >>>>>>>>>>>
            for asset in self.current_assets_to_buy:
                # We buy the asset only if we have enough cash plus we still have enough liquidity.
                # The minimal threshold liquidity ensures to still have some cash after buying the asset.
                mininmal_threshold_liquidity = self.minimal_liquidity_ratio * PF.equity_value / 100
                if PF.balance > (self.quote_ticket_amount + mininmal_threshold_liquidity):
                    # We buy the asset only if the maximal equity per asset ratio is not reached.
                    asset_equity_ratio = PF.get_value(symbol=asset, quote='USDT') / PF.equity_value
                    if asset_equity_ratio < self.maximal_equity_per_asset_ratio:
                        PF.buy(symbol=asset, amount_quote=self.quote_ticket_amount, timestamp=self.current_timestamp)
            # ASSETS TO SELL >>>>>>>>>>>
            for asset in self.current_assets_to_sell:
                # We sell the asset only if we own the asset.
                if asset in PF.positive_balance_assets_list:
                    PF.sell(symbol=asset, amount_quote=self.quote_ticket_amount, timestamp=self.current_timestamp)
    
    def dispatch_signals_in_portfolios(self) -> None:
        """
        In order to make the signals available in the Portfolios, we dispatch the signals_df and volatility_df in the Portfolios.

        """
        for PF in self.Portfolios:
            PF.signals_df = self.signals_df
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