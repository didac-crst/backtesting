from dataclasses import dataclass
from typing import Optional, Literal, Union

import pandas as pd

@dataclass
class Strategy:
    """
    Dataclass defining the basic structure of a strategy.

    """

    name: str
    description: str
    assets_symbols: list[str]
    initial_equity: float
    portfolio_symbol: str = "USDT"

    
