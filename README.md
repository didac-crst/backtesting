````
# Backtesting

**Backtesting** is a lightweight, composable Python toolkit that lets you model portfolios and rigorously evaluate rule-based trading strategies against historical market data.  
It is designed for engineers, quantitative researchers, and data-driven investors who need transparent, reproducible backtests without the drag of heavyweight monolithic frameworks.

---

## Key Features

| Feature | What it gives you | Where to look |
|---------|------------------|---------------|
| **Object-oriented portfolio model** | `Portfolio`, `Asset`, `Currency` classes with audit-ready transaction ledger | `src/backtesting/portfolio.py`, `asset.py`, `currency.py`, `ledger.py` |
| **Strategy abstraction** | Create a strategy by subclassing `TradingStrategy`; run multi-period evaluations with `MultiPeriodBacktest` | `trading_strategy.py` |
| **Unit-aware analytics** | `pandas_units` integrates pint units with pandas for type-safe calculations | `pandas_units.py` |
| **Tutorial notebooks** | End-to-end examples in Jupyter (see `tutorial/`) | `tutorial/portfolio_introduction.ipynb` |
| **Python ≥ 3.11, zero heavy deps** | Pure-Python core, only standard libraries plus `pandas` | `requirements.txt` |

---

## Installation

```bash
# Recommended: create a fresh virtual environment first
python -m venv .venv && source .venv/bin/activate

# From PyPI (when released)
pip install backtesting

# Or from source
git clone https://github.com/<your-org>/backtesting.git
cd backtesting
pip install -e .
```

---

## Quick Start

```python
import pandas as pd
from backtesting import Portfolio, TradingStrategy, MultiPeriodBacktest

class SMACrossover(TradingStrategy):
    short = 20
    long  = 50

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        fast = prices.rolling(self.short).mean()
        slow = prices.rolling(self.long).mean()
        return (fast > slow).astype(int)  # 1 = long, 0 = flat

# Load your price series (index must be datetime) e.g. from CSV
prices = pd.read_csv("EURUSD_daily.csv", index_col=0, parse_dates=True)["close"]

portfolio = Portfolio(base_currency="EUR", cash=10_000)
strategy  = SMACrossover()
bt        = MultiPeriodBacktest(portfolio, strategy, prices)

tearsheet = bt.run()
tearsheet.plot()          # equity curve & metrics
print(tearsheet.summary)  # table of risk/return stats
```

See the [tutorial notebooks](tutorial/) for a deeper walk-through.

---

## Project Layout

```
backtesting/
├── src/
│   └── backtesting/           # Core package
│       ├── portfolio.py
│       ├── trading_strategy.py
│       ├── ...                # other helpers
├── tutorial/                  # Jupyter notebooks
├── tests/                     # (coming soon)
├── requirements.txt
└── pyproject.toml
```

---

## Running Tests

```bash
pip install -r requirements.txt
pytest -q
```

---

## Contributing

Pull requests are welcome!  
If you find a bug or want to suggest a feature, please open an issue first ensuring it hasn’t been reported.

1. Fork the repo & create your branch: `git checkout -b feature/my-awesome-feature`  
2. Commit your changes with clear messages.  
3. Run the test suite (`pytest`) and keep coverage ≥ 90 %.  
4. Submit a PR and describe **why** and **how** you changed things.

Please follow [Conventional Commits](https://www.conventionalcommits.org) and run `pre-commit install` after cloning to keep the style consistent.

---

## Roadmap

- [ ] Vectorised execution engine for intraday data  
- [ ] Plug-in metric system (Sharpe, Sortino, max drawdown, etc.)  
- [ ] Slippage & transaction-cost models  
- [ ] Exchange adapters for live paper-trading  
- [ ] Interactive Streamlit dashboard  

---

## Disclaimer

This project is **for research and educational purposes only** and **does not constitute financial advice**.  
Trading involves substantial risk; use these tools responsibly and at your own risk.

---

## License

Backtesting is distributed under the **MIT** License. see [`LICENSE`](LICENSE).
````
