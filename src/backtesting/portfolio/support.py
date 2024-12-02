from time import time
from typing import Optional, Callable, Union

import numpy as np
import prettytable as pt
from faker import Faker


def now_ms() -> int:
    """
    Function to get the current time in milliseconds.

    """
    return int(time() * 1000)


def check_positive(func: Callable) -> Callable:
    """
    Decorator to check if the arguments of a function are positive.

    Mainly used to check if the amount of a transaction is positive.

    """
    def wrapper(self, *args, **kwargs):
        for i, arg in enumerate(args):
            if isinstance(arg, (int, float)) and not isinstance(arg, bool) and arg <= 0:
                raise ValueError(f"Argument {i+1} must be positive")
        for key, arg in kwargs.items():
            if isinstance(arg, (int, float)) and not isinstance(arg, bool) and arg <= 0:
                raise ValueError(f"Argument '{key}' must be positive")
        return func(self, *args, **kwargs)

    return wrapper

def get_random_name() -> str:
    """
    Function to generate a fake name.

    """
    fake = Faker()
    name = fake.name().replace(' ', '_').replace('.', '').lower()
    return name

# Display functions -----------------------------------------------------------


def display_percentage(amount: float) -> str:
    """
    Function to display a percentage with two decimal places.

    """
    amount100 = amount * 100
    return display_price(amount100) + "%"


def display_integer(amount: int, unit: Optional[str] = None) -> str:
    """
    Function to display an integer with commas.

    """
    text = format(amount, ",d")
    if unit is not None:
        text += f" {unit}"
    return text


def display_price(amount: float, unit: Optional[str] = None) -> str:
    """
    Function to display a price with two decimal places and commas.

    If the amount is less than 1, the decimals are displayed according to the magnitude of the amount.

    """
    text = format(amount, ",.2f")
    if amount > 0:
        log_figure = np.log10(amount)
        factor = int(np.ceil(-log_figure))
        if factor > 0:
            depth = factor + 1
            text = f"{amount:.{depth}f}"
    if unit is not None:
        text += f" {unit}"
    return text


def add_padding_table(table: pt.PrettyTable, padding: int) -> str:
    """
    Function to add padding to a pretty table on each new line to align it with the rest of the text.

    """
    table_str = table.get_string()
    table_str = table_str.replace("\n", "\n" + " " * padding)
    table_str = " " * padding + table_str
    return table_str


def display_pretty_table(data: list[list[str]], padding: int = 0) -> str:
    """
    Function to display a pretty table with the data provided.

    """
    table = pt.PrettyTable()
    table.field_names = data[0]
    payload = data[1:]
    # Sort the assets by the last column, where the quote value is stored
    sorted_assets = sorted(payload, key=lambda asset: asset[-1], reverse=True)
    for row in sorted_assets:
        # Remove the last element `.pop()` of the row, which is the quote value
        # It is not displayed in the table
        row.pop()
        table.add_row(row)
    for field in table.field_names:
        if field == table.field_names[0]:
            table.align[field] = "l"
        else:
            table.align[field] = "r"
    if padding > 0:
        table_str = add_padding_table(table, padding)
    else:
        table_str = table.get_string()
    return table_str


# Plotting functions ----------------------------------------------------------


def thousands(x: Union[int, float], pos) -> str:
    """
    Function to format the y-axis tick labels with commas for plotting.

    This is used by FuncFormatter in matplotlib.

    """
    return f"{x:,.0f}"


def to_percent(x: Union[int, float], pos) -> str:
    """
    Function to format the y-axis tick labels as percentages for plotting.

    This is used by FuncFormatter in matplotlib.

    """
    return f"{x * 100:.0f}%"
