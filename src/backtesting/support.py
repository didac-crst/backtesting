import itertools 
import math
import random
from time import time
from typing import Optional, Callable, Union

import numpy as np
import prettytable as pt
from faker import Faker

COLOR_PALETTE_DICT = {
    "Blue": "#1f77b4",
    "Sky Blue": "#87ceeb",
    "Cyan": "#17becf",
    "Turquoise Blue": "#00ced1",
    "Green": "#2ca02c",
    "Emerald Green": "#50c878",
    "Yellow-Green": "#bcbd22",
    "Olive Green": "#dbdb8d",
    "Golden Yellow": "#ffce60",
    "Mustard Yellow": "#e5ae38",
    "Red": "#b62728",
    "Cherry Red": "#e4002b",
    "Coral Pink": "#ff6f61",
    "Pink": "#e377c2",
    "Light Pink": "#f7b6e2",
    "Peach": "#f5b78f",
    "Purple": "#9467bd",
    "Lavender": "#9b59b6",
    "Lilac": "#cbb5d4",
    "Brown": "#8c564b",
    "Taupe": "#c49c94",
    "Beige": "#e7bc94",
    "Gray": "#7f7f7f",
}

def now_ms() -> int:
    """
    Function to get the current time in milliseconds.

    """
    return int(time() * 1000)


def max_2_numbers(a, b):
    """
    Function to get the maximum of two values, ignoring NaN values.
    
    """
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    return max(a, b)

def move_to_end(lst, element):
    """
    Function to move an element to the end of a list.
    
    """
    if element in lst:
        lst.remove(element)
        lst.append(element)
    return lst

def check_property_update(func: Callable) -> Callable:
    """
    Decorator to check if properties have been updated.
    
    """
    def wrapper(self, *args, **kwargs):
        # Use func.__name__ to get the current method's name
        self_name = func.__name__
        # Check if the property is a key in the dictionary
        # If not, it has never been calculated
        if self_name not in self._properties_evolution_id:
            recalculate = True
        # If the property has been calculated but the evolution_id has changed
        # We need to recalculate the property
        elif self._properties_evolution_id[self_name] != self.evolution_id:
            recalculate = True
        # Otherwise, the property is up-to-date
        else:
            recalculate = False
        # If the property needs to be recalculated
        # Call the original method
        if recalculate:
            # Update the properties cache
            self._properties_cached[self_name] = func(self, *args, **kwargs)
            # Update the properties evolution_id
            self._properties_evolution_id[self_name] = self.evolution_id
        # Return the property cached
        return self._properties_cached[self_name]
    return wrapper


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

def get_coloured_markers() -> list[tuple[str, str]]:
    """
    Function to get a list of coloured markers for plotting.

    """
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'v', 'D', '*', '+', 'x', 'p', 'h']
    combinations = list(itertools.product(markers, colors))
    return combinations

# Display functions -----------------------------------------------------------


def display_percentage(amount: float) -> str:
    """
    Function to display a percentage with two decimal places.

    """
    amount100 = amount * 100
    return display_price(amount100) + " %"


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
    elif amount == 0:
        text = "---"
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


def display_pretty_table(data: list[list[str]], quote_currency:str, padding: int = 0, sorting_columns: int = 0) -> str:
    """
    Function to display a pretty table with the data provided.
    
    Sorting columns are the last columns of the table, which are used to sort the table.
    After sorting, the last columns are removed from the table.

    """
    table = pt.PrettyTable()
    table.field_names = data[0][:-sorting_columns] # Get rid of the sorting columns
    payload = data[1:]
    # Here we sort the table by the last columns
    if sorting_columns > 0:
        payload = sorted(payload, key=lambda asset: tuple(asset[-i-1] for i in range(sorting_columns)), reverse=True)
    for row in payload:
        # Remove the sorting columns
        row = row[:-sorting_columns]
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

def to_percent_log_growth(x: Union[int, float], pos) -> str:
    """
    Function to format the y-axis tick labels as percentages for plotting.

    This is used by FuncFormatter in matplotlib.

    """
    return f"{(x-1) * 100:.0f}%"
