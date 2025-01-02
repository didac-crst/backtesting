from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

# Display functions -----------------------------------------------------------

def display_unit(amount: float, unit: Optional[str] = None) -> str:
    """
    Function to display a unit with two decimal places and commas.

    If the amount is less than 1, the decimals are displayed according to the magnitude of the amount.

    """
    if unit == '%':
        amount = amount * 100
    if isinstance(amount, int):
        if np.isnan(amount):
            text = "N/A"
            unit = ""
        elif amount == 0:
            text = "---"
        else:
            text = format(int(amount), ",d")
    elif isinstance(amount, float):
        if np.isnan(amount):
            text = "N/A"
            unit = ""
        # This is to avoid a lot of decimals when the amount is very small
        elif np.abs(amount) > 0:
            log_figure = np.log10(np.abs(amount))
            factor = int(np.ceil(-log_figure))
            if factor > 0:
                depth = factor + 1
                text = f"{amount:.{depth}f}"
            else:
                text = f"{amount:,.2f}"
        elif amount == 0:
            text = "---"
        else:
            text = f"{amount:,.2f}"
    else:
        text = str(amount)
    if unit is not None:
        text += f"{unit}"
    return text

def display_unit_df(data_df: pd.Series, units_df: pd.Series) -> pd.DataFrame:
    """
    Function that returns a DataFrame with the data and units concatenated.
    
    """
    output_df = pd.DataFrame()
    for col in data_df.columns:
        data_col = data_df[col].rename('data')
        units_col = units_df[col].rename('units')
        concatenated = pd.concat([data_col, units_col], axis=1)
        output_df[col] = concatenated.apply(lambda row: display_unit(row['data'], row['units']), axis=1)
    return output_df


# Classes ---------------------------------------------------------------------

@dataclass
class SeriesUnits:
    """
    Class that emulates a pandas Series with units.
    
    """
    data: pd.Series
    units: pd.Series
    
    def _check_indeces(self) -> None:
        """
        Method that checks that units and data have the same index.
        
        """
        if not self.data.index.equals(self.units.index):
            raise ValueError("The indices of data and units must be the same.")
    
    def _reformat_units(self) -> None:
        """
        Method adding a space before the units if they are not percentages."""
        for i in self.units.items():
            if i[1] != '%':
                self.units[i[0]] = f" {i[1]}"
    
    def __post_init__(self) -> None:
        self.units = self.units.copy()
        self.name = self.data.name
        self._name_units = '_units'
        self._check_indeces()
        self._reformat_units()
        # Rename the units column
        self.units = self.units.rename(self._name_units)
    
    @property
    def _display_data_units(self) -> pd.Series:
        """
        Method that returns a Series with the data and units concatenated.
        
        """
        concatenated = pd.concat([self.data, self.units], axis=1)
        concatenated.columns = [self.name, self._name_units]
        return concatenated.apply(lambda row: display_unit(row[self.name], row[self._name_units]), axis=1)
    
    @property
    def D(self) -> pd.Series:
        """
        Method that returns the _display_data_units property.
        
        """
        return self._display_data_units

@dataclass
class DataFrameUnits:
    """
    Class that emulates a pandas DataFrame with units.
    
    """
    df_dict: dict[str, SeriesUnits]
    
    def _check_equal_units(self, df: pd.DataFrame) -> None:
        """
        Method that checks that all series have the same units.
        
        """
        all_equal = (df.nunique(axis=1) == 1).all()  # Check if all values in each row are the same
        if not all_equal:
            raise ValueError("Units among columns are not equal.")
    
    def _reduce_units_to_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Method that reduces the units DataFrame to a Series, assuming all units are the same for each column.
        
        """
        self._check_equal_units(df)
        units = df.iloc[:,0]
        units.rename('_units', inplace=True)
        return units
    
    def _extract_data(self) -> None:
        """
        Method that extracts the data and units from the SeriesUnits objects and adapts them to a DataFrame.
        
        """
        df = pd.DataFrame()
        units_df = pd.DataFrame()
        columns = []
        for key, value in self.df_dict.items():
            df[key] = value.data
            units_df[key] = value.units
            columns.append(key)
        self.df = df
        self.units = self._reduce_units_to_series(units_df)
        self._columns = columns
    
    def __post_init__(self) -> None:
        self._extract_data()
    
    @property
    def _display_data_units(self) -> pd.DataFrame:
        """
        Method that returns a DataFrame with the data and units concatenated.
        
        """
        # output_df = pd.DataFrame()
        data_df = self.df.copy()
        units_df = pd.DataFrame([self.units.rename(i) for i in data_df.columns]).T
        output_df = display_unit_df(data_df, units_df)
        # for col in data_df.columns:
        #     data_col = data_df[col].rename('data')
        #     units_col = units_df[col].rename('units')
        #     concatenated = pd.concat([data_col, units_col], axis=1)
        #     output_df[col] = concatenated.apply(lambda row: display_unit(row['data'], row['units']), axis=1)
        output_df = output_df.T
        output_df.set_index('Name', inplace=True)
        output_df = output_df.T
        return output_df
    
    def describe(self, drop_columns:Optional[list[str]] = None) -> pd.DataFrame:
        """
        Method that returns a DataFrame with the dataframe statistics and adds units.
        
        """
        data_df = self.df.copy()
        if drop_columns is not None:
            data_df = data_df.drop(index=drop_columns)
        data_df = data_df.T
        described_data = pd.DataFrame()
        described_units = pd.DataFrame()
        described_data['count'] = data_df.count()
        described_data['mean'] = data_df.mean()
        described_data['std'] = data_df.std()
        described_data['min'] = data_df.min().astype(float)
        described_data['25%'] = data_df.quantile(0.25)
        described_data['50%'] = data_df.quantile(0.5)
        described_data['75%'] = data_df.quantile(0.75)
        described_data['max'] = data_df.max().astype(float)
        described_units = pd.DataFrame([self.units.rename(i) for i in described_data.columns]).T
        if drop_columns is not None:
            described_units = described_units.drop(index=drop_columns)
        described_units['count'] = ''
        output_df = display_unit_df(described_data, described_units)
        return output_df
    
    @property
    def D(self):
        return self._display_data_units