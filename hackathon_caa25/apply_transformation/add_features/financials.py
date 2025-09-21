from typing import Union

from numpy import nan
from pandas import DataFrame

DEFAULT_COLUMNS_REVENUES = ["CA1", "CA2", "CA3"]


def format_revenues(
    data: DataFrame, columns_revenues: Union[list[str], None] = None
) -> DataFrame:
    """
    Formats the revenues data by converting categorical values to numerical
    and calculating total and mean revenues.
    Args:
        data (pd.DataFrame): DataFrame containing revenue columns.
    Returns:
        pd.DataFrame: DataFrame with formatted revenue columns.
    """

    if columns_revenues is None:
        columns_revenues = DEFAULT_COLUMNS_REVENUES

    # revenues ratios
    data[columns_revenues] = data[columns_revenues].replace(0, nan)
    data["CA_TOT"] = data[columns_revenues].sum(axis=1).replace(0, nan)
    data["CA_MEAN"] = data[columns_revenues].mean(axis=1).replace(0, nan)

    # replace NaN with 0
    for col in columns_revenues:
        data[col + "_CA_TOT"] = data[col].divide(data["CA_TOT"])
        data[col + "_CA_MEAN"] = data[col].divide(data["CA_MEAN"])

    return data
