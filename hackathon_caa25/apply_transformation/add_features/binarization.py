from typing import Union

from pandas import DataFrame


COLUMNS_TO_BINARIZE = [
    "DEROG1",
    "DEROG6",
    "DEROG7",
    "DEROG9",
    "DEROG10",
    "DEROG11",
    "KAPITAL36",
    "KAPITAL38",
    "KAPITAL39",
]


def one_hot_encode(
    data: DataFrame, columns_to_binarize: Union[list[str], None] = None
) -> DataFrame:
    """
    One-hot encodes specified columns in the DataFrame.
    Args:
        data (DataFrame): The input DataFrame containing the columns to be binarized.
        columns_to_binarize (list[str], optional): List of column names to binarize.
            If None, defaults to COLUMNS_TO_BINARIZE.
    Returns:
        DataFrame: The DataFrame with specified columns binarized.
    """

    if columns_to_binarize is None:
        columns_to_binarize = COLUMNS_TO_BINARIZE

    for column in columns_to_binarize:
        data[column] = (data[column] == "O").astype(int)

    return data
