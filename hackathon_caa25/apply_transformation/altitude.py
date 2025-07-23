from typing import Union

from pandas import DataFrame

from hackathon_caa25.apply_transformation import categorical_conversion

ALTITUDE_COLUMNS = [
    "ALTITUDE_1",
    "ALTITUDE_2",
    "ALTITUDE_3",
    "ALTITUDE_4",
    "ALTITUDE_5",
]


def format_altitude(
    data: DataFrame, altitude_columns: Union[list[str], None] = None
) -> DataFrame:
    """
    Format the altitude data by converting categorical features to numerical
    and calculating the total altitude and its ratios.

    Args:
        data (DataFrame): The input DataFrame containing the altitude data.
        altitude_columns (list[str], optional): List of column names to be used for
            altitude. If None, defaults to ALTITUDE_COLUMNS.
    Returns:
        DataFrame: The formatted DataFrame with additional columns for altitude.
    """

    if altitude_columns is None:
        altitude_columns = ALTITUDE_COLUMNS

    data = categorical_conversion(data, altitude_columns)

    # converting to numerica
    altitude_columns_num = [col + "_num" for col in altitude_columns]

    # sum of altitude
    data["ALTITUDE_TOT"] = data[altitude_columns_num].sum(axis=1)

    for col in altitude_columns:
        data[col + "_ALT_TOT"] = data[col + "_num"].divide(
            data["ALTITUDE_TOT"]
        )

    return data
