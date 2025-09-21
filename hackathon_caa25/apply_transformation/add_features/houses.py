from typing import Union

from pandas import DataFrame

from hackathon_caa25.apply_transformation.utils_transformation import (
    categorical_conversion,
)

COLUMNS_MENAGES = [
    "MEN_PAUV",
    "MEN_1IND",
    "MEN_5IND",
    "MEN_PROP",
    "MEN_FMP",
    "MEN_COLL",
    "MEN_MAIS",
    "MEN",
]


def format_menages(
    data: DataFrame, columns_menages: Union[list[str], None] = None
) -> DataFrame:
    """
    Format the menages data by converting categorical features to numerical
    and calculating the total number of menages and their ratios.

    Args:
        data (DataFrame): The input DataFrame containing the menages data.
        columns_menages (list[str], optional): List of column names to be used for
            menages. If None, defaults to COLUMNS_MENAGES.
    Returns:
        DataFrame: The formatted DataFrame with additional columns for menages.
    """

    if columns_menages is None:
        columns_menages = COLUMNS_MENAGES

    # converting to numerical
    data = categorical_conversion(data, columns_menages)

    # total number of menage
    data["MEN_TOT"] = data[
        [col + "_num" for col in columns_menages if col != "MEN"]
    ].sum(axis=1)
    for column in columns_menages:
        data[column + "_MEN_TOT"] = data[column + "_num"].divide(
            data["MEN_TOT"]
        )
        if column != "MEN":
            data[column + "_MEN"] = data[column + "_num"].divide(
                data["MEN_num"]
            )
    return data


HOUSING_COLUMNS = [
    "LOG_AVA1",
    "LOG_A1_A2",
    "LOG_A2_A3",
    "LOG_APA3",
    "LOG_INC",
    "LOG_SOC",
]


def format_housing(
    data: DataFrame, housing_columns: Union[list[str], None] = None
) -> DataFrame:
    """Format the housing data by converting categorical features to numerical
    and calculating the total number of housing and their ratios.

    Args:
        data (DataFrame): The input DataFrame containing the housing data.
        housing_columns (Union[list[str], None], optional): List of column names to be used for
            housing. If None, defaults to HOUSING_COLUMNS.

    Returns:
        DataFrame: The formatted DataFrame with additional columns for housing.
    """

    if housing_columns is None:
        housing_columns = HOUSING_COLUMNS

    data = categorical_conversion(data, housing_columns)

    housing_columns_num = [col + "_num" for col in housing_columns]

    # total number of housing
    data["LOG_TOT"] = data[
        [col for col in housing_columns_num if not col.startswith("LOG_SOC")]
    ].sum(axis=1)

    # proportion of each type of housing
    for col in housing_columns_num:
        data[col + "_LOG_TOT"] = data[col].divide(data["LOG_TOT"])

    # vetusty of buildings
    data["LOG_VETUSTE"] = (
        data["LOG_AVA1_num"] * 80
        + data["LOG_A1_A2_num"] * 50
        + data["LOG_A2_A3_num"] * 30
        + data["LOG_APA3_num"] * 10
    ).divide(data["LOG_TOT"])

    return data
