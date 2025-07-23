from typing import Union

from pandas import DataFrame

from hackathon_caa25.apply_transformation import categorical_conversion

INDIVIDUALS_COLUMNS = [
    "IND",
    "IND_0_Y1",
    "IND_Y1_Y2",
    "IND_Y2_Y3",
    "IND_Y3_Y4",
    "IND_Y4_Y5",
    "IND_Y5_Y6",
    "IND_Y6_Y7",
    "IND_Y7_Y8",
    "IND_Y8_Y9",
    "IND_Y9",
    "IND_INC",
    "IND_SNV",
]


def format_individuals(
    data: DataFrame, columns_individuals: Union[list[str], None] = None
) -> DataFrame:
    """
    Format the individuals data by converting categorical features to numerical
    and calculating the total number of individuals and their ratios.

    Args:
        data (DataFrame): The input DataFrame containing the individuals data.
        columns_individuals (list[str], optional): List of column names to be used for
            individuals. If None, defaults to INDIVIDUALS_COLUMNS.
    Returns:
        DataFrame: The formatted DataFrame with additional columns for individuals.
    """

    # converting to num
    if columns_individuals is None:
        columns_individuals = INDIVIDUALS_COLUMNS

    data = categorical_conversion(data, columns_individuals)

    # total number of individuals
    columns_individuals_num = [col + "_num" for col in columns_individuals]

    # total number of ind
    ind_tot_columns = [
        col
        for col in columns_individuals_num
        if col != "IND" and col != "IND_SNV_num"
    ]
    data["IND_TOT"] = data[ind_tot_columns].sum(axis=1)

    for col in columns_individuals:
        data[col + "_IND_TOT"] = data[col + "_num"].divide(data["IND_TOT"])
        if col != "IND":
            data[col + "_IND"] = data[col + "_num"].divide(data["IND_num"])
        if col != "IND_SNV":
            data[col + "_IND_SNV"] = data[col + "_num"].divide(
                data["IND_SNV_num"]
            )

    return data
