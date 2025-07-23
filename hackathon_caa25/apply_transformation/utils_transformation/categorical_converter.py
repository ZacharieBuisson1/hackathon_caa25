from pandas import concat, DataFrame

CATEGORICAL_CONVERTER = {
    "01. <= 10": 1,
    "02. <= 20": 10,
    "03. <= 30": 20,
    "04. <= 40": 30,
    "05. <= 50": 40,
    "06. <= 60": 50,
    "07. <= 70": 60,
    "08. <= 80": 70,
    "09. <= 90": 80,
    "10. > 90": 90,
    #
    "01. <= 17204": 17204,
    "02. <= 153098": 153098,
    "03. <= 670263": 670263,
    #
    "01. <= 21873": 21873,
    "02. <= 24733": 24733,
    "03. <= 29681": 29681,
    "04. >= 29681": 29681,
    #
    "01. <= 33995": 33995,
    "02. <= 318560": 318560,
    "03. <= 1340814": 1340814,
    # ALTITUDE
    "01. <= 190": 1,
    "02. <= 438": 190,
    "03. <= 794": 438,
    "04. >= 794": 794,
    "01. <= 239": 1,
    "02. <= 588": 239,
    "03. <= 1186": 588,
    "04. >= 1186": 1186,
    "01. <= 236": 1,
    "02. <= 579": 236,
    "03. <= 1178": 579,
    "04. >= 1178": 1178,
    "01. <= 143": 1,
    "02. <= 333": 143,
    "03. <= 645": 333,
    "04. >= 645": 645,
    "01. <= 337": 1,
    "02. <= 840": 337,
    "03. <= 1810": 840,
    "04. >= 1810": 1810,
}


def categorical_conversion(data: DataFrame, cols: list[str]) -> DataFrame:
    """Converts categorical features to numerical values.

    Args:
        data (DataFrame): The input DataFrame containing the categorical columns.
        cols (list[str]): List of column names to convert.

    Returns:
        DataFrame: The DataFrame with new numerical columns appended.
    """
    converted_data = DataFrame(
        {c + "_num": data[c].replace(CATEGORICAL_CONVERTER) for c in cols}
    )
    return concat([data, converted_data], axis=1)
