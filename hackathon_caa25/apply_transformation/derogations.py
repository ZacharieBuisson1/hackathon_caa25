from pandas import DataFrame


def format_derog13(value: str) -> int:
    """
    Format DEROG13 values to numerical representation.
    Args:
        value (str): The value to format, expected to be "D12", "D18", or other.
    Returns:
        int: 1 if "D12", 2 if "D18", otherwise 0.
    """
    return {"D12": 1, "D18": 2}.get(value, 0)


def format_on(value: str) -> int:
    """Format DEROG8 and DEROG3 values to numerical representation.
    Args:
        value (str): The value to format, expected to be "O" or "N".
    Returns:
        int: 1 if "O", 0 if "N".
    """
    return int(value == "O")


def format_derog16(value: str) -> int:
    """Format DEROG16 values to numerical representation.
    Args:
        value (str): The value to format, expected to be "SA" or other.
    Returns:
        int: 1 if "SA", otherwise 0.
    """

    return int(value == "SA")


def format_derog14(value: str) -> int:
    """Format DEROG14 values to numerical representation.
    Args:
        value (str): The value to format, expected to be "D14" or other.
    Returns:
        int: 1 if "D14", otherwise 0.
    """
    return int(value == "D14")


def format_derogations(data: DataFrame) -> DataFrame:
    """
    Format derogation columns in the DataFrame.
    Args:
        data (DataFrame): The input DataFrame containing derogation columns.
    Returns:
        DataFrame: The DataFrame with formatted derogation columns.
    """

    data["DEROG13_formatted"] = data["DEROG13"].map(format_derog13)
    data["DEROG8_formatted"] = data["DEROG8"].map(format_on)
    data["DEROG3_formatted"] = data["DEROG3"].map(format_on)
    data["DEROG16_formatted"] = data["DEROG16"].map(format_derog16)
    data["DEROG14_formatted"] = data["DEROG14"].map(format_derog14)
    return data
