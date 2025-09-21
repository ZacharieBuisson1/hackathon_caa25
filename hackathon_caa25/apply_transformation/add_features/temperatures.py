from pandas import DataFrame


def cross_temperature_data(data: DataFrame) -> DataFrame:
    """Crosses temperature data columns in the DataFrame.
    Args:
        data (DataFrame): The input DataFrame containing temperature data columns.
    Returns:
        DataFrame: The DataFrame with new columns created by crossing temperature data.
    """

    columns_to_cross = [
        col for col in data.columns if "_MMAX_A" in col or "_MM_A" in col
    ]
    for col1 in columns_to_cross:
        for col2 in columns_to_cross:
            if col1 != col2:

                # Check if col1 and col2 are related to the same month
                if (col1.endswith("_MM_A") and col2.endswith("_MMAX_A")) or (
                    col1.endswith("_MM_A_Y") and col2.endswith("_MMAX_A_Y")
                ):
                    # Remove the "_MM_A" or "_MMAX_A" suffix to compare
                    if col1.replace("_MM_A", "") == col2.replace(
                        "_MMAX_A", ""
                    ):

                        # Create a new column with the cross data
                        data[col1 + "_" + col2] = data[col1] + "_" + data[col2]

    return data
