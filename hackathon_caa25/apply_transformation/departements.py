"""Module to map departement codes to their corresponding regions."""

from warnings import warn

from pandas import DataFrame

from hackathon_caa25.config import DEPARTEMENT_TO_REGION


def get_region(departement_code: str) -> str:
    """
    Returns the region name for a given departement code.
    If the departement code is not found, it returns 'Unknown' with the code.

    Args:
        departement_code (str): The departement code as a string.
    Returns:
        str: The name of the region or 'Unknown' with the departement code.
    """
    found = DEPARTEMENT_TO_REGION.get(
        str(int(departement_code)).zfill(2), f"Unknown {departement_code}"
    )

    if "Unknown" in found:
        warn(f"Unknown departement code: {departement_code}")

    return found


def format_zone(data: DataFrame) -> DataFrame:
    """
    Formats the zone information in the DataFrame.

    Args:
        data (DataFrame): The input DataFrame containing a 'ZONE' column.

    Returns:
        DataFrame: The DataFrame with an additional 'ZONE_REGION' column
        containing the region names corresponding to the 'ZONE' codes.
    """
    # convert zone to region
    data["ZONE_REGION"] = data["ZONE"].apply(get_region)

    # ensure ZONE is a string with leading zeros
    data["ZONE"] = data["ZONE"].apply(lambda u: str(u).zfill(2))

    return data
