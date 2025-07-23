"""Module to process fire-related data and add it to the dataset.
This module includes functions to format department codes, load fire data,
count event types per zone, and add fire-related information to the dataset.

Source : https://bdiff.agriculture.gouv.fr/indicateurs/cartes"""

from pandas import DataFrame, read_csv
from numpy import nan

from hackathon_caa25.config import (
    TOTAL_SURFACE_2023,
    TOTAL_SURFACE_5Y,
    SURFACE_OVER_FOREST,
    FIRE_EXTINCTION_RATES,
)


# Format zone (pad with zeros, handle Corsica codes)
def format_zone(dept: str) -> str:
    """Format the department code to a two-character string, handling Corsica codes.
    Args:
        dept (str): The department code, which may be a number or a string
            like '2A' or '2B'.
    Returns:
        str: A two-character string representing the department code, with
            '2A' and '2B' mapped to '20'.
    """
    dept_str = str(dept).zfill(2)
    if dept_str in ["2A", "2B"]:
        return "20"
    return dept_str


def get_incendies_natures(path: str = "hackathon_caa25/data/") -> DataFrame:
    """Load fire data, count event types per zone, and filter department codes.

    Args:
        path (str): Path to the directory containing 'Incendies.csv'.

    Returns:
        pd.DataFrame: DataFrame with counts of event types per department zone.
    """
    # read the dataset
    incendies = read_csv(path + "Incendies.csv", sep=",")

    # convert zone to a two-character string and handle Corsica codes
    incendies["zone"] = incendies["Département"].apply(format_zone)

    # group by zone and count occurrences of each nature, filling NaN with "Unknown"
    incendies_natures = (
        incendies.assign(Nature=incendies["Nature"].fillna("Unknown"))
        .groupby("zone")["Nature"]
        .value_counts()
        .unstack(fill_value=nan)
        .reset_index()
    )

    # rename columns to have a consistent format
    incendies_natures = incendies_natures[
        incendies_natures["zone"].str.len() < 3
    ]

    return incendies_natures


def add_incendies_info(data: DataFrame) -> DataFrame:
    """Add fire-related information to the dataset.

    Args:
        data (DataFrame): The input DataFrame containing fire data.

    Returns:
        DataFrame: The input DataFrame with additional columns
            for fire-related information.
    """

    data["total_surface_2023"] = data["ZONE"].apply(
        lambda u: TOTAL_SURFACE_2023.get(u, "<10ha")
    )
    data["total_surface_5y"] = data["ZONE"].apply(
        lambda u: TOTAL_SURFACE_5Y.get(u, ">200ha")
    )
    data["surface_over_forest"] = data["ZONE"].apply(
        lambda u: SURFACE_OVER_FOREST.get(u, "<0.05")
    )
    data["fire_extinction_rates"] = data["ZONE"].apply(
        lambda u: FIRE_EXTINCTION_RATES.get(u, "Aucun feu")
    )
    data["total_surface_crossed"] = (
        data["total_surface_2023"] + "__" + data["total_surface_5y"]
    )
    data["nb_casernes_extinction_rate"] = (
        data["NB_CASERNES"] + "__" + data["fire_extinction_rates"]
    )
    data["zone_vent_extinction_rate"] = (
        data["ZONE_VENT"].astype(str) + "__" + data["fire_extinction_rates"]
    )

    # getting incendies natures
    incendies_natures = get_incendies_natures()

    # adding to dataset
    data = data.join(incendies_natures.fillna(0).set_index("zone"), on="ZONE")

    return data
