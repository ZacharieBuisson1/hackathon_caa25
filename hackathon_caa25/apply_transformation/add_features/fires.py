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
from hackathon_caa25.create_dataset.bdiff import get_bdiff_incendies


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

    # getting incendies natures from data source BDIFF...
    incendies_natures = get_bdiff_incendies()

    # adding to dataset
    data = data.join(incendies_natures.fillna(0).set_index("zone"), on="ZONE")

    return data
