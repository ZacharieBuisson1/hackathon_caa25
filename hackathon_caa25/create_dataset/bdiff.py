"""Module to process fire-related data and add it to the dataset.
This module includes functions to format department codes, load fire data,
count event types per zone, and add fire-related information to the dataset.

Source : https://bdiff.agriculture.gouv.fr/indicateurs/cartes"""

import os

from pandas import DataFrame, read_csv
from numpy import nan


def get_bdiff_incendies(path: str = "hackathon_caa25/data/") -> DataFrame:
    """Load fire data, count event types per zone, and filter department codes.

    Args:
        path (str): Path to the directory containing 'Incendies.csv'.

    Returns:
        pd.DataFrame: DataFrame with counts of event types per department zone.
    """
    # read the dataset
    complete_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "data",
        "Incendies.csv",
    )
    incendies = read_csv(complete_path, sep=",")

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
