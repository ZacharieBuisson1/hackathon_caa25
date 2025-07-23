from pandas import DataFrame, read_csv
from numpy import nan


# https://bdiff.agriculture.gouv.fr/indicateurs/cartes
def invert_dict_list(mapping: dict[str, list[int]]) -> dict[str, str]:
    """Invert a dict of lists to a dict mapping str(zfilled value) -> key.

    Args:
        mapping (dict[str, list[int]]): A dictionary where keys are strings and values are
            lists of integers.

    Returns:
        dict[str, str]: A dictionary where each key is a zero-padded string representation of
            the integers from the lists, and each value is the corresponding key from the input
            dictionary.
    """
    return {
        str(value).zfill(2): key
        for key, values in mapping.items()
        for value in values
    }


FIRE_EXTINCTION_RATES_RAW = {
    "Aucun feu": [],
    ">85%": [
        13,
        83,
        84,
        30,
        40,
        19,
        63,
        43,
        73,
        74,
        68,
        88,
        21,
        18,
        85,
        49,
        72,
        35,
        29,
        14,
        76,
        60,
        8,
        59,
    ],
    "70-85%": [
        20,
        11,
        66,
        4,
        6,
        5,
        26,
        38,
        47,
        33,
        24,
        16,
        17,
        87,
        86,
        56,
        41,
        45,
        89,
        77,
    ],
    "50-70%": [64, 34, 48, 7, 69, 71, 58, 23, 37, 44, 22, 27, 91, 57, 67],
    "<50%": [9, 31, 81, 12, 32, 82, 46, 15, 36, 53, 28, 10, 52, 54],
}
FIRE_EXTINCTION_RATES = invert_dict_list(FIRE_EXTINCTION_RATES_RAW)

TOTAL_SURFACE_2023_RAW = {
    "Aucun feu": [
        50,
        61,
        78,
        95,
        75,
        80,
        62,
        2,
        51,
        55,
        70,
        25,
        39,
        1,
        3,
        42,
        79,
        65,
    ],
    "<10ha": [],
    "10-20ha": [27, 28, 89, 21, 17, 15, 46, 47, 32, 81, 38, 84],
    "20-50ha": [26, 9, 24, 41, 56, 52, 88],
    "50-100ha": [83, 12, 33, 40, 91],
    "100-200ha": [4, 5, 30, 48, 64, 82, 54],
    ">200ha": [6, 20, 13, 7, 34, 11, 66],
}
TOTAL_SURFACE_2023 = invert_dict_list(TOTAL_SURFACE_2023_RAW)

TOTAL_SURFACE_5Y_RAW = {
    "Aucun feu": [22, 56, 44, 78, 95, 75, 51, 59],
    "<10ha": [
        29,
        50,
        14,
        61,
        53,
        76,
        80,
        62,
        60,
        2,
        8,
        91,
        77,
        10,
        54,
        57,
        67,
        88,
        68,
        70,
        25,
        90,
        74,
        73,
        71,
        69,
        36,
        85,
        79,
    ],
    "10-20ha": [35, 37, 18, 23, 42, 39],
    "20-50ha": [
        81,
        82,
        32,
        47,
        16,
        87,
        19,
        63,
        43,
        3,
        58,
        89,
        21,
        52,
        1,
        38,
        5,
        49,
        72,
        28,
    ],
    "50-100ha": [17, 86, 41, 24, 12, 26, 45],
    "100-200ha": [64, 65, 31, 9, 46, 15, 48],
    ">200ha": [],
}
TOTAL_SURFACE_5Y = invert_dict_list(TOTAL_SURFACE_5Y_RAW)


SURFACE_OVER_FOREST_RAW = {
    "Absence de feu": [
        50,
        61,
        80,
        62,
        2,
        51,
        55,
        70,
        25,
        39,
        1,
        3,
        42,
        79,
        65,
    ],
    "<0.05": [],
    "0.05-0.1": [54, 48, 7, 5, 6],
    "0.1-0.2": [82, 11, 34, 13, 20],
    "0.5-2": [66],
}
SURFACE_OVER_FOREST = invert_dict_list(SURFACE_OVER_FOREST_RAW)


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
