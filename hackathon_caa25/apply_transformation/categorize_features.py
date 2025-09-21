from pandas import DataFrame

from hackathon_caa25.logger import setup_logger

ORDINALS_FEATURES = [
    "NB_CASERNES",
    "BDTOPO_BAT_MAX_HAUTEUR",
    "HAUTEUR_MAX",
    "HAUTEUR",
    "BDTOPO_BAT_MAX_HAUTEUR_MAX",
    "MEN_SURF",
    "IND_SNV",
    "IND_INC",
    "IND_Y9",
    "IND_0_Y1",
    "IND",
    "LOG_SOC",
    "LOG_INC",
    "LOG_APA3",
    "LOG_AVA1",
    "MEN_MAIS",
    "MEN_COLL",
    "MEN_FMP",
    "MEN_PROP",
    "MEN_PAUV",
    "MEN",
    "COEFASS",
    "DISTANCE_111",
    "DISTANCE_112",
    "DISTANCE_121",
    "DISTANCE_122",
    "DISTANCE_123",
    "DISTANCE_124",
    "DISTANCE_131",
    "DISTANCE_132",
    "DISTANCE_133",
    "DISTANCE_141",
    "DISTANCE_142",
    "DISTANCE_211",
    "DISTANCE_212",
    "DISTANCE_213",
    "DISTANCE_221",
    "DISTANCE_222",
    "DISTANCE_223",
    "DISTANCE_231",
    "DISTANCE_242",
    "DISTANCE_243",
    "DISTANCE_244",
    "DISTANCE_311",
    "DISTANCE_312",
    "DISTANCE_313",
    "DISTANCE_321",
    "DISTANCE_322",
    "DISTANCE_323",
    "DISTANCE_324",
    "DISTANCE_331",
    "DISTANCE_332",
    "DISTANCE_333",
    "DISTANCE_334",
    "DISTANCE_335",
    "DISTANCE_411",
    "DISTANCE_412",
    "DISTANCE_421",
    "DISTANCE_422",
    "DISTANCE_423",
    "DISTANCE_511",
    "DISTANCE_512",
    "DISTANCE_521",
    "DISTANCE_522",
    "DISTANCE_523",
    "PROPORTION_11",
    "PROPORTION_12",
    "PROPORTION_13",
    "PROPORTION_14",
    "PROPORTION_21",
    "PROPORTION_22",
    "PROPORTION_23",
    "PROPORTION_24",
    "PROPORTION_31",
    "PROPORTION_32",
    "PROPORTION_33",
    "PROPORTION_41",
    "PROPORTION_42",
    "PROPORTION_51",
    "PROPORTION_52",
    "MEN_1IND",
    "MEN_5IND",
    "LOG_A1_A2",
    "LOG_A2_A3",
    "IND_Y1_Y2",
    "IND_Y2_Y3",
    "IND_Y3_Y4",
    "IND_Y4_Y5",
    "IND_Y5_Y6",
    "IND_Y6_Y7",
    "IND_Y7_Y8",
    "IND_Y8_Y9",
    "DISTANCE_1",
    "DISTANCE_2",
    "ALTITUDE_1",
    "ALTITUDE_2",
    "ALTITUDE_3",
    "ALTITUDE_4",
    "ALTITUDE_5",
    "NBJTX25_MM_A",
    "NBJTX25_MMAX_A",
    "NBJTX25_MSOM_A",
    "NBJTX0_MM_A",
    "NBJTX0_MMAX_A",
    "NBJTX0_MSOM_A",
    "NBJTXI27_MM_A",
    "NBJTXI27_MMAX_A",
    "NBJTXI27_MSOM_A",
    "NBJTXS32_MM_A",
    "NBJTXS32_MMAX_A",
    "NBJTXS32_MSOM_A",
    "NBJTXI20_MM_A",
    "NBJTXI20_MMAX_A",
    "NBJTXI20_MSOM_A",
    "NBJTX30_MM_A",
    "NBJTX30_MMAX_A",
    "NBJTX30_MSOM_A",
    "NBJTX35_MM_A",
    "NBJTX35_MMAX_A",
    "NBJTX35_MSOM_A",
    "NBJTN10_MM_A",
    "NBJTN10_MMAX_A",
    "NBJTN10_MSOM_A",
    "NBJTNI10_MM_A",
    "NBJTNI10_MMAX_A",
    "NBJTNI10_MSOM_A",
    "NBJTN5_MM_A",
    "NBJTN5_MMAX_A",
    "NBJTN5_MSOM_A",
    "NBJTNS25_MM_A",
    "NBJTNS25_MMAX_A",
    "NBJTNS25_MSOM_A",
    "NBJTNI15_MM_A",
    "NBJTNI15_MMAX_A",
    "NBJTNI15_MSOM_A",
    "NBJTNI20_MM_A",
    "NBJTNI20_MMAX_A",
    "NBJTNI20_MSOM_A",
    "NBJTNS20_MM_A",
    "NBJTNS20_MMAX_A",
    "NBJTNS20_MSOM_A",
    "NBJTMS24_MM_A",
    "NBJTMS24_MMAX_A",
    "NBJTMS24_MSOM_A",
    "TAMPLIAB_VOR_MM_A",
    "TAMPLIAB_VOR_MMAX_A",
    "TAMPLIM_VOR_MM_A",
    "TAMPLIM_VOR_MMAX_A",
    "TM_VOR_MM_A",
    "TM_VOR_MMAX_A",
    "TMM_VOR_MM_A",
    "TMM_VOR_MMAX_A",
    "TMMAX_VOR_MM_A",
    "TMMAX_VOR_MMAX_A",
    "TMMIN_VOR_MM_A",
    "TMMIN_VOR_MMAX_A",
    "TN_VOR_MM_A",
    "TN_VOR_MMAX_A",
    "TNAB_VOR_MM_A",
    "TNAB_VOR_MMAX_A",
    "TNMAX_VOR_MM_A",
    "TNMAX_VOR_MMAX_A",
    "TX_VOR_MM_A",
    "TX_VOR_MMAX_A",
    "TXAB_VOR_MM_A",
    "TXAB_VOR_MMAX_A",
    "TXMIN_VOR_MM_A",
    "TXMIN_VOR_MMAX_A",
    "NBJFF10_MM_A",
    "NBJFF10_MMAX_A",
    "NBJFF10_MSOM_A",
    "NBJFF16_MM_A",
    "NBJFF16_MMAX_A",
    "NBJFF16_MSOM_A",
    "NBJFF28_MM_A",
    "NBJFF28_MMAX_A",
    "NBJFF28_MSOM_A",
    "NBJFXI3S10_MM_A",
    "NBJFXI3S10_MMAX_A",
    "NBJFXI3S10_MSOM_A",
    "NBJFXI3S16_MM_A",
    "NBJFXI3S16_MMAX_A",
    "NBJFXI3S16_MSOM_A",
    "NBJFXI3S28_MM_A",
    "NBJFXI3S28_MMAX_A",
    "NBJFXI3S28_MSOM_A",
    "NBJFXY8_MM_A",
    "NBJFXY8_MMAX_A",
    "NBJFXY8_MSOM_A",
    "NBJFXY10_MM_A",
    "NBJFXY10_MMAX_A",
    "NBJFXY10_MSOM_A",
    "NBJFXY15_MM_A",
    "NBJFXY15_MMAX_A",
    "NBJFXY15_MSOM_A",
    "FFM_VOR_MM_A",
    "FFM_VOR_MMAX_A",
    "FXI3SAB_VOR_MM_A",
    "FXI3SAB_VOR_MMAX_A",
    "FXIAB_VOR_MM_A",
    "FXIAB_VOR_MMAX_A",
    "FXYAB_VOR_MM_A",
    "FXYAB_VOR_MMAX_A",
    "FFM_VOR_COM_MM_A_Y",
    "FFM_VOR_COM_MMAX_A_Y",
    "FXI3SAB_VOR_COM_MM_A_Y",
    "FXI3SAB_VOR_COM_MMAX_A_Y",
    "NBJRR50_MM_A",
    "NBJRR50_MMAX_A",
    "NBJRR50_MSOM_A",
    "NBJRR1_MM_A",
    "NBJRR1_MMAX_A",
    "NBJRR1_MSOM_A",
    "NBJRR5_MM_A",
    "NBJRR5_MMAX_A",
    "NBJRR5_MSOM_A",
    "NBJRR10_MM_A",
    "NBJRR10_MMAX_A",
    "NBJRR10_MSOM_A",
    "NBJRR30_MM_A",
    "NBJRR30_MMAX_A",
    "NBJRR30_MSOM_A",
    "NBJRR100_MM_A",
    "NBJRR100_MMAX_A",
    "NBJRR100_MSOM_A",
    "RR_VOR_MM_A",
    "RR_VOR_MMAX_A",
    "RRAB_VOR_MM_A",
    "RRAB_VOR_MMAX_A",
    "TAILLE1",
    "TAILLE2",
]


ORDINALS_MAPPINGS = {
    "CARACT4": [
        "absence de surface",
        "Surface de moins d",
        "Surface entre 501",
        "Surface entre 1001",
        "Surface entre 1501",
        "Surface de plus de",
    ],
    "SURFACE4": [
        "0",
        "500",
        "1000",
        "1500",
        "2000",
        "2500",
        "3000",
        "3500",
        "4000",
        "4500",
        "5000",
        "5500",
        "6000",
        "6500",
        "7000",
        "7000+",
    ],
    "SURFACE6": [
        "0",
        "500",
        "1000",
        "1500",
        "2000",
        "2500",
        "3000",
        "3500",
        "4000",
        "4500",
        "5000",
        "5500",
        "6000",
        "6500",
        "7000",
        "7000+",
    ],
    "total_surface_2023": [
        "Aucun feu",
        "<10ha",
        "10-20ha",
        "20-50ha",
        "50-100ha",
        "100-200ha",
        ">200ha",
    ],
    "total_surface_5y": [
        "Aucun feu",
        "<10ha",
        "10-20ha",
        "20-50ha",
        "50-100ha",
        "100-200ha",
        ">200ha",
    ],
    "surface_over_forest": [
        "Absence de feu",
        "<0.05",
        "0.05-0.1",
        "0.1-0.2",
        "0.5-2",
    ],
    "fire_extinction_rates": [
        "Aucun feu",
        "<50%",
        "50-70%",
        "70-85%",
        ">85%",
    ],
}

# get the columns that are to be removed
COLUMNS_TO_REMOVE = (
    ["FREQ", "CM", "CHARGE", "ANNEE_ASSURANCE"]
    + ["TARGET"]
    + [
        "DEROG13",  # no values
        "DEROG14",  # no values
        "DEROG16",  # no values
        "DEROG13_formatted",
        "DEROG8_formatted",
        "DEROG3_formatted",
        "DEROG16_formatted",
        "DEROG14_formatted",
        "KAPITAL_MAX",
        "KAPITAL_SUM",
    ]
)


def categorize_features(
    dataframe: DataFrame,
    ordinals: list = ORDINALS_FEATURES,
    ordinal_mappings: dict = ORDINALS_MAPPINGS,
    columns_to_remove: list = COLUMNS_TO_REMOVE,
) -> tuple:
    """Retype features in the dataframe into categorical, numerical, and ordinal.
    Args:
        dataframe (DataFrame): The input dataframe containing features.
        ordinals (list): List of column names to be treated as ordinal features.
        ordinal_mappings (dict): Dictionary mapping ordinal columns to their
            ordered categories.
        columns_to_remove (list): List of column names to be removed from the
            feature lists.
    Returns:
        tuple: A tuple containing three elements:
            - List of categorical column names.
            - List of numerical column names.
            - Dictionary of ordinal column names and their ordered categories.
    """

    # setup logger
    logger = setup_logger(__name__)
    logger.info("Retyping features...")

    # get the columns that are categorical
    categorical_columns = dataframe.select_dtypes(include=["object"]).columns

    # get the columns that are numerical
    numerical_columns = dataframe.select_dtypes(
        include=["int64", "float64"]
    ).columns

    # get the unique values for each ordinal column
    ordinal_columns = {
        col: list(dataframe[col].value_counts().sort_index().index)
        for col in ordinals
        if col in dataframe.columns
    }
    ordinal_columns["PROPORTION_32"] += ["10. > 90"]

    # update the ordinal columns with the mappings
    ordinal_columns.update(ordinal_mappings)

    # remove useless columns from dataframe
    columns_to_remove = [c for c in dataframe.columns if "MMSOM" in c]
    columns_to_remove = [
        col for col in columns_to_remove if col in dataframe.columns
    ]

    # removing columns from the lists
    categorical_columns = [
        col
        for col in categorical_columns
        if col not in columns_to_remove and col not in ordinal_columns
    ] + ["TYPERS"]
    numerical_columns = [
        col
        for col in numerical_columns
        if col not in columns_to_remove
        and col not in ordinal_columns
        and col not in categorical_columns
    ]
    logger.info(
        "Columns to be removed: %s", [col for col in columns_to_remove]
    )
    logger.info(
        "Categorical columns: %s, number of features :%s",
        categorical_columns,
        len(categorical_columns),
    )
    logger.info(
        "Numerical columns: %s, number of features :%s",
        numerical_columns,
        len(numerical_columns),
    )
    logger.info(
        "Ordinal columns: %s, number of features :%s",
        list(ordinal_columns.keys()),
        len(ordinal_columns),
    )

    return categorical_columns, numerical_columns, ordinal_columns
