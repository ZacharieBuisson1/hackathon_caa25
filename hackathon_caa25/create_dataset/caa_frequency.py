from pandas import DataFrame

from hackathon_caa25.logger import setup_logger


def create_target_caa_frequency(
    dataset: DataFrame, target_name: str = "TARGET"
) -> DataFrame:
    """
    Create the target column for the CAA frequency dataset.

    Args:
        dataset (DataFrame): The input dataset containing the 'zone' column.

    Returns:
        DataFrame: The dataset with the target column added.
    """
    logger = setup_logger(__name__)
    logger.info("Creating target column for CAA frequency dataset")

    # Create a new column 'TARGET' based on the 'zone' column
    dataset[target_name] = (
        dataset["FREQ"] * dataset["ANNEE_ASSURANCE"]
    ).astype(int)

    logger.info("Target column created successfully")
    logger.info(
        dataset[target_name].value_counts(normalize=False).sort_index()
    )

    return dataset
