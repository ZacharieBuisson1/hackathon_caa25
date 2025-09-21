"""
Create train and test datasets for the CAA hackathon.
This script reads the training and test data files, joins them with the target columns,
and returns DataFrames for both the training and test datasets.
It assumes the data files are located in a specific directory structure relative to the script.
The training data includes features and target columns, while the test data includes only features.
The script prints the shapes of the resulting DataFrames for verification.
"""

from pathlib import Path
import os

from pandas import DataFrame, read_csv

from hackathon_caa25.logger import setup_logger

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = os.path.join(REPO_ROOT, "data")


def read_train_test_dataset(
    data_path: str = DATA_PATH,
    train_file_name: str = "train_input_Z61KlZo.csv",
    target_file_name: str = "train_output_DzPxaPY.csv",
    test_file_name: str = "test_input_5qJzHrr.csv",
) -> tuple[DataFrame, DataFrame]:
    """
    Add the target columns to the data DataFrame.

    Args:
        data_path (str): Path to the directory containing the data files.
        train_file_name (str): Name of the training data file.
        target_file_name (str): Name of the target data file.
        test_file_name (str): Name of the test data file.

    Returns:
        DataFrame: A DataFrame containing the training data with target columns joined.
        DataFrame: A DataFrame containing the test data.
    """
    logger = setup_logger(__name__)
    logger.info("Creating train and test datasets")

    # add the given train data
    data = read_csv(data_path + train_file_name)
    data.set_index("ID", inplace=True)
    logger.info("x_train %s", data.shape)

    # add the target columns
    target = read_csv(data_path + target_file_name)
    target.set_index("ID", inplace=True)
    logger.info("y_train %s", target.shape)

    # join the target columns to the data DataFrame
    train = data.join(target.drop("ANNEE_ASSURANCE", axis=1))

    # add test dataset
    test = read_csv(data_path + test_file_name)
    test.set_index("ID", inplace=True)
    logger.info("x_test %s", test.shape)

    return train, test
