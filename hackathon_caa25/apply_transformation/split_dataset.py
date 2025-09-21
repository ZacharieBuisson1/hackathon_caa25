import numpy as np
from pandas import DataFrame

from collections import Counter
from sklearn.model_selection import train_test_split

from hackathon_caa25.logger import setup_logger


def split_dataset_with_weights(
    data: DataFrame, target_col: str = "TARGET"
) -> tuple:
    """Splits the dataset into training and development sets with stratification and computes sample weights.

    Args:
        data (DataFrame): The input dataset containing features and target variable.
        target_col (str): The name of the target column in the dataset.

    Returns:
        Tuple: x_train, x_dev, y_train, y_dev, w_train, w_dev
    """
    logger = setup_logger(__name__)
    logger.info("Starting dataset split.")

    # Transform the target variable to ensure binary classification
    y_transform = lambda u: u.where(
        u <= 1, 2
    )  # Transform the target variable to ensure binary classification

    # Compute class frequencies
    class_counts = Counter(y_transform(data[target_col]))
    total_samples = len(data[target_col])

    # Compute inverse frequency class weights
    class_weights = {
        cls: total_samples / (len(class_counts) * count)
        for cls, count in class_counts.items()
    }
    logger.info("Class weights: %s", class_weights)

    # Assign sample weights based on target values
    logger.info("Assigning sample weights.")
    weights = np.array(
        [class_weights[label] for label in y_transform(data[target_col])]
    )

    # Train-test split
    logger.info("Performing train-test split.")
    x_train, x_dev, y_train, y_dev, w_train, w_dev = train_test_split(
        data,
        data[target_col],
        weights,
        test_size=0.2,
        random_state=42,
        stratify=y_transform(data[target_col]),
    )
    logger.info("y_train mean: %f", y_train.mean())
    logger.info("y_dev mean: %f", y_dev.mean())

    return x_train, x_dev, y_train, y_dev, w_train, w_dev
