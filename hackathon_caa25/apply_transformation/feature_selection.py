from AutoCarver import Features, MulticlassCarver
from AutoCarver.selectors import (
    ClassificationSelector,
    CramervFilter,
    KruskalMeasure,
    SpearmanFilter,
    TschuprowtMeasure,
)
from pandas import DataFrame, Series

from hackathon_caa25.logger import setup_logger


def remove_one_modality_features(dataframe):
    """
    Remove features with only one modality from the DataFrame.
    Args:
        dataframe (DataFrame): The input DataFrame.
    Returns:
        DataFrame: The DataFrame with one-modality features removed.
    """
    one_modality_columns = [
        col for col in dataframe.columns if dataframe[col].nunique() <= 1
    ]
    dataframe = dataframe.drop(columns=one_modality_columns)
    return dataframe


def select_numerical_features_with_autocarver(
    x_train: DataFrame, y_train: Series, numerical_columns: list[str]
):
    """
    Select numerical features using AutoCarver.
    Args:
        x_train (DataFrame): The training feature set.
        y_train (Series): The target variable for training.
        numerical_columns (list[str]): List of numerical feature column names.
    Returns:
        list[str]: List of selected numerical feature column names.
    """

    logger = setup_logger(__name__)
    logger.info("Starting feature selection with AutoCarver.")

    # defining the features to select
    quantitative_features = Features(quantitatives=numerical_columns)
    logger.info(
        "Number of quantitative features: %d", len(quantitative_features)
    )

    # defining the target association measures and threshold used to select features
    measures = [KruskalMeasure(threshold=1)]

    # defining the inter-feature association measures and threshold used to filter out features
    filters = [SpearmanFilter(threshold=0.9)]

    # initiating the selector
    selector = ClassificationSelector(
        features=quantitative_features,
        measures=measures,
        filters=filters,
        n_best_per_type=50,
        max_num_features_per_chunk=250,
        verbose=True,
    )

    y_transform = lambda u: u.where(u <= 1, 2)

    # feature selection on train data
    best_quantitative_features = selector.select(x_train, y_transform(y_train))
    logger.info(
        "Number of selected quantitative features: %d",
        len(best_quantitative_features),
    )

    logger.info(
        "Best selected quantitative features: %s", best_quantitative_features
    )

    return best_quantitative_features.versions


def select_categorical_features_with_autocarver(
    x_train: DataFrame,
    y_train: Series,
    carver: MulticlassCarver,
):
    """
    Select categorical features using AutoCarver.
    Args:
        x_train (DataFrame): The training feature set.
        y_train (Series): The target variable for training.
        categorical_columns (list[str]): List of categorical feature column names.
    Returns:
        list[str]: List of selected categorical feature column names.
    """

    logger = setup_logger(__name__)
    logger.info("Starting feature selection with AutoCarver.")

    # defining the features to select
    qualitative_features = Features(carver.features.versions)
    logger.info(
        "number of qualitative features: %d", len(qualitative_features)
    )

    # defining the target association measures and threshold used to select features
    measures = [TschuprowtMeasure(threshold=0.005)]

    # defining the inter-feature association measures and threshold used to filter out features
    filters = [CramervFilter(threshold=0.9)]

    # initiating the selector
    selector = ClassificationSelector(
        features=qualitative_features,
        measures=measures,
        filters=filters,
        n_best_per_type=50,
        max_num_features_per_chunk=1000,
        verbose=True,
    )

    y_transform = lambda u: u.where(u <= 1, 2)

    # feature selection on train data
    best_categorical_features = selector.select(x_train, y_transform(y_train))
    logger.info(
        "Number of selected categorical features: %d",
        len(best_categorical_features),
    )

    logger.info(
        "Best selected categorical features: %s", best_categorical_features
    )

    return best_categorical_features.versions
