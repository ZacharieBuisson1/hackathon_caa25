import json
import os

from AutoCarver import Features, MulticlassCarver
from AutoCarver.selectors import (
    ClassificationSelector,
    KruskalMeasure,
    SpearmanFilter,
)
from pandas import DataFrame

from hackathon_caa25.logger import setup_logger
from hackathon_caa25.apply_transformation.processor import Processor
from hackathon_caa25.apply_transformation.split_dataset import (
    split_dataset_with_weights,
)
from hackathon_caa25.apply_transformation.categorize_features import (
    categorize_features,
)
from hackathon_caa25.apply_transformation.feature_selection import (
    select_numerical_features_with_autocarver,
    select_categorical_features_with_autocarver,
)


def train_frequency_model(
    dataframe: DataFrame,
    target_name: str = "TARGET",
    autocarver_freq_version: str = "010",
) -> DataFrame:

    logger = setup_logger(__name__)
    logger.info("Starting training of frequency model.")

    # Create a new target column "FREQ" based on the condition
    dataframe[target_name] = (
        dataframe["FREQ"] * dataframe["ANNEE_ASSURANCE"]
    ).astype(int)
    logger.info("Target column 'TARGET' created")
    logger.info(
        "Target distribution:\n%s", dataframe[target_name].value_counts()
    )

    # stratified sampling to keep the same distribution of the target
    logger.info("Performing stratified sampling.")
    x_train, x_dev, y_train, y_dev, w_train, w_dev = (
        split_dataset_with_weights(dataframe, target_col=target_name)
    )
    logger.info("x_train shape: %s", x_train.shape)
    logger.info("x_dev shape: %s", x_dev.shape)
    logger.info("y_train shape: %s", y_train.shape)
    logger.info("y_dev shape: %s", y_dev.shape)
    logger.info("w_train shape: %s", w_train.shape)
    logger.info("w_dev shape: %s", w_dev.shape)

    # process the data before training
    processor = Processor()
    x_train = processor.fit_transform(x_train)
    x_dev = processor.transform(x_dev)
    dataframe = processor.transform(dataframe)
    logger.info("Data processing completed.")
    logger.info("x_train shape after processing: %s", x_train.shape)
    logger.info("x_dev shape after processing: %s", x_dev.shape)
    logger.info("dataframe shape after processing: %s", dataframe.shape)

    # separate features into categorical, numerical and ordinal
    categorical_columns, numerical_columns, ordinal_columns = (
        categorize_features(x_train)
    )

    # define the auto-carver features
    autocarver_carver_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        f"{autocarver_freq_version}_freq_carver.json",
    )
    if os.path.exists(autocarver_carver_path):
        logger.info(
            "Loading existing AutoCarver from %s", autocarver_carver_path
        )
        carver = MulticlassCarver.load(autocarver_carver_path)
        logger.info("AutoCarver loaded successfully.")

    else:
        features = Features(
            categoricals=categorical_columns, ordinals=ordinal_columns
        )
        carver = MulticlassCarver(
            features=features,
            min_freq=0.02,
            max_n_mod=5,
            dropna=False,
            copy=False,
            verbose=False,
        )

        logger.info("Training new AutoCarver.")
        y_transform = lambda u: u.where(u <= 1, 2)

        x_train = carver.fit_transform(
            x_train,
            y_transform(y_train),
            X_dev=x_dev,
            y_dev=y_transform(y_dev),
        )
        logger.info("AutoCarver training completed.")

        # save the carver model
        carver.save(
            autocarver_carver_path,
            light_mode=True,
        )

    # transform the entire dataframe
    x_train = carver.transform(x_train)
    x_dev = carver.transform(x_dev)
    logger.info("Train & dev transformed using AutoCarver.")

    # feature selection
    autocarver_selector_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        f"{autocarver_freq_version}_freq_best_features.json",
    )
    if os.path.exists(autocarver_selector_path):
        logger.info(
            "Loading existing AutoCarver selector from %s",
            autocarver_selector_path,
        )
        with open(
            autocarver_selector_path,
            "r",
            encoding="utf-8",
        ) as json_file:
            best_features = json.load(json_file)
        logger.info(
            "Number of selected features: %d",
            len(best_features),
        )

    else:
        logger.info("No existing list of selected features found.")
        best_quantitative_features = select_numerical_features_with_autocarver(
            x_train, y_train, numerical_columns
        )
        best_categorical_features = (
            select_categorical_features_with_autocarver(
                x_train, y_train, carver
            )
        )
        best_features = best_quantitative_features + best_categorical_features
        logger.info(
            "Number of selected quantitative features: %d",
            len(best_quantitative_features),
        )
        logger.info(
            "Number of selected categorical features: %d",
            len(best_categorical_features),
        )
        logger.info(
            "Total number of selected features: %d", len(best_features)
        )

        # save the list of best features
        with open(
            autocarver_selector_path,
            "w",
            encoding="utf-8",
        ) as json_file:
            json.dump(best_features, json_file)
        logger.info(
            "List of best features saved to %s", autocarver_selector_path
        )

    return dataframe


# def inference_frequency_model():

#     model = frequency_model()
#     model.fit(x_train, y_train)
#     preds = model.predict(x_test)
#     return preds
