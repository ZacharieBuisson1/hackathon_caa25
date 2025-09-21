from collections import Counter

from numpy import array
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from hackathon_caa25.logger import setup_logger
from hackathon_caa25.apply_transformation.processor import Processor
from hackathon_caa25.apply_transformation.split_dataset import (
    split_dataset_with_weights,
)


def train_frequency_model(
    dataframe: DataFrame, targets: DataFrame, target_name: str = "TARGET"
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

    # train a frequency model

    return dataframe


# def inference_frequency_model():

#     model = frequency_model()
#     model.fit(x_train, y_train)
#     preds = model.predict(x_test)
#     return preds
