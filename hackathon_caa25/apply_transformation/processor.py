from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

from hackathon_caa25.apply_transformation.add_features import (
    add_incendies_info,
    cross_temperature_data,
    format_altitude,
    format_derogations,
    format_housing,
    format_individuals,
    format_menages,
    format_revenues,
    format_zone,
    one_hot_encode,
)


class Processor(BaseEstimator, TransformerMixin):
    """Processor class to apply transformations on the dataset."""

    def __init__(self):
        self.log_means = None
        self.log_vetuste_means = None
        self.men_means = None
        self.ind_means = None
        self.ind_snv_means = None
        self.altitude_means = None

    def fit(self, data: DataFrame):
        """Fit the processor to the data.
        Args:
            data (DataFrame): The input DataFrame containing the data to fit.
        Returns:
            self: The fitted processor instance.
        """
        x = data.copy()
        x = format_zone(x)
        x = format_housing(x)
        x = format_menages(x)
        x = format_individuals(x)
        x = format_altitude(x)

        self.log_means = (
            x.groupby(["ZONE_REGION"])["LOG_TOT"].mean().sort_values()
        )
        self.log_vetuste_means = (
            x.groupby(["ZONE_REGION"])["LOG_VETUSTE"].mean().sort_values()
        )
        self.men_means = (
            x.groupby(["ZONE_REGION"])["MEN_TOT"].mean().sort_values()
        )
        self.ind_means = (
            x.groupby(["ZONE_REGION"])["IND_TOT"].mean().sort_values()
        )
        self.ind_snv_means = (
            x.groupby(["ZONE_REGION"])["IND_SNV_num"].mean().sort_values()
        )
        self.altitude_means = (
            x.groupby(["ZONE_REGION"])["ALTITUDE_TOT"].mean().sort_values()
        )
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        """Transform the data using the fitted processor.
        Args:
            data (DataFrame): The input DataFrame containing the data to transform.
        Returns:
            DataFrame: The transformed DataFrame with additional features.
        """

        # getting region
        data = format_zone(data)

        # revenues ratios
        data = format_revenues(data)

        # HOUSING TYPES
        data = format_housing(data)
        data["LOG_REGION"] = data["LOG_TOT"].divide(
            data["ZONE_REGION"].map(self.log_means)
        )
        data["LOG_VETUSTE_REGION"] = data["LOG_VETUSTE"].divide(
            data["ZONE_REGION"].map(self.log_vetuste_means)
        )
        # houses
        data = format_menages(data)
        data["MEN_REGION"] = data["MEN_TOT"].divide(
            data["ZONE_REGION"].map(self.men_means)
        )

        # individuals information
        data = format_individuals(data)
        data["IND_REGION"] = data["IND_TOT"].divide(
            data["ZONE_REGION"].map(self.ind_means)
        )
        data["IND_SNV_REGION"] = data["IND_SNV_num"].divide(
            data["ZONE_REGION"].map(self.ind_means)
        )

        # ALTITUDE
        data = format_altitude(data)
        data["ALTITUDE_REGION"] = data["ALTITUDE_TOT"].divide(
            data["ZONE_REGION"].map(self.altitude_means)
        )

        # crossing temperature data
        data = cross_temperature_data(data)

        # one hot encoding
        data = one_hot_encode(data)

        # adding fire data
        data = add_incendies_info(data)
        data["VENT_x_CASERNES"] = (
            data["ZONE_VENT"].astype(str) + "__" + data["NB_CASERNES"]
        )

        # kapital data
        kapital_cols = [
            c
            for c in data.columns
            if c.startswith("KAPITAL") and data[c].dtype != "object"
        ]
        data["KAPITAL_SUM"] = data[kapital_cols].sum(axis=1)
        data["KAPITAL_MAX"] = data[kapital_cols].max(axis=1)

        # ohe for derogations
        data = format_derogations(data)

        return data
