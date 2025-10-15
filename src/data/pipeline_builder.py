"""Utilities to build sklearn preprocessing pipelines.

This module provides PipelineBuilder which constructs a ColumnTransformer-based
preprocessing pipeline from a PipelineConfig instance.
"""

from typing import Literal

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder

from .config import DataPipelineConfig, FeatureScalingConfig


class PipelineBuilder:
    """Builder for sklearn preprocessing pipelines.

    The builder converts a PipelineConfig into a scikit-learn Pipeline
    containing a ColumnTransformer preprocessor.
    """

    @classmethod
    def build(cls, config: DataPipelineConfig) -> Pipeline:
        """Build preprocessing pipeline based on config.

        Args:
            config: Pipeline configuration object

        Returns:
            A sklearn Pipeline with a ColumnTransformer preprocessor
        """
        transformers = []

        if config.numerical_std and config.numerical_std.features:
            num_std_pipeline = cls._create_numerical_pipeline(config.numerical_std)
            transformers.append(("num_std", num_std_pipeline, config.numerical_std.features))

        if config.numerical_robust and config.numerical_robust.features:
            num_robust_pipeline = cls._create_numerical_pipeline(config.numerical_robust)
            transformers.append(("num_robust", num_robust_pipeline, config.numerical_robust.features))

        if config.numerical_minmax and config.numerical_minmax.features:
            num_minmax_pipeline = cls._create_numerical_pipeline(config.numerical_minmax)
            transformers.append(("num_minmax", num_minmax_pipeline, config.numerical_minmax.features))

        if config.categorical and config.categorical.features:
            cat_pipeline = cls._create_categorical_pipeline(config.categorical, config.handle_unknown)
            transformers.append(("cat", cat_pipeline, config.categorical.features))

        if config.binary and config.binary.features:
            bin_pipeline = cls._create_binary_pipeline(config.binary, config.handle_unknown)
            transformers.append(("bin", bin_pipeline, config.binary.features))

        column_transformer = ColumnTransformer(transformers=transformers, remainder=config.remainder)
        pipeline = Pipeline(steps=[("preprocessor", column_transformer)])

        return pipeline

    @staticmethod
    def _create_numerical_pipeline(cfg: FeatureScalingConfig) -> Pipeline:
        """Create a numerical preprocessing pipeline.

        The pipeline applies imputation according to cfg.impute_strategy and an
        optional scaler as specified by cfg.scaler.

        Args:
            cfg: Feature scaling configuration

        Returns:
            A sklearn Pipeline for numerical features
        """
        steps = []
        impute_kwargs = {"strategy": cfg.impute_strategy}
        if cfg.impute_strategy == "constant":
            impute_kwargs["fill_value"] = cfg.fill_value
        steps.append(("imputer", SimpleImputer(**impute_kwargs)))

        if cfg.scaler == "standard":
            steps.append(("scaler", StandardScaler()))
        elif cfg.scaler == "minmax":
            steps.append(("scaler", MinMaxScaler()))
        elif cfg.scaler == "robust":
            steps.append(("scaler", RobustScaler()))

        return Pipeline(steps=steps)

    @staticmethod
    def _create_categorical_pipeline(cfg: FeatureScalingConfig, handle_unknown: Literal["ignore", "error"]) -> Pipeline:
        """Create a categorical preprocessing pipeline.

        The pipeline imputes missing values then applies one-hot encoding. The
        encoder's unknown handling is controlled by handle_unknown.

        Args:
            cfg: Feature scaling configuration
            handle_unknown: how to handle unknown categories in the encoder

        Returns:
            A sklearn Pipeline for categorical features
        """
        steps = []
        impute_kwargs = {"strategy": cfg.impute_strategy}
        if cfg.impute_strategy == "constant":
            impute_kwargs["fill_value"] = cfg.fill_value
        steps.append(("imputer", SimpleImputer(**impute_kwargs)))

        encoder = OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)
        steps.append(("encoder", encoder))

        return Pipeline(steps=steps)

    @staticmethod
    def _create_binary_pipeline(cfg: FeatureScalingConfig, handle_unknown: Literal["ignore", "error"]) -> Pipeline:
        """Create a pipeline for binary categorical features.

        The pipeline imputes then encodes binary categories as a single indicator
        column using OneHotEncoder with drop='if_binary'.

        Args:
            cfg: Feature scaling configuration
            handle_unknown: how to handle unknown categories in the encoder

        Returns:
            A sklearn Pipeline for binary categorical features
        """
        steps = []
        impute_kwargs = {"strategy": cfg.impute_strategy}
        if cfg.impute_strategy == "constant":
            impute_kwargs["fill_value"] = cfg.fill_value
        steps.append(("imputer", SimpleImputer(**impute_kwargs)))

        encoder = OneHotEncoder(handle_unknown=handle_unknown, drop="if_binary", sparse_output=False)
        steps.append(("encoder", encoder))

        return Pipeline(steps=steps)