"""Utilities to build sklearn preprocessing pipelines.

This module provides PipelineBuilder which constructs a ColumnTransformer-based
preprocessing pipeline from a PipelineConfig instance.
"""

from typing import Any, Dict, List, Literal, Tuple

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FunctionTransformer, Pipeline
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

        if config.lambda_func:
            lambda_pipeline = cls._create_lambda_pipeline(config.lambda_func)
            transformers.append(("lambda", lambda_pipeline, list(config.lambda_func.keys())))
        
        if config.custom_transformers:
            custom_pipeline = cls._create_custom_transformer_pipeline(config.custom_transformers)
            transformers.append(("custom", custom_pipeline, list(config.custom_transformers.keys())))

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
    
    @staticmethod
    def _create_lambda_pipeline(lambda_funcs: Dict[str, str]) -> Pipeline:
        """Create a pipeline that applies lambda functions to features.

        Args:
            lambda_funcs: Dictionary mapping feature names to lambda function strings

        Returns:
            A sklearn Pipeline that applies the lambda functions
        """
        steps: List[Tuple[str, FunctionTransformer]] = []
        for name, expr in lambda_funcs.items():
            fn = eval(expr) if isinstance(expr, str) else expr
            if not callable(fn):
                raise TypeError(f"Lambda spec must be callable or valid lambda string, got: {type(fn)}")
            steps.append((name, FunctionTransformer(func=fn, validate=False)))
        return Pipeline(steps=steps)
    
    @staticmethod
    def _create_custom_transformer_pipeline(custom_transformers: Dict[str, Dict[str, str]]) -> Pipeline:
        """Create a pipeline that applies custom transformers to features.

        Args:
            custom_transformers: Dictionary mapping feature names to transformer specs

        Returns:
            A sklearn Pipeline that applies the custom transformers
        """
        steps: List[Tuple[str, Any]] = []
        for name, spec in custom_transformers.items():
            transformer_type = spec.get("type")
            params = spec.get("params", {})
            if transformer_type == "function":
                fn = eval(params["function"]) if isinstance(params["function"], str) else params["function"]
                if not callable(fn):
                    raise TypeError(f"Custom transformer function must be callable or valid string, got: {type(fn)}")
                transformer = FunctionTransformer(func=fn, validate=False)
            else:
                raise ValueError(f"Unsupported custom transformer type: {transformer_type}")
            steps.append((name, transformer))
        return Pipeline(steps=steps)