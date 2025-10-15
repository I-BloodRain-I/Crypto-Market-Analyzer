"""Utilities for defining and persisting preprocessing pipeline configuration.

This module defines lightweight Pydantic models that describe how features
should be imputed and scaled and how a preprocessing pipeline should be
constructed. The models are intentionally simple and JSON-serializable so
configurations can be stored alongside experiments and re-loaded when
reconstructing pipelines.

Example:
    config = PipelineConfig(
        numerical_std=FeatureScalingConfig(features=['price', 'volume'], impute_strategy='mean', scaler='standard')
    )
    config.save('configs/pipeline.json')
    loaded = PipelineConfig.load('configs/pipeline.json')
"""

import json
from pathlib import Path
from typing import List, Literal, Optional, Union

import pandas as pd
from pydantic import BaseModel


class FeatureScalingConfig(BaseModel):
    """Configuration for imputing and scaling a group of features.

    This model encapsulates the imputation strategy and the scaler choice to be
    applied to a named list of features when building a preprocessing
    pipeline. It is intentionally minimal so it can be embedded inside a
    higher-level pipeline configuration.

    Attributes:
        features: The list of feature names this configuration applies to.
        impute_strategy: Strategy to use for imputing missing values.
        scaler: Which scaler to apply after imputation.
        fill_value: Value used when `impute_strategy` is set to "constant".
    """
    features: List[str]
    impute_strategy: Literal["mean", "median", "most_frequent", "constant"] = "median"
    scaler: Literal["standard", "minmax", "robust", "none"] = "standard"
    fill_value: Union[int, float, str, None] = 0  # Used if strategy is "constant"


class DataPipelineConfig(BaseModel):
    """Top-level configuration for building data preprocessing pipelines.

    Use this model to declare how different groups of features should be
    processed (standardization, min-max scaling, robust scaling, categorical
    encoding, etc.), how to treat unknown categories, and how to handle any
    remainder columns when composing a ColumnTransformer.

    Attributes:
        numerical_std: Configuration for features that should be standardized.
        numerical_robust: Configuration for features that should use a robust scaler.
        numerical_minmax: Configuration for features that should be min-max scaled.
        categorical: Configuration for categorical features (imputation/encoding).
        binary: Configuration for binary features.
        handle_unknown: How to handle unknown categories encountered at transform time.
        rolling_window: Optional rolling window size to apply rolling mean on numeric features before scaling.
        remainder: What to do with columns not targeted by any transformer (drop or passthrough).
    """
    numerical_std: Optional[FeatureScalingConfig] = None
    numerical_robust: Optional[FeatureScalingConfig] = None
    numerical_minmax: Optional[FeatureScalingConfig] = None
    categorical: Optional[FeatureScalingConfig] = None
    binary: Optional[FeatureScalingConfig] = None
    handle_unknown: Literal["ignore", "error"] = "ignore"
    rolling_window: Optional[int] = None  # If set, apply rolling mean with this window size
    remainder: Literal["drop", "passthrough"] = "passthrough"

    @staticmethod
    def load(path: Union[str, Path]) -> "DataPipelineConfig":
        """Load a PipelineConfig instance from a JSON file.

        Args:
            path: Path to a JSON file containing the pipeline configuration.

        Returns:
            A PipelineConfig instance populated from the JSON content.

        Raises:
            FileNotFoundError: If the given path does not exist.
            ValueError: If the provided file is not a JSON file.
        """
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        if path.suffix != ".json":
            raise ValueError(f"Config file must be a JSON file: {path}")

        with open(path) as f:
            loaded_json = json.load(f)

        return DataPipelineConfig(
            numerical_std=FeatureScalingConfig(
                **loaded_json["numerical_std"]
            ) if loaded_json["numerical_std"] else None,
            numerical_robust=FeatureScalingConfig(
                **loaded_json["numerical_robust"]
            ) if loaded_json["numerical_robust"] else None,
            numerical_minmax=FeatureScalingConfig(
                **loaded_json["numerical_minmax"]
            ) if loaded_json["numerical_minmax"] else None,
            categorical=FeatureScalingConfig(
                **loaded_json["categorical"]
            ) if loaded_json["categorical"] else None,
            binary=FeatureScalingConfig(
                **loaded_json["binary"]
            ) if loaded_json["binary"] else None
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """Persist the configuration to a JSON file.

        Args:
            path: Destination path where the JSON representation will be written.

        Raises:
            ValueError: If the destination path does not use a .json extension.
        """
        if isinstance(path, str):
            path = Path(path)
        if path.suffix != ".json":
            raise ValueError(f"Config file must be a JSON file: {path}")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=4)
