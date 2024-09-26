from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder
from feature_engine.imputation import AddMissingIndicator, CategoricalImputer, MeanMedianImputer
from feature_engine.selection import DropFeatures
from feature_engine.transformation import LogTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, MinMaxScaler

from config.core import config
from regression_model.processing import features as pp

price_pipe = Pipeline(
    [
        # ===== IMPUTATION =====
        # impute categorical variables with string missing
        (
            "missing_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=config.feature_config.categorical_vars_with_na_missing,
            ),
        ),
        (
            "frequent_imputation",
            CategoricalImputer(
                imputation_method="frequent",
                variables=config.feature_config.categorical_vars_with_na_frequent,
            ),
        ),
        # add missing indicator
        (
            "missing_indicator",
            AddMissingIndicator(variables=config.feature_config.numerical_vars_with_na),
        ),
        # impute numerical variables with the mean
        (
            "mean_imputation",
            MeanMedianImputer(
                imputation_method="mean",
                variables=config.feature_config.numerical_vars_with_na,
            ),
        ),
        # == TEMPORAL VARIABLES ====
        (
            "elapsed_time",
            pp.TemporalVariableTransformer(
                variables=config.feature_config.temporal_vars,
                reference_variable=config.feature_config.ref_var,
            ),
        ),
        ("drop_features", DropFeatures(features_to_drop=[config.feature_config.ref_var])),
        # ==== VARIABLE TRANSFORMATION =====
        ("log", LogTransformer(variables=config.feature_config.numericals_log_vars)),
        (
            "binarizer",
            SklearnTransformerWrapper(
                transformer=Binarizer(threshold=0),
                variables=config.feature_config.binarize_vars,
            ),
        ),
        # === mappers ===
        (
            "mapper_qual",
            pp.Mapper(
                variables=config.feature_config.qual_vars,
                mappings=config.feature_config.qual_mappings,
            ),
        ),
        (
            "mapper_exposure",
            pp.Mapper(
                variables=config.feature_config.exposure_vars,
                mappings=config.feature_config.exposure_mappings,
            ),
        ),
        (
            "mapper_finish",
            pp.Mapper(
                variables=config.feature_config.finish_vars,
                mappings=config.feature_config.finish_mappings,
            ),
        ),
        (
            "mapper_garage",
            pp.Mapper(
                variables=config.feature_config.garage_vars,
                mappings=config.feature_config.garage_mappings,
            ),
        ),
        # == CATEGORICAL ENCODING
        (
            "rare_label_encoder",
            RareLabelEncoder(
                tol=0.01, n_categories=1, variables=config.feature_config.categorical_vars
            ),
        ),
        # encode categorical variables using the target mean
        (
            "categorical_encoder",
            OrdinalEncoder(
                encoding_method="ordered",
                variables=config.feature_config.categorical_vars,
            ),
        ),
        ("scaler", MinMaxScaler()),
        (
            "Lasso",
            Lasso(
                alpha=config.feature_config.alpha,
                random_state=config.feature_config.random_state,
            ),
        ),
    ]
)