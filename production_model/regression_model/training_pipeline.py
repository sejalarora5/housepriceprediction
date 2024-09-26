import numpy as np
from production_model.regression_model.pipeline import price_pipe
from processing.data_manager import load_dataset, save_pipeline
from sklearn.model_selection import train_test_split

from config.core import config


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.feature_config.features],  # predictors
        data[config.feature_config.target],
        test_size=config.feature_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.feature_config.random_state,
    )
    y_train = np.log(y_train)

    # fit model
    price_pipe.fit(X_train, y_train)

    # persist trained model
    save_pipeline(pipeline_to_persist=price_pipe)


if __name__ == "__main__":
    run_training()