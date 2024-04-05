# data manipulation
import pandas as pd

# data processing
from sklearn.preprocessing import StandardScaler

# ML models
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    StratifiedKFold,
    learning_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# for saving the model
import pickle


def load_clean_data() -> pd.DataFrame:
    """Load fetal health data and remove duplicates"""
    data = pd.read_csv("data/fetal_health.csv").drop_duplicates()
    return data


def train_models(X, y):
    """
    A function to train machine learning models on the provided data.

    Parameters:
    - data: a DataFrame containing the dataset with the target variable 'fetal_health'

    Returns:
    - random_forest_model: a trained Random Forest Classifier model
    - gbc_model: a trained Gradient Boosting Classifier model
    """

    # splitting the data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # training the models
    # Random Forest
    random_forest = RandomForestClassifier(
        criterion="gini",
        min_samples_leaf=1,
        min_samples_split=6,
        n_estimators=400,
        random_state=42,
    )
    random_forest_model = random_forest.fit(X_train, y_train)

    # Gradient boost Classifier
    gbc = GradientBoostingClassifier(
        learning_rate=1,
        loss="log_loss",
        max_depth=5,
        n_estimators=500,
        random_state=42,
    )

    gbc_model = gbc.fit(X_train, y_train)

    return random_forest_model, gbc_model


def preprocess(df):
    """Split data and scale features."""
    X = df.drop("fetal_health", axis=1)
    y = df["fetal_health"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler


def save_models(rf, gb, scaler):
    """Save trained models and scaler."""
    with open("model/RandomForestClassifier.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open("model/GradientBoostingClassifier.pkl", "wb") as f:
        pickle.dump(gb, f)
    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


def main():
    """Load, scale, train, and save machine learning models."""
    df = load_clean_data()
    X, y, scaler = preprocess(df)
    RandomForest_model, GradientBoosting_model = train_models(X, y)
    save_models(RandomForest_model, GradientBoosting_model, scaler)


if __name__ == "__main__":
    main()
