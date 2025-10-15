from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline


RANDOM_STATE = 70134


class QuestionnaireToFeatures(BaseEstimator, TransformerMixin):
    """Custom transformer to convert questionnaire responses to behavioral features."""

    def __init__(self):
        self.questionnaire_mapping = {
            "gender": ("gender", str),
            "age": ("age_years", float),
            "total_income": ("total_income", float),
            "employment_status": ("income_type", str),
            "years_employed": ("years_employed", float),
            "education": ("education_level", str),
            "family_status": ("family_status", str),
            "num_children": ("num_children", float),
            "num_family_members": ("num_family_members", float),
            "owns_car": ("owns_car", self._convert_yn),
            "owns_housing": ("owns_housing", self._convert_yn),
            "housing_type": ("housing_type", str),
        }
        self.expected_columns = None

    @staticmethod
    def _convert_yn(value):
        """Convert Yes/No to Y/N"""
        if isinstance(value, str):
            return "Y" if value.lower() in ["yes", "y"] else "N"
        return value

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.expected_columns = X.columns.tolist()
        return self

    def transform(self, X):
        if isinstance(X, dict):
            features = {}
            for q_key, value in X.items():
                if q_key in self.questionnaire_mapping:
                    feature_name, transformer = self.questionnaire_mapping[q_key]
                    features[feature_name] = transformer(value)

            if self.expected_columns:
                df = pd.DataFrame([features])
                for col in self.expected_columns:
                    if col not in df.columns:
                        df[col] = np.nan
                return df[self.expected_columns]
            return pd.DataFrame([features])

        return X


class BehavioralDataPreprocessor(BaseEstimator, TransformerMixin):
    """Transform behavioral data, feature engineering, and column renaming."""

    def __init__(self):
        self.income_median = None

    def fit(self, X, y=None):
        """Learn imputation values from training data."""
        if "amt_income_total" in X.columns:
            self.income_median = X["amt_income_total"].median()
        return self

    def transform(self, X):
        """Apply preprocessing transformations."""
        df = X.copy()

        if "days_birth" in df.columns:
            df["age_years"] = -df["days_birth"] / 365.25
            df = df.drop(columns=["days_birth"])

        if "days_employed" in df.columns:
            df["years_employed"] = np.where(
                df["days_employed"] > 0, 0, -df["days_employed"] / 365.25
            )
            df = df.drop(columns=["days_employed"])

        column_mapping = {
            "code_gender": "gender",
            "flag_own_car": "owns_car",
            "flag_own_realty": "owns_housing",
            "cnt_children": "num_children",
            "amt_income_total": "total_income",
            "name_income_type": "income_type",
            "name_education_type": "education_level",
            "name_family_status": "family_status",
            "name_housing_type": "housing_type",
            "cnt_fam_members": "num_family_members",
        }

        df = df.rename(
            columns={k: v for k, v in column_mapping.items() if k in df.columns}
        )

        if "total_income" in df.columns and self.income_median is not None:
            df["total_income"] = df["total_income"].fillna(self.income_median)

        return df


class BehavioralTraitsModel:
    """ML-based behavioral traits analysis using Pipeline architecture."""

    def __init__(
        self,
        data_directory: str | Path = "data",
        random_state: int = RANDOM_STATE,
    ):
        self.data_directory = Path(data_directory)
        self.random_state = random_state
        self.job_stability_pipeline = None
        self.payment_behavior_pipeline = None
        self.responsibility_pipeline = None
        self.feature_names = None
        self.categorical_features = None
        self.numerical_features = None
        self.is_trained = False

        self._define_feature_groups()

    def _define_feature_groups(self):
        """Define categorical and numerical features."""
        self.categorical_features = [
            "gender",
            "owns_car",
            "owns_housing",
            "income_type",
            "education_level",
            "family_status",
            "housing_type",
        ]

        self.numerical_features = [
            "age_years",
            "years_employed",
            "num_children",
            "num_family_members",
            "total_income",
        ]

    def _build_pipeline(
        self,
        target_type: str,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        max_depth: int = 5,
    ) -> Pipeline:
        """Build preprocessing and modeling pipeline for specific behavioral trait."""

        if target_type == "job_stability":
            selected_categorical = [
                "gender",
                "income_type",
                "education_level",
                "owns_housing",
            ]
            selected_numerical = ["age_years", "years_employed", "total_income"]
        elif target_type == "payment_behavior":
            selected_categorical = ["family_status", "owns_car", "owns_housing"]
            selected_numerical = ["total_income", "num_children", "num_family_members"]
        else:
            selected_categorical = ["education_level", "family_status", "owns_car"]
            selected_numerical = ["age_years", "total_income"]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    StandardScaler(),
                    selected_numerical,
                ),
                (
                    "cat",
                    OneHotEncoder(
                        sparse_output=False,
                        handle_unknown="ignore",
                        drop=None,
                    ),
                    selected_categorical,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        pipeline = Pipeline(
            [
                ("preprocessing", preprocessor),
                (
                    "regressor",
                    GradientBoostingRegressor(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        return pipeline

    def load_and_preprocess_data(
        self, file_name: str = "application_train.parquet"
    ) -> pd.DataFrame:
        """Load and preprocess data from file."""
        file_path = self.data_directory / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        df = pd.read_parquet(file_path)
        df.columns = df.columns.str.lower()

        behavioral_features = [
            "code_gender",
            "days_birth",
            "days_employed",
            "flag_own_car",
            "flag_own_realty",
            "cnt_children",
            "amt_income_total",
            "name_income_type",
            "name_education_type",
            "name_family_status",
            "name_housing_type",
            "cnt_fam_members",
            "amt_credit",
            "flag_work_phone",
            "flag_emp_phone",
            "flag_phone",
            "flag_email",
        ]

        available_features = [f for f in behavioral_features if f in df.columns]
        df = df[available_features]

        preprocessor = BehavioralDataPreprocessor()
        df = preprocessor.fit_transform(df)
        return df

    def create_behavioral_targets(self, df: pd.DataFrame) -> dict:
        """Create proper behavioral targets."""
        targets = {}

        if "years_employed" in df.columns:
            employment_years = df["years_employed"].clip(0, 20)  # Cap at 20 years
            targets["job_stability"] = employment_years * 5  # Scale to 0-100

        if "total_income" in df.columns:
            log_income = np.log1p(df["total_income"])
            income_percentile = (log_income - log_income.min()) / (
                log_income.max() - log_income.min()
            )
            targets["payment_behavior"] = income_percentile * 100

        if all(col in df.columns for col in ["age_years", "education_level"]):
            # Responsibility score based on age and education
            age_score = (df["age_years"] - 18) / (70 - 18) * 50

            education_mapping = {
                "Lower secondary": 10,
                "Secondary / secondary special": 20,
                "Incomplete higher": 30,
                "Higher education": 40,
                "Academic degree": 50,
            }
            education_score = df["education_level"].map(education_mapping).fillna(20)

            targets["responsibility"] = np.clip(age_score + education_score, 0, 100)

        return targets

    def train(
        self,
        data: pd.DataFrame,
        test_size: float = 0.2,
    ) -> dict:
        """Train behavioral trait models using Pipeline architecture."""
        targets = self.create_behavioral_targets(data)
        results = {}

        if "job_stability" in targets:
            self.job_stability_pipeline = self._build_pipeline("job_stability")

            X = data[self.categorical_features + self.numerical_features].fillna(0)
            y = targets["job_stability"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )

            self.job_stability_pipeline.fit(X_train, y_train)

            train_score = self.job_stability_pipeline.score(X_train, y_train)
            test_score = self.job_stability_pipeline.score(X_test, y_test)

            results["job_stability"] = {
                "train_r2": train_score,
                "test_r2": test_score,
                "samples": len(X),
            }

        if "payment_behavior" in targets:
            self.payment_behavior_pipeline = self._build_pipeline("payment_behavior")

            X = data[self.categorical_features + self.numerical_features].fillna(0)
            y = targets["payment_behavior"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )

            self.payment_behavior_pipeline.fit(X_train, y_train)

            train_score = self.payment_behavior_pipeline.score(X_train, y_train)
            test_score = self.payment_behavior_pipeline.score(X_test, y_test)

            results["payment_behavior"] = {
                "train_r2": train_score,
                "test_r2": test_score,
                "samples": len(X),
            }

        if "responsibility" in targets:
            self.responsibility_pipeline = self._build_pipeline("responsibility")

            X = data[self.categorical_features + self.numerical_features].fillna(0)
            y = targets["responsibility"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )

            self.responsibility_pipeline.fit(X_train, y_train)

            train_score = self.responsibility_pipeline.score(X_train, y_train)
            test_score = self.responsibility_pipeline.score(X_test, y_test)

            results["responsibility"] = {
                "train_r2": train_score,
                "test_r2": test_score,
                "samples": len(X),
            }

        self.is_trained = True
        return results

    def predict_traits(self, client_data: dict | pd.DataFrame) -> dict:
        """Predict behavioral traits using trained pipelines."""
        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train() first.")

        if isinstance(client_data, dict):
            transformer = QuestionnaireToFeatures()
            client_df = transformer.transform(client_data)
        else:
            client_df = client_data.copy()

        for col in (self.categorical_features or []) + (self.numerical_features or []):
            if col not in client_df.columns:
                if col in (self.numerical_features or []):
                    client_df[col] = 0.0
                else:
                    client_df[col] = "Unknown"

        for col in self.categorical_features or []:
            if col in client_df.columns:
                if client_df[col].dtype == bool or str(client_df[col].iloc[0]) in [
                    "True",
                    "False",
                ]:
                    client_df[col] = client_df[col].astype(str)

        results = {}

        if self.job_stability_pipeline is not None:
            pred = self.job_stability_pipeline.predict(client_df)[0]
            results["job_stability"] = max(0, min(100, pred))

        if self.payment_behavior_pipeline is not None:
            pred = self.payment_behavior_pipeline.predict(client_df)[0]
            results["payment_behavior"] = max(0, min(100, pred))

        if self.responsibility_pipeline is not None:
            pred = self.responsibility_pipeline.predict(client_df)[0]
            results["responsibility"] = max(0, min(100, pred))

        scores = [v for v in results.values() if v is not None]
        results["overall_behavioral_score"] = (
            sum(scores) / len(scores) if scores else 50
        )

        return results

    def save(self, filepath: str | Path):
        """Save trained models."""
        model_data = {
            "job_stability_pipeline": self.job_stability_pipeline,
            "payment_behavior_pipeline": self.payment_behavior_pipeline,
            "responsibility_pipeline": self.responsibility_pipeline,
            "categorical_features": self.categorical_features,
            "numerical_features": self.numerical_features,
            "is_trained": self.is_trained,
            "random_state": self.random_state,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Behavioral traits models saved to {filepath}")

    def load(self, filepath: str | Path):
        """Load trained models."""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.job_stability_pipeline = model_data["job_stability_pipeline"]
        self.payment_behavior_pipeline = model_data["payment_behavior_pipeline"]
        self.responsibility_pipeline = model_data["responsibility_pipeline"]
        self.categorical_features = model_data["categorical_features"]
        self.numerical_features = model_data["numerical_features"]
        self.is_trained = model_data["is_trained"]
        self.random_state = model_data.get("random_state", RANDOM_STATE)

        print(f"Behavioral traits models loaded from {filepath}")


def train_behavioral_model():
    """Train and save the behavioral traits model."""
    model = BehavioralTraitsModel()
    data = model.load_and_preprocess_data("application_train.parquet")
    results = model.train(data)

    model_path = Path("src/assets/behavioral_traits_model.pkl")
    model.save(model_path)

    return model


if __name__ == "__main__":
    train_behavioral_model()
