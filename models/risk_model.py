from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
import shap

import matplotlib.pyplot as plt


RANDOM_STATE = 70134


class QuestionnaireToFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.questionnaire_mapping = {
            "gender": ("gender", self._convert_gender),
            "age": ("age_years", float),
            "total_income": ("total_income", float),
            "employment_status": ("income_type", str),
            "years_employed": ("years_employed", float),
            "education_level": ("education_level", str),
            "family_status": ("family_status", str),
            "num_children": ("num_children", float),
            "num_family_members": ("num_family_members", float),
            "owns_car": ("owns_car", self._convert_binary),
            "owns_housing": ("owns_housing", self._convert_binary),
            "housing_type": ("housing_type", str),
            "contract_type": ("contract_type", self._convert_contract_type),
            "credit_amount": ("credit_amount", float),
            "loan_annuity": ("loan_annuity", float),
        }
        self.expected_columns = None

    @staticmethod
    def _convert_binary(value):
        """Convert binary values to 0 or 1."""
        if value is None or value == '':
            return 0
        if isinstance(value, str):
            return 1 if value.lower() in ["yes", "y", "true", "1"] else 0
        if isinstance(value, (int, float)):
            return 1 if value else 0
        return 0

    @staticmethod
    def _convert_gender(value):
        """Convert gender to binary (1 for Male, 0 for Female)."""
        if value is None or value == '':
            return 0
        if isinstance(value, str):
            return 1 if value.upper() in ["M", "MALE"] else 0
        if isinstance(value, (int, float)):
            return int(value) if value in [0, 1] else 0
        return 0
    
    @staticmethod
    def _convert_contract_type(value):
        """Convert contract type to binary (1 for Cash loans, 0 for Revolving loans)."""
        if value is None or value == '':
            return 1  # Default to Cash loans
        if isinstance(value, str):
            return 1 if "cash" in value.lower() else 0
        if isinstance(value, (int, float)):
            return int(value) if value in [0, 1] else 1
        return 1

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
                    try:
                        features[feature_name] = transformer(value)
                    except (ValueError, TypeError) as e:
                        # If conversion fails, keep original value for categorical or set nan for numeric
                        if transformer in (float, int):
                            features[feature_name] = np.nan
                        else:
                            features[feature_name] = value

            if self.expected_columns:
                df = pd.DataFrame([features])
                for col in self.expected_columns:
                    if col not in df.columns:
                        df[col] = np.nan
                
                # Ensure proper dtypes for each column type
                numeric_cols = ["age_years", "years_employed", "num_children", "num_family_members", 
                                "total_income", "credit_amount", "loan_annuity"]
                binary_cols = ["gender", "owns_car", "owns_housing", "contract_type"]
                
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                for col in binary_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                
                return df[self.expected_columns]
            return pd.DataFrame([features])

        return X


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.loan_annuity_median = None
        self.num_family_members_median = None

    def fit(self, X, y=None):
        if "amt_annuity" in X.columns:
            self.loan_annuity_median = X["amt_annuity"].median()
        if "cnt_fam_members" in X.columns:
            self.num_family_members_median = X["cnt_fam_members"].median()
        return self

    def transform(self, X):
        df = X.copy()

        if "days_birth" in df.columns:
            df["age_years"] = -df["days_birth"] / 365
            df = df.drop(columns=["days_birth"])

        if "days_employed" in df.columns:
            df["years_employed"] = np.where(
                -df["days_employed"] / 365 < 0,
                0,
                -df["days_employed"] / 365,
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
            "name_contract_type": "contract_type",
            "amt_credit": "credit_amount",
            "amt_annuity": "loan_annuity",
            "target": "defaulted",
        }

        df = df.rename(
            columns={k: v for k, v in column_mapping.items() if k in df.columns}
        )

        if "gender" in df.columns:
            df["gender"] = df["gender"].replace({"XNA": "M"})
            df = df[df["gender"].isin(["M", "F"])]
            df["gender"] = (df["gender"] == "M").astype(int)

        if "owns_car" in df.columns:
            df["owns_car"] = (df["owns_car"] == "Y").astype(int)

        if "owns_housing" in df.columns:
            df["owns_housing"] = (df["owns_housing"] == "Y").astype(int)

        if "contract_type" in df.columns:
            df["contract_type"] = (df["contract_type"] == "Cash loans").astype(int)

        if "loan_annuity" in df.columns and self.loan_annuity_median is not None:
            df["loan_annuity"] = df["loan_annuity"].fillna(self.loan_annuity_median)

        if (
            "num_family_members" in df.columns
            and self.num_family_members_median is not None
        ):
            df["num_family_members"] = df["num_family_members"].fillna(
                self.num_family_members_median
            )

        expected_order = [
            "age_years",
            "years_employed",
            "num_children",
            "num_family_members",
            "total_income",
            "credit_amount",
            "loan_annuity",
            "gender",
            "owns_car",
            "owns_housing",
            "contract_type",
            "income_type",
            "education_level",
            "family_status",
            "housing_type",
            "defaulted",
        ]
        
        available_columns = [col for col in expected_order if col in df.columns]
        return df[available_columns]


class RiskModel:
    def __init__(
        self,
        data_directory: str | Path = "data",
        random_state: int = RANDOM_STATE,
    ):
        self.data_directory = Path(data_directory)
        self.random_state = random_state
        self.optimal_threshold = 0.5

        self.pipeline = None
        self.feature_names = None
        self.categorical_features = None
        self.numerical_features = None
        self.shap_explainer = None
        self.is_trained = False

        self._define_feature_groups()

    def _define_feature_groups(self):
        self.binary_features = [
            "gender",
            "owns_car",
            "owns_housing",
            "contract_type",
        ]

        self.categorical_features = [
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
            "credit_amount",
            "loan_annuity",
        ]

    def _build_pipeline(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        scaler_type: str = "robust",
    ) -> ImbPipeline:
        if scaler_type == "robust":
            numerical_scaler = RobustScaler()
        elif scaler_type == "power":
            numerical_scaler = PowerTransformer(method="yeo-johnson", standardize=True)
        else:
            numerical_scaler = StandardScaler()

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numerical_scaler,
                    self.numerical_features,
                ),
                (
                    "bin",
                    "passthrough",
                    self.binary_features,
                ),
                (
                    "cat",
                    OneHotEncoder(
                        sparse_output=False,
                        handle_unknown="ignore",
                        drop="first",
                    ),
                    self.categorical_features,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        pipeline = ImbPipeline(
            [
                ("preprocessing", preprocessor),
                (
                    "resampling",
                    SMOTETomek(
                        sampling_strategy="auto",
                        random_state=self.random_state,
                        smote=SMOTE(
                            k_neighbors=7,
                            random_state=self.random_state,
                        ),
                        tomek=TomekLinks(
                            sampling_strategy="auto",
                        ),
                    ),
                ),
                (
                    "classifier",
                    GradientBoostingClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        subsample=1.0,
                        random_state=self.random_state,
                        verbose=0,
                    ),
                ),
            ]
        )

        return pipeline

    def load_and_preprocess_data(
        self, file_name: str = "application_train.parquet"
    ) -> pd.DataFrame:
        """
        Load and preprocess data from file.
        """
        file_path = self.data_directory / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        df = pd.read_parquet(file_path)
        df.columns = df.columns.str.lower()

        questionnaire_features = [
            "code_gender",
            "flag_own_car",
            "flag_own_realty",
            "cnt_children",
            "amt_income_total",
            "name_income_type",
            "name_education_type",
            "name_family_status",
            "name_housing_type",
            "days_birth",
            "days_employed",
            "cnt_fam_members",
            "name_contract_type",
            "amt_credit",
            "amt_annuity",
            "target",
        ]
        df = df[questionnaire_features]
        preprocessor = DataPreprocessor()
        df = preprocessor.fit_transform(df)
        return df

    def train(
        self,
        data: pd.DataFrame,
        target_col: str = "defaulted",
        test_size: float = 0.2,
        tune_hyperparameters: bool = False,
    ) -> dict:
        print("TRAINING PIPELINE-BASED RISK MODEL")
        print("=" * 70)

        X = data.drop(columns=[target_col])
        y = data[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        print(f"\nTrain set: {X_train.shape[0]:,} samples")
        print(f"Test set: {X_test.shape[0]:,} samples")
        print(f"Default rate (train): {y_train.mean():.3f}")
        print("Resampling strategy: SMOTETomek (oversample + clean)")
        print(f"Hyperparameter tuning: {tune_hyperparameters}")

        print("\nBuilding imblearn.Pipeline with SMOTETomek...")
        self.pipeline = self._build_pipeline()
        print(f"Pipeline steps: {list(self.pipeline.named_steps.keys())}")

        if tune_hyperparameters:
            print("\nWrapping pipeline in GridSearchCV for hyperparameter tuning...")
            param_grid = {
                "classifier__n_estimators": [100],
                "classifier__learning_rate": [0.1],
                "classifier__max_depth": [7],
                "resampling__sampling_strategy": ["auto"],  # Tune SMOTE ratio
                "resampling__smote__k_neighbors": [7],  # Tune k_neighbors
            }

            print(f"  Parameter grid: {param_grid}")
            total_combinations = 1
            for param_values in param_grid.values():
                total_combinations *= len(param_values)
            print(f"  Total combinations: {total_combinations}")

            grid_search = GridSearchCV(
                estimator=self.pipeline,
                param_grid=param_grid,
                cv=3,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=1,
                return_train_score=True,
            )

            print("\nRunning GridSearchCV...")
            grid_search.fit(X_train, y_train)

            print(f"\nBest parameters found:")
            for param, value in grid_search.best_params_.items():
                print(f"    {param}: {value}")
            print(f"\nBest CV ROC-AUC: {grid_search.best_score_:.4f}")

            self.pipeline = grid_search.best_estimator_
        else:
            print("\nTraining pipeline...")
            self.pipeline.fit(X_train, y_train)

        print("✓ Training complete!")

        preprocessor = self.pipeline.named_steps["preprocessing"]
        cat_encoder = preprocessor.named_transformers_["cat"]

        numerical_names = self.numerical_features
        binary_names = self.binary_features
        categorical_names = cat_encoder.get_feature_names_out(
            self.categorical_features
        ).tolist()
        self.feature_names = numerical_names + binary_names + categorical_names

        print(f"Total features: {len(self.feature_names)}")
        print(f"  - Numerical: {len(numerical_names)}")
        print(f"  - Binary: {len(binary_names)}")
        print(f"  - Categorical (one-hot): {len(categorical_names)}")

        # Evaluate on test set
        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
        y_pred = self.pipeline.predict(X_test)

        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Find optimal threshold
        self.optimal_threshold = self._find_optimal_threshold(y_test, y_pred_proba)
        y_pred_optimal = (y_pred_proba >= self.optimal_threshold).astype(int)
        balanced_acc_optimal = balanced_accuracy_score(y_test, y_pred_optimal)
        f1_optimal = f1_score(y_test, y_pred_optimal)

        print("EVALUATION METRICS")
        print("=" * 70)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"Optimal Threshold: {self.optimal_threshold:.3f}")
        print("\nDefault Threshold (0.5):")
        print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"\nOptimal Threshold ({self.optimal_threshold:.3f}):")
        print(f"  Balanced Accuracy: {balanced_acc_optimal:.4f}")
        print(f"  F1 Score: {f1_optimal:.4f}")

        print("\nInitializing SHAP explainer...")
        X_train_transformed = self.pipeline.named_steps["preprocessing"].transform(
            X_train
        )

        # Use a sample for SHAP to speed up initialization
        sample_size = min(100, X_train_transformed.shape[0])
        sample_indices = np.random.choice(
            X_train_transformed.shape[0], sample_size, replace=False
        )
        X_sample = X_train_transformed[sample_indices]

        self.shap_explainer = shap.TreeExplainer(
            self.pipeline.named_steps["classifier"],
            X_sample,
        )
        print("SHAP explainer initialized")

        self.is_trained = True

        return {
            "roc_auc": roc_auc,
            "balanced_accuracy": balanced_acc,
            "f1_score": f1,
            "optimal_threshold": self.optimal_threshold,
            "balanced_accuracy_optimal": balanced_acc_optimal,
            "f1_score_optimal": f1_optimal,
            "n_features": len(self.feature_names),
        }

    def _find_optimal_threshold(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray
    ) -> float:
        """Find optimal classification threshold by maximizing F1 score."""
        thresholds = np.arange(0.1, 0.9, 0.01)
        f1_scores = []

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1_scores.append(f1_score(y_true, y_pred))

        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx]

    def predict(self, client_data: dict) -> dict:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        transformer = QuestionnaireToFeatures()
        # Use the exact order that matches training data (from DataPreprocessor)
        expected_order = [
            "age_years",
            "years_employed",
            "num_children",
            "num_family_members",
            "total_income",
            "credit_amount",
            "loan_annuity",
            "gender",
            "owns_car",
            "owns_housing",
            "contract_type",
            "income_type",
            "education_level",
            "family_status",
            "housing_type",
        ]
        transformer.expected_columns = expected_order

        X = transformer.transform(client_data)

        risk_proba = self.pipeline.predict_proba(X)[0, 1]
        risk_score = int(risk_proba * 1000)

        if risk_proba < 0.3:
            risk_category = "Low Risk"
            color = "green"
            recommendation = "Approved"
        elif risk_proba < 0.6:
            risk_category = "Medium Risk"
            color = "orange"
            recommendation = "Review Required"
        else:
            risk_category = "High Risk"
            color = "red"
            recommendation = "Declined"

        return {
            "risk_probability": risk_proba,
            "risk_score": risk_score,
            "risk_category": risk_category,
            "recommendation": recommendation,
            "color": color,
        }

    def predict_with_explanations(self, client_data: dict, top_n: int = 10) -> dict:
        result = self.predict(client_data)

        if self.shap_explainer is not None:
            transformer = QuestionnaireToFeatures()
            # Use the exact order that matches training data (from DataPreprocessor)
            expected_order = [
                "age_years",
                "years_employed",
                "num_children",
                "num_family_members",
                "total_income",
                "credit_amount",
                "loan_annuity",
                "gender",
                "owns_car",
                "owns_housing",
                "contract_type",
                "income_type",
                "education_level",
                "family_status",
                "housing_type",
            ]
            transformer.expected_columns = expected_order
            X = transformer.transform(client_data)

            X_transformed = self.pipeline.named_steps["preprocessing"].transform(X)

            shap_values = self.shap_explainer.shap_values(X_transformed)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            feature_contributions = []
            for i, (feature, value) in enumerate(
                zip(self.feature_names, shap_values[0])
            ):
                feature_contributions.append(
                    {
                        "feature": feature,
                        "contribution": float(value),
                        "abs_contribution": abs(float(value)),
                    }
                )

            feature_contributions.sort(
                key=lambda x: x["abs_contribution"], reverse=True
            )

            result["shap_explanations"] = feature_contributions[:top_n]
            result["base_value"] = float(self.shap_explainer.expected_value)

        return result

    def save(self, filepath: str = "pipeline_risk_model.pkl"):
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
    
        if Path(filepath).is_absolute():
            save_path = Path(filepath)
        else:
            save_path = self.data_directory.parent / filepath

        model_data = {
            "pipeline": self.pipeline,
            "feature_names": self.feature_names,
            "binary_features": self.binary_features,
            "categorical_features": self.categorical_features,
            "numerical_features": self.numerical_features,
            "shap_explainer": self.shap_explainer,
            "optimal_threshold": self.optimal_threshold,
            "random_state": self.random_state,
            "is_trained": self.is_trained,
        }

        with open(save_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to: {save_path}")

    def load(self, filepath: str = "pipeline_risk_model.pkl"):
        load_path = self.data_directory.parent / filepath

        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        with open(load_path, "rb") as f:
            model_data = pickle.load(f)

        if not isinstance(model_data, dict):
            raise ValueError(f"Expected dict, got {type(model_data)}")
        
        available_keys = list(model_data.keys())
        
        if "pipeline" in model_data:
            self.pipeline = model_data["pipeline"]
        else:
            raise KeyError(f"Model file doesn't have 'pipeline' key. Available keys: {available_keys}")

        self.feature_names = model_data["feature_names"]
        self.binary_features = model_data.get("binary_features", ["gender", "owns_car", "owns_housing"])
        self.categorical_features = model_data["categorical_features"]
        self.numerical_features = model_data["numerical_features"]
        self.shap_explainer = model_data.get("shap_explainer")
        self.optimal_threshold = model_data.get("optimal_threshold", 0.5)
        self.random_state = model_data.get("random_state", RANDOM_STATE)
        self.is_trained = model_data.get("is_trained", True)

        print(f"Model loaded from: {load_path}")

    def plot_feature_importance(self, top_n: int = 15):
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting importance")

        classifier = self.pipeline.named_steps["classifier"]
        importances = classifier.feature_importances_

        importance_df = (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": importances,
                }
            )
            .sort_values("importance", ascending=False)
            .head(top_n)
        )

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importance_df)), importance_df["importance"])
        plt.yticks(range(len(importance_df)), importance_df["feature"])
        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Feature Importances")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

        return importance_df


def main():
    """Example usage of the RiskModel."""
    print("PIPELINE RISK MODEL - EXAMPLE USAGE")
    print("=" * 70)
    model = RiskModel(data_directory=Path(__file__).parent.parent / "data")
    data = model.load_and_preprocess_data(file_name="application_train.parquet")
    model.train(data=data, tune_hyperparameters=True)

    print("TESTING PREDICTION")
    print("=" * 70)

    test_client = {
        "gender": "Female",
        "age": 35,
        "total_income": 150000,
        "employment_status": "Working",
        "years_employed": 8,
        "education": "Higher education",
        "family_status": "Married",
        "num_children": 2,
        "num_family_members": 4,
        "owns_car": "Yes",
        "owns_housing": "Yes",
        "housing_type": "House / apartment",
        "contract_type": "Cash loans",
        "credit_amount": 600000,
        "loan_annuity": 30000,
    }

    prediction = model.predict_with_explanations(client_data=test_client)

    print(f"\nRisk Score: {prediction['risk_score']}/1000")
    print(f"Risk Probability: {prediction['risk_probability']:.3f}")
    print(f"Risk Category: {prediction['risk_category']}")
    print(f"Recommendation: {prediction['recommendation']}")

    if "shap_explanations" in prediction:
        print("\nTop Contributing Features:")
        for i, feat in enumerate(prediction["shap_explanations"][:5], 1):
            print(f"  {i}. {feat['feature']}: {feat['contribution']:+.4f}")

    model.save(filepath="risk_prediction_model.pkl")

    print("\nDone!")


if __name__ == "__main__":
    main()
