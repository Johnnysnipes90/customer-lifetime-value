import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    RETURN_CLASSIFIER_FILE,
    SHAP_EXPLAINER_FILE,
    VALUE_REGRESSOR_FILE,
)
from src.utils import load_model


class CLVPredictor:
    """Two-stage CLV predictor with SHAP explainability."""

    def __init__(self) -> None:
        self.classifier = load_model(RETURN_CLASSIFIER_FILE)
        self.regressor = load_model(VALUE_REGRESSOR_FILE)
        self.shap_explainer = load_model(SHAP_EXPLAINER_FILE)

    def _prepare_input(self, input_data: dict) -> pd.DataFrame:
        """Create a one-row DataFrame in the correct feature order."""
        df = pd.DataFrame([input_data])
        df = df[FEATURE_COLUMNS]
        return df

    def predict(self, input_data: dict) -> dict:
        """Return probability, conditional value, and expected CLV."""
        X = self._prepare_input(input_data)

        return_probability = float(self.classifier.predict_proba(X)[:, 1][0])

        conditional_value_log = float(self.regressor.predict(X)[0])
        conditional_value = float(np.expm1(conditional_value_log))

        expected_clv = float(return_probability * conditional_value)

        return {
            "return_probability": round(return_probability, 6),
            "predicted_value_if_return": round(conditional_value, 2),
            "expected_clv": round(expected_clv, 2),
        }

    def explain_prediction(self, input_data: dict, top_n: int = 5) -> dict:
        """
        Return SHAP-based explanation for the conditional regression prediction.
        """
        X = self._prepare_input(input_data)

        shap_values = self.shap_explainer(X)
        values = shap_values.values[0]

        feature_impacts = []
        for feature, raw_value, shap_value in zip(
            FEATURE_COLUMNS, X.iloc[0].values, values
        ):
            feature_impacts.append(
                {
                    "feature": feature,
                    "feature_value": (
                        float(raw_value)
                        if isinstance(raw_value, (int, float, np.integer, np.floating))
                        else raw_value
                    ),
                    "shap_value": float(shap_value),
                    "direction": "increase" if shap_value >= 0 else "decrease",
                }
            )

        feature_impacts = sorted(
            feature_impacts,
            key=lambda x: abs(x["shap_value"]),
            reverse=True,
        )

        return {
            "top_factors": feature_impacts[:top_n],
            "all_factors": feature_impacts,
        }
