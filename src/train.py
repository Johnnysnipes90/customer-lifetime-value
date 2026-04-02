import numpy as np
import pandas as pd
import shap
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from src.config import (
    CLASSIFIER_PARAMS,
    FEATURE_COLUMNS,
    METRICS_FILE,
    MODELING_TABLE_FILE,
    REGRESSOR_PARAMS,
    RETURN_CLASSIFIER_FILE,
    SHAP_BACKGROUND_FILE,
    SHAP_EXPLAINER_FILE,
    VALUE_REGRESSOR_FILE,
)
from src.utils import save_dataframe, save_json, save_model


def load_modeling_data(file_path=MODELING_TABLE_FILE) -> pd.DataFrame:
    """Load customer-level modeling table."""
    return pd.read_csv(file_path)


def train_baseline_and_single_stage(df_model: pd.DataFrame) -> dict:
    """Train baseline and single-stage regression benchmark."""
    X = df_model[FEATURE_COLUMNS].copy()
    y_raw = df_model["future_clv"].copy()
    y = np.log1p(y_raw)

    X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = train_test_split(
        X,
        y,
        y_raw,
        test_size=0.2,
        random_state=42,
    )

    baseline_model = DummyRegressor(strategy="mean")
    baseline_model.fit(X_train, y_train)

    baseline_pred_log = baseline_model.predict(X_test)
    baseline_pred_raw = np.expm1(baseline_pred_log)

    baseline_metrics = {
        "model": "DummyRegressor_mean",
        "mae_raw": float(mean_absolute_error(y_test_raw, baseline_pred_raw)),
        "rmse_raw": float(np.sqrt(mean_squared_error(y_test_raw, baseline_pred_raw))),
        "r2_log": float(r2_score(y_test, baseline_pred_log)),
    }

    single_stage_model = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
    )
    single_stage_model.fit(X_train, y_train)

    single_stage_pred_log = single_stage_model.predict(X_test)
    single_stage_pred_raw = np.expm1(single_stage_pred_log)

    single_stage_metrics = {
        "model": "XGBRegressor_single_stage",
        "mae_raw": float(mean_absolute_error(y_test_raw, single_stage_pred_raw)),
        "rmse_raw": float(
            np.sqrt(mean_squared_error(y_test_raw, single_stage_pred_raw))
        ),
        "r2_log": float(r2_score(y_test, single_stage_pred_log)),
    }

    return {
        "X_test": X_test,
        "y_test_raw": y_test_raw,
        "baseline_metrics": baseline_metrics,
        "single_stage_metrics": single_stage_metrics,
    }


def train_return_classifier(df_model: pd.DataFrame) -> tuple[XGBClassifier, dict]:
    """Train stage-1 return classifier."""
    df_model = df_model.copy()
    df_model["will_return"] = (df_model["future_clv"] > 0).astype(int)

    X_cls = df_model[FEATURE_COLUMNS]
    y_cls = df_model["will_return"]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cls,
        y_cls,
        test_size=0.2,
        random_state=42,
    )

    clf = XGBClassifier(**CLASSIFIER_PARAMS)
    clf.fit(X_train_c, y_train_c)

    y_pred_c = clf.predict(X_test_c)
    y_proba_c = clf.predict_proba(X_test_c)[:, 1]

    class_report = classification_report(y_test_c, y_pred_c, output_dict=True)

    classifier_metrics = {
        "model": "XGBClassifier_return_stage",
        "roc_auc": float(roc_auc_score(y_test_c, y_proba_c)),
        "accuracy": float(class_report["accuracy"]),
        "precision_class_0": float(class_report["0"]["precision"]),
        "recall_class_0": float(class_report["0"]["recall"]),
        "f1_class_0": float(class_report["0"]["f1-score"]),
        "precision_class_1": float(class_report["1"]["precision"]),
        "recall_class_1": float(class_report["1"]["recall"]),
        "f1_class_1": float(class_report["1"]["f1-score"]),
    }

    return clf, classifier_metrics


def train_conditional_regressor(df_model: pd.DataFrame) -> tuple[XGBRegressor, dict]:
    """Train stage-2 regressor on returning customers only."""
    df_reg = df_model[df_model["future_clv"] > 0].copy()

    X_reg = df_reg[FEATURE_COLUMNS]
    y_reg = np.log1p(df_reg["future_clv"])

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg,
        y_reg,
        test_size=0.2,
        random_state=42,
    )

    reg_model = XGBRegressor(**REGRESSOR_PARAMS)
    reg_model.fit(X_train_r, y_train_r)

    pred_log = reg_model.predict(X_test_r)
    pred_raw = np.expm1(pred_log)
    y_test_raw_r = np.expm1(y_test_r)

    reg_metrics = {
        "model": "XGBRegressor_conditional",
        "mae_raw": float(mean_absolute_error(y_test_raw_r, pred_raw)),
        "rmse_raw": float(np.sqrt(mean_squared_error(y_test_raw_r, pred_raw))),
    }

    return reg_model, reg_metrics


def evaluate_two_stage_system(
    clf: XGBClassifier,
    reg_model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test_raw: pd.Series,
) -> dict:
    """Evaluate final expected CLV system."""
    return_prob = clf.predict_proba(X_test)[:, 1]
    clv_pred_log = reg_model.predict(X_test)
    clv_pred = np.expm1(clv_pred_log)

    final_pred = return_prob * clv_pred

    final_metrics = {
        "model": "Two_Stage_Expected_CLV",
        "mae_raw": float(mean_absolute_error(y_test_raw, final_pred)),
        "rmse_raw": float(np.sqrt(mean_squared_error(y_test_raw, final_pred))),
    }

    return final_metrics


def save_shap_artifacts(reg_model: XGBRegressor, df_model: pd.DataFrame) -> None:
    """
    Create and save SHAP explainer artifacts for the conditional regression model.
    """
    df_reg = df_model[df_model["future_clv"] > 0].copy()
    X_reg = df_reg[FEATURE_COLUMNS].copy()

    # Keep a manageable background sample
    background = X_reg.sample(n=min(200, len(X_reg)), random_state=42).copy()

    explainer = shap.Explainer(reg_model, background)

    save_model(explainer, SHAP_EXPLAINER_FILE)
    save_dataframe(background, SHAP_BACKGROUND_FILE)


def train_pipeline() -> dict:
    """Train full CLV system and save artifacts."""
    df_model = load_modeling_data()

    benchmark_results = train_baseline_and_single_stage(df_model)
    clf, classifier_metrics = train_return_classifier(df_model)
    reg_model, regression_stage_metrics = train_conditional_regressor(df_model)

    final_metrics = evaluate_two_stage_system(
        clf=clf,
        reg_model=reg_model,
        X_test=benchmark_results["X_test"],
        y_test_raw=benchmark_results["y_test_raw"],
    )

    metrics_payload = {
        "baseline_metrics": benchmark_results["baseline_metrics"],
        "single_stage_metrics": benchmark_results["single_stage_metrics"],
        "classifier_metrics": classifier_metrics,
        "conditional_regression_metrics": regression_stage_metrics,
        "final_two_stage_metrics": final_metrics,
        "feature_columns": FEATURE_COLUMNS,
        "final_prediction_formula": "P(return) * E(value | return)",
    }

    save_model(clf, RETURN_CLASSIFIER_FILE)
    save_model(reg_model, VALUE_REGRESSOR_FILE)
    save_shap_artifacts(reg_model, df_model)
    save_json(metrics_payload, METRICS_FILE)

    return metrics_payload


if __name__ == "__main__":
    results = train_pipeline()
    print("Training complete.")
    print(results["final_two_stage_metrics"])
