
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import pandas as pd
def plot_predictive(y_test_orig: float, y_pred_lr_orig: float, y_pred_rf_orig: float, num_features: str, cat_features: str, bin_features: str, rf_pipeline) -> None:  
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Predicted vs. Actual
    ax = axes[0]
    ax.scatter(y_test_orig, y_pred_lr_orig, alpha=0.3, label="Linear Regression", color="steelblue", s=15)
    ax.scatter(y_test_orig, y_pred_rf_orig, alpha=0.3, label="Random Forest",     color="tomato",    s=15)
    max_val = y_test_orig.quantile(0.95)
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, label="Perfect Prediction")
    ax.set_xlim(0, max_val); ax.set_ylim(0, max_val)
    ax.set_xlabel("Actual Price (CHF)")
    ax.set_ylabel("Predicted Price (CHF)")
    ax.set_title("Predicted vs. Actual")
    ax.legend()

    # Plot 2: Feature Importance
    ax = axes[1]
    cat_names = list(rf_pipeline.named_steps["preprocessing"]
                    .named_transformers_["cat"]
                    .named_steps["encoder"]          
                    .get_feature_names_out(cat_features))
    all_feature_names = num_features + cat_names + bin_features

    importances = pd.Series(
        rf_pipeline.named_steps["model"].feature_importances_,
        index=all_feature_names
    ).sort_values(ascending=True).tail(15)

    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Top 15 Feature Importances (RF)")
    ax.set_xlabel("Importance")

    # Plot 3: Residuals
    ax = axes[2]
    residuals = y_test_orig - y_pred_rf_orig
    ax.scatter(y_pred_rf_orig, residuals, alpha=0.3, color="tomato", s=15)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlim(0, y_test_orig.quantile(0.95))
    ax.set_xlabel("Predicted Price (CHF)")
    ax.set_ylabel("Residual (CHF)")
    ax.set_title("Residuals (RF)")

    plt.suptitle("Airbnb Price Prediction – Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()