import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OrdinalEncoder

EXCLUDE_COLS = {
    "price",
    "price_tier",
    "cluster_quality",
    "cluster_market",
    "estimated_revenues_l365d",
}

TIER_LABELS = ["budget", "mid-range", "premium"]


def bin_price(series: pd.Series) -> pd.Series:
    q33 = series.quantile(1 / 3)
    q66 = series.quantile(2 / 3)
    return pd.cut(
        series,
        bins=[-np.inf, q33, q66, np.inf],
        labels=TIER_LABELS,
    )


def prepare_features(df: pd.DataFrame):
    df = df.copy()
    df["price_tier"] = bin_price(df["price"])
    df = df.dropna(subset=["price_tier"])

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    # Binary: amenity flags and other 0/1 int columns
    binary_cols = [
        c for c in feature_cols
        if c.startswith("amenity_")
        or c in {"bathroom_shared", "bathroom_private", "bathroom_half",
                 "host_verif_email", "host_verif_phone", "host_verif_work_email"}
    ]

    # Categorical: string/category columns
    cat_cols = [
        c for c in feature_cols
        if str(df[c].dtype) in {"object", "category", "string"}
        and c not in binary_cols
    ]

    # Continuous: everything else numeric
    continuous_cols = [
        c for c in feature_cols
        if c not in binary_cols and c not in cat_cols
    ]

    X_cont = df[continuous_cols].apply(pd.to_numeric, errors="coerce")
    X_bin  = df[binary_cols].apply(pd.to_numeric, errors="coerce")
    X_cat  = df[cat_cols].copy()

    # encode categoricals as integer codes for CategoricalNB
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat_enc = pd.DataFrame(
        enc.fit_transform(X_cat.astype(str)),
        columns=cat_cols,
        index=df.index,
    )

    # combine to find valid rows (no NaN in continuous or binary)
    X_check = pd.concat([X_cont, X_bin], axis=1)
    X_check = X_check.dropna(axis=1, how="all")
    mask = X_check.notna().all(axis=1)

    X_cont_clean = X_check.loc[mask]
    X_bin_clean  = X_bin.loc[mask, [c for c in binary_cols if c in X_bin.columns]]
    X_cat_clean  = X_cat_enc.loc[mask]
    y            = df.loc[mask, "price_tier"].astype(str)
    prices       = df.loc[mask, "price"]

    # number of categories per feature from the full dataset - needed so
    # CategoricalNB doesn't go out of bounds on unseen test categories
    min_categories = (X_cat_enc.max() + 1).astype(int).tolist()

    return X_cont_clean, X_bin_clean, X_cat_clean, y, prices, min_categories


class MixedNaiveBayes:
    """
    Combines GaussianNB (continuous), BernoulliNB (binary), and CategoricalNB (categorical)
    by multiplying their class-conditional likelihoods. Valid under the Naive Bayes
    independence assumption.
    """

    def __init__(self):
        self.gaussian    = GaussianNB()
        self.bernoulli   = BernoulliNB()
        self.categorical = None
        self.classes_    = None

    def fit(self, X_cont, X_bin, X_cat, y, min_categories):
        self.classes_ = np.array(sorted(y.unique()))
        self.gaussian.fit(X_cont, y)
        self.bernoulli.fit(X_bin, y)
        self.categorical = CategoricalNB(min_categories=min_categories)
        self.categorical.fit(np.clip(X_cat.values.astype(int), 0, None), y)
        return self

    def predict_log_proba(self, X_cont, X_bin, X_cat):
        log_p  = self.gaussian.predict_log_proba(X_cont)
        log_p += self.bernoulli.predict_log_proba(X_bin)
        log_p += self.categorical.predict_log_proba(
            np.clip(X_cat.values.astype(int), 0, None)
        )
        # each sub-model adds log P(class) once → counted 3 times, subtract 2
        log_p -= 2 * np.log(self.gaussian.class_prior_)
        return log_p

    def predict(self, X_cont, X_bin, X_cat):
        log_p = self.predict_log_proba(X_cont, X_bin, X_cat)
        return self.classes_[np.argmax(log_p, axis=1)]


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=TIER_LABELS)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=TIER_LABELS, yticklabels=TIER_LABELS, ax=ax,
    )
    ax.set_xlabel("Predicted tier")
    ax.set_ylabel("Actual tier")
    ax.set_title("Mixed Naive Bayes - Confusion Matrix (price tier)")
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_price_by_predicted_tier(prices_test: pd.Series, y_test, y_pred, df: pd.DataFrame):
    q33 = df["price"].quantile(1 / 3)
    q66 = df["price"].quantile(2 / 3)

    plot_df = pd.DataFrame({
        "actual_price": prices_test.values,
        "predicted_tier": y_pred,
        "correct": np.array(y_test) == np.array(y_pred),
    })

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.stripplot(
        data=plot_df, x="predicted_tier", y="actual_price",
        order=TIER_LABELS, hue="correct",
        palette={True: "#5b9bd5", False: "#d9534f"},
        alpha=0.4, size=3, jitter=True, ax=ax,
    )
    ax.axhline(q33, color="black", linestyle="--", linewidth=1.0, label=f"33rd pct ({q33:.0f} CHF)")
    ax.axhline(q66, color="grey",  linestyle="--", linewidth=1.0, label=f"66th pct ({q66:.0f} CHF)")
    ax.set_ylim(0, df["price"].quantile(0.99))
    ax.set_xlabel("Predicted tier")
    ax.set_ylabel("Actual price [CHF/night]")
    ax.set_title("Actual price distribution per predicted tier\nBlue = correct, Red = misclassified")
    ax.legend(title="Correct", loc="upper left")
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_tier_distribution(df: pd.DataFrame):
    tiers = bin_price(df["price"]).astype(str)
    counts = tiers.value_counts().reindex(TIER_LABELS)
    thresholds = (df["price"].quantile(1 / 3), df["price"].quantile(2 / 3))

    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color=["#5b9bd5", "#f0a500", "#d9534f"], width=0.6)
    ax.set_xlabel("Price tier")
    ax.set_ylabel("Number of listings")
    ax.set_title(
        f"Price tier distribution\n"
        f"budget < {thresholds[0]:.0f} CHF ≤ mid-range < {thresholds[1]:.0f} CHF ≤ premium"
    )
    ax.set_xticklabels(TIER_LABELS, rotation=0)
    plt.tight_layout()
    plt.show()
    return fig, ax
