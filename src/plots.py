import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# ------------------------------------------
# 4.1.4. Correlation Matrix
# ------------------------------------------

# function for prepping the dataset
def prepare_corr_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].fillna(False).astype(int)

    df = df.iloc[:, :-30].select_dtypes(include="number")

    drop_cols = [
        "latitude", "longitude", "host_total_listings_count",
        "host_has_profile_pic", "calculated_host_listings_count_private_rooms",
        "nr_bathrooms", "calculated_host_listings_count_shared_rooms",
        "calculated_host_listings_count_entire_homes",
        "calculated_host_listings_count", "host_about_len",
        "description_len", "neighborhood_overview_len",
        "days_since_first_review", "days_since_last_review",
        "host_in_switzerland", "host_in_zurich",
        "host_verif_email", "host_verif_phone", "host_verif_work_email",
        "minimum_minimum_nights", "maximum_minimum_nights",
        "minimum_maximum_nights", "maximum_maximum_nights",
        "minimum_nights_avg_ntm", "maximum_nights_avg_ntm"
    ]

    return df.drop(columns=drop_cols, errors="ignore")

# creating the correlation matrix
def plot_corr_heatmap(df: pd.DataFrame) -> None:
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(15, 10), dpi=300)

    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        square=True,
        annot_kws={"size": 5},
        ax=ax
    )

    ax.set_title("Correlation Heatmap Airbnb Dataset", fontsize=14)
    ax.set_xticks(np.arange(len(corr.columns)) + 0.5)
    ax.set_yticks(np.arange(len(corr.index)) + 0.5)

    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(corr.index, fontsize=6)

    plt.tight_layout()
    plt.show()

# ------------------------------------------
# 4.2.1. Box Pliot and  Stacked Bar Chart
# ------------------------------------------

# restructure df to get columns by number of rooms
def restructure_housing_by_rooms(df: pd.DataFrame) -> pd.DataFrame:
    room_data = {}
    df = df.set_index("quarlang_")
    for col in df.columns:
        match = re.search(r"(\d+)-Zimmer", col)
        if match:
            room = int(match.group(1))
            room_data.setdefault(room, []).append(col)

    housing_rooms = pd.DataFrame({
        room: df[cols].sum(axis=1)   # or mean depending on meaning
        for room, cols in room_data.items()
    })
    return housing_rooms

# double plot with boxplot and stacked bar chart
def box_and_stacked_housing_stock(df: pd.DataFrame) -> pd.DataFrame:

    # initiate the plot canvas
    fig, axs = plt.subplots(
        1, 2,
        figsize=(14, 5),
        gridspec_kw={"width_ratios": [5, 9]}
    )

    # left: boxplot
    sns.boxplot(ax=axs[0], data=df)
    axs[0].set_xlabel("Number of rooms")
    axs[0].set_ylabel("Count per quartier")
    axs[0].set_title("Boxplot by number of rooms")
    axs[0].set_ylim(0,0.5)

    # right: stacked bar chart
    df.plot(
        kind="bar",
        stacked=True,
        ax=axs[1]
    )
    axs[1].set_xlabel("")
    axs[1].set_ylabel("Share of housing stock")
    axs[1].legend(
        title="Room size",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )
    axs[1].set_title("Stacked bar chart: share of objects per quartier according to number of rooms")

    plt.show()
