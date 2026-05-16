import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import statsmodels.api as sm
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import OLSInfluence

import folium

# ------------------------------------------
# 4.1.1. Airbnb/ Price
# ------------------------------------------

def plot_price_hist_box(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(df["price"], bins=1000)
    lim=1600
    ax.set_xlim(0, lim)
    ax.set_xlabel("Price [CHF/night]")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Airbnb listing prices")

    median = df["price"].median()
    q25 = df["price"].quantile(0.25)
    q75 = df["price"].quantile(0.75)
    ax.axvline(median, color="red", linestyle="--", linewidth=1.2, label=f"Median: {median:.0f} CHF")
    ax.axvline(q25, color="orange", linestyle="--", linewidth=1.0, label=f"25th pct: {q25:.0f} CHF")
    ax.axvline(q75, color="orange", linestyle="--", linewidth=1.0, label=f"75th pct: {q75:.0f} CHF")
    ax.legend()

    n_cropped = (df["price"] > lim).sum()
    ax.annotate(f"{n_cropped} listings above {lim:.0f} CHF not shown",
                xy=(0.98, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=8, color="grey")

    plt.tight_layout()
    plt.show()

# ------------------------------------------
# 4.1.2. Airbnb/ Reviews
# 4.4. Comparing rental prices and housing stock variance
# ------------------------------------------

def plot_multiple_violinplot(df: pd.DataFrame, cols: list, col_names: list, ylim_min: float, ylim_max: float, title: str = None, ylabel: str = None, grid: bool = False) -> None:
    plot_width = len(cols) * 1.57 + 1
    fig, ax = plt.subplots(figsize=(plot_width, 4))
    sns.violinplot(ax=ax, data=df[cols], order=cols, inner="box", cut=0)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(col_names)
    ax.set_ylim(ylim_min, ylim_max)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if grid:
        plt.grid(zorder=0, alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_multiple_boxplot(df: pd.DataFrame, cols: list, col_names: list, ylim_min: float, ylim_max: float, title: str = None, ylabel: str = None, grid: bool = False) -> None:
    plot_width = len(cols) * 1.57 + 1
    fig, ax = plt.subplots(figsize=(plot_width,4))
    sns.boxplot(ax=ax, data=df[cols],
                order=cols
                )
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(col_names)
    ax.set_ylim(ylim_min,ylim_max)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if grid == True:
        plt.grid(zorder=0,
                 alpha=0.4)
    plt.tight_layout()
    plt.show()

# ------------------------------------------
# 4.1.4. Airbnb/ Correlation Matrix
# ------------------------------------------

# function for prepping the dataset
def prepare_corr_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].fillna(False).astype(int)

    amenity_cols = [c for c in df.columns if c.startswith("amenity_")]
    df = df.drop(columns=amenity_cols).select_dtypes(include="number")

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

def plot_amenity_price_corr(df: pd.DataFrame) -> None:
    amenity_cols = [c for c in df.columns if c.startswith("amenity_")]
    corrs = (
        df[amenity_cols]
        .corrwith(df["price"])
        .rename(lambda c: c.replace("amenity_", ""))
    )
    count_corr = df["amenities_count"].corr(df["price"])
    corrs["total count"] = count_corr
    corrs = corrs.sort_values()

    colors = ["#d9534f" if v < 0 else "#5b9bd5" for v in corrs]
    colors[list(corrs.index).index("total count")] = "#2ca02c"

    fig, ax = plt.subplots(figsize=(len(corrs) * 0.35 + 1, 7))
    ax.bar(corrs.index, corrs.values, color=colors)    
    ax.axhline(0, color="black", linewidth=0.8)  
    ax.set_ylabel("Pearson correlation with price")  
    ax.set_title("Correlation between amenities and listing price")
    ax.tick_params(axis="x", rotation=90)   
    ax.grid(alpha=0.3)   
    plt.tight_layout()
    plt.show()


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
# 4.2.1. Housing Stock/ Box Plot and  Stacked Bar Chart
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
    # sorting the df by 1 room
    df = df.sort_values(1, ascending=False)
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
    axs[1].set_title("Share of objects per quartier according to number of rooms")

    plt.show()

# ------------------------------------------
# 4.2.2. Housing Stock/ Airbnbs per square kilometer
# 4.3.1. Rental Prices/ Price for each quartier
# ------------------------------------------

def plot_per_quartier(df: gpd.GeoDataFrame, column: str, title: str, ylabel: str, grid: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(12,5))
    df[column].plot(
            kind="bar",
            stacked=False,
            ax=ax,
            zorder=3
        )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if grid:
        plt.grid(zorder=0)
    plt.tight_layout()
    plt.show()


# ===========================================================
# ===========Clustering pipeline (Chapter 5)=================
# ===========================================================
# ------------------------------------------
# 5.4. K-Means sweep (elbow + silhouette)
# ------------------------------------------
def plot_kmeans_sweep(
    sweep_a: pd.DataFrame,
    sweep_b: pd.DataFrame,
    labels: tuple[str, str],
) -> None:
    # 2x2 grid: top row inertia (elbow), bottom row silhouette; left col = sweep_a, right col = sweep_b
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for col, sweep, label in zip([0, 1], [sweep_a, sweep_b], labels):
        ax_elbow = axes[0][col]
        ax_sil   = axes[1][col]

        ax_elbow.plot(sweep["k"], sweep["inertia"], "o-")
        ax_elbow.set_title(f"Elbow — {label}")
        ax_elbow.set_xlabel("k")
        ax_elbow.set_ylabel("Inertia")
        ax_elbow.grid(alpha=0.3)

        ax_sil.plot(sweep["k"], sweep["silhouette"], "o-", color="orange")
        ax_sil.set_title(f"Silhouette — {label}")
        ax_sil.set_xlabel("k")
        ax_sil.set_ylabel("Silhouette score")
        ax_sil.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ------------------------------------------
# 5.4. / 5.5. Cluster scatter in 2D space (PCA or UMAP)
# ------------------------------------------
def plot_clusters_2d(
    X_a: np.ndarray, labels_a: np.ndarray,
    X_b: np.ndarray, labels_b: np.ndarray,
    titles: tuple[str, str],
    axis_labels: tuple[str, str] = ("Component 1", "Component 2"),
) -> None:
    # Side-by-side 2D scatter of two cluster solutions, colored by label
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, X, labels, title in zip(axes, [X_a, X_b], [labels_a, labels_b], titles):
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            ax.scatter(X[mask, 0], X[mask, 1], s=10, alpha=0.7, label=f"Cluster {cluster_id}")
        ax.set_title(title)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.legend(markerscale=2, fontsize=8)

    plt.tight_layout()
    plt.show()


# ------------------------------------------
# 5.6. Cluster review heatmap
# ------------------------------------------
def plot_cluster_review_heatmap(
    profile: pd.DataFrame,
    review_cols: list[str],
    title: str,
) -> None:
    # Heatmap of cluster x review-dimension means
    data = profile[review_cols]
    fig, ax = plt.subplots(figsize=(len(review_cols) * 1.2 + 1, len(data) * 0.8 + 1))
    sns.heatmap(
        data,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "Mean score"},
    )
    ax.set_title(title)
    ax.set_ylabel("Cluster")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


# ------------------------------------------
# 5.7. Geographic distribution of clusters
# ------------------------------------------
def zurich_map_clusters(
    airbnb_gdf: "gpd.GeoDataFrame",
    forest: "gpd.GeoDataFrame",
    quartiere: "gpd.GeoDataFrame",
    cluster_col: str,
    title: str = "",
) -> "folium.Map":
    # Folium map of listings colored by cluster label
    # Mirrors the structure of zurich_map_eda() in src/maps.py
    # (forest layer, quartier borders, listings layer, layer control)
    # Uses categorical color per cluster instead of log-price colormap
    PALETTE = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868", 3: "#C44E52", 4: "#8172B3"}

    m = folium.Map(
        location=[47.3769, 8.5417],
        zoom_start=12,
        min_zoom=12,
        max_zoom=17,
        tiles="CartoDB dark-matter",
    )

    folium.GeoJson(forest, name="Forested areas",
                   style_function=lambda x: {"color": "green", "weight": 1,
                                             "opacity": 0.5, "fillOpacity": 0.2}
                   ).add_to(m)

    folium.GeoJson(quartiere, name="Quartier borders",
                   style_function=lambda x: {"color": "white", "weight": 1,
                                             "opacity": 0.7, "fillOpacity": 0},
                   tooltip=folium.GeoJsonTooltip(fields=["qname"], aliases=["Quartier:"])
                   ).add_to(m)

    listings_layer = folium.FeatureGroup(name="Airbnb Listings")
    for _, row in airbnb_gdf.iterrows():
        color = PALETTE.get(int(row[cluster_col]), "#999999")
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=3,
            stroke=False,
            fill=True,
            fill_color=color,
            fill_opacity=1,
            tooltip=f"Cluster {int(row[cluster_col])}",
        ).add_to(listings_layer)
    listings_layer.add_to(m)

    legend_items = "".join(
        f'<i style="background:{PALETTE[k]};width:10px;height:10px;display:inline-block;margin-right:4px"></i>Cluster {k}<br>'
        for k in sorted(PALETTE)
    )
    legend_html = f"""
    <div style="position:fixed;bottom:50px;left:50px;z-index:9999;
                background-color:rgba(0,0,0,0.7);color:white;padding:10px;
                border-radius:6px;font-size:14px;">
    <b>{title}</b><br>{legend_items}
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)
    return m


def plot_cluster_share_per_quartier(
    df: pd.DataFrame,
    cluster_col: str,
    quartier_col: str,
    title: str,
) -> None:
    # Stacked bar chart: x=quartier, y=share (%), stacks=cluster
    counts = (
        df.groupby([quartier_col, cluster_col])
        .size()
        .unstack(fill_value=0)
    )
    shares = counts.div(counts.sum(axis=1), axis=0) * 100
    shares = shares.sort_index()

    fig, ax = plt.subplots(figsize=(14, 5))
    shares.plot(kind="bar", stacked=True, ax=ax, colormap="tab10", width=0.85)
    ax.set_xlabel("")
    ax.set_ylabel("Share (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    plt.show()
