import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import OLSInfluence
from shapely.geometry import Point
from scipy import stats


# ------------------------------------------
# 5.3. linear regression
# ------------------------------------------

def plot_fit_resid(X: pd.DataFrame, y: pd.DataFrame) -> None:
    # linear regression model
    # model
    reg = sm.OLS(y, X).fit()
    # -------------------------------------------
    # parameter calculation
    X2 = X["airbnb_density"]
    infl = OLSInfluence(reg)
    fitted = reg.fittedvalues
    resid = reg.resid
    resid_norm = reg.get_influence().resid_studentized_internal
    resid_abs_norm_sqr = np.sqrt(np.abs(resid_norm))
    resid_abs = np.abs(resid)
    resid_stud = infl.resid_studentized_internal.to_numpy()
    leverage = infl.hat_matrix_diag
    cooks = infl.cooks_distance[0]
    inter, s = reg.params
    line = s * X + inter
    # -------------------------------------------
    # plot setup
    fig, axs = plt.subplots(1,5, figsize=(18, 4))
    # -------------------------------------------
    # scatter plot of regression line
    axs[0].scatter(X2,y,
                   marker="o",
                   s=25,
                   facecolors="none",
                   edgecolors="grey")
    axs[0].plot(X2, line, lw=1, color='red', alpha=0.8)
    axs[0].set_xlabel("Predictor")
    x_pad = 0.05 * (max(X2) - min(X2))
    y_pad = 0.05 * (max(y) - min(y))
    axs[0].set_xlim(min(X2) - x_pad, max(X2) + x_pad)
    axs[0].set_ylim(min(y) - y_pad, max(y) + y_pad)
    axs[0].set_ylabel("Response")
    axs[0].set_title("1) Linear Regression Model")
    # -------------------------------------------
    # Residuals vs. fitted plot
    sns.residplot(x=fitted,
                  y=resid,
                  lowess=True,
                  scatter_kws={
                      "marker": "o",
                      "s":25,
                      "facecolors": "none",
                      "edgecolors": "grey"
                      },
                  line_kws={
                      'color': 'red',
                      'lw': 1,
                      'alpha': 0.8
                      },
                  ax=axs[1])
    axs[1].axhline(0,
                   color="darkgrey",
                   linewidth=1,
                   linestyle="--")
    axs[1].set_xlabel("Fitted values")
    axs[1].set_ylabel("Residuals")
    axs[1].set_title("2) Residuals vs Fitted")
    # -------------------------------------------
    # Normal Q-Q Plot
    QQ = ProbPlot(resid_norm)
    QQ.qqplot(line='45',
                          lw=1,
                          marker="o",
                          markersize=5,
                          markerfacecolor="none",
                          markeredgecolor="grey",
                          ax=axs[2])
    axs[2].lines[1].set_alpha(0.8)
    axs[2].set_xlabel("Theoretical Quantiles")
    axs[2].set_ylabel("Standardized Residuals")
    axs[2].set_title("3) Normal Q-Q")
    # -------------------------------------------
    # Scale-Location
    axs[3].scatter(fitted,resid_abs_norm_sqr,
                   marker="o",
                   s=25,
                   facecolors="none",
                   edgecolors="grey")
    sns.regplot(x=fitted, y=resid_abs_norm_sqr,
              scatter=False,
              ci=False,
              lowess=True,
              line_kws={"color": "red", "lw": 1, "alpha": 0.8},
              ax=axs[3])
    axs[3].set_xlabel("Fitted values")
    axs[3].set_ylabel(r"$\sqrt{|Standardized Residuals|}$")
    axs[3].set_title("4) Scale-Location")
    # -------------------------------------------
    # Residuals vs. Leverage
    threshold = 4 / len(fitted)
    influential_points = np.where(cooks > threshold)[0]

    axs[4].scatter(
        leverage,
        resid_stud,
        s=80 * cooks,
        marker="o",
        facecolors="none",
        edgecolors="grey"
    )
    if influential_points.size > 0:
        for i in influential_points:
            axs[4].annotate(i, (leverage[i], resid_stud[i]))
    axs[4].set_xlabel("Leverage")
    axs[4].set_ylabel("Studentized Residuals")
    axs[4].set_title("5) Influence Plot")

    
    

    plt.tight_layout()
    plt.show()

    print(reg.summary())


# ------------------------------------------
# 5.4.1. calculating distance
# ------------------------------------------

# create GeoDataFrame with 4 different center distances
def get_center_gdf(gdf: gpd.GeoDataFrame, center_data: dict):
    # gdf with center coordinates
    center_gdf = gpd.GeoDataFrame(center_data, crs="EPSG:2056")

    # get central points for each quartier
    gdf["geometry_center"] = gdf.representative_point()

    # convert to GeoSeries with geometry type as points, not polygons
    gdf = gpd.GeoDataFrame(
        gdf,
        geometry="geometry_center",
        crs="EPSG:2056"
    )
    
    # calculate distances to different CBDs
    for i, row in center_gdf.iterrows():
        gdf[f"dist_{row['name']}"] = (gdf.geometry.distance(row.geometry))
    
    return gdf


# ------------------------------------------
# 5.4.2. Partial correlations
# ------------------------------------------

# helper function to get residuals from a lin reg
def get_residuals(y, x):
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return model.resid

# calculating partial correlations
def get_part_corr(gdf: gpd.GeoDataFrame, y1_col: str, y2_col: str, naive_r: float, naive_p: float) -> pd.DataFrame:
    center_definitions = {
        "Hauptbahnhof": "dist_hb",
        "Paradeplatz":  "dist_paradeplatz",
        "Bellevue":     "dist_bellevue",
        "Grossmünster": "dist_grossmunster",
    }
    
    results = []

    for label, dist_col in center_definitions.items():
        dist = gdf[dist_col]
        
        resid_rent   = get_residuals(gdf[y1_col], dist)
        resid_airbnb = get_residuals(gdf[y2_col], dist)
        
        partial_r, partial_p = stats.pearsonr(resid_rent, resid_airbnb)
        
        results.append({
            "Center Definition": label,
            "Partial r":         round(partial_r, 3),
            "p-value":           round(partial_p, 4),
            "Significant":       partial_p < 0.05,
        })
    results_df = pd.DataFrame(results)

    # print summary table
    print("Summary")
    print(f"  Naive r:  {naive_r:.3f} (p = {naive_p:.4f})")
    print(results_df.to_string(index=False))

    return results


# ------------------------------------------
# 5.4.3. Visualisation
# ------------------------------------------

def plot_bar_r(naive_r: float, results: pd.DataFrame) -> None:
    center_definitions = {
        "Hauptbahnhof": "dist_hb",
        "Paradeplatz":  "dist_paradeplatz",
        "Bellevue":     "dist_bellevue",
        "Grossmünster": "dist_grossmunster",
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ["Naive"] + list(center_definitions.keys())
    r_values = [naive_r] + [r["Partial r"] for r in results]
    colors = ["steelblue"] + ["crimson"] * 4

    bars = ax.bar(labels, r_values, color=colors, alpha=0.8, edgecolor="k", linewidth=0.5)

    # Add value labels on top of bars
    for bar, val in zip(bars, r_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 1)
    ax.axhline(naive_r, color="steelblue", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_ylabel("Pearson r")
    ax.set_title("Naive vs. Partial Correlation\n(Airbnb Density ~ Rent, controlling for distance to center)")
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.show()