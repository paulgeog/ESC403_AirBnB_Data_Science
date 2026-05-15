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
import libpysal
from esda.moran import Moran


# ------------------------------------------
# 5.3. linear regression
# ------------------------------------------

def plot_fit_resid(X: pd.DataFrame, y: pd.DataFrame) -> None:
    # linear regression model
    # model
    reg = sm.OLS(y, X).fit()
    # -------------------------------------------
    # parameter calculation
    X2 = X.iloc[:, 1]  # grabs first predictor column, skipping the constant
    infl = OLSInfluence(reg)
    fitted = reg.fittedvalues
    resid = reg.resid
    resid_norm = reg.get_influence().resid_studentized_internal
    resid_abs_norm_sqr = np.sqrt(np.abs(resid_norm))
    resid_abs = np.abs(resid)
    resid_stud = infl.resid_studentized_internal.to_numpy()
    leverage = infl.hat_matrix_diag
    cooks = infl.cooks_distance[0]
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
    sort_idx = X2.argsort()
    axs[0].plot(X2.iloc[sort_idx], reg.fittedvalues.iloc[sort_idx], lw=1, color='red', alpha=0.8)
    #axs[0].plot(X2, reg.fittedvalues, lw=1, color='red', alpha=0.8)
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
# 5.3.3. Analysis with CI
# ------------------------------------------
def bootstrap_ols(X, y, n_boot=10000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    boot_coefs = np.empty((n_boot, X.shape[1]))
    boot_r2 = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        model = sm.OLS(y.iloc[idx], X.iloc[idx]).fit()
        boot_coefs[i] = model.params
        boot_r2[i] = model.rsquared

    lower = (100 - ci) / 2
    upper = 100 - lower

    ci_coefs = np.percentile(boot_coefs, [lower, upper], axis=0)
    ci_r2 = np.percentile(boot_r2, [lower, upper])

    return boot_coefs, boot_r2, ci_coefs, ci_r2


# ------------------------------------------
# 5.3.4. Moran's I
# ------------------------------------------
def get_morans_weights(gdf: gpd.GeoDataFrame):
    w = libpysal.weights.KNN.from_dataframe(gdf, use_index=True, k=8)
    w.transform = "r"
    return w

def moran(X, y, w):
    model = sm.OLS(y, X).fit()
    residuals = model.resid

    moran = Moran(residuals, w)  
    print(f"Moran's I: {moran.I}")
    print(f"P-Value: {moran.p_sim}")

# ------------------------------------------
# 5.4.1. calculating distance
# ------------------------------------------

# create GeoDataFrame with 4 different center distances
def get_center_gdf(gdf: gpd.GeoDataFrame, center_data: dict):
    # gdf with center coordinates
    center_gdf = gpd.GeoDataFrame(center_data, crs="EPSG:2056")

    # get central points for each quartier
    gdf["geometry_center"] = gdf.representative_point()
    
    # calculate distances to different CBDs
    for i, row in center_gdf.iterrows():
        gdf[f"dist_{row['name']}"] = (gdf["geometry_center"].distance(row.geometry))
    
    return gdf


# ------------------------------------------
# 5.4.2. Partial correlations
# ------------------------------------------

# helper function to get residuals from a lin reg
def get_residuals(y, x):
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return model.resid

# helper function for bootstrapping
def bootstrap_r(x, y, n_boot=10000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_r = np.empty(n_boot)
    
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_r[i] = stats.pearsonr(x[idx], y[idx])[0]
    
    lower = (100 - ci) / 2
    upper = 100 - lower
    ci_low, ci_high = np.percentile(boot_r, [lower, upper])
    
    return boot_r, ci_low, ci_high

# calculating partial correlations
def get_part_corr(gdf: gpd.GeoDataFrame, y1_col: str, y2_col: str, naive_r: float, naive_p: float, n_boot: int = 10000):
    center_definitions = {
        "Hauptbahnhof": "dist_hb",
        "Paradeplatz":  "dist_paradeplatz",
        "Bellevue":     "dist_bellevue",
        "Grossmünster": "dist_grossmunster",
    }
    results = []
    residuals = {}  # new: store residuals per center

    for label, dist_col in center_definitions.items():
        dist = gdf[dist_col]
        resid_y1 = get_residuals(gdf[y1_col], dist)
        resid_y2 = get_residuals(gdf[y2_col], dist)
        partial_r, partial_p = stats.pearsonr(resid_y1, resid_y2)
        _, ci_low, ci_high = bootstrap_r(resid_y1, resid_y2, n_boot=n_boot)
        results.append({
            "Center Definition": label,
            "Partial r":         round(partial_r, 3),
            "Partial r²":        round(partial_r**2, 3),
            "95% CI (r²)":       f"[{ci_low**2:.2f}, {ci_high**2:.2f}]",
            "p-value":           round(partial_p, 4),
            "Significant":       partial_p < 0.05,
        })
        residuals[label] = {
            "dist":     np.array(dist),
            "y1":       np.array(gdf[y1_col]),
            "y2":       np.array(gdf[y2_col]),
            "resid_y1": np.array(resid_y1),
            "resid_y2": np.array(resid_y2),
        }

    results_df = pd.DataFrame(results)
    print("Summary")
    print(f"  Naive r:  {naive_r:.3f} (p = {naive_p:.4f})")
    print(results_df.to_string(index=False))
    return results_df, residuals  # now returns both

# ------------------------------------------
# 5.4.2. Partial Correlation visualisation
# ------------------------------------------

def plot_partial_corr(results_df: pd.DataFrame, residuals: dict, y1_col: str, y2_col: str):
    n = len(residuals)
    fig, axs = plt.subplots(n, 3, figsize=(20, 5 * n))
    fig.suptitle("Partial Correlations by City Center", fontsize=14, y=1.01)

    def draw_panel(ax, x_vals, y_vals, x_label, y_label):
        X = sm.add_constant(x_vals)
        model = sm.OLS(y_vals, X).fit()
        fitted = np.array(model.fittedvalues)
        x_sorted = np.sort(x_vals)

        # residual lines
        for xi, yi, fi in zip(x_vals, y_vals, fitted):
            ax.plot([xi, xi], [yi, fi], color='red', lw=0.8, alpha=0.6)

        # scatter points
        ax.scatter(x_vals, y_vals, marker='o', s=25,
                facecolors='none', edgecolors='grey', zorder=3)

        # regression line
        ax.plot(x_sorted, model.params[0] + model.params[1] * x_sorted,
                color='black', lw=1, alpha=0.8, zorder=2)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    for row, (label, data) in enumerate(residuals.items()):
        partial_r = results_df.loc[results_df["Center Definition"] == label, "Partial r"].values[0]

        draw_panel(axs[row, 0], data["dist"], data["y1"],
                   x_label=f"Distance to {label} (m)", y_label=y1_col)
        axs[row, 0].set_title(f"{label} — {y1_col} vs Distance")

        draw_panel(axs[row, 1], data["dist"], data["y2"],
                   x_label=f"Distance to {label} (m)", y_label=y2_col)
        axs[row, 1].set_title(f"{label} — {y2_col} vs Distance")

        draw_panel(axs[row, 2], data["resid_y2"], data["resid_y1"],
                   x_label=f"Residuals ({y2_col} ~ distance)",
                   y_label=f"Residuals ({y1_col} ~ distance)")
        axs[row, 2].set_title(f"{label} — Partial r = {partial_r:.3f}")

    plt.tight_layout()
    plt.show()

# ------------------------------------------
# 5.4.3. Visualisation
# ------------------------------------------

def plot_bar_r(naive_r: float, naive_ci_low_r2: float, naive_ci_high_r2: float, results: list) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    labels    = ["Naive"] + [r["Center Definition"] for r in results]
    r2_values = [naive_r**2] + [r["Partial r²"] for r in results]
    ci_lows   = [naive_ci_low_r2]  + [float(r["95% CI (r²)"].strip("[]").split(",")[0]) for r in results]
    ci_highs  = [naive_ci_high_r2] + [float(r["95% CI (r²)"].strip("[]").split(",")[1]) for r in results]

    colors = ["steelblue"] + ["crimson"] * 4
    bars = ax.bar(labels, r2_values, color=colors, alpha=0.8, edgecolor="k", linewidth=0.5)

    for i, (bar, val) in enumerate(zip(bars, r2_values)):
        ci_l, ci_h = ci_lows[i], ci_highs[i]

        # error bars
        ax.errorbar(
            bar.get_x() + bar.get_width() / 2, val,
            yerr=[[val - ci_l], [ci_h - val]],
            fmt="none", color="black", capsize=4, linewidth=1.5
        )
        # r² value on top of bar
        ax.text(
            bar.get_x() + bar.get_width() / 2 - 0.2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )
        # lower CI at bottom of error bar
        if colors[i] == "steelblue":
            ci_col = "black"
        else:
            ci_col = "white"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ci_l - 0.02,
            f"{ci_l:.2f}",
            ha="center", va="top", fontsize=7, color=ci_col
        )
        # upper CI at top of error bar
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ci_h + 0.02,
            f"{ci_h:.2f}",
            ha="center", va="bottom", fontsize=7, color="black"
        )

    ax.set_ylim(0, 1)
    ax.axhline(naive_r**2, color="steelblue", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_ylabel("Pearson r²")
    ax.set_title("Naive vs. Partial Correlation\n(Airbnb Density ~ Rent, controlling for distance to center)")
    ax.tick_params(axis="x")
    ax.set_xlabel("Different definitions for city center")
    plt.tight_layout()
    plt.show()