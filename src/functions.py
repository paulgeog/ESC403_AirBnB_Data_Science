import pandas as pd

# -------------------------------------------------
# 4.1.1. Airbnb/ Price
# -------------------------------------------------
def min_max_mean_med_var(col: pd.Series, name: str) -> str:
    minimum = col.min()
    maximum = col.max()
    mean = col.mean()
    median = col.median()
    var = col.var()

    print(f"{name}:\nMin: {minimum}\nMax: {maximum}\nMean: {mean:.2f}\nMedian: {median}\nVariance: {var:.2f}")