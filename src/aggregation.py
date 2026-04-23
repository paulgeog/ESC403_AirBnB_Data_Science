import pandas as pd

def quartiere_standardized_housing(df: pd.DataFrame) -> pd.DataFrame:
    df_q = df.pivot_table(index=["quarlang", "quarsort", "kreissort"],
                          columns=["eigentuemersszpubl1lang", "anzzimmerlevel2lang_nodm"],
                          values="anzwhgstat",
                          aggfunc="sum"
                          ).reset_index()
    
    df_q = df_q.fillna(0)

    # add total housing stock by room number
    room_types = [
        "1-Zimmer",
        "2-Zimmer",
        "3-Zimmer",
        "4-Zimmer",
        "5-Zimmer",
        "6-Zimmer und mehr"
        ]
    for room in room_types:
        cols = [c for c in df_q.columns if room in c]
        df_q[f"total_{room}"] = df_q[cols].sum(axis=1)

    # add total housing stock by quartier
    df_q["total_units"] = df_q[[f"total_{r}" for r in room_types]].sum(axis=1)
    df_q = df_q.reset_index()
    df_q.index.name = None

    return df_q


def quartiere_standardized_rental(df: pd.DataFrame) -> pd.DataFrame:
    df_q = df.pivot_table(
        index=["gliederunglang", "gliederungsort"],
        columns="is_gemeinnuetzig",
        values="qu50",
        aggfunc="mean"
    )
    df_q = df_q.rename(columns={
        0: "rent_non_gemeinnuetzig",
        1: "rent_gemeinnuetzig"
    })
    df_q["rent_mean_both"] = df_q[
        ["rent_non_gemeinnuetzig", "rent_gemeinnuetzig"]
    ].mean(axis=1)
    df_q = df_q.reset_index()
    df_q.index.name = None
    return df_q