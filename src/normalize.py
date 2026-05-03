import ast
import re
from typing import Any, Iterable, List, Optional

import pandas as pd


# =========================================================
# Helpers
# =========================================================
def clean_column_names(data):
    data = data.copy()
    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace(r"[^\w]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return data


def pct_to_float(x: Any) -> Optional[float]:
    if pd.isna(x):
        return pd.NA
    if isinstance(x, (int, float)):
        return float(x)
    x = str(x).strip()
    if x == "":
        return pd.NA
    x = x.replace("%", "").strip()
    try:
        return float(x) / 100.0
    except Exception:
        return pd.NA


def price_to_float(x: Any) -> Optional[float]:
    if pd.isna(x):
        return pd.NA
    if isinstance(x, (int, float)):
        return float(x)

    x = str(x).strip()
    if x == "":
        return pd.NA

    # Remove currency symbols and separators like $1,234.00
    x = re.sub(r"[^\d.,-]", "", x)

    # Handle common formatting
    if x.count(",") > 0 and x.count(".") > 0:
        x = x.replace(",", "")
    elif x.count(",") > 0 and x.count(".") == 0:
        x = x.replace(",", ".")

    try:
        return float(x)
    except Exception:
        return pd.NA


def tf_to_bool(x: Any) -> Optional[bool]:
    if pd.isna(x):
        return pd.NA
    if isinstance(x, bool):
        return x

    x = str(x).strip().lower()
    if x in {"t", "true", "yes", "y", "1"}:
        return True
    if x in {"f", "false", "no", "n", "0"}:
        return False
    return pd.NA


def parse_list_string(x: Any) -> List[Any]:
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    if not isinstance(x, str):
        return []

    try:
        val = ast.literal_eval(x)
        return val if isinstance(val, list) else []
    except Exception:
        return []


def extract_country(x: Any) -> Optional[str]:
    if pd.isna(x):
        return pd.NA

    x = str(x).strip()
    if x == "":
        return pd.NA

    parts = [p.strip() for p in x.split(",") if p.strip()]
    if not parts:
        return pd.NA

    country = parts[-1]

    # Known messy values discussed
    if country in {"NY", "HI"}:
        return country

    return country


def contains_zurich(x: Any) -> Optional[bool]:
    if pd.isna(x):
        return pd.NA
    x = str(x).strip().lower()
    if x == "":
        return pd.NA
    return "zurich" in x or "zürich" in x


def _safe_text_len(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.len()


# =========================================================
# Amenity feature helpers
# =========================================================
def _has_wifi(items: List[str]) -> int:
    return int(any("wifi" in item for item in items))


def _has_tv(items: List[str]) -> int:
    return int(any(("tv" in item) or ("hdtv" in item) for item in items))


def _has_free_parking(items: List[str]) -> int:
    return int(any(("parking" in item) and ("free" in item) for item in items))


def _has_paid_parking(items: List[str]) -> int:
    return int(any(("parking" in item) and ("paid" in item) for item in items))


def _has_air_conditioning(items: List[str]) -> int:
    return int(any("air conditioning" in item for item in items))


def _has_laundry(items: List[str]) -> int:
    return int(any(
        (
            (("washer" in item) and ("dishwasher" not in item))
            or
            (("dryer" in item) and ("hair dryer" not in item))
        )
        for item in items
    ))


def _has_kitchen(items: List[str]) -> int:
    return int(any(item == "kitchen" for item in items))


def _has_dishwasher(items: List[str]) -> int:
    return int(any("dishwasher" in item for item in items))


def _has_elevator(items: List[str]) -> int:
    return int(any("elevator" in item for item in items))


def _has_pets_allowed(items: List[str]) -> int:
    return int(any("pets allowed" in item for item in items))


def _has_bathtub(items: List[str]) -> int:
    return int(any("bathtub" in item for item in items))


def _has_refrigerator(items: List[str]) -> int:
    return int(any(("refrigerator" in item) or ("fridge" in item) for item in items))


def _has_oven(items: List[str]) -> int:
    return int(any("oven" in item for item in items))


def _has_stove(items: List[str]) -> int:
    return int(any("stove" in item for item in items))


def _has_freezer(items: List[str]) -> int:
    return int(any("freezer" in item for item in items))


def _has_microwave(items: List[str]) -> int:
    return int(any("microwave" in item for item in items))


def _has_self_checkin(items: List[str]) -> int:
    return int(any(
        ("self check-in" in item) or ("lockbox" in item) or ("smart lock" in item) or ("keypad" in item)
        for item in items
    ))


def _has_toaster(items: List[str]) -> int:
    return int(any("toaster" in item for item in items))


def _has_outdoor_dining(items: List[str]) -> int:
    return int(any("outdoor dining area" in item for item in items))


def _has_outdoor_space(items: List[str]) -> int:
    return int(any(
        ("patio" in item) or ("balcony" in item) or ("backyard" in item)
        for item in items
    ))


def _has_blender(items: List[str]) -> int:
    return int(any("blender" in item for item in items))


def _has_lake_access(items: List[str]) -> int:
    return int(any("lake access" in item for item in items))


def _has_rice_maker(items: List[str]) -> int:
    return int(any("rice maker" in item for item in items))


def _has_board_games(items: List[str]) -> int:
    return int(any("board games" in item for item in items))


def _has_city_view(items: List[str]) -> int:
    return int(any("city skyline view" in item for item in items))


def _has_coffee_maker(items: List[str]) -> int:
    return int(any("coffee maker" in item for item in items))


def _has_workspace(items: List[str]) -> int:
    return int(any("dedicated workspace" in item for item in items))


def _has_hair_dryer(items: List[str]) -> int:
    return int(any("hair dryer" in item for item in items))


def _has_heating(items: List[str]) -> int:
    return int(any("heating" in item for item in items))


def _has_housekeeping(items: List[str]) -> int:
    return int(any("housekeeping" in item for item in items))


# =========================================================
# Main function
# =========================================================
def normalize_airbnb(airbnb_df: pd.DataFrame) -> pd.DataFrame:
    df = airbnb_df.copy()
    df =clean_column_names(df)

    # -------------------------
    # Reference date
    # -------------------------
    if "calendar_last_scraped" in df.columns and df["calendar_last_scraped"].notna().any():
        reference_date = pd.Timestamp(df["calendar_last_scraped"].dropna().iloc[0])
    else:
        reference_date = pd.Timestamp.today().normalize()

    # -------------------------
    # Text length features before dropping text
    # -------------------------
    for col in ["host_about", "description", "neighborhood_overview"]:
        if col in df.columns:
            df[f"{col}_len"] = _safe_text_len(df[col])

    # -------------------------
    # Drop noisy/raw columns
    # -------------------------
    to_drop = [
        "name",
        "host_name",
        "description",
        "neighborhood_overview",
        "host_about",
        "id",
        "listing_url",
        "scrape_id",
        "last_scraped",
        "source",
        "picture_url",
        "host_id",
        "host_url",
        "host_thumbnail_url",
        "host_picture_url",
        "calendar_last_scraped",
        "license",
        "calendar_updated",
        "host_neighbourhood",
        "neighbourhood",
    ]
    df = df.drop(columns=[c for c in to_drop if c in df.columns], errors="ignore")

    # -------------------------
    # Target cleaning
    # -------------------------
    if "price" in df.columns:
        df = df[df["price"].notna()].copy()
        df["price"] = df["price"].apply(price_to_float)

    # -------------------------
    # Dates -> elapsed days
    # -------------------------
    date_cols = ["host_since", "first_review", "last_review"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "host_since" in df.columns:
        df["host_tenure_days"] = (reference_date - df["host_since"]).dt.days

    if "first_review" in df.columns:
        df["days_since_first_review"] = (reference_date - df["first_review"]).dt.days

    if "last_review" in df.columns:
        df["days_since_last_review"] = (reference_date - df["last_review"]).dt.days

    df = df.drop(columns=[c for c in date_cols if c in df.columns], errors="ignore")

    # -------------------------
    # Percent columns
    # -------------------------
    for col in ["host_response_rate", "host_acceptance_rate"]:
        if col in df.columns:
            df[col] = df[col].apply(pct_to_float)

    # -------------------------
    # Bool-like columns
    # -------------------------
    bool_cols = [
        "host_is_superhost",
        "host_has_profile_pic",
        "host_identity_verified",
        "instant_bookable",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].apply(tf_to_bool).astype("boolean")

    # -------------------------
    # Host location features
    # -------------------------
    if "host_location" in df.columns:
        df["host_country"] = df["host_location"].apply(extract_country)
        df["host_country"] = df["host_country"].replace(["NY", "HI"], pd.NA)

        df["host_in_switzerland"] = df["host_country"].eq("Switzerland").astype("Float64")
        df["host_in_zurich"] = df["host_location"].apply(contains_zurich).astype("Float64")

        df = df.drop(columns=["host_location"], errors="ignore")

    # -------------------------
    # Host verifications
    # -------------------------
    if "host_verifications" in df.columns:
        verif_lists = df["host_verifications"].apply(parse_list_string)
        df["host_verifications_count"] = verif_lists.apply(len)
        df["host_verif_email"] = verif_lists.apply(lambda x: int("email" in x))
        df["host_verif_phone"] = verif_lists.apply(lambda x: int("phone" in x))
        df["host_verif_work_email"] = verif_lists.apply(lambda x: int("work_email" in x))
        df = df.drop(columns=["host_verifications"], errors="ignore")

    # -------------------------
    # Bathrooms
    # -------------------------
    if "bathrooms_text" in df.columns:
        s = df["bathrooms_text"].astype(str).str.lower()

        df["bathroom_shared"] = s.str.contains("shared", na=False).astype(int)
        df["bathroom_private"] = s.str.contains("private", na=False).astype(int)
        df["bathroom_half"] = s.str.contains("half-bath", na=False).astype(int)

        extracted = s.str.extract(r"(\d+(\.\d+)?)")[0].astype(float)

        if "bathrooms" in df.columns:
            df["nr_bathrooms"] = df["bathrooms"].fillna(extracted)
        else:
            df["nr_bathrooms"] = extracted

        df = df.drop(columns=["bathrooms_text"], errors="ignore")

    # -------------------------
    # Response time as ordered categorical
    # -------------------------
    if "host_response_time" in df.columns:
        df["host_response_time"] = df["host_response_time"].astype("string").str.strip()

        response_order = [
            "within an hour",
            "within a few hours",
            "within a day",
            "a few days or more",
        ]
        df["host_response_time"] = pd.Categorical(
            df["host_response_time"],
            categories=response_order,
            ordered=True,
        )

    # -------------------------
    # Cast selected text cols to category
    # -------------------------
    cat_cols = [
        "host_country",
        "neighbourhood_cleansed",
        "neighbourhood_group_cleansed",
        "room_type",
        "property_type",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # -------------------------
    # Amenities -> engineered features
    # -------------------------
    if "amenities" in df.columns:
        amenities_series = df["amenities"].copy()
        amenity_lists = amenities_series.apply(parse_list_string)
        amenity_lists_norm = amenity_lists.apply(
            lambda items: [str(x).strip().lower() for x in items]
        )

        df["amenities_count"] = amenity_lists_norm.apply(len)

        df["amenity_wifi"] = amenity_lists_norm.apply(_has_wifi)
        df["amenity_tv"] = amenity_lists_norm.apply(_has_tv)
        df["amenity_free_parking"] = amenity_lists_norm.apply(_has_free_parking)
        df["amenity_paid_parking"] = amenity_lists_norm.apply(_has_paid_parking)
        df["amenity_air_conditioning"] = amenity_lists_norm.apply(_has_air_conditioning)
        df["amenity_laundry"] = amenity_lists_norm.apply(_has_laundry)

        df["amenity_kitchen"] = amenity_lists_norm.apply(_has_kitchen)
        df["amenity_dishwasher"] = amenity_lists_norm.apply(_has_dishwasher)
        df["amenity_elevator"] = amenity_lists_norm.apply(_has_elevator)
        df["amenity_pets_allowed"] = amenity_lists_norm.apply(_has_pets_allowed)

        df["amenity_bathtub"] = amenity_lists_norm.apply(_has_bathtub)
        df["amenity_refrigerator"] = amenity_lists_norm.apply(_has_refrigerator)
        df["amenity_oven"] = amenity_lists_norm.apply(_has_oven)
        df["amenity_stove"] = amenity_lists_norm.apply(_has_stove)
        df["amenity_freezer"] = amenity_lists_norm.apply(_has_freezer)
        df["amenity_microwave"] = amenity_lists_norm.apply(_has_microwave)

        df["amenity_self_checkin"] = amenity_lists_norm.apply(_has_self_checkin)
        df["amenity_toaster"] = amenity_lists_norm.apply(_has_toaster)
        df["amenity_outdoor_dining"] = amenity_lists_norm.apply(_has_outdoor_dining)
        df["amenity_outdoor_space"] = amenity_lists_norm.apply(_has_outdoor_space)

        df["amenity_blender"] = amenity_lists_norm.apply(_has_blender)
        df["amenity_lake_access"] = amenity_lists_norm.apply(_has_lake_access)
        df["amenity_rice_maker"] = amenity_lists_norm.apply(_has_rice_maker)
        df["amenity_board_games"] = amenity_lists_norm.apply(_has_board_games)
        df["amenity_city_view"] = amenity_lists_norm.apply(_has_city_view)

        df["amenity_coffee_maker"] = amenity_lists_norm.apply(_has_coffee_maker)
        df["amenity_workspace"] = amenity_lists_norm.apply(_has_workspace)
        df["amenity_hair_dryer"] = amenity_lists_norm.apply(_has_hair_dryer)
        df["amenity_heating"] = amenity_lists_norm.apply(_has_heating)
        df["amenity_housekeeping"] = amenity_lists_norm.apply(_has_housekeeping)

        df = df.drop(columns=["amenities"], errors="ignore")

    return df


def normalize_housing(housing_df: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    df = housing_df.copy()
    df = clean_column_names(df)

    # drop columns you explicitly dropped in the notebook
    df = df.drop(columns=["datenstandcd"], errors="ignore")

    # categorical columns from the notebook
    cat_cols = [
        "anzzimmerlevel2lang_nodm",
        "eigentuemersszpubl1lang",
        "kreislang",
        "quarlang",
    ]

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    if year is not None and "stichtagdatjahr" in df.columns:
        df = df[df["stichtagdatjahr"] == year]

    return df


def normalize_rental(rental_df: pd.DataFrame,
                     year: int | None = None,
                     brutto: bool | None = None,
                     sqm: bool | None = None,
                     level: int | None = None,
                     cat_zimmer: bool = False
                     ) -> pd.DataFrame:
    df = rental_df.copy()
    df = clean_column_names(df)

    # -------------------------
    # time
    # -------------------------
    if "stichtagdatmonat" in df.columns:
        df["stichtagdatmonat"] = pd.to_datetime(
            df["stichtagdatmonat"].astype(str).str.replace(".", "-", regex=False),
            format="%Y-%m",
            errors="coerce",
        )

    if "stichtagdatjahr" in df.columns:
        df["is_april_2024"] = (df["stichtagdatjahr"] == 2024).astype("bool")

    df = df.drop(columns=["stichtagdatjahr", "stichtagdatmonat"], errors="ignore")

    # -------------------------
    # raumeinheit
    # -------------------------
    if "raumeinheitlang" in df.columns:
        df["raumeinheitlang"] = df["raumeinheitlang"].astype("category")
    # raumeinheitsort will be dropped further down in the code, once it is not needed anymore

    # -------------------------
    # binary columns
    # -------------------------
    if "gemeinnuetziglang" in df.columns:
        df["is_gemeinnuetzig"] = df["gemeinnuetziglang"].map({
            "Gemeinnützig": True,
            "Nicht gemeinnützig": False,
        }).astype("bool")

    df = df.drop(columns=["gemeinnuetzigsort", "gemeinnuetziglang"], errors="ignore")

    if "einheitlang" in df.columns:
        df["is_per_sqm"] = df["einheitlang"].map({
            "Quadratmeter": True,
            "Wohnung": False,
        }).astype("bool")

    df = df.drop(columns=["einheitsort", "einheitlang"], errors="ignore")

    if "preisartlang" in df.columns:
        df["is_brutto"] = df["preisartlang"].map({
            "brutto": True,
            "netto": False,
        }).astype("bool")

    df = df.drop(columns=["preisartsort", "preisartlang"], errors="ignore")

    # -------------------------
    # zimmer
    # -------------------------
    if cat_zimmer == True:
        if "zimmerlang" in df.columns:
            df["zimmerlang"] = (
                df["zimmerlang"]
                .astype("string")
            )

            room_order = ["2 Zimmer", "3 Zimmer", "4 Zimmer"]
            df["zimmerlang"] = pd.Categorical(
                df["zimmerlang"],
                categories=room_order,
                ordered=True,
            )
    df = df.drop(columns=["zimmersort"], errors="ignore")
    # -------------------------
    # gliederung
    # -------------------------
    if "gliederunglang" in df.columns:
        gl = df["gliederunglang"].astype("string").str.strip()

    # -------------------------
    # categories
    # -------------------------
    cat_cols = [
        "raumeinheitlang",
        "zimmerlang",
        "gliederung_city",
        "gliederung_kreis",
        "gliederung_area",
        "gliederung_contract_age",
        "gliederung_quartier",
    ]

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")


    # -------------------------
    # apply parameter masks
    # -------------------------
    if year is not None and "is_april_2024" in df.columns:
        if year == 2024:
            year_bool = True
        if year == 2022:
            year_bool = False
        if year not in [2022,2024]:
            raise ValueError("Valid years: 2022, 2024")
        df = df[df["is_april_2024"] == year_bool]

    if brutto is not None and "is_brutto" in df.columns:
        df = df[df["is_brutto"] == brutto]
    
    if sqm is not None and "is_per_sqm" in df.columns:
        df = df[df["is_per_sqm"] == sqm]

    if level is not None and "raumeinheitsort" in df.columns:
        if level not in range(1,6,1):
            raise ValueError("Valid leves: 1,2,4,5")
        df = df[df["raumeinheitsort"] == level]
        df = df[df["gliederungsort"] > 0]

    # remove raumeinheitsort, now that we don't need it anymore for filtering levels
    df = df.drop(columns=["raumeinheitsort"], errors="ignore")

    return df