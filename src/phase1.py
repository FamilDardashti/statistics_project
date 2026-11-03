from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "alzheimer"
OUT = ROOT / "reports" / "phase1"
CLEAN = OUT / "cleaned_data"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"

UDS_FILE = DATA / "udslong_imputed.csv"
MRI_FILE = DATA / "mriscanclean.csv"
ADJ_FILE = DATA / "combined_adjacency_matrix.csv"


# Selected variables
UDS_COLUMNS = [
    "NACCID", "NACCAGE", "SEX", "EDUC", "NACCUDSD", "CDRSUM",
    "CDRGLOB", "MOCATOTS", "MEMORY", "ORIENT", "NACCGDS", "NACCBMI",
]

CONTROL_COLUMNS = [
    "NACCID", "DIABETES", "HYPERTEN", "HYPERCHO", "DEP2YRS",
    "ANXIETY", "APNEA", "EDUC", "INDEPEND",
]

CONTROL_CATEGORICAL = [
    "DIABETES", "HYPERTEN", "HYPERCHO", "DEP2YRS", "ANXIETY", "APNEA",
]

MRI_REGIONS = [
    "ENTORHINAL", "PARAHIPPOCAMPAL", "FUSIFORM", "MIDDLETEMPORAL",
    "INFERIORTEMPORAL", "PRECUNEUS", "POSTERIORCINGULATE",
    "INFERIORPARIETAL",
]

BMI_MIN, BMI_MAX = 14, 50
CONTROL_MIN, CONTROL_MAX = 0, 2
FLOOR_CONTROL_VALUES = True

VISIT_COLUMNS = [
    "NACCVNUM", "VISITNUM", "VISIT_ID", "VISITID", "VISIT_DATE",
    "EXAMDATE", "NACCYOD", "NACCAVST", "VISIT",
]


# Helpers
def make_dirs():
    for folder in (CLEAN, TABLES, FIGURES):
        folder.mkdir(parents=True, exist_ok=True)


def check_columns(frame, columns, name):
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")


def log_cleaning(log, variable, issue, count, action):
    log.append({
        "variable": variable,
        "issue": issue,
        "count": int(count),
        "action": action,
    })


def numeric_summary(frame, columns):
    rows = []
    for col in columns:
        x = pd.to_numeric(frame[col], errors="coerce").dropna()
        if x.empty:
            continue
        q1, median, q3 = x.quantile([0.25, 0.50, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append({
            "variable": col,
            "n": len(x),
            "mean": x.mean(),
            "std": x.std(),
            "median": median,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "min": x.min(),
            "max": x.max(),
            "iqr_outliers_n": int(((x < low) | (x > high)).sum()),
            "iqr_lower_fence": low,
            "iqr_upper_fence": high,
        })
    return pd.DataFrame(rows)


def frequency_table(frame, columns):
    rows = []
    for col in columns:
        counts = frame[col].value_counts(dropna=True, sort=False)
        n = int(frame[col].notna().sum())
        for value, count in counts.items():
            rows.append({
                "variable": col,
                "category_or_code": value,
                "count": int(count),
                "percent_nonmissing": 100 * count / n if n else np.nan,
                "missing_n": int(frame[col].isna().sum()),
            })
    return pd.DataFrame(rows)


def missingness_table(named_frames):
    rows = []
    for name, raw, clean in named_frames:
        for col in clean.columns:
            if col == "NACCID":
                continue
            if col in raw.columns:
                raw_missing = int(raw[col].isna().sum())
            elif name == "MRI regions":
                left = pd.to_numeric(raw[f"LH_{col}_GVOL"], errors="coerce")
                right = pd.to_numeric(raw[f"RH_{col}_GVOL"], errors="coerce")
                raw_missing = int((left.isna() | right.isna()).sum())
            else:
                raw_missing = np.nan
            clean_missing = int(clean[col].isna().sum())
            n = len(clean)
            rows.append({
                "matrix": name,
                "variable": col,
                "raw_missing_n": raw_missing,
                "cleaned_missing_n": clean_missing,
                "cleaned_missing_percent": 100 * clean_missing / n if n else np.nan,
            })
    return pd.DataFrame(rows)


def build_variable_map():
    entries = [
        ("NACCID", "UDS/MRI", "Identifier", "Nominal", "Join key", "Participant ID"),
        ("NACCAGE", "UDS", "Demographic", "Quantitative continuous", "Covariate", "Years"),
        ("SEX", "UDS", "Demographic", "Categorical nominal", "Covariate/group", "1/2 codes in supplied file"),
        ("EDUC", "UDS", "Demographic", "Quantitative discrete", "Covariate", "Years of education"),
        ("NACCUDSD", "UDS", "Diagnostic", "Categorical", "Grouping variable", "Codes retained; labels need dictionary"),
        ("CDRSUM", "UDS", "Cognitive", "Score", "Main outcome candidate", "CDR sum score"),
        ("CDRGLOB", "UDS", "Cognitive", "Ordinal score", "Cognitive descriptor", "Global CDR score"),
        ("MOCATOTS", "UDS", "Cognitive", "Score", "Secondary outcome", "0-30; 88 and out-of-range set missing"),
        ("MEMORY", "UDS", "Cognitive", "Ordinal score", "Cognitive descriptor", "CDR memory domain"),
        ("ORIENT", "UDS", "Cognitive", "Ordinal score", "Cognitive descriptor", "CDR orientation domain"),
        ("NACCGDS", "UDS", "Psychological", "Score", "Complementary variable", "0-15; 88 and out-of-range set missing"),
        ("NACCBMI", "UDS", "Physical/medical", "Quantitative continuous", "Complementary variable", "kg/m^2; project range 14-50"),
        ("DIABETES", "UDS", "Medical history", "Categorical code", "Control variable", "Provisional 0-2; verify dictionary"),
        ("HYPERTEN", "UDS", "Medical history", "Categorical code", "Control variable", "Provisional 0-2; verify dictionary"),
        ("HYPERCHO", "UDS", "Medical history", "Categorical code", "Control variable", "Provisional 0-2; verify dictionary"),
        ("DEP2YRS", "UDS", "Medical history", "Categorical code", "Control variable", "Provisional 0-2; verify dictionary"),
        ("ANXIETY", "UDS", "Medical history", "Categorical code", "Control variable", "Provisional 0-2; verify dictionary"),
        ("APNEA", "UDS", "Medical history", "Categorical code", "Control variable", "Provisional 0-2; verify dictionary"),
        ("INDEPEND", "UDS", "Functional", "Ordinal/code-based", "Control variable", "Observed range 1-4; 9 set missing"),
    ]
    for region in MRI_REGIONS:
        entries.append((
            region, "MRI", "Structural imaging", "Quantitative continuous",
            "MRI feature", "Mean of left and right hemisphere GVOL; confirm unit",
        ))
    return pd.DataFrame(entries, columns=[
        "variable", "source", "domain", "measurement_type", "role", "unit_or_coding",
    ])


def clean_uds(uds_raw, cleaning_log):
    work = uds_raw.copy()
    numeric_cols = list(dict.fromkeys(UDS_COLUMNS[1:] + CONTROL_COLUMNS[1:]))
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    bad = work["SEX"].notna() & ~work["SEX"].isin([1, 2])
    log_cleaning(cleaning_log, "SEX", "Code outside 1/2", bad.sum(), "Set to NaN")
    work.loc[bad, "SEX"] = np.nan

    rules = {
        "EDUC": (None, None, [99]),
        "MOCATOTS": (0, 30, [88]),
        "NACCGDS": (0, 15, [88]),
        "NACCBMI": (BMI_MIN, BMI_MAX, [888.8]),
    }
    for col, (low, high, special) in rules.items():
        mask = work[col].isin(special)
        if low is not None:
            mask |= work[col].notna() & (work[col] < low)
        if high is not None:
            mask |= work[col].notna() & (work[col] > high)
        count = int(mask.sum())
        work.loc[mask, col] = np.nan
        log_cleaning(cleaning_log, col, f"Special code {special} or outside valid range", count, "Set to NaN")

    uds_matrix = work[UDS_COLUMNS].copy()

    controls = work[CONTROL_COLUMNS].copy()
    for col in CONTROL_CATEGORICAL:
        x = controls[col]
        invalid = x.notna() & (
            (x < CONTROL_MIN) | (x > CONTROL_MAX) | x.eq(9)
        )
        log_cleaning(
            cleaning_log, col, "Outside provisional 0-2 range or code 9",
            invalid.sum(), "Set to NaN",
        )
        controls.loc[invalid, col] = np.nan

        fractional = controls[col].notna() & (controls[col] % 1 != 0)
        n_fractional = int(fractional.sum())
        if FLOOR_CONTROL_VALUES:
            controls.loc[controls[col].notna(), col] = np.floor(
                controls.loc[controls[col].notna(), col]
            )
            log_cleaning(
                cleaning_log, col, "Fractional values", n_fractional,
                "Floor to integer; verify imputation method",
            )

    x = controls["INDEPEND"]
    invalid = x.notna() & ((x < 1) | (x > 4) | x.eq(9))
    log_cleaning(cleaning_log, "INDEPEND", "Code 9 or outside observed range 1-4", invalid.sum(), "Set to NaN")
    controls.loc[invalid, "INDEPEND"] = np.nan

    return uds_matrix, controls


def build_mri_matrix(mri_raw, cleaning_log):
    result = pd.DataFrame({"NACCID": mri_raw["NACCID"].astype(str)})
    for region in MRI_REGIONS:
        left = pd.to_numeric(mri_raw[f"LH_{region}_GVOL"], errors="coerce")
        right = pd.to_numeric(mri_raw[f"RH_{region}_GVOL"], errors="coerce")
        invalid_left = left.notna() & (left <= 0)
        invalid_right = right.notna() & (right <= 0)
        log_cleaning(
            cleaning_log, region, "Zero or negative hemispheric GVOL",
            int(invalid_left.sum() + invalid_right.sum()),
            "Set invalid hemisphere to NaN; require both sides",
        )
        left = left.mask(left <= 0)
        right = right.mask(right <= 0)
        result[region] = (left + right) / 2
    return result


def validate_adjacency(adjacency_raw):
    a = adjacency_raw.copy()
    a.index = a.index.astype(str)
    a.columns = a.columns.astype(str)
    if a.shape != (62, 62):
        raise ValueError(f"Adjacency matrix must be 62x62, got {a.shape}")
    if not a.index.is_unique or not a.columns.is_unique:
        raise ValueError("Adjacency labels are not unique")
    if list(a.index) != list(a.columns):
        raise ValueError("Adjacency row and column labels/order do not match")
    a = a.apply(pd.to_numeric, errors="coerce")
    values = a.to_numpy()
    if np.isnan(values).any() or not np.isin(values, [0, 1]).all():
        raise ValueError("Adjacency matrix must contain only 0 and 1")
    if not np.array_equal(values, values.T):
        raise ValueError("Adjacency matrix is not symmetric")
    return a.astype(int)


def make_merge_audit(uds_raw, mri_raw, merged, diagnosis_check):
    uds_ids = pd.Index(uds_raw["NACCID"].astype(str).unique())
    mri_ids = pd.Index(mri_raw["NACCID"].astype(str).unique())
    shared = uds_ids.intersection(mri_ids)
    return pd.DataFrame([
        ("UDS rows", len(uds_raw)),
        ("UDS unique participants", uds_raw["NACCID"].nunique()),
        ("UDS duplicated ID rows", int(uds_raw["NACCID"].duplicated().sum())),
        ("MRI rows", len(mri_raw)),
        ("MRI unique participants", mri_raw["NACCID"].nunique()),
        ("MRI duplicated ID rows", int(mri_raw["NACCID"].duplicated().sum())),
        ("Participants in both files", len(shared)),
        ("UDS participants without MRI record", len(uds_ids.difference(mri_ids))),
        ("MRI participants without UDS record", len(mri_ids.difference(uds_ids))),
        ("Final UDS+MRI participants", len(merged)),
        ("NACCUDSD disagreements between files", int(diagnosis_check["mismatch"].sum())),
        ("Participants with CDRSUM and Entorhinal available", int(merged[["CDRSUM", "ENTORHINAL"]].notna().all(axis=1).sum())),
    ], columns=["item", "count"])


def visit_tables(uds_raw, mri_raw):
    summary_rows = []
    distribution_rows = []
    visit_distribution_rows = []

    for name, frame in (("UDS", uds_raw), ("MRI", mri_raw)):
        record_counts = frame.groupby("NACCID").size()
        visit_cols = [c for c in VISIT_COLUMNS if c in frame.columns]

        if visit_cols:
            visit_key = visit_cols[0]
            visit_counts = frame.groupby("NACCID")[visit_key].nunique(dropna=True)
            visit_counts = visit_counts[visit_counts > 0]
            for n_visits, n_people in visit_counts.value_counts().sort_index().items():
                visit_distribution_rows.append({
                    "file": name,
                    "visits_per_participant": int(n_visits),
                    "number_of_participants": int(n_people),
                })
        else:
            visit_counts = pd.Series(dtype="float64")

        summary_rows.append({
            "file": name,
            "rows": len(frame),
            "unique_participants": frame["NACCID"].nunique(),
            "participants_with_multiple_rows": int((record_counts > 1).sum()),
            "min_rows_per_participant": int(record_counts.min()) if len(record_counts) else 0,
            "max_rows_per_participant": int(record_counts.max()) if len(record_counts) else 0,
            "visit_key_used": visit_cols[0] if visit_cols else "Not available",
            "visit_counts_available": bool(visit_cols),
            "min_visits_per_participant": int(visit_counts.min()) if len(visit_counts) else np.nan,
            "max_visits_per_participant": int(visit_counts.max()) if len(visit_counts) else np.nan,
        })

        for n_records, n_people in record_counts.value_counts().sort_index().items():
            distribution_rows.append({
                "file": name,
                "records_per_participant": int(n_records),
                "number_of_participants": int(n_people),
            })

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(distribution_rows, columns=[
            "file", "records_per_participant", "number_of_participants"
        ]),
        pd.DataFrame(visit_distribution_rows, columns=[
            "file", "visits_per_participant", "number_of_participants"
        ]),
    )


def make_mri_availability_comparison(uds_matrix, mri_raw):
    mri_ids = set(mri_raw["NACCID"].astype(str))
    data = uds_matrix.copy()
    data["MRI_RECORD_AVAILABLE"] = data["NACCID"].astype(str).isin(mri_ids)

    numeric_rows = []
    for group_value, group_label in ((True, "MRI available"), (False, "MRI unavailable")):
        group = data.loc[data["MRI_RECORD_AVAILABLE"] == group_value]
        for col in ["NACCAGE", "EDUC", "NACCBMI", "CDRSUM", "MOCATOTS"]:
            x = group[col].dropna()
            numeric_rows.append({
                "mri_group": group_label,
                "group_n": len(group),
                "variable": col,
                "valid_n": len(x),
                "mean": x.mean() if len(x) else np.nan,
                "median": x.median() if len(x) else np.nan,
                "std": x.std() if len(x) else np.nan,
                "q1": x.quantile(0.25) if len(x) else np.nan,
                "q3": x.quantile(0.75) if len(x) else np.nan,
            })

    cat_rows = []
    for group_value, group_label in ((True, "MRI available"), (False, "MRI unavailable")):
        group = data.loc[data["MRI_RECORD_AVAILABLE"] == group_value]
        for col in ["SEX", "NACCUDSD"]:
            counts = group[col].value_counts(dropna=False, sort=False)
            for category, count in counts.items():
                cat_rows.append({
                    "mri_group": group_label,
                    "group_n": len(group),
                    "variable": col,
                    "category_or_code": category,
                    "count": int(count),
                    "percent_of_group": 100 * count / len(group) if len(group) else np.nan,
                })
            if len(group) == 0:
                cat_rows.append({
                    "mri_group": group_label, "group_n": 0, "variable": col,
                    "category_or_code": np.nan, "count": 0, "percent_of_group": np.nan,
                })

    return pd.DataFrame(numeric_rows), pd.DataFrame(cat_rows)


def create_figures(analysis, adjacency):
    paths = []

    fig, ax = plt.subplots(figsize=(8, 5))
    for sex, label in ((1, "Male (1)"), (2, "Female (2)")):
        x = analysis.loc[analysis["SEX"] == sex, "NACCAGE"].dropna()
        ax.hist(x, bins=18, alpha=0.45, label=label, edgecolor="black")
    ax.set(title="Age distribution by sex", xlabel="Age (years)", ylabel="Participants")
    ax.legend()
    fig.tight_layout()
    paths.append(save_plot(fig, "01_age_by_sex.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    for sex, label in ((1, "Male (1)"), (2, "Female (2)")):
        x = analysis.loc[analysis["SEX"] == sex, "EDUC"].dropna()
        ax.hist(x, bins=12, alpha=0.45, label=label, edgecolor="black")
    ax.set(title="Education by sex", xlabel="Years of education", ylabel="Participants")
    ax.legend()
    fig.tight_layout()
    paths.append(save_plot(fig, "02_education_by_sex.png"))

    fig, ax = plt.subplots(figsize=(7, 5))
    groups = [analysis.loc[analysis["SEX"] == sex, "CDRSUM"].dropna() for sex in (1, 2)]
    ax.boxplot(groups, showfliers=True)
    ax.set_xticks([1, 2], ["Male (1)", "Female (2)"])
    ax.set(title="CDRSUM by sex", xlabel="Sex", ylabel="CDRSUM score")
    fig.tight_layout()
    paths.append(save_plot(fig, "03_cdrsum_by_sex.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    pd.crosstab(analysis["NACCUDSD"], analysis["SEX"]).plot(kind="bar", ax=ax)
    ax.set(title="NACCUDSD codes by sex", xlabel="NACCUDSD code", ylabel="Participants")
    ax.legend(title="SEX code")
    fig.tight_layout()
    paths.append(save_plot(fig, "04_naccudsd_by_sex.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    for sex, label in ((1, "Male (1)"), (2, "Female (2)")):
        x = analysis.loc[analysis["SEX"] == sex, "NACCBMI"].dropna()
        ax.hist(x, bins=16, alpha=0.45, label=label, edgecolor="black")
    ax.set(title="BMI by sex", xlabel="BMI (kg/m²)", ylabel="Participants")
    ax.legend()
    fig.tight_layout()
    paths.append(save_plot(fig, "05_bmi_by_sex.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(analysis["ENTORHINAL"].dropna(), bins=20, edgecolor="black")
    ax.set(
        title="Entorhinal mean hemispheric gray-matter volume",
        xlabel="Mean LH/RH GVOL (source unit to confirm)", ylabel="Participants",
    )
    fig.tight_layout()
    paths.append(save_plot(fig, "06_entorhinal_volume.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    pair = analysis[["ENTORHINAL", "CDRSUM"]].dropna()
    ax.scatter(pair["ENTORHINAL"], pair["CDRSUM"], alpha=0.45)
    ax.set(
        title="Entorhinal volume and CDRSUM",
        xlabel="Mean LH/RH entorhinal GVOL (source unit to confirm)",
        ylabel="CDRSUM score",
    )
    fig.tight_layout()
    paths.append(save_plot(fig, "07_entorhinal_vs_cdrsum.png"))

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(adjacency.to_numpy(), interpolation="nearest", aspect="auto")
    ax.set(title="Provided 62 × 62 adjacency matrix", xlabel="Region index", ylabel="Region index")
    fig.colorbar(image, ax=ax, label="Adjacency value")
    fig.tight_layout()
    paths.append(save_plot(fig, "08_adjacency_matrix.png"))

    return paths


def save_plot(fig, filename):
    path = FIGURES / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    make_dirs()
    cleaning_log = []

    for path in (UDS_FILE, MRI_FILE, ADJ_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    uds_raw = pd.read_csv(UDS_FILE, dtype={"NACCID": str})
    mri_raw = pd.read_csv(MRI_FILE, dtype={"NACCID": str})
    adjacency_raw = pd.read_csv(ADJ_FILE, index_col=0)

    check_columns(uds_raw, list(dict.fromkeys(UDS_COLUMNS + CONTROL_COLUMNS[1:])), "UDS")
    mri_required = ["NACCID"] + [
        f"{side}_{region}_GVOL"
        for region in MRI_REGIONS for side in ("LH", "RH")
    ]
    check_columns(mri_raw, mri_required, "MRI")

    for name, frame in (("UDS", uds_raw), ("MRI", mri_raw)):
        if frame["NACCID"].isna().any():
            raise ValueError(f"{name} contains missing NACCID values")
        if frame["NACCID"].duplicated().any():
            raise ValueError(f"{name} contains repeated NACCID values; visit selection is needed")

    uds_matrix, control_matrix = clean_uds(uds_raw, cleaning_log)
    mri_matrix = build_mri_matrix(mri_raw, cleaning_log)
    adjacency = validate_adjacency(adjacency_raw)

    analysis = uds_matrix.merge(
        mri_matrix, on="NACCID", how="inner", validate="one_to_one"
    )

    diagnosis_check = uds_raw[["NACCID", "NACCUDSD"]].merge(
        mri_raw[["NACCID", "NACCUDSD"]], on="NACCID", how="inner",
        validate="one_to_one", suffixes=("_UDS", "_MRI"),
    )
    uds_diag = diagnosis_check["NACCUDSD_UDS"]
    mri_diag = diagnosis_check["NACCUDSD_MRI"]
    diagnosis_check["mismatch"] = (
        uds_diag.isna() != mri_diag.isna()
    ) | (uds_diag.notna() & mri_diag.notna() & uds_diag.ne(mri_diag))

    audit = make_merge_audit(uds_raw, mri_raw, analysis, diagnosis_check)
    visit_summary, record_distribution, visit_distribution = visit_tables(uds_raw, mri_raw)

    uds_matrix.to_csv(CLEAN / "01_uds_matrix.csv", index=False)
    control_matrix.to_csv(CLEAN / "02_control_matrix.csv", index=False)
    mri_matrix.to_csv(CLEAN / "03_mri_regions_matrix.csv", index=False)
    adjacency.to_csv(CLEAN / "04_adjacency_matrix.csv", index=True)
    analysis.to_csv(CLEAN / "05_uds_mri_analysis_dataset.csv", index=False)

    numeric_cols = [
        "NACCAGE", "EDUC", "CDRSUM", "MOCATOTS", "NACCGDS", "NACCBMI",
    ] + MRI_REGIONS
    numeric_stats = numeric_summary(analysis, numeric_cols)

    cat_source = uds_matrix.merge(
        control_matrix.drop(columns=["EDUC"]), on="NACCID", how="left",
        validate="one_to_one",
    )
    categorical_cols = [
        "SEX", "NACCUDSD", "CDRGLOB", "MEMORY", "ORIENT", *CONTROL_CATEGORICAL,
        "INDEPEND",
    ]
    frequencies = frequency_table(cat_source, categorical_cols)

    missingness = missingness_table([
        ("UDS", uds_raw, uds_matrix),
        ("Controls", uds_raw, control_matrix),
        ("MRI regions", mri_raw, mri_matrix),
    ])

    numeric_comparison, categorical_comparison = make_mri_availability_comparison(
        uds_matrix, mri_raw
    )

    pair = analysis[["CDRSUM", "ENTORHINAL"]].dropna()
    joint_stats = numeric_summary(pair, ["CDRSUM", "ENTORHINAL"])
    joint_stats["complete_pair_n"] = len(pair)

    sex_rows = []
    for sex in (1, 2):
        group = analysis.loc[analysis["SEX"] == sex]
        for col in ("NACCAGE", "EDUC", "NACCBMI", "CDRSUM"):
            x = group[col].dropna()
            sex_rows.append({
                "SEX_code": sex, "variable": col, "n": len(x),
                "mean": x.mean() if len(x) else np.nan,
                "median": x.median() if len(x) else np.nan,
                "std": x.std() if len(x) else np.nan,
                "q1": x.quantile(0.25) if len(x) else np.nan,
                "q3": x.quantile(0.75) if len(x) else np.nan,
            })
    sex_comparison = pd.DataFrame(sex_rows)
    diagnosis_by_sex = pd.crosstab(analysis["NACCUDSD"], analysis["SEX"], dropna=False)

    variable_map = build_variable_map()
    cleaning_log_df = pd.DataFrame(cleaning_log)

    audit.to_csv(TABLES / "merge_audit.csv", index=False)
    visit_summary.to_csv(TABLES / "visit_structure.csv", index=False)
    record_distribution.to_csv(TABLES / "records_per_participant.csv", index=False)
    visit_distribution.to_csv(TABLES / "visits_per_participant.csv", index=False)
    variable_map.to_csv(TABLES / "variable_map.csv", index=False)
    missingness.to_csv(TABLES / "missingness.csv", index=False)
    numeric_stats.to_csv(TABLES / "numeric_summary.csv", index=False)
    frequencies.to_csv(TABLES / "categorical_frequencies.csv", index=False)
    numeric_comparison.to_csv(TABLES / "mri_availability_numeric_comparison.csv", index=False)
    categorical_comparison.to_csv(TABLES / "mri_availability_categorical_comparison.csv", index=False)
    joint_stats.to_csv(TABLES / "joint_uds_mri_summary.csv", index=False)
    sex_comparison.to_csv(TABLES / "descriptive_comparison_by_sex.csv", index=False)
    diagnosis_by_sex.to_csv(TABLES / "naccudsd_by_sex.csv", index=True)
    cleaning_log_df.to_csv(TABLES / "cleaning_log.csv", index=False)

    figure_paths = create_figures(analysis, adjacency)

    n_mri = len(mri_raw)
    n_uds = len(uds_raw)
    n_common = len(set(uds_raw["NACCID"]) & set(mri_raw["NACCID"]))
    n_no_mri = len(set(uds_raw["NACCID"]) - set(mri_raw["NACCID"]))
    n_complete_pair = len(pair)
    visit_keys = visit_summary["visit_key_used"].drop_duplicates().tolist()
    if all(key == "Not available" for key in visit_keys):
        visit_note = "No visit identifier/date field was supplied; visit counts cannot be recovered."
    else:
        visit_note = "Visit key(s): " + ", ".join(str(key) for key in visit_keys)

    print("\nPhase 1 outputs created")
    print(f"UDS rows: {n_uds} | MRI rows: {n_mri} | matched participants: {n_common}")
    print(f"UDS participants without MRI record: {n_no_mri}")
    print(f"Complete CDRSUM + Entorhinal pairs: {n_complete_pair}")
    print(f"Visit check: {visit_note}")
    print(f"Cleaned matrices: {CLEAN}")
    print(f"Tables: {TABLES}")
    print(f"Figures: {len(figure_paths)} files in {FIGURES}")

    print("\nDone")
    print("- UDS, control, MRI-region and adjacency matrices created.")
    print("- UDS/MRI matching and NACCUDSD consistency checked.")
    print("- Missingness, descriptive summaries and category frequencies calculated.")
    print("- MRI availability comparison and UDS-MRI joint summary created.")
    print("- Participant record counts and visit metadata availability checked.")
    print("- Descriptive plots saved.")

    print("\nStill to do")
    print("- Confirm UDS/MRI visit-time alignment and the visit-selection rule with the TA.")
    print("- Confirm control-variable codes and fractional values against the data dictionary and imputation method.")
    print("- Confirm MRI volume units and the meaning of the adjacency matrix.")
    print("- Interpret the tables and plots, then finalize the statistical question and project card.")
    print("- Write the Phase 1 report and update the repository README manually.")


if __name__ == "__main__":
    main()
