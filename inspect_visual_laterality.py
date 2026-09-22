from pathlib import Path

import pandas as pd


# ============================================================
# MALECNS VISUAL / MOTOR LATERALITY INSPECTOR
#
# Amaç:
# - LPLC2 / LC4 nöronlarının annotation bilgilerini görmek
# - LEFT / RIGHT metadata gerçekten var mı bulmak
# - descending ve vnc_motor tarafında laterality ipucu aramak
#
# Bu script hiçbir simulation davranışını değiştirmez.
# Sadece veri inceler.
# ============================================================


ROOT = Path(__file__).resolve().parent

SIM_NEURONS = (
    ROOT
    / "data"
    / "processed"
    / "simulation-neurons.parquet"
)

RAW_ANNOTATIONS = (
    ROOT
    / "data"
    / "raw"
    / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)


print("=" * 100)
print("MALECNS VISUAL / MOTOR LATERALITY INSPECTOR")
print("=" * 100)


# ============================================================
# LOAD FILES
# ============================================================

if not SIM_NEURONS.exists():

    raise FileNotFoundError(
        f"Bulunamadı:\n{SIM_NEURONS}"
    )


print()
print("Loading simulation neurons...")

sim = pd.read_parquet(
    SIM_NEURONS
)

print(
    f"simulation-neurons rows: {len(sim):,}"
)


annotations = None


if RAW_ANNOTATIONS.exists():

    print()
    print("Loading raw body annotations...")

    annotations = pd.read_feather(
        RAW_ANNOTATIONS
    )

    print(
        f"body annotations rows: {len(annotations):,}"
    )

else:

    print()
    print(
        "Raw body annotations bulunamadı."
    )

    print(
        "Sadece simulation-neurons.parquet incelenecek."
    )


# ============================================================
# COLUMN REPORT
# ============================================================

def print_columns(
    title,
    frame
):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    for i, column in enumerate(
        frame.columns
    ):

        print(
            f"[{i:02d}] "
            f"{column}"
            f" | dtype={frame[column].dtype}"
        )


print_columns(
    "SIMULATION NEURONS COLUMNS",
    sim
)


if annotations is not None:

    print_columns(
        "RAW ANNOTATION COLUMNS",
        annotations
    )


# ============================================================
# BODY ID COLUMN DETECTION
# ============================================================

def find_body_id_column(
    frame
):

    candidates = [
        "bodyId",
        "bodyid",
        "body_id",
        "body",
        "id",
    ]


    lower_map = {
        str(column).lower(): column
        for column in frame.columns
    }


    for candidate in candidates:

        if candidate.lower() in lower_map:

            return lower_map[
                candidate.lower()
            ]


    return None


sim_body_col = find_body_id_column(
    sim
)


annotation_body_col = None


if annotations is not None:

    annotation_body_col = find_body_id_column(
        annotations
    )


print()
print("=" * 100)
print("BODY ID COLUMNS")
print("=" * 100)

print(
    "simulation:",
    sim_body_col
)

print(
    "annotations:",
    annotation_body_col
)


# ============================================================
# MERGE RICH ANNOTATIONS
# ============================================================

merged = sim.copy()


if (
    annotations is not None
    and sim_body_col is not None
    and annotation_body_col is not None
):

    print()
    print(
        "Merging simulation neurons + raw annotations..."
    )


    raw = annotations.copy()


    if annotation_body_col != sim_body_col:

        raw = raw.rename(
            columns={
                annotation_body_col:
                    sim_body_col
            }
        )


    duplicate_columns = [
        column
        for column in raw.columns
        if (
            column in merged.columns
            and column != sim_body_col
        )
    ]


    if duplicate_columns:

        raw = raw.rename(
            columns={
                column:
                    f"raw_{column}"
                for column in duplicate_columns
            }
        )


    merged = merged.merge(
        raw,
        on=sim_body_col,
        how="left"
    )


print(
    f"Merged rows: {len(merged):,}"
)


# ============================================================
# STRING SEARCH HELPERS
# ============================================================

def text_columns(
    frame
):

    result = []


    for column in frame.columns:

        dtype = frame[column].dtype


        if (
            dtype == object
            or
            pd.api.types.is_string_dtype(
                dtype
            )
        ):

            result.append(
                column
            )


    return result


def search_rows(
    frame,
    terms
):

    strings = text_columns(
        frame
    )


    mask = pd.Series(
        False,
        index=frame.index
    )


    for column in strings:

        values = (
            frame[column]
            .fillna("")
            .astype(str)
        )


        column_mask = pd.Series(
            False,
            index=frame.index
        )


        for term in terms:

            column_mask |= values.str.contains(
                term,
                case=False,
                regex=False
            )


        mask |= column_mask


    return frame.loc[
        mask
    ].copy()


# ============================================================
# POTENTIAL LATERALITY COLUMNS
# ============================================================

laterality_keywords = [
    "side",
    "later",
    "hemi",
    "soma",
    "root",
    "left",
    "right",
    "instance",
]


potential_laterality_columns = []


for column in merged.columns:

    lower = str(
        column
    ).lower()


    if any(
        keyword in lower
        for keyword in laterality_keywords
    ):

        potential_laterality_columns.append(
            column
        )


print()
print("=" * 100)
print("POTENTIAL LATERALITY COLUMNS")
print("=" * 100)


if potential_laterality_columns:

    for column in potential_laterality_columns:

        print(
            "-",
            column
        )

else:

    print(
        "İsimden laterality çağrıştıran kolon bulunamadı."
    )


# ============================================================
# VISUAL TARGETS
# ============================================================

visual = search_rows(
    merged,
    [
        "LPLC2",
        "LC4",
    ]
)


print()
print("=" * 100)
print("LPLC2 / LC4 MATCHES")
print("=" * 100)

print(
    "Rows:",
    len(visual)
)


# ============================================================
# CHOOSE USEFUL COLUMNS
# ============================================================

preferred_columns = [
    sim_body_col,
    "type",
    "instance",
    "class",
    "subclass",
    "superclass",
    "group",
    "side",
    "somaSide",
    "rootSide",
    "hemisphere",
    "laterality",
    "cell_type",
    "cellType",
    "raw_type",
    "raw_instance",
    "raw_class",
    "raw_subclass",
    "raw_superclass",
    "raw_group",
    "raw_side",
    "raw_somaSide",
    "raw_rootSide",
    "raw_hemisphere",
    "raw_laterality",
]


visual_columns = []


for column in preferred_columns:

    if (
        column is not None
        and column in visual.columns
        and column not in visual_columns
    ):

        visual_columns.append(
            column
        )


for column in potential_laterality_columns:

    if column not in visual_columns:

        visual_columns.append(
            column
        )


if visual_columns:

    print()
    print(
        visual[
            visual_columns
        ].to_string(
            index=False,
            max_rows=400
        )
    )

else:

    print()
    print(
        "Gösterilecek annotation kolonu bulunamadı."
    )


# ============================================================
# UNIQUE LATERALITY VALUES
# ============================================================

print()
print("=" * 100)
print("LPLC2 / LC4 — LATERALITY UNIQUE VALUES")
print("=" * 100)


for column in potential_laterality_columns:

    if column not in visual.columns:

        continue


    values = (
        visual[column]
        .dropna()
        .astype(str)
        .unique()
    )


    if len(values) == 0:

        continue


    print()
    print(
        f"{column}:"
    )


    for value in sorted(
        values
    )[:100]:

        print(
            "   ",
            value
        )


# ============================================================
# EXPLICIT LEFT / RIGHT TEXT SEARCH
# ============================================================

left_rows = search_rows(
    visual,
    [
        "left",
        "_l",
        "-l",
    ]
)


right_rows = search_rows(
    visual,
    [
        "right",
        "_r",
        "-r",
    ]
)


print()
print("=" * 100)
print("VISUAL EXPLICIT LEFT / RIGHT SEARCH")
print("=" * 100)

print(
    "Possible LEFT matches:",
    len(left_rows)
)

print(
    "Possible RIGHT matches:",
    len(right_rows)
)


# ============================================================
# DESCENDING NEURONS
# ============================================================

descending = search_rows(
    merged,
    [
        "descending",
        "DNp",
        "DNg",
        "DNa",
    ]
)


print()
print("=" * 100)
print("DESCENDING MATCHES")
print("=" * 100)

print(
    "Rows:",
    len(descending)
)


descending_columns = []


for column in preferred_columns:

    if (
        column is not None
        and column in descending.columns
        and column not in descending_columns
    ):

        descending_columns.append(
            column
        )


for column in potential_laterality_columns:

    if column not in descending_columns:

        descending_columns.append(
            column
        )


if descending_columns:

    print()
    print(
        descending[
            descending_columns
        ].head(
            250
        ).to_string(
            index=False
        )
    )


# ============================================================
# MOTOR NEURONS
# ============================================================

motor = search_rows(
    merged,
    [
        "vnc_motor",
        "motor",
        "DLMn",
        "DVMn",
        "TTMn",
        "MNad",
    ]
)


print()
print("=" * 100)
print("MOTOR MATCHES")
print("=" * 100)

print(
    "Rows:",
    len(motor)
)


motor_columns = []


for column in preferred_columns:

    if (
        column is not None
        and column in motor.columns
        and column not in motor_columns
    ):

        motor_columns.append(
            column
        )


for column in potential_laterality_columns:

    if column not in motor_columns:

        motor_columns.append(
            column
        )


if motor_columns:

    print()
    print(
        motor[
            motor_columns
        ].head(
            250
        ).to_string(
            index=False
        )
    )


# ============================================================
# SAVE REPORTS
# ============================================================

output_dir = (
    ROOT
    / "laterality_reports"
)

output_dir.mkdir(
    exist_ok=True
)


visual_file = (
    output_dir
    / "visual_lplc2_lc4.csv"
)


descending_file = (
    output_dir
    / "descending_candidates.csv"
)


motor_file = (
    output_dir
    / "motor_candidates.csv"
)


visual.to_csv(
    visual_file,
    index=False
)


descending.to_csv(
    descending_file,
    index=False
)


motor.to_csv(
    motor_file,
    index=False
)


print()
print("=" * 100)
print("REPORTS SAVED")
print("=" * 100)

print(
    visual_file
)

print(
    descending_file
)

print(
    motor_file
)


print()
print("=" * 100)
print("INSPECTION COMPLETE")
print("=" * 100)

print(
    "Bu script sadece annotation metadata inceler."
)

print(
    "LEFT/RIGHT eşleşmesini henüz simulation'a bağlamaz."
)

print("=" * 100)