import numpy as np

from lif_engine_realtime import RealtimeMaleCNSLIF


# ============================================================
# AYARLAR
# ============================================================

W_SYN = 0.110


# ============================================================
# BRAIN
# ============================================================

print("=" * 100)
print("MALECNS CENTRAL / ENDOGENOUS DRIVE INSPECTOR V2")
print("=" * 100)


brain = RealtimeMaleCNSLIF()

brain.w_syn = W_SYN


neurons = brain.neurons.copy()


print()
print("Nöron:", brain.N)
print("Bağlantı:", len(brain.indices))


# ============================================================
# KOLONLAR
# ============================================================

print("\n" + "=" * 100)
print("NEURON DATAFRAME KOLONLARI")
print("=" * 100)


for column in neurons.columns:

    print(column)


# ============================================================
# GEREKLİ KOLON KONTROLÜ
# ============================================================

required_columns = [
    "bodyId",
    "type",
    "superclass",
]


for column in required_columns:

    if column not in neurons.columns:

        raise RuntimeError(
            f"Gerekli kolon bulunamadı: {column}"
        )


# ============================================================
# SUPERCLASS COUNTS
# ============================================================

print("\n" + "=" * 100)
print("SUPERCLASS COUNTS")
print("=" * 100)


superclass_values = (

    neurons["superclass"]

    .fillna("")

    .astype(str)

)


superclass_counts = (

    superclass_values
    .replace("", "<EMPTY>")
    .value_counts()

)


print(
    superclass_counts.to_string()
)


# ============================================================
# TYPE COUNTS
# ============================================================

print("\n" + "=" * 100)
print("TOP TYPE COUNTS")
print("=" * 100)


type_counts = (

    neurons["type"]

    .fillna("")

    .astype(str)

    .replace("", "<EMPTY>")

    .value_counts()

)


print(
    type_counts.head(100).to_string()
)


# ============================================================
# CONNECTIVITY
#
# Current matrix yapımızda:
# CSC column = presynaptic neuron
#
# diff(indptr) = outgoing edge count
# bincount(indices) = incoming edge count
# ============================================================

out_degree = np.diff(
    brain.indptr
).astype(
    np.int32
)


in_degree = np.bincount(

    brain.indices,

    minlength=brain.N

).astype(
    np.int32
)


neurons["_out_degree"] = (
    out_degree
)

neurons["_in_degree"] = (
    in_degree
)


# ============================================================
# CONNECTIVITY GENEL İSTATİSTİK
# ============================================================

print("\n" + "=" * 100)
print("TÜM AĞ CONNECTIVITY")
print("=" * 100)


print(
    "Out-degree median:",
    float(
        np.median(
            out_degree
        )
    )
)

print(
    "Out-degree p75:",
    float(
        np.percentile(
            out_degree,
            75
        )
    )
)

print(
    "Out-degree p90:",
    float(
        np.percentile(
            out_degree,
            90
        )
    )
)

print(
    "Out-degree p95:",
    float(
        np.percentile(
            out_degree,
            95
        )
    )
)

print(
    "Out-degree max:",
    int(
        out_degree.max()
    )
)


print()


print(
    "In-degree median:",
    float(
        np.median(
            in_degree
        )
    )
)

print(
    "In-degree p75:",
    float(
        np.percentile(
            in_degree,
            75
        )
    )
)

print(
    "In-degree p90:",
    float(
        np.percentile(
            in_degree,
            90
        )
    )
)

print(
    "In-degree p95:",
    float(
        np.percentile(
            in_degree,
            95
        )
    )
)

print(
    "In-degree max:",
    int(
        in_degree.max()
    )
)


# ============================================================
# CENTRAL ADAY MASKESİ
#
# Burada:
#
# - tüm sensory superclass'ları çıkarıyoruz
# - motor çıkarıyoruz
# - descending çıkarıyoruz
#
# Henüz visual_projection vb. grupları çıkarmıyoruz.
# Önce dağılımı görmek istiyoruz.
# ============================================================

superclass_array = (
    superclass_values.to_numpy()
)


sensory_mask = np.array(

    [
        value.endswith(
            "_sensory"
        )

        for value in superclass_array
    ],

    dtype=np.bool_

)


central_mask = (
    ~sensory_mask
)


central_mask &= (
    ~brain.motor_mask
)


central_mask &= (
    ~brain.descending_mask
)


central_indices = np.flatnonzero(
    central_mask
)


print("\n" + "=" * 100)
print("CENTRAL CANDIDATES")
print("=" * 100)


print(
    "Sensory excluded:",
    int(
        sensory_mask.sum()
    )
)


print(
    "Motor excluded:",
    int(
        brain.motor_mask.sum()
    )
)


print(
    "Descending excluded:",
    int(
        brain.descending_mask.sum()
    )
)


print(
    "Central candidate:",
    len(
        central_indices
    )
)


# ============================================================
# CENTRAL CONNECTIVITY DAĞILIMI
# ============================================================

central_out = out_degree[
    central_indices
]


central_in = in_degree[
    central_indices
]


print("\nCentral out-degree:")


print(
    "median:",
    float(
        np.median(
            central_out
        )
    )
)

print(
    "p25:",
    float(
        np.percentile(
            central_out,
            25
        )
    )
)

print(
    "p50:",
    float(
        np.percentile(
            central_out,
            50
        )
    )
)

print(
    "p75:",
    float(
        np.percentile(
            central_out,
            75
        )
    )
)

print(
    "p90:",
    float(
        np.percentile(
            central_out,
            90
        )
    )
)

print(
    "p95:",
    float(
        np.percentile(
            central_out,
            95
        )
    )
)

print(
    "max:",
    int(
        central_out.max()
    )
)


print("\nCentral in-degree:")


print(
    "median:",
    float(
        np.median(
            central_in
        )
    )
)

print(
    "p25:",
    float(
        np.percentile(
            central_in,
            25
        )
    )
)

print(
    "p50:",
    float(
        np.percentile(
            central_in,
            50
        )
    )
)

print(
    "p75:",
    float(
        np.percentile(
            central_in,
            75
        )
    )
)

print(
    "p90:",
    float(
        np.percentile(
            central_in,
            90
        )
    )
)

print(
    "p95:",
    float(
        np.percentile(
            central_in,
            95
        )
    )
)

print(
    "max:",
    int(
        central_in.max()
    )
)


# ============================================================
# CENTRAL SUPERCLASS DAĞILIMI
# ============================================================

print("\n" + "=" * 100)
print("CENTRAL SUPERCLASS DISTRIBUTION")
print("=" * 100)


central_superclasses = (

    neurons.iloc[
        central_indices
    ]["superclass"]

    .fillna("")

    .astype(str)

    .replace("", "<EMPTY>")

    .value_counts()

)


print(
    central_superclasses.to_string()
)


# ============================================================
# BALANCED CENTRAL POPULATION
#
# Aşırı hub seçmiyoruz.
#
# out-degree:
# p50 - p75
#
# in-degree:
# en az p25
#
# Amaç:
# - çok zayıf değil
# - aşırı güçlü hub değil
# - ağın içinde gerçekten bağlı
# ============================================================

out_low = float(

    np.percentile(
        central_out,
        50
    )

)


out_high = float(

    np.percentile(
        central_out,
        75
    )

)


in_low = float(

    np.percentile(
        central_in,
        25
    )

)


balanced_mask = (

    central_mask

    &

    (
        out_degree
        >=
        out_low
    )

    &

    (
        out_degree
        <=
        out_high
    )

    &

    (
        in_degree
        >=
        in_low
    )

)


balanced_indices = np.flatnonzero(
    balanced_mask
)


print("\n" + "=" * 100)
print("BALANCED CENTRAL CANDIDATES")
print("=" * 100)


print(
    "Out-degree range:",
    round(
        out_low,
        2
    ),
    "-",
    round(
        out_high,
        2
    )
)


print(
    "Minimum in-degree:",
    round(
        in_low,
        2
    )
)


print(
    "Balanced candidate:",
    len(
        balanced_indices
    )
)


# ============================================================
# BALANCED SUPERCLASS
# ============================================================

print("\n" + "=" * 100)
print("BALANCED SUPERCLASS DISTRIBUTION")
print("=" * 100)


balanced_superclasses = (

    neurons.iloc[
        balanced_indices
    ]["superclass"]

    .fillna("")

    .astype(str)

    .replace("", "<EMPTY>")

    .value_counts()

)


print(
    balanced_superclasses.to_string()
)


# ============================================================
# BALANCED TRANSMITTER
# ============================================================

if "predicted_nt" in neurons.columns:

    print("\n" + "=" * 100)
    print("BALANCED PREDICTED NT")
    print("=" * 100)


    nt_counts = (

        neurons.iloc[
            balanced_indices
        ]["predicted_nt"]

        .fillna("")

        .astype(str)

        .replace("", "<EMPTY>")

        .value_counts()

    )


    print(
        nt_counts.to_string()
    )


# ============================================================
# BALANCED NT ROLE
# ============================================================

if "nt_role" in neurons.columns:

    print("\n" + "=" * 100)
    print("BALANCED NT ROLE")
    print("=" * 100)


    role_counts = (

        neurons.iloc[
            balanced_indices
        ]["nt_role"]

        .fillna("")

        .astype(str)

        .replace("", "<EMPTY>")

        .value_counts()

    )


    print(
        role_counts.to_string()
    )


# ============================================================
# BALANCED TYPE COUNTS
# ============================================================

print("\n" + "=" * 100)
print("BALANCED TOP TYPES")
print("=" * 100)


balanced_types = (

    neurons.iloc[
        balanced_indices
    ]["type"]

    .fillna("")

    .astype(str)

    .replace("", "<EMPTY>")

    .value_counts()

)


print(
    balanced_types.head(100).to_string()
)


# ============================================================
# ÖRNEK NÖRONLAR
# ============================================================

print("\n" + "=" * 100)
print("ÖRNEK 60 BALANCED CENTRAL NÖRON")
print("=" * 100)


display_columns = [

    "neuron_idx",

    "bodyId",

    "type",

    "superclass",

]


if "predicted_nt" in neurons.columns:

    display_columns.append(
        "predicted_nt"
    )


if "nt_role" in neurons.columns:

    display_columns.append(
        "nt_role"
    )


display_columns.extend(

    [
        "_out_degree",
        "_in_degree",
    ]

)


sample = neurons.iloc[
    balanced_indices[:60]
]


print(

    sample[
        display_columns
    ].to_string(
        index=False
    )

)


# ============================================================
# EN YÜKSEK CENTRAL OUT-DEGREE
#
# Sadece incelemek için.
# Bunları doğrudan drive etmeyeceğiz.
# ============================================================

print("\n" + "=" * 100)
print("TOP 30 CENTRAL OUT-DEGREE HUB")
print("=" * 100)


central_sorted = central_indices[

    np.argsort(

        out_degree[
            central_indices
        ]

    )[::-1]

]


top_hubs = neurons.iloc[
    central_sorted[:30]
]


print(

    top_hubs[
        display_columns
    ].to_string(
        index=False
    )

)


print("\n" + "=" * 100)
print("INSPECTION BİTTİ")
print("=" * 100)