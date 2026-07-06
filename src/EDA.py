import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_all = pd.read_pickle("data/processed_df.pkl")


print("=" * 60)
print("Dataset Information")
print("=" * 60)

print(df_all.info())

print("\nMissing Values:")
print(df_all.isnull().sum())

print("\nStatistical Summary:")
print(df_all.describe())

# Remove missing values if any
df_all.dropna(inplace=True)


label_names = {
    1: "Baseline",
    2: "Stress",
    3: "Amusement"
}

df_all["label_name"] = df_all["label"].map(label_names)


signal_cols = [
    "ACC_mag_mean",
    "ECG_mean",
    "EMG_mean",
    "EDA_mean",
    "Temp_mean",
    "Resp_mean"
]


fig, axes = plt.subplots(2, 3, figsize=(16,8))

axes = axes.flatten()

for i, col in enumerate(signal_cols):

    axes[i].hist(df_all[col], bins=40)

    axes[i].set_title(col)

plt.tight_layout()

plt.show()


plt.figure(figsize=(7,5))

sns.countplot(
    data=df_all,
    x="label_name",
    order=["Baseline","Stress","Amusement"]
)

plt.title("Class Distribution")

plt.xlabel("Class")

plt.ylabel("Count")

plt.tight_layout()

plt.show()


fig, axes = plt.subplots(2,3, figsize=(18,10))

axes = axes.flatten()

for i,col in enumerate(signal_cols):

    sns.boxplot(

        data=df_all,

        x="label_name",

        y=col,

        order=["Baseline","Stress","Amusement"],

        ax=axes[i],

        showfliers=False

    )

    axes[i].set_title(col)

    axes[i].tick_params(axis='x',rotation=25)

plt.tight_layout()

plt.show()

df_grouped = df_all.groupby("label_name")[signal_cols].mean()

df_grouped.T.plot(

    kind="bar",

    figsize=(14,6)

)

plt.title("Mean Feature Values")

plt.tight_layout()

plt.show()


key_signals = [
    "ECG_mean",
    "EDA_mean",
    "Temp_mean",
    "Resp_mean"
]

fig, axes = plt.subplots(1, 4, figsize=(20, 6))

for i, col in enumerate(key_signals):

    sns.violinplot(
        data=df_all,
        x="label_name",
        y=col,
        order=["Baseline", "Stress", "Amusement"],
        ax=axes[i]
    )

    axes[i].set_title(col)
    axes[i].tick_params(axis="x", rotation=25)

plt.tight_layout()

plt.show()


plt.figure(figsize=(10,8))

corr = df_all[signal_cols].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    square=True
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()

plt.show()


df_sample = (
    df_all
    .groupby("label_name", group_keys=False)
    .sample(n=500, random_state=42)
)

sns.pairplot(

    df_sample[

        [
            "ECG_mean",
            "EDA_mean",
            "Temp_mean",
            "Resp_mean",
            "label_name"
        ]

    ],

    hue="label_name",

    diag_kind="kde"

)

plt.show()



print("\nCorrelation with Label:\n")

corr_label = (
    df_all
    .drop(columns=["subject", "label_name"])
    .corr(numeric_only=True)["label"]
    .sort_values(ascending=False)
)

print(corr_label.head(20))

# ==========================================
# Dataset Summary
# ==========================================

print("\nFinal Dataset Shape:")

print(df_all.shape)

print("\nColumns:")

print(df_all.columns.tolist())

