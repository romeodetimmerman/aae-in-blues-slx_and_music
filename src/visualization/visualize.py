import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix

# set seaborn style
sns.set_style("white")
sns.set_context("talk")

# import data
df = pd.read_csv("../../data/interim/corpus_studio_pre_processed.csv")
X_test = pd.read_csv("../../data/processed/X_test.csv", na_filter=False, index_col=0)
y_test = pd.read_csv("../../data/processed/y_test.csv", index_col=0)

# create group variable
df["group"] = df["time"] + df["social_group"]

# import model
model = CatBoostClassifier()
model.load_model(fname="../../models/catboost_model.json", format="json")

# initialize shap
shap.initjs()

# calculate shap values
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# get predictions
y_pred = model.predict(X_test)


# generate confusion matrix with final model
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=["absent", "present"], columns=["absent", "present"])
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt="g")
plt.title("confusion matrix")
plt.ylabel("actual values")
plt.xlabel("predicted values")
plt.savefig("../../figures/confusion_matrix.png", dpi=600)
plt.show()


# mean shap values for each feature
shap.plots.bar(shap_values, show=False)
plt.savefig("../../figures/shap_bar.png", dpi=600, bbox_inches="tight")
plt.show()


# waterfall plot for individual predictions
shap.plots.waterfall(shap_values[15], show=False)
plt.savefig("../../figures/shap_waterfall_15.png", dpi=600, bbox_inches="tight")
plt.show()


# waterfall plot for individual predictions
shap.plots.waterfall(shap_values[40], show=False)
plt.savefig("../../figures/shap_waterfall_40.png", dpi=600, bbox_inches="tight")
plt.show()


# boxplot
feature_values = shap_values[:, "aae_feature"].values  # get shap values and data
feature_data = shap_values[:, "aae_feature"].data

feature_list = X_test["aae_feature"].unique()  # get list of features

feature_groups = []  # split shap values based on feature
for a in feature_list:
    relevant_values = feature_values[feature_data == a]
    feature_groups.append(relevant_values)

mean_shap_values = [
    np.mean(values) for values in feature_groups
]  # calculate the mean SHAP value for each feature

sorted_indices = np.argsort(
    mean_shap_values
)  # sort features based on their mean SHAP values
sorted_feature_list = feature_list[sorted_indices]
sorted_feature_groups = [feature_groups[i] for i in sorted_indices]

plt.boxplot(sorted_feature_groups, labels=sorted_feature_list, vert=False)
plt.xlabel("Shapley values", size=15)
plt.ylabel("feature", size=15)
plt.xticks(rotation=0)
plt.savefig("../../figures/shap_box_feature.png", dpi=600, bbox_inches="tight")
plt.show()


# artist means
blues_artist_mean = (
    df.groupby(["artist", "group"])["aae_realization"]
    .mean()
    .reset_index()
    .sort_values(by="aae_realization", ascending=False)
)  # calculate group means

custom_colors = sns.color_palette("muted", 9)  # define custom colors for each group

groups = [
    "1960sAA",
    "1960snonAA_US",
    "1960snonAA_nonUS",
    "1980sAA",
    "1980snonAA_US",
    "1980snonAA_nonUS",
    "2010sAA",
    "2010snonAA_US",
    "2010snonAA_nonUS",
]

g = sns.catplot(
    data=blues_artist_mean,
    x="aae_realization",
    y="artist",
    hue="group",
    hue_order=groups,
    palette=custom_colors,
    height=13.5,
    aspect=1.75,
    s=75,
    legend=False,
)  # create the catplot

group_means = blues_artist_mean.groupby("group")["aae_realization"].mean()
for group, color in zip(group_means.index, custom_colors):
    mean_value = group_means[group]
    plt.axvline(
        mean_value, color=color, linestyle="--", linewidth=3
    )  # calculate group means and add as horizontal lines

plt.legend(
    title="group",
    labels=groups,
    bbox_to_anchor=(0.9, 0.65),
    fontsize=20,
    title_fontsize=25,
    frameon=False,
)

g.set_axis_labels("mean AAE realization", "artist", fontsize=25)
g.set_xticklabels(fontsize=22.5)
g.set_yticklabels(fontsize=20)

plt.tight_layout()
plt.xlim(0.5, 1)
plt.savefig("../../figures/mean_aae_realizations.png", dpi=600)
plt.show()


# point plot by group
g = sns.catplot(
    data=df,
    y="aae_feature",
    x="aae_realization",
    col="group",
    col_wrap=3,
    col_order=[
        "1960sAA",
        "1960snonAA_US",
        "1960snonAA_nonUS",
        "1980sAA",
        "1980snonAA_US",
        "1980snonAA_nonUS",
        "2010sAA",
        "2010snonAA_US",
        "2010snonAA_nonUS",
    ],
    order=[
        "ing ultimas",
        "ai monophthongization",
        "post-vocalic r",
        "post-consonantal d",
        "post-consonantal t",
        "auxiliary verb",
        "third person singular",
        "zero copula",
    ],
    kind="point",
    errorbar="ci",
)
g.set_xlabels("mean AAE realization")
g.set_ylabels("AAE feature")
plt.savefig("../../figures/point_plots_by_group.png", dpi=600)
plt.show()


# point plot by group and song type
g = sns.catplot(
    data=df,
    y="aae_feature",
    x="aae_realization",
    col="group",
    col_wrap=3,
    hue="song_type",
    col_order=[
        "1960sAA",
        "1960snonAA_US",
        "1960snonAA_nonUS",
        "1980sAA",
        "1980snonAA_US",
        "1980snonAA_nonUS",
        "2010sAA",
        "2010snonAA_US",
        "2010snonAA_nonUS",
    ],
    order=[
        "ing ultimas",
        "ai monophthongization",
        "post-vocalic r",
        "post-consonantal d",
        "post-consonantal t",
        "auxiliary verb",
        "third person singular",
        "zero copula",
    ],
    kind="point",
    errorbar="ci",
)
g.set_xlabels("mean AAE realization")
g.set_ylabels("AAE feature")
plt.savefig("../../figures/point_plots_by_group_and_song_type.png", dpi=600)
plt.show()


# point plot by group, artist and song type
g = sns.catplot(
    data=df,
    y="artist",
    x="aae_realization",
    col="group",
    col_wrap=3,
    hue="song_type",
    col_order=[
        "1960sAA",
        "1960snonAA_US",
        "1960snonAA_nonUS",
        "1980sAA",
        "1980snonAA_US",
        "1980snonAA_nonUS",
        "2010sAA",
        "2010snonAA_US",
        "2010snonAA_nonUS",
    ],
    kind="point",
    errorbar="ci",
    sharey=False,
    aspect=1.25,
)
g.set_xlabels("mean AAE realization")
g.set_ylabels("artist")
g.set(
    xlim=(0, 1),
    xticks=[0, 0.25, 0.5, 0.75, 1],
    xticklabels=["0", "0,25", "0.5", "0.75", "1"],
)
plt.savefig("../../figures/point_plots_by_group_artist_and_song_type.png", dpi=600)
plt.show()
