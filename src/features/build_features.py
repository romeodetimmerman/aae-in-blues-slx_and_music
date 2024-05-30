import pandas as pd
from sklearn.model_selection import train_test_split

# set seed
seed = 42

# import data
df = pd.read_csv("../../data/interim/corpus_studio_pre_processed.csv")
df = df.fillna("")
df.isna().sum()
df.head()

# drop redundant column
df.drop(columns=["performance_type", "year"], inplace=True)

# set target
y = df["aae_realization"]
X = df.drop(columns="aae_realization")

# create test set
X_train_temp, X_test, y_train_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=seed
)

# create validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train_temp,
    y_train_temp,
    test_size=0.17647,
    stratify=y_train_temp,
    random_state=seed,
)

# save sets
X_train.to_csv("../../data/processed/X_train.csv")
X_val.to_csv("../../data/processed/X_val.csv")
X_test.to_csv("../../data/processed/X_test.csv")
y_train.to_csv("../../data/processed/y_train.csv")
y_val.to_csv("../../data/processed/y_val.csv")
y_test.to_csv("../../data/processed/y_test.csv")
