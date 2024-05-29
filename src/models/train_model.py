import pandas as pd
import optuna
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

# set seed
seed = 42

# import data
X_train = pd.read_csv("../../data/processed/X_train.csv", na_filter=False, index_col=0)
X_val = pd.read_csv("../../data/processed/X_val.csv", na_filter=False, index_col=0)
X_test = pd.read_csv("../../data/processed/X_test.csv", na_filter=False, index_col=0)
y_train = pd.read_csv("../../data/processed/y_train.csv", index_col=0)
y_val = pd.read_csv("../../data/processed/y_val.csv", index_col=0)
y_test = pd.read_csv("../../data/processed/y_test.csv", index_col=0)

# define cat features
cat_features = list(X_train.select_dtypes("object").columns)
X_train[cat_features] = X_train[cat_features].astype("category")
X_val[cat_features] = X_val[cat_features].astype("category")
X_test[cat_features] = X_test[cat_features].astype("category")


# catboost study
def catboost_objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }

    model = CatBoostClassifier(
        **params,
        early_stopping_rounds=100,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        cat_features=cat_features,
        verbose=0,
    )

    preds = model.predict(X_val)
    score = f1_score(y_val, preds)
    return score


# xgboost study
def xgboost_objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }

    model = XGBClassifier(
        **params,
        enable_categorical=True,
        early_stopping_rounds=100,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    preds = model.predict(X_val)
    score = f1_score(y_val, preds)
    return score


# catboost
catboost_study = optuna.create_study(direction="maximize")
catboost_study.optimize(catboost_objective, n_trials=50, timeout=600)

print()
print("catboost - best trial:")
catboost_best_trial = catboost_study.best_trial
print(f"\tbest score: {catboost_best_trial.value}")
print("\tbest params: ")
for key, value in catboost_best_trial.params.items():
    print(f"\t\t{key}: {value}")
print()

# xgboost
xgboost_study = optuna.create_study(direction="maximize")
xgboost_study.optimize(xgboost_objective, n_trials=50, timeout=600)

print()
print("xgboost - best trial:")
xgboost_best_trial = xgboost_study.best_trial
print(f"\tbest score: {xgboost_best_trial.value}")
print("\tbest params: ")
for key, value in xgboost_best_trial.params.items():
    print(f"\t\t{key}: {value}")
print()

# training final models
final_catboost_model = CatBoostClassifier(
    **catboost_study.best_params, early_stopping_rounds=100
)
final_catboost_model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    cat_features=cat_features,
    verbose=0,
)

final_xgboost_model = XGBClassifier(
    **xgboost_study.best_params, enable_categorical=True, early_stopping_rounds=100
)
final_xgboost_model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

# evaluating final models
catboost_preds = final_catboost_model.predict(X_test)
catboost_score = f1_score(y_test, catboost_preds)
print(f"catboost - f1-score of the final model is {catboost_score}")

xgboost_preds = final_xgboost_model.predict(X_test)
xgboost_score = f1_score(y_test, xgboost_preds)
print(f"xgboost - f1-score of the final model is {xgboost_score}")

# saving catboost model
final_catboost_model.save_model(fname="../models/catboost_model.json", format="json")
