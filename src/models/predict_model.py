import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report

# import data
X_test = pd.read_csv("../data/processed/X_test.csv", na_filter=False, index_col=0)
y_test = pd.read_csv("../data/processed/y_test.csv", index_col=0)

# import model
model = CatBoostClassifier()
model.load_model(fname="../models/catboost_model.json", format="json")

# get predictions
y_pred = model.predict(X_test)

# get classification report
print(classification_report(y_true=y_test, y_pred=y_pred))
