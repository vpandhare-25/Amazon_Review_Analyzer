from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from xgboost import plot_importance
from pathlib import Path
from matplotlib import pyplot
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

model_dir = Path("./model")
baseline_model_path = model_dir / "review_classifier.pkl"
model = joblib.load(baseline_model_path)
label_map = {"OR": 1, "CG": 0}

df = pd.read_csv("processed-dataset.csv")
X = pd.get_dummies(df.drop(columns=["label", "text_", "cleaned_text"]), columns=["category"]) # Prepare features (X) by removing text columns and the label column
y = df["label"].map(label_map)
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'feature': X.columns,'importance': importances}).sort_values('importance', ascending=False)

print(feature_importance_df)
plot_importance(model)
pyplot.show()




parameters = {"n_estimators": [50, 100, 200, 500],"learning_rate": [0.1, 0.3, 0.6, 1.0],"max_depth": [3, 6, 10],"reg_alpha": [0.0, 0.1, 0.5, 1.0],"reg_lambda": [0.1, 0.5, 1.0, 1.5]}

xgb_model = XGBClassifier(use_label_encoder=False,eval_metric="logloss",random_state=42)
grid_search = GridSearchCV(estimator=xgb_model,param_grid=parameters,scoring="roc_auc",cv=5,n_jobs=-1,verbose=1,return_train_score=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 42, test_size = 0.2)
grid_search.fit(X_train, y_train)


best_params = best_params = { "learning_rate": 0.3,"max_depth": 3,"n_estimators": 500,"reg_alpha": 0.5,   "reg_lambda": 1.0}
best_model = grid_search.best_estimator_
best_score = grid_search.best_score_

print("\n" + "=" * 50)
print("GRID SEARCH RESULTS")
print("=" * 50)
print(f"Best parameters: {best_params}")
print(f"Best cross-validation AUC score: {best_score:.4f}")
print("=" * 50)


best_pred = best_model.predict(X_test)
best_prob = best_model.predict_proba(X_test)[:, 1]

print("\nBest Model Performance on Test Set:")
print("Classification Report:\n", classification_report(y_test, best_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, best_pred))
print(f"Test AUC Score: {roc_auc_score(y_test, best_prob):.4f}")

baseline_pred = model.predict(X_test)
baseline_prob = model.predict_proba(X_test)[:, 1]
print("\n" + "=" * 50)
print("BASELINE vs BEST MODEL COMPARISON")
print("=" * 50)
print(f"Baseline AUC: {roc_auc_score(y_test, baseline_prob):.4f}")
print(f"Best Model AUC: {roc_auc_score(y_test, best_prob):.4f}")



thresh = 0.020           # Only keep features with importance >= 0.020
selection = SelectFromModel(model, threshold=thresh, prefit=True)
select_X_train = selection.transform(X_train)
model = XGBClassifier(**best_params,n_estimators=100, max_depth=4, use_label_encoder=False, eval_metric="logloss", random_state=42)
model.fit(select_X_train, y_train)
y_pred = model.predict(select_X_train)
y_prob = model.predict_proba(select_X_train)[:,1]

# Print detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate and print AUC score
auc_score = roc_auc_score(y_test, y_prob)
print(f"\nAUC Score: {auc_score:.4f}")

