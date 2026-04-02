import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from pathlib import Path
import joblib
import json

df = pd.read_csv("processed-dataset.csv")
print(f"Dataset loaded with {len(df)} reviews")
print(f"Columns in dataset: {list(df.columns)}")

#The pd.get_dummies() function converts categorical variables (like product categories) into numerical columns that our model can understand.
X = pd.get_dummies(df.drop(columns=["label", "text_", "cleaned_text"]), columns=["category"]) # Prepare features (X) by removing text columns and the label column

# Convert labels to numbers: OR (Original/real Reviews) = 1, CG (Computer-Generated) = 0
label_map = {"OR": 1, "CG": 0}
y = df["label"].map(label_map)
print(f"Features shape: {X.shape}")
print(f"Labels distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 42, test_size = 0.2)
print(f"Training set size: {len(X_train)} reviews")
print(f"Test set size: {len(X_test)} reviews")

# Create XGBoost classifier
# n_estimators=100: Creates 100 decision trees. A decision tree is a sort of flowchart of choices and their outcomes made by the model, along with their probabilities
# max_depth=4: Limits how deep each decision tree can be (prevents overfittin;/ g). The end, or leaf node, of each tree is the final decision for that branch.
# eval_metric="logloss": Uses logarithmic loss to measure performance during training
model = XGBClassifier(n_estimators=100, max_depth=4, use_label_encoder=False, eval_metric="logloss", random_state=42)
print("Training the model...")

# Train the model
model.fit(X_train, y_train)
print("Model training completed!")

# Make predictions on the test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1] # Get probabilities for the positive class

# Print detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate and print AUC score
auc_score = roc_auc_score(y_test, y_prob)
print(f"\nAUC Score: {auc_score:.4f}")

# Create directory for saving the model
model_dir = Path("./model")
model_dir.mkdir(parents=True, exist_ok=True)


joblib.dump(model, model_dir / "review_classifier.pkl") # Save the trained model
feature_names = X.columns.tolist() # Save feature names for future use
with open(model_dir / "feature_names.json", "w") as f: json.dump(feature_names, f)

model_metadata = {"test_auc_score": float(auc_score), "num_features": len(feature_names), "label_mapping": label_map, "training_samples": len(X_train), "test_samples": len(X_test)} #Save model metadata
with open(model_dir / "model_metadata.json", "w") as f: json.dump(model_metadata, f, indent=2)

print(f"\nModel saved successfully in '{model_dir}' directory!")

#Save the best model
joblib.dump(selection_model, model_dir / "selection_model.pkl")
feature_names = X.columns.tolist()

#Save the model metadata
model_metadata = {
    "best_params": best_params,
    "best_cv_score": float(best_score), # may need to remove if grid search is commented out
    "test_auc_best": float(roc_auc_score(y_test, best_prob)), # may need to remove if grid search is commented out
    "num_original_features": len(feature_names),}

with open(model_dir / "selection_metadata.json", "w") as f:
  json.dump(model_metadata, f, indent=2)

