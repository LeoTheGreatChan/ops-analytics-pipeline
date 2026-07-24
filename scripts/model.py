import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

df = pd.read_pickle('../data/processed/cleaned_data.pkl')

# ---- FEATURE SET ----
cat_features = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Time_Band', 'Weekday']
num_features = ['Agent_Age', 'Agent_Rating', 'Distance_km', 'Is_Weekend', 'Is_Holiday']

model_df = df[cat_features + num_features + ['SLA_Breach']].copy()

# drop rows with null Distance_km (bad geocodes) for model training specifically
before = len(model_df)
model_df = model_df.dropna(subset=['Distance_km']).reset_index(drop=True)
dropped_for_model = before - len(model_df)

model_df['Is_Weekend'] = model_df['Is_Weekend'].astype(int)
model_df['Is_Holiday'] = model_df['Is_Holiday'].astype(int)

X = pd.get_dummies(model_df[cat_features + num_features], columns=cat_features, drop_first=False)
y = model_df['SLA_Breach'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_leaf=20,
    class_weight='balanced', random_state=42, n_jobs=-1
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# aggregate one-hot feature importances back to original feature names
importances = pd.Series(clf.feature_importances_, index=X.columns)
agg_importance = {}
for feat in cat_features + num_features:
    matching = [c for c in X.columns if c == feat or c.startswith(feat + '_')]
    agg_importance[feat] = importances[matching].sum()
agg_importance = pd.Series(agg_importance).sort_values(ascending=False)

print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
print(f"Rows dropped for modelling (null Distance_km): {dropped_for_model}")
print(f"Baseline breach rate (test set): {y_test.mean()*100:.1f}%")
print()
print(f"Accuracy: {acc*100:.1f}%")
print(f"F1 score: {f1:.3f}")
print()
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=['On-time','Breach']))
print()
print("Feature importances (aggregated by original feature):")
print(agg_importance.round(4))

import pickle
with open('../data/processed/model.pkl', 'wb') as f:
    pickle.dump({'model': clf, 'columns': list(X.columns), 'accuracy': acc, 'f1': f1,
                 'feature_importance': agg_importance}, f)
