import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

model_dir = "ml_agent_burnout/models"

print("Loading training data...")
df = pd.read_csv(f'ml_agent_burnout/data/training_data.csv')

# Feature columns (exclude metadata and target)
feature_cols = [
    'utilization_rate', 'avg_handle_time_mins', 'escalation_rate', 'backlog',
    'sla_breaches', 'reopened_tickets', 'total_tickets_handled',
    'avg_utilization_7d', 'avg_utilization_30d', 'avg_handle_time_7d', 'avg_handle_time_30d',
    'avg_backlog_7d', 'avg_backlog_30d', 'utilization_trend', 'escalation_trend',
    'stress_index', 'performance_declining', 'persistent_backlog', 'workload_velocity',
    'tenure_months'
]

X = df[feature_cols].fillna(0)
y = df['burnout_risk_label']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\nDataset: {len(X)} samples, {len(feature_cols)} features")
print(f"Classes: {le.classes_}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")

# Train XGBoost
print("\nTraining XGBoost model...")
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model trained! Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model
joblib.dump(model, f'{model_dir}/burnout_model.pkl')
joblib.dump(le, f'{model_dir}/label_encoder.pkl')
joblib.dump(feature_cols, f'{model_dir}/feature_columns.pkl')

print("\n✅ Model saved to models/")