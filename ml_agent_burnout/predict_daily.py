import pandas as pd
import joblib
from clickhouse_driver import Client
from datetime import date, timedelta
from pathlib import Path
import sys

print("\n" + "="*80)
print("Agent Burnout Prediction - Daily Inference")
print("="*80 + "\n")

model_dir = "ml_agent_burnout/models"


try:
# Load model
    print("Loading model...")
    model = joblib.load(f'{model_dir}/burnout_model.pkl')
    le = joblib.load(f'{model_dir}/label_encoder.pkl')
    feature_cols = joblib.load(f'{model_dir}/feature_columns.pkl')
    print(f"✅ Model loaded successfully (trained classes: {le.classes_})")
except FileNotFoundError as e:
    print(f"❌ Error: Model files not found. Please run train_model.py first.")
    print(f"   Missing file: {e.filename}")
    sys.exit(1)

# Connect to ClickHouse
print("\nConnecting to ClickHouse...")
client = Client(host='localhost', port=9005, user='admin', password='1234567', database='pbapp_poc')

# Get latest simulation date (matches producer's last_simulation_date.txt)
result = client.execute("SELECT MAX(date_id) FROM fact_agent_performance")
latest_simulation_date = result[0][0]

if latest_simulation_date is None:
    print("❌ No data in fact_agent_performance. Run producer first.")
    sys.exit(1)

end_date = date.fromisoformat(str(latest_simulation_date))
start_date = end_date - timedelta(days=30)

print(f"✅ Latest simulation date: {end_date}")
print(f"   Fetching data from {start_date} to {end_date}")

# Check if prediction already exists for this date
result = client.execute(f"SELECT COUNT(*) FROM fact_agent_burnout_prediction WHERE date_id = '{end_date}'")
existing_count = result[0][0]

if existing_count > 0:
    print(f"\n⚠️  Prediction already exists for {end_date} ({existing_count} records)")
    print("   Skipping to avoid duplicates. Run producer.py to advance to next day.")
    sys.exit(0)

# Fetch latest 30 days of agent performance data
print("\n📊 Fetching agent performance data...")
query = f"""
SELECT 
    date_id, agent_id, department_id,
    total_tickets_handled, avg_handle_time_mins, escalation_rate,
    backlog, reopened_tickets, utilization_rate, sla_breaches
FROM fact_agent_performance
WHERE date_id BETWEEN '{start_date}' AND '{end_date}'
ORDER BY agent_id, date_id
"""

df = pd.DataFrame(client.execute(query), columns=[
    'date_id', 'agent_id', 'department_id', 'total_tickets_handled',
    'avg_handle_time_mins', 'escalation_rate', 'backlog', 'reopened_tickets',
    'utilization_rate', 'sla_breaches'
])

print(f"✅ Loaded {len(df)} records for {df['agent_id'].nunique()} agents")

if df.empty:
    print("❌ No agent performance data found. Run producer first.")
    sys.exit(1)

# Convert dates
df['date_id'] = pd.to_datetime(df['date_id'])

# ===== FEATURE ENGINEERING =====
print("\n🔧 Calculating features...")
df = df.sort_values(['agent_id', 'date_id'])

# Rolling averages (7-day and 30-day)
for window in [7, 30]:
    df[f'avg_utilization_{window}d'] = df.groupby('agent_id')['utilization_rate'].rolling(window, min_periods=3).mean().reset_index(0, drop=True)
    df[f'avg_handle_time_{window}d'] = df.groupby('agent_id')['avg_handle_time_mins'].rolling(window, min_periods=3).mean().reset_index(0, drop=True)
    df[f'avg_backlog_{window}d'] = df.groupby('agent_id')['backlog'].rolling(window, min_periods=3).mean().reset_index(0, drop=True)

# Trends
df['utilization_trend'] = df.groupby('agent_id')['utilization_rate'].diff(7)
df['escalation_trend'] = df.groupby('agent_id')['escalation_rate'].diff(7)

# Composite metrics
df['stress_index'] = df['escalation_rate'] * 0.3 + (df['sla_breaches'] / 10).clip(upper=1.0) * 0.4 + df['utilization_rate'] * 0.3
df['performance_declining'] = (df['avg_handle_time_7d'] > df['avg_handle_time_30d'] * 1.15).astype(int)
df['persistent_backlog'] = (df['backlog'] > 10).astype(int)
df['workload_velocity'] = df.groupby('agent_id')['backlog'].diff(7)
df['tenure_months'] = 12  # Placeholder

# Get most recent record per agent
df_latest = df.groupby('agent_id').tail(1).reset_index(drop=True)
df_latest = df_latest.fillna(0)

print(f"✅ Features calculated for {len(df_latest)} agents")

# ===== GENERATE PREDICTIONS =====
print("\n🔮 Generating burnout predictions...")

X = df_latest[feature_cols]
y_pred_encoded = model.predict(X)
y_pred_labels = le.inverse_transform(y_pred_encoded)
y_pred_proba = model.predict_proba(X)

# Confidence scores
confidence_scores = y_pred_proba.max(axis=1)

# Calculate weighted burnout risk score (0.0 - 1.0)
# Base weights for risk categories
risk_weights = {
    'Low': 0.0,
    'Medium': 0.33,
    'High': 0.67,
    'Critical': 1.0
}

# Get class indices
class_indices = {class_name: list(le.classes_).index(class_name) for class_name in le.classes_}

burnout_risk_scores = []
for i, proba_row in enumerate(y_pred_proba):
    row = df_latest.iloc[i]
    
    # Base weighted score from probabilities
    base_score = (
        proba_row[class_indices['Low']] * risk_weights['Low'] +
        proba_row[class_indices['Medium']] * risk_weights['Medium'] +
        proba_row[class_indices['High']] * risk_weights['High'] +
        proba_row[class_indices['Critical']] * risk_weights['Critical']
    )
    
    # ✅ CONTEXTUAL ADJUSTMENTS (create variance within categories)
    
    # 1️⃣ Workload Intensity Factor (±0.15)
    utilization = row['utilization_rate']
    backlog = row['backlog']
    workload_factor = 0.0
    
    if utilization > 0.95:  # Extreme workload
        workload_factor += 0.10
    elif utilization > 0.85:
        workload_factor += 0.05
    elif utilization < 0.60:  # Underutilized
        workload_factor -= 0.05

    if backlog > 15:
        workload_factor += 0.05
    elif backlog > 10:
        workload_factor += 0.03
    
    # 2️⃣ Performance Decline Factor (±0.12)
    decline_factor = 0.0
    if row['performance_declining'] == 1:
        decline_factor += 0.08
    
    if row['utilization_trend'] > 0.10:  # Rapidly increasing utilization
        decline_factor += 0.04
    elif row['utilization_trend'] < -0.10:  # Improving
        decline_factor -= 0.04
    
    # 3️⃣ Quality Issues Factor (±0.10)
    quality_factor = 0.0
    escalation_rate = row['escalation_rate']
    sla_breaches = row['sla_breaches']
    
    if escalation_rate > 0.25:
        quality_factor += 0.06
    elif escalation_rate > 0.15:
        quality_factor += 0.03
    
    if sla_breaches >= 5:
        quality_factor += 0.04
    elif sla_breaches >= 3:
        quality_factor += 0.02
    
    # 4️⃣ Backlog Persistence Factor (±0.08)
    persistence_factor = 0.0
    if row['persistent_backlog'] == 1:
        persistence_factor += 0.05
    
    workload_velocity = row['workload_velocity']
    if workload_velocity > 5:  # Backlog growing
        persistence_factor += 0.03
    elif workload_velocity < -5:  # Backlog shrinking
        persistence_factor -= 0.03
    
    # 5️⃣ Handle Time Efficiency Factor (±0.08)
    efficiency_factor = 0.0
    avg_handle_7d = row['avg_handle_time_7d']
    avg_handle_30d = row['avg_handle_time_30d']
    
    if avg_handle_7d > avg_handle_30d * 1.20:  # Slowing down significantly
        efficiency_factor += 0.05
    elif avg_handle_7d > avg_handle_30d * 1.10:
        efficiency_factor += 0.03
    elif avg_handle_7d < avg_handle_30d * 0.90:  # Getting faster
        efficiency_factor -= 0.03
    
    # 6️⃣ Ticket Volume Stress Factor (±0.07)
    volume_factor = 0.0
    total_tickets = row['total_tickets_handled']
    reopened = row['reopened_tickets']
    
    if total_tickets > 10:  # High volume day
        volume_factor += 0.04
    elif total_tickets < 3:  # Low volume
        volume_factor -= 0.03
    
    if reopened > 3:  # Many reopened tickets
        volume_factor += 0.03
    
    # ✅ COMBINE ALL FACTORS
    contextual_adjustments = (
        workload_factor +
        decline_factor +
        quality_factor +
        persistence_factor +
        efficiency_factor +
        volume_factor
    )
    
    # Final score with contextual adjustments
    final_score = base_score + contextual_adjustments
    
    # Clamp to valid range [0.0, 1.0]
    final_score = max(0.0, min(1.0, final_score))
    
    burnout_risk_scores.append(final_score)

print(f"✅ Predictions generated with contextual risk scoring")

# ===== CALCULATE PRODUCTIVITY INDEX =====
print("\n📊 Calculating Agent Productivity Index...")

productivity_scores = []
for i, row in df_latest.iterrows():
    # ✅ 1. Throughput Score (0.0 - 0.30) - Volume handled
    total_tickets = row['total_tickets_handled']
    
    # Benchmark: 8 tickets/day is 100%
    throughput_score = min(0.30, (total_tickets / 8.0) * 0.30)
    
    # ✅ 2. Quality Score (0.0 - 0.35) - Low escalation & reopened tickets
    escalation_rate = row['escalation_rate']
    reopened_tickets = row['reopened_tickets']
    
    # Penalize high escalation (ideal: < 0.10)
    escalation_penalty = max(0, escalation_rate - 0.10) * 0.5
    
    # Penalize reopened tickets (ideal: 0)
    reopen_penalty = (reopened_tickets / max(1, total_tickets)) * 0.3
    
    quality_score = max(0.0, 0.35 - escalation_penalty - reopen_penalty)
    
    # ✅ 3. Efficiency Score (0.0 - 0.25) - Handle time vs benchmark
    avg_handle_time = row['avg_handle_time_mins']
    avg_handle_30d = row['avg_handle_time_30d']
    
    # Ideal: current handle time <= 30-day average
    if avg_handle_time <= avg_handle_30d * 0.90:
        efficiency_score = 0.25  # Faster than usual
    elif avg_handle_time <= avg_handle_30d * 1.10:
        efficiency_score = 0.20  # On par
    elif avg_handle_time <= avg_handle_30d * 1.20:
        efficiency_score = 0.15  # Slightly slower
    else:
        efficiency_score = 0.10  # Significantly slower
    
    # ✅ 4. Utilization Contribution (0.0 - 0.10) - Not overworked or underworked
    utilization = row['utilization_rate']
    
    # Optimal range: 0.70 - 0.85
    if 0.70 <= utilization <= 0.85:
        utilization_contribution = 0.10
    elif 0.60 <= utilization < 0.70 or 0.85 < utilization <= 0.90:
        utilization_contribution = 0.07
    else:
        utilization_contribution = 0.03  # Too low or too high
    
    # ✅ COMBINE COMPONENTS
    productivity_index = (
        throughput_score +
        quality_score +
        efficiency_score +
        utilization_contribution
    )
    
    # Clamp to [0.0, 1.0]
    productivity_index = max(0.0, min(1.0, productivity_index))
    
    productivity_scores.append(productivity_index)

print(f"✅ Productivity Index calculated for {len(productivity_scores)} agents")

# ===== PREPARE DATA FOR CLICKHOUSE =====
print("\n📦 Preparing prediction records with ALL engineered features...")

predictions = []
for i, row in df_latest.iterrows():
    pred_label = y_pred_labels[i]
    pred_encoded = int(y_pred_encoded[i])
    
    # Extract all metrics
    base_utilization = float(row['utilization_rate'])
    backlog_count = int(row['backlog'])
    escalation_rate = float(row['escalation_rate'])
    stress_index = float(row['stress_index'])
    sla_breaches = int(row['sla_breaches'])
    handle_time = float(row['avg_handle_time_mins'])
    total_tickets = int(row['total_tickets_handled'])
    reopened = int(row['reopened_tickets'])
    
    # Calculate workload reduction
    if pred_label == 'Critical':
        reduction = 35
        if base_utilization > 0.95:
            reduction += 5
        if backlog_count > 20:
            reduction += 5
        reduction = min(50, reduction)
    elif pred_label == 'High':
        reduction = 20
        if base_utilization > 0.90:
            reduction += 5
        if backlog_count > 15:
            reduction += 3
        reduction = min(30, reduction)
    elif pred_label == 'Medium':
        reduction = 10
        if base_utilization > 0.85:
            reduction += 3
        if backlog_count > 10:
            reduction += 2
        if escalation_rate > 0.15:
            reduction += 2
        reduction = min(15, reduction)
    else:  # Low
        reduction = 0
        if base_utilization > 0.80:
            reduction = 3
        elif backlog_count > 8:
            reduction = 2
        elif stress_index > 0.55:
            reduction = 2
    
    predictions.append({
        'prediction_id': i + 1,
        'date_id': end_date,
        'agent_id': int(row['agent_id']),
        
        # Risk Scores
        'burnout_risk_score': float(burnout_risk_scores[i]),
        'productivity_index': float(productivity_scores[i]),
        'risk_category': pred_label,
        'risk_category_encoded': pred_encoded,
        
        # Current State
        'current_utilization': base_utilization,
        'current_backlog': backlog_count,
        'current_escalation_rate': escalation_rate,
        'current_sla_breaches': sla_breaches,
        'current_stress_index': stress_index,
        'current_handle_time_mins': handle_time,
        'current_total_tickets': total_tickets,
        'current_reopened_tickets': reopened,
        
        # Engineered Features (7-day)
        'avg_utilization_7d': float(row['avg_utilization_7d']),
        'avg_handle_time_7d': float(row['avg_handle_time_7d']),
        'avg_backlog_7d': float(row['avg_backlog_7d']),
        
        # Engineered Features (30-day)
        'avg_utilization_30d': float(row['avg_utilization_30d']),
        'avg_handle_time_30d': float(row['avg_handle_time_30d']),
        'avg_backlog_30d': float(row['avg_backlog_30d']),
        
        # Trends
        'utilization_trend': float(row['utilization_trend']),
        'escalation_trend': float(row['escalation_trend']),
        
        # Flags
        'performance_declining': int(row['performance_declining']),
        'persistent_backlog': int(row['persistent_backlog']),
        'workload_velocity': float(row['workload_velocity']),
        
        # Recommendations
        'recommended_action': {
            'Low': 'Continue monitoring',
            'Medium': 'Proactive check-in scheduled',
            'High': 'Redistribute 5-8 cases immediately',
            'Critical': 'URGENT: Mandatory rest + workload redistribution'
        }[pred_label],
        'workload_reduction_pct': reduction,
        
        # Model Metadata
        'confidence_score': float(confidence_scores[i]),
        'model_version': 'v1.0_xgboost'
    })

# Ensure at least 1 Critical case if total agents >= 10
total_agents = len(predictions)
critical_count = sum(1 for p in predictions if p['risk_category'] == 'Critical')

if total_agents >= 10 and critical_count == 0:
    highest_risk_idx = max(range(len(predictions)), key=lambda i: predictions[i]['burnout_risk_score'])
    
    if predictions[highest_risk_idx]['burnout_risk_score'] > 0.6:
        predictions[highest_risk_idx]['risk_category'] = 'Critical'
        predictions[highest_risk_idx]['risk_category_encoded'] = list(le.classes_).index('Critical')
        predictions[highest_risk_idx]['workload_reduction_pct'] = 35
        predictions[highest_risk_idx]['recommended_action'] = 'URGENT: Mandatory rest + workload redistribution'
        print(f"\n⚠️  Forced agent {predictions[highest_risk_idx]['agent_id']} to Critical (risk score: {predictions[highest_risk_idx]['burnout_risk_score']:.3f})")

# ===== INSERT INTO CLICKHOUSE =====
print(f"\n💾 Inserting {len(predictions)} predictions into ClickHouse...")

client.execute("""
    INSERT INTO fact_agent_burnout_prediction (
        prediction_id, date_id, agent_id,
        burnout_risk_score, productivity_index, risk_category, risk_category_encoded,
        current_utilization, current_backlog, current_escalation_rate,
        current_sla_breaches, current_stress_index, current_handle_time_mins,
        current_total_tickets, current_reopened_tickets,
        avg_utilization_7d, avg_handle_time_7d, avg_backlog_7d,
        avg_utilization_30d, avg_handle_time_30d, avg_backlog_30d,
        utilization_trend, escalation_trend,
        performance_declining, persistent_backlog, workload_velocity,
        recommended_action, workload_reduction_pct,
        confidence_score, model_version
    ) VALUES
""", predictions)

print("✅ Predictions inserted successfully!")

print("\n" + "="*80)
print("✅ Daily prediction complete!")
print("="*80)
print(f"\n📅 Prediction date: {end_date}")
print(f"📊 Total agents: {len(predictions)}")
print(f"🎯 Model version: v1.0_xgboost")
print(f"💡 Predictions sync with producer's simulation date")
print("="*80 + "\n")