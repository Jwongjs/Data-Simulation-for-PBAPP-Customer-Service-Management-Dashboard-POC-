"""
Backfill predictions for all historical dates with agent performance data
Run this once to create evenly-spaced predictions
"""

from clickhouse_driver import Client
import pandas as pd
import joblib
from datetime import date, timedelta
from pathlib import Path
import sys

print("\n" + "="*80)
print("🔙 Agent Burnout Prediction - Historical Backfill")
print("="*80 + "\n")

# Load model
model_dir = "ml_agent_burnout/models"
try:
    model = joblib.load(f'{model_dir}/burnout_model.pkl')
    le = joblib.load(f'{model_dir}/label_encoder.pkl')
    feature_cols = joblib.load(f'{model_dir}/feature_columns.pkl')
    print(f"✅ Model loaded successfully")
except FileNotFoundError as e:
    print(f"❌ Error: Model files not found. Run train_model.py first.")
    sys.exit(1)

# Connect to ClickHouse
client = Client(host='localhost', port=9005, user='admin', password='1234567', database='pbapp_poc')

# Get all dates with agent performance data
result = client.execute("""
    SELECT DISTINCT date_id 
    FROM fact_agent_performance 
    WHERE date_id >= (SELECT MIN(date_id) FROM fact_agent_performance) + INTERVAL 30 DAY
    ORDER BY date_id
""")

all_dates = [date.fromisoformat(str(row[0])) for row in result]

if not all_dates:
    print("❌ Need at least 30 days of data in fact_agent_performance")
    sys.exit(1)

print(f"✅ Found {len(all_dates)} dates with sufficient historical data")
print(f"   Date range: {all_dates[0]} to {all_dates[-1]}")

# Check existing predictions
result = client.execute("SELECT DISTINCT date_id FROM fact_agent_burnout_prediction")
existing_dates = {date.fromisoformat(str(row[0])) for row in result}

dates_to_predict = [d for d in all_dates if d not in existing_dates]

if not dates_to_predict:
    print("\n✅ All dates already have predictions!")
    sys.exit(0)

print(f"\n🔮 Generating predictions for {len(dates_to_predict)} missing dates...")

# Process each date
for idx, prediction_date in enumerate(dates_to_predict, 1):
    start_date = prediction_date - timedelta(days=30)
    
    print(f"\n[{idx}/{len(dates_to_predict)}] Processing: {prediction_date}")
    
    # Fetch 30 days of data
    query = f"""
    SELECT 
        date_id, agent_id, department_id,
        total_tickets_handled, avg_handle_time_mins, escalation_rate,
        backlog, reopened_tickets, utilization_rate, sla_breaches
    FROM fact_agent_performance
    WHERE date_id BETWEEN '{start_date}' AND '{prediction_date}'
    ORDER BY agent_id, date_id
    """
    
    df = pd.DataFrame(client.execute(query), columns=[
        'date_id', 'agent_id', 'department_id', 'total_tickets_handled',
        'avg_handle_time_mins', 'escalation_rate', 'backlog', 'reopened_tickets',
        'utilization_rate', 'sla_breaches'
    ])
    
    if df.empty or len(df) < 60:  # Need sufficient data
        print(f"   ⚠️  Insufficient data ({len(df)} rows), skipping...")
        continue
    
    # Feature engineering
    df['date_id'] = pd.to_datetime(df['date_id'])
    df = df.sort_values(['agent_id', 'date_id'])
    
    for window in [7, 30]:
        df[f'avg_utilization_{window}d'] = df.groupby('agent_id')['utilization_rate'].rolling(window, min_periods=3).mean().reset_index(0, drop=True)
        df[f'avg_handle_time_{window}d'] = df.groupby('agent_id')['avg_handle_time_mins'].rolling(window, min_periods=3).mean().reset_index(0, drop=True)
        df[f'avg_backlog_{window}d'] = df.groupby('agent_id')['backlog'].rolling(window, min_periods=3).mean().reset_index(0, drop=True)
    
    df['utilization_trend'] = df.groupby('agent_id')['utilization_rate'].diff(7)
    df['escalation_trend'] = df.groupby('agent_id')['escalation_rate'].diff(7)
    df['stress_index'] = df['escalation_rate'] * 0.3 + (df['sla_breaches'] / 10).clip(upper=1.0) * 0.4 + df['utilization_rate'] * 0.3
    df['performance_declining'] = (df['avg_handle_time_7d'] > df['avg_handle_time_30d'] * 1.15).astype(int)
    df['persistent_backlog'] = (df['backlog'] > 10).astype(int)
    df['workload_velocity'] = df.groupby('agent_id')['backlog'].diff(7)
    df['tenure_months'] = 12
    
    df_latest = df.groupby('agent_id').tail(1).reset_index(drop=True).fillna(0)
    
    if len(df_latest) < 10:
        print(f"   ⚠️  Only {len(df_latest)} agents, skipping...")
        continue
    
    # Generate predictions (simplified version - add your contextual logic)
    X = df_latest[feature_cols]
    y_pred_encoded = model.predict(X)
    y_pred_labels = le.inverse_transform(y_pred_encoded)
    y_pred_proba = model.predict_proba(X)
    
    predictions = []
    for i, row in df_latest.iterrows():
        pred_label = y_pred_labels[i]
        pred_encoded = int(y_pred_encoded[i])
        
        # Simplified - add your full contextual adjustment logic here
        burnout_risk_score = float(y_pred_proba[i].max())
        productivity_index = 0.75  # Placeholder
        
        workload_reduction_pct = {'Critical': 35, 'High': 20, 'Medium': 10, 'Low': 0}.get(pred_label, 0)
        recommended_action = {
            'Critical': 'Immediate intervention',
            'High': 'Redistribute workload',
            'Medium': 'Proactive check-in',
            'Low': 'Continue monitoring'
        }.get(pred_label, '')
        
        predictions.append({
            'prediction_id': int(row['agent_id']),
            'date_id': prediction_date,
            'agent_id': int(row['agent_id']),
            'burnout_risk_score': burnout_risk_score,
            'productivity_index': productivity_index,
            'risk_category': pred_label,
            'risk_category_encoded': pred_encoded,
            'current_utilization': float(row['utilization_rate']),
            'current_backlog': int(row['backlog']),
            'current_escalation_rate': float(row['escalation_rate']),
            'current_sla_breaches': int(row['sla_breaches']),
            'current_stress_index': float(row['stress_index']),
            'current_handle_time_mins': float(row['avg_handle_time_mins']),
            'current_total_tickets': int(row['total_tickets_handled']),
            'current_reopened_tickets': int(row['reopened_tickets']),
            'avg_utilization_7d': float(row['avg_utilization_7d']),
            'avg_handle_time_7d': float(row['avg_handle_time_7d']),
            'avg_backlog_7d': float(row['avg_backlog_7d']),
            'avg_utilization_30d': float(row['avg_utilization_30d']),
            'avg_handle_time_30d': float(row['avg_handle_time_30d']),
            'avg_backlog_30d': float(row['avg_backlog_30d']),
            'utilization_trend': float(row['utilization_trend']),
            'escalation_trend': float(row['escalation_trend']),
            'performance_declining': int(row['performance_declining']),
            'persistent_backlog': int(row['persistent_backlog']),
            'workload_velocity': float(row['workload_velocity']),
            'recommended_action': recommended_action,
            'workload_reduction_pct': workload_reduction_pct,
            'confidence_score': burnout_risk_score,
            'model_version': 'v1.0_xgboost'
        })
    
    # Insert
    if predictions:
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
        print(f"   ✅ Inserted {len(predictions)} predictions")

print("\n" + "="*80)
print("✅ Backfill complete!")
print(f"📊 Total dates processed: {len(dates_to_predict)}")
print("="*80 + "\n")