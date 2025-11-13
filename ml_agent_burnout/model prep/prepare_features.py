import pandas as pd
import numpy as np

csv_dir = "ml_agent_burnout/data/"

print("Loading data...")
df_perf = pd.read_csv(f'{csv_dir}agent_performance_90d.csv')
df_tickets = pd.read_csv(f'{csv_dir}agent_tickets_90d.csv')
df_meta = pd.read_csv(f'{csv_dir}agent_metadata.csv')

# Merge datasets
df = df_perf.merge(df_tickets, on=['date_id', 'agent_id'], how='left')
df = df.merge(df_meta[['agent_id', 'role_title', 'hire_date']], on='agent_id', how='left')

# Convert date columns
df['date_id'] = pd.to_datetime(df['date_id'])
df['hire_date'] = pd.to_datetime(df['hire_date'])

# Fill missing values
df = df.fillna(0)

print(f"Merged data: {len(df)} rows, {len(df['agent_id'].unique())} agents")

# ===== FEATURE ENGINEERING =====
print("\nCalculating features...")

# 1. Rolling averages (7-day and 30-day)
df = df.sort_values(['agent_id', 'date_id'])

for window in [7, 30]:
    df[f'avg_utilization_{window}d'] = df.groupby('agent_id')['utilization_rate'].rolling(window, min_periods=3).mean().reset_index(0, drop=True)
    df[f'avg_handle_time_{window}d'] = df.groupby('agent_id')['avg_handle_time_mins'].rolling(window, min_periods=3).mean().reset_index(0, drop=True)
    df[f'avg_backlog_{window}d'] = df.groupby('agent_id')['backlog'].rolling(window, min_periods=3).mean().reset_index(0, drop=True)

# 2. Trend indicators (change over last 7 days)
df['utilization_trend'] = df.groupby('agent_id')['utilization_rate'].diff(7)
df['escalation_trend'] = df.groupby('agent_id')['escalation_rate'].diff(7)

# 3. Composite stress index
df['stress_index'] = (
    df['escalation_rate'] * 0.3 + 
    (df['sla_breaches'] / 10).clip(upper=1.0) * 0.4 + 
    df['utilization_rate'] * 0.3
)

# 4. Performance decline flag
df['performance_declining'] = (df['avg_handle_time_7d'] > df['avg_handle_time_30d'] * 1.15).astype(int)

# 5. Persistent backlog flag
df['persistent_backlog'] = (df['backlog'] > 10).astype(int)

# 6. Workload velocity (backlog change)
df['workload_velocity'] = df.groupby('agent_id')['backlog'].diff(7)

# 7. Agent tenure (months)
df['tenure_months'] = ((df['date_id'] - df['hire_date']).dt.days / 30).astype(int)

# Drop rows with NaN (insufficient data for rolling windows)
df = df.dropna()

print(f"Features calculated. Dataset now has {len(df)} rows.")

# ===== BURNOUT LABELING (RELAXED THRESHOLDS) =====
print("\nAssigning burnout risk labels...")

def label_burnout_risk(row):
    """
    Rule-based labeling with balanced distribution
    Target: ~40% Low, ~35% Medium, ~15% High, ~10% Critical
    """
    
    # ✅ CRITICAL: 2+ severe indicators
    critical_conditions = [
        row['utilization_rate'] >= 0.90,   # Very high utilization
        row['escalation_rate'] >= 0.22,    # High escalation
        row['backlog'] >= 12,              # High backlog
        row['sla_breaches'] >= 5,          # Multiple SLA breaches
        row['stress_index'] >= 0.65        # High stress
    ]
    
    if sum(critical_conditions) >= 2:
        return 'Critical'
    
    # Also Critical if extreme single indicator
    if (row['utilization_rate'] >= 0.95 or 
        row['escalation_rate'] >= 0.30 or 
        row['backlog'] >= 20):
        return 'Critical'
    
    # ✅ HIGH: 2+ moderate indicators
    high_conditions = [
        row['utilization_rate'] >= 0.78,   # High utilization
        row['escalation_rate'] >= 0.12,    # Moderate escalation
        row['backlog'] >= 7,               # Moderate backlog
        row['sla_breaches'] >= 3,          # Some SLA breaches
        row['persistent_backlog'] == 1,
        row['performance_declining'] == 1,
        row['stress_index'] >= 0.50        # Moderate stress
    ]
    
    if sum(high_conditions) >= 2:
        return 'High'
    
    # ✅ MEDIUM: 1+ mild indicators
    medium_conditions = [
        row['utilization_rate'] >= 0.65,   # Moderate utilization
        row['backlog'] >= 4,               # Some backlog
        row['stress_index'] >= 0.40,       # Mild stress
        row['escalation_rate'] >= 0.08,    # Some escalation
        row['sla_breaches'] >= 1           # At least 1 SLA breach
    ]
    
    if sum(medium_conditions) >= 1:
        return 'Medium'
    
    # ✅ LOW: Everything else
    return 'Low'

df['burnout_risk_label'] = df.apply(label_burnout_risk, axis=1)

# Check initial distribution
print("\n📊 Initial label distribution:")
print(df['burnout_risk_label'].value_counts(normalize=True).sort_index())
print("\nAbsolute counts:")
print(df['burnout_risk_label'].value_counts().sort_index())

# ===== FORCE BALANCED DISTRIBUTION =====
required_labels = ['Low', 'Medium', 'High', 'Critical']
missing_labels = set(required_labels) - set(df['burnout_risk_label'].unique())

if missing_labels or (df['burnout_risk_label'] == 'Low').sum() == 0:
    print(f"\n⚠️  Forcing balanced distribution...")
    
    # Sort by stress index
    df_sorted = df.sort_values('stress_index', ascending=False)
    
    # Target distribution (adjust indices to create all 4 classes)
    total_samples = len(df)


    ### Force distribution: 5% Critical, 12% High, 35% Medium, 48% Low
    critical_count = max(int(total_samples * 0.05), 1)
    high_count = max(int(total_samples * 0.12), 1)
    medium_count = max(int(total_samples * 0.35), 1)
    # Remaining will be Low
    
    # Reset all labels
    df['burnout_risk_label'] = 'Low'  # Start with all Low
    
    # Assign Critical (top 10%)
    critical_indices = df_sorted.head(critical_count).index
    df.loc[critical_indices, 'burnout_risk_label'] = 'Critical'
    
    # Assign High (next 15%)
    high_indices = df_sorted.iloc[critical_count:critical_count + high_count].index
    df.loc[high_indices, 'burnout_risk_label'] = 'High'
    
    # Assign Medium (next 35%)
    medium_indices = df_sorted.iloc[critical_count + high_count:critical_count + high_count + medium_count].index
    df.loc[medium_indices, 'burnout_risk_label'] = 'Medium'
    
    # Remaining are Low (bottom ~40%)
    print(f"   ✅ Forced {critical_count} Critical, {high_count} High, {medium_count} Medium")

print("\n✅ Final label distribution:")
print(df['burnout_risk_label'].value_counts(normalize=True).sort_index())
print("\nAbsolute counts:")
print(df['burnout_risk_label'].value_counts().sort_index())

# Verify all 4 classes exist
final_classes = set(df['burnout_risk_label'].unique())
if len(final_classes) == 4:
    print(f"\n✅ All 4 classes present: {sorted(final_classes)}")
else:
    print(f"\n❌ ERROR: Only {len(final_classes)} classes found: {sorted(final_classes)}")
    print("   Please check the labeling logic!")

# Save training dataset
df.to_csv(f'{csv_dir}training_data.csv', index=False)
print(f"\n✅ Training data saved: {len(df)} rows")
print(f"   Low: {(df['burnout_risk_label'] == 'Low').sum()} samples ({(df['burnout_risk_label'] == 'Low').sum() / len(df) * 100:.1f}%)")
print(f"   Medium: {(df['burnout_risk_label'] == 'Medium').sum()} samples ({(df['burnout_risk_label'] == 'Medium').sum() / len(df) * 100:.1f}%)")
print(f"   High: {(df['burnout_risk_label'] == 'High').sum()} samples ({(df['burnout_risk_label'] == 'High').sum() / len(df) * 100:.1f}%)")
print(f"   Critical: {(df['burnout_risk_label'] == 'Critical').sum()} samples ({(df['burnout_risk_label'] == 'Critical').sum() / len(df) * 100:.1f}%)")