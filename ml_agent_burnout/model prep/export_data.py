"""
Export 90 days of agent performance data from ClickHouse
Uses the LATEST simulation date from ClickHouse (not system date)
"""

from clickhouse_driver import Client
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

# ClickHouse connection
client = Client(host='localhost', port=9005, user='admin', password='1234567', database='pbapp_poc')

print("\n" + "="*80)
print("📊 ClickHouse Data Export for ML Training")
print("="*80 + "\n")

# ===== STEP 1: Find the LATEST date in the simulation data =====
print("🔍 Detecting latest simulation date...")

result = client.execute("SELECT MAX(date_id) FROM fact_agent_performance")
latest_date_in_db = result[0][0]

if latest_date_in_db is None:
    print("❌ No data in fact_agent_performance!")
    print("⚠️  Please run the producer first:")
    print("   cd data_sources/clickhouse_direct")
    print("   python producer.py")
    exit(1)

# Convert to Python date object
end_date = date.fromisoformat(str(latest_date_in_db))
start_date = end_date - timedelta(days=90)

print(f"✅ Latest simulation date: {end_date}")
print(f"✅ Exporting data from {start_date} to {end_date} (last 90 days)")

# ===== STEP 2: Verify date range coverage =====
result = client.execute("SELECT MIN(date_id), MAX(date_id), COUNT(DISTINCT date_id) FROM fact_agent_performance")
min_date, max_date, total_days = result[0]

print(f"\n📊 Available data in ClickHouse:")
print(f"   Earliest date: {min_date}")
print(f"   Latest date: {max_date}")
print(f"   Total days: {total_days}")

if total_days < 90:
    print(f"\n⚠️  Warning: Only {total_days} days of data available (need 90 for training)")
    print(f"   Adjusting to use all {total_days} days...")
    start_date = date.fromisoformat(str(min_date))
    end_date = date.fromisoformat(str(max_date))

print(f"\n📅 Final export range: {start_date} to {end_date}")
print("="*80 + "\n")

# ===== STEP 3: Export agent performance data =====
print("📊 Exporting agent performance data...")

query1 = f"""
SELECT 
    date_id, agent_id, department_id,
    total_tickets_handled, avg_handle_time_mins, escalation_rate,
    backlog, reopened_tickets, utilization_rate, sla_breaches
FROM fact_agent_performance
WHERE date_id BETWEEN '{start_date}' AND '{end_date}'
ORDER BY agent_id, date_id
"""

result = client.execute(query1)
df1 = pd.DataFrame(result, columns=[
    'date_id', 'agent_id', 'department_id', 'total_tickets_handled',
    'avg_handle_time_mins', 'escalation_rate', 'backlog', 'reopened_tickets',
    'utilization_rate', 'sla_breaches'
])

output_dir = Path(__file__).parent / "data"
output_dir.mkdir(exist_ok=True)

df1.to_csv(output_dir / 'agent_performance_90d.csv', index=False)
print(f"✅ Exported {len(df1)} agent performance records")
print(f"   Date range: {df1['date_id'].min()} to {df1['date_id'].max()}")
print(f"   Agents covered: {df1['agent_id'].nunique()}")

# ===== STEP 4: Export agent metadata =====
print("\n📊 Exporting agent metadata...")

query2 = "SELECT agent_id, agent_name, department_id, role_title, hire_date FROM dim_agent"
result = client.execute(query2)
df2 = pd.DataFrame(result, columns=['agent_id', 'agent_name', 'department_id', 'role_title', 'hire_date'])
df2.to_csv(output_dir / 'agent_metadata.csv', index=False)
print(f"✅ Exported {len(df2)} agent records")

# ===== STEP 5: Export ticket metrics (aggregated by agent) =====
print("\n📊 Exporting ticket data...")

query3 = f"""
SELECT 
    date_id, agent_id,
    COUNT(*) AS total_tickets,
    AVG(satisfaction_score) AS avg_csat,
    SUM(reopened) AS reopened_count,
    SUM(sla_compliance = 0) AS sla_breach_count
FROM fact_tickets
WHERE date_id BETWEEN '{start_date}' AND '{end_date}'
GROUP BY date_id, agent_id
ORDER BY agent_id, date_id
"""

result = client.execute(query3)
df3 = pd.DataFrame(result, columns=[
    'date_id', 'agent_id', 'total_tickets', 'avg_csat', 'reopened_count', 'sla_breach_count'
])
df3.to_csv(output_dir / 'agent_tickets_90d.csv', index=False)
print(f"✅ Exported {len(df3)} ticket records")
print(f"   Date range: {df3['date_id'].min()} to {df3['date_id'].max()}")

# ===== SUMMARY =====
print("\n" + "="*80)
print("✅ Data export complete!")
print("="*80)
print(f"📁 Output directory: {output_dir}")
print(f"   ✅ agent_performance_90d.csv ({len(df1)} rows)")
print(f"   ✅ agent_tickets_90d.csv ({len(df3)} rows)")
print(f"   ✅ agent_metadata.csv ({len(df2)} rows)")
print(f"\n📊 Coverage:")
print(f"   Date range: {start_date} to {end_date}")
print(f"   Total days: {(end_date - start_date).days + 1}")
print(f"   Agents: {df1['agent_id'].nunique()}")
print(f"   Avg records per agent: {len(df1) / df1['agent_id'].nunique():.1f}")
print("="*80 + "\n")

print("💡 Next step: Run prepare_features.py")