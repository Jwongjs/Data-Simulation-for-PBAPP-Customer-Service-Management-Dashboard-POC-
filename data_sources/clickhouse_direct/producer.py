from multiprocessing.pool import RUN
from clickhouse_driver import Client
from faker import Faker
import json
import random
import time
import os
from datetime import datetime, date, timedelta
from pathlib import Path
import collections
import sys

fake = Faker()

# Load water facility configuration
CONFIG_PATH = Path(__file__).parent.parent / "water_iot" / "config.json"
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

WATER_FACILITIES = config["facilities"]
SENSOR_CONFIGS = config["sensors"]

# ClickHouse connection
CLICKHOUSE_HOST = 'localhost'
CLICKHOUSE_PORT = 9005
CLICKHOUSE_USER = 'admin'
CLICKHOUSE_PASSWORD = '1234567'
CLICKHOUSE_DB = 'pbapp_poc'

# Reference data
DISTRICTS = [
    "Georgetown", "Bayan Lepas", "Batu Ferringhi", "Tanjung Bungah",
    "Air Itam", "Balik Pulau", "Teluk Bahang", "Butterworth", "Bukit Mertajam"
]

TARIFF_CATEGORIES = ["Domestic", "Commercial", "Industrial", "Government", "Non-Profit"]
BILLING_STATUSES = ["Paid", "Pending", "Overdue"]
TICKET_PRIORITIES = ["Low", "Medium", "High", "Critical"]
ISSUE_CATEGORIES = ["Billing Dispute", "Water Quality", "Leakage", "No Water", "Pressure Issues", 
                   "Meter Reading", "Payment Problem", "Account Management", "Service Request", "Technical Support"]
ISSUE_CHANNELS = ["Phone", "Email", "Web Portal", "Mobile App", "In Person", "Social Media"]
DEPARTMENTS = [
    "Water Operations", 
    "Engineering", 
    "Water Quality Lab", 
    "Customer Service", 
    "Billing", 
    "Field Maintenance",
    "Distribution Management"
]

ROLE_TITLES = [
    "Agent", "Senior Agent", "Team Lead", "Supervisor", 
    "Specialist", "Analyst", "Coordinator", "Manager"
]

# ============================================================================
# DATE TRACKING FOR TIME-COHERENT SIMULATION
# ============================================================================

def get_next_simulation_date():
    """Get the next date to simulate, persisting between runs"""
    date_file = Path(__file__).parent / "last_simulation_date.txt"
    
    if date_file.exists():
        with open(date_file, "r") as f:
            last_str = f.read().strip()
        try:
            # Correctly parse the YYYY-MM-DD format
            last_date = datetime.strptime(last_str, '%Y-%m-%d').date()
        except (ValueError, TypeError):
            # Handle empty or malformed file
            last_date = date(2025, 1, 1) - timedelta(days=1)
        next_date = last_date + timedelta(days=1)
    else:
        # Start from January 1, 2025 if no previous date
        next_date = date(2025, 1, 1)
    
    # Retry file write with backoff (handles OneDrive lock)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(date_file, 'w') as f:
                f.write(next_date.strftime('%Y-%m-%d'))
            break  # Success
        except PermissionError as e:
            if attempt < max_retries - 1:
                print(f"⚠️  File locked, retrying in 1 second... (attempt {attempt + 1}/{max_retries})")
                time.sleep(1)
            else:
                print(f"❌ Could not write to {date_file} after {max_retries} attempts")
                print(f"   Error: {e}")
                print(f"   Continuing with date: {next_date}")
                # Continue execution even if file write fails
    
    return next_date

# ============================================================================
# STRONG ASSUMPTIONS FOR REALISTIC DATA SIMULATION
# ============================================================================

# Customer profiles for consistent behavior patterns
CUSTOMER_PROFILES = {}

# District-Facility Mapping
DISTRICT_FACILITY_MAP = {}

# Data Volume Configuration
DATA_VOLUMES = {
    'billing': {'distribution': {'Domestic': 0.65, 'Commercial': 0.20, 'Industrial': 0.10, 'Government': 0.03, 'Non-Profit': 0.02}},
    'tickets': {
        'distribution': {'Low': 0.40, 'Medium': 0.35, 'High': 0.20, 'Critical': 0.05},
        'category_distribution': {
            'Billing Dispute': 0.25,
            'Leakage': 0.20,
            'Water Quality': 0.15,
            'No Water': 0.10,
            'Pressure Issues': 0.10,
            'Payment Problem': 0.05,
            'Meter Reading': 0.05,
            'Account Management': 0.04,
            'Service Request': 0.03,
            'Technical Support': 0.03
        },
        'channel_distribution': {
            'Phone': 0.45,
            'Web Portal': 0.25,
            'Mobile App': 0.15,
            'Email': 0.10,
            'In Person': 0.03,
            'Social Media': 0.02
        }
    }
}

# Reference data store
REFERENCE_DATA = {
    'customers': {},  # Will store customer_id -> customer data
    'facilities': {},  # Will store facility_id -> facility data
    'time': {},       # Will store date_id -> time data
    'departments': {}, # Will store department_id -> department data
    'agents': {}      # Will store agent_id -> agent data
}

def connect_clickhouse():
    """Connect to ClickHouse"""
    return Client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        user=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DB
    )

def get_customer_profile(customer_id):
    """Return consistent behavior profile for a specific customer"""
    if customer_id not in CUSTOMER_PROFILES:
        # Use customer_id as seed for deterministic randomness
        random.seed(customer_id)
        
        CUSTOMER_PROFILES[customer_id] = {
            'consumption_variance': random.uniform(0.8, 1.2),
            'payment_reliability': random.choices(['reliable', 'occasional_late', 'problematic'], 
                                               weights=[0.7, 0.2, 0.1])[0],
            'ticket_frequency': random.choices(['low', 'medium', 'high'],
                                            weights=[0.8, 0.15, 0.05])[0],
            'satisfaction_baseline': random.uniform(3.0, 5.0),
            'primary_district': random.choice(DISTRICTS),
            'tariff_category': random.choices(TARIFF_CATEGORIES, 
                                           weights=[0.7, 0.15, 0.1, 0.03, 0.02])[0]
        }
        
    return CUSTOMER_PROFILES[customer_id]

def get_seasonal_factor(date_obj):
    """Return seasonal multiplier for water consumption"""
    # Penang has a dry season (Dec-Mar) and rainy season
    month = date_obj.month
    
    if month in [12, 1, 2, 3]:  # Dry season - higher consumption
        return random.uniform(1.1, 1.3)
    elif month in [4, 5]:  # Transition
        return random.uniform(0.9, 1.1)
    elif month in [6, 7, 8, 9, 10]:  # Rainy season - lower consumption
        return random.uniform(0.7, 0.9)
    else:  # Transition
        return random.uniform(0.9, 1.1)

def build_district_facility_map():
    """Build mapping between districts and facilities"""
    for idx, (facility_name, facility_info) in enumerate(WATER_FACILITIES.items(), start=1):
        district = facility_info['district']
        if district not in DISTRICT_FACILITY_MAP:
            DISTRICT_FACILITY_MAP[district] = []
        DISTRICT_FACILITY_MAP[district].append(idx)  # facility_id

# ============================================================================
# DIMENSION TABLE POPULATION
# ============================================================================
def ensure_dim_time_up_to(client, target_date: date):
    """Append dim_time rows up to target_date (inclusive), no duplicates."""
    result = client.execute("SELECT toString(max(date_id)) FROM dim_time")
    max_date_str = result[0][0]
    
    start_date = None
    
    if max_date_str is None or max_date_str == '1970-01-01':
        # Table is empty or has bad data, start from the beginning
        start_date = date(2025, 1, 1)
    else:
        # Parse the string date
        max_date_val = datetime.strptime(max_date_str, '%Y-%m-%d').date()
        
        if max_date_val < target_date:
            start_date = max_date_val + timedelta(days=1)

    if start_date is None:
        # Already up to date, nothing to do
        return
    
    # Build records using dictionaries with explicit field names
    records = []
    current_date = start_date
    while current_date <= target_date:
        records.append((
            current_date,
            current_date,
            current_date.day,
            current_date.month,
            (current_date.month - 1) // 3 + 1,
            current_date.year
        ))
        current_date += timedelta(days=1)
    
    if records:
        # Use execute with column specification
        client.execute(
            """
            INSERT INTO dim_time (date_id, full_date, day, month, quarter, year) VALUES
            """,
            records
        )
        print(f"   ✅ Appended {len(records)} date records into dim_time (from {start_date} to {target_date})")

def populate_dim_time(client):
    """Ensures dim_time is seeded on the very first run."""
    print("\n📅 Populating dim_time (incremental check)...")
    
    # Check if already populated
    result = client.execute("SELECT COUNT(*) FROM dim_time")
    if result[0][0] > 0:
        print(f"   ✅ dim_time already has {result[0][0]} records, skipping initial seed.")
        return

    # If table is empty, seed it up to the day before the simulation starts
    print("   ⏳ First run detected. Seeding initial dim_time...")
    initial_seed_date = date(2025, 1, 1)
    ensure_dim_time_up_to(client, initial_seed_date)

def populate_dim_customer(client, n=1000):
    """Populate dim_customer with sample customers"""
    print(f"\n👥 Populating dim_customer with {n} customers...")
    
    # Check if already populated
    result = client.execute("SELECT COUNT(*) FROM dim_customer")
    if result[0][0] > 0:
        print(f"   ✅ dim_customer already has {result[0][0]} records, skipping...")
        return
    
    records = []
    for i in range(1, n + 1):
        customer_id = 1000 + i
        profile = get_customer_profile(customer_id)
        
        records.append({
            'customer_id': customer_id,
            'account_no': f'ACC-{100000 + i}',
            'customer_name': fake.name(),
            'district': profile['primary_district'],
            'tariff_category': profile['tariff_category'],
            'customer_segment': random.choice(['Residential', 'Business', 'Government', 'Industrial'])
        })
    
    client.execute("""
        INSERT INTO dim_customer (customer_id, account_no, customer_name, district, tariff_category, customer_segment) VALUES
    """, records)
    
    print(f"   ✅ Inserted {n} customers into dim_customer")

def populate_dim_facility(client):
    """Populate dim_facility with water facilities from config"""
    print("\n🏭 Populating dim_facility...")
    
    # Check if already populated
    result = client.execute("SELECT COUNT(*) FROM dim_facility")
    if result[0][0] > 0:
        print(f"   ✅ dim_facility already has {result[0][0]} records, skipping...")
        return
    
    records = []
    for idx, (facility_name, facility_info) in enumerate(WATER_FACILITIES.items(), start=1):
        records.append({
            'facility_id': idx,
            'facility_name': facility_name,
            'facility_type': facility_info['type'],
            'district': facility_info['district'],
            'latitude': facility_info['location']['latitude'],
            'longitude': facility_info['location']['longitude']
        })
    
    client.execute("""
        INSERT INTO dim_facility (facility_id, facility_name, facility_type, district, latitude, longitude) VALUES
    """, records)
    
    print(f"   ✅ Inserted {len(records)} facilities into dim_facility")

def populate_dim_department(client):
    """Populate dim_department with organizational departments"""
    print("\n🏢 Populating dim_department...")
    
    # Check if already populated
    result = client.execute("SELECT COUNT(*) FROM dim_department")
    if result[0][0] > 0:
        print(f"   ✅ dim_department already has {result[0][0]} records, skipping...")
        return
    
    records = []
    for idx, dept in enumerate(DEPARTMENTS, start=1):
        records.append({
            'department_id': idx,
            'department_name': dept,
            'manager_name': fake.name(),
            'team_size': random.randint(5, 50)
        })
    
    client.execute("""
        INSERT INTO dim_department (department_id, department_name, manager_name, team_size) VALUES
    """, records)
    
    print(f"   ✅ Inserted {len(records)} departments into dim_department")

def populate_dim_agent(client):
    """Populate dim_agent with customer service agents"""
    print("\n👨‍💼 Populating dim_agent...")
    
    # Check if already populated
    result = client.execute("SELECT COUNT(*) FROM dim_agent")
    if result[0][0] > 0:
        print(f"   ✅ dim_agent already has {result[0][0]} records, skipping...")
        return
    
    records = []
    # Focus on customer service and billing departments for agents
    service_dept_ids = [4, 5]  # Customer Service and Billing departments
    
    for i in range(1, 21):
        # Deterministic department assignment based on agent_id
        if i <= 10:
            department_id = service_dept_ids[0]
        else:
            department_id = service_dept_ids[1]
        
        # Use consistent skill level based on agent_id
        random.seed(i)  # Ensure consistent results
        experience_level = i % 4  # 0-3
        
        # Role title depends on experience level
        if experience_level == 0:
            role = "Agent"
            hire_years_ago = random.randint(0, 1)
        elif experience_level == 1:
            role = "Senior Agent"
            hire_years_ago = random.randint(2, 3)
        elif experience_level == 2:
            role = "Team Lead"
            hire_years_ago = random.randint(4, 5)
        else:            
            role = "Supervisor"
            hire_years_ago = random.randint(6, 10)

        hire_date = date.today() - timedelta(days=hire_years_ago*365)

        records.append({
            'agent_id': i,
            'agent_name': fake.name(),
            'department_id': department_id,
            'role_title': role,
            'hire_date': hire_date,
            'status': 'Active'
        })
    
    client.execute("""
        INSERT INTO dim_agent (agent_id, agent_name, department_id, role_title, hire_date, status) VALUES
    """, records)
    
    print(f"   ✅ Inserted {len(records)} agents into dim_agent")

def populate_reference_data(client):
    """Populate reference data structures from dimension tables"""
    print("\n🔄 Loading reference data from dimensions...")
    
    # Load customers
    for row in client.execute("SELECT * FROM dim_customer"):
        customer_id = row[0]
        REFERENCE_DATA['customers'][customer_id] = {
            'customer_id': row[0],
            'account_no': row[1],
            'customer_name': row[2],
            'district': row[3],
            'tariff_category': row[4],
            'customer_segment': row[5]
        }
    
    # Load facilities
    for row in client.execute("SELECT * FROM dim_facility"):
        facility_id = row[0]
        REFERENCE_DATA['facilities'][facility_id] = {
            'facility_id': row[0],
            'facility_name': row[1],
            'facility_type': row[2],
            'district': row[3],
            'latitude': row[4],
            'longitude': row[5]
        }
    
    # Load departments
    for row in client.execute("SELECT * FROM dim_department"):
        department_id = row[0]
        REFERENCE_DATA['departments'][department_id] = {
            'department_id': row[0],
            'department_name': row[1],
            'manager_name': row[2],
            'team_size': row[3]
        }
    
    # Load agents
    for row in client.execute("SELECT * FROM dim_agent"):
        agent_id = row[0]
        REFERENCE_DATA['agents'][agent_id] = {
            'agent_id': row[0],
            'agent_name': row[1],
            'department_id': row[2],
            'role_title': row[3],
            'hire_date': row[4],
            'status': row[5],
            'efficiency': 0.7 + (agent_id / 20) * 0.3  # Efficiency increases with agent_id (0.7-1.0)
        }
    
    # Load time dimension - we'll fetch dates as needed
    print(f"   ✅ Loaded {len(REFERENCE_DATA['customers'])} customers")
    print(f"   ✅ Loaded {len(REFERENCE_DATA['facilities'])} facilities")
    print(f"   ✅ Loaded {len(REFERENCE_DATA['departments'])} departments")
    print(f"   ✅ Loaded {len(REFERENCE_DATA['agents'])} agents")

# ============================================================================
# FACT TABLE STREAMING INSERTS - DAILY SIMULATION
# ============================================================================

def insert_fact_tickets(client, simulation_date, n=80):
    """Insert RESOLVED customer service tickets for a specific date with agent metrics tracking"""

    # Track metrics by agent for this day's tickets
    agent_metrics = {i: {
        'total_tickets': 0,
        'reopened': 0,
        'sla_breached': 0,
        'handle_time_mins': [],
        'backlog_count': 0,
        'escalated_count': 0
    } for i in range(1, 21)}

    # For realistic distribution, calculate tickets per agent based on experience
    tickets_per_agent = []
    remaining_tickets = n

    for agent_id in range(1, 21):
        agent = REFERENCE_DATA['agents'].get(agent_id, {})
        role = agent.get('role_title', '')

        if "Supervisor" in role:
            base_tickets = int(n * 0.08)
        elif "Team Lead" in role:
            base_tickets = int(n * 0.07)
        elif "Senior" in role:
            base_tickets = int(n * 0.055)
        else:
            base_tickets = int(n * 0.04)

        agent_tickets = max(1, int(base_tickets * random.uniform(0.8, 1.2)))
        tickets_per_agent.append(agent_tickets)
        remaining_tickets -= agent_tickets

    while remaining_tickets > 0:
        for i in range(20):
            if remaining_tickets > 0:
                tickets_per_agent[i] += 1
                remaining_tickets -= 1
            else:
                break

    # ✅ REVERTED: Use simple counter for ticket_id (no date prefix)
    result = client.execute(f"""
        SELECT COALESCE(MAX(ticket_id), 0) 
        FROM fact_tickets
    """)
    last_ticket_id = result[0][0]
    
    ticket_counter = last_ticket_id + 1

    resolved_tickets = []
    backlog_tickets = []
    
    for agent_id, num_tickets in enumerate(tickets_per_agent, 1):
        agent_data = REFERENCE_DATA['agents'].get(agent_id, {})
        department_id = agent_data.get('department_id', 4)
        role = agent_data.get('role_title', '')

        for _ in range(num_tickets):
            # Select customer
            high_freq_customers = [cid for cid, profile in CUSTOMER_PROFILES.items()
                               if profile.get('ticket_frequency') == 'high']
            medium_freq_customers = [cid for cid, profile in CUSTOMER_PROFILES.items()
                                  if profile.get('ticket_frequency') == 'medium']

            if high_freq_customers and random.random() < 0.4:
                customer_id = random.choice(high_freq_customers)
            elif medium_freq_customers and random.random() < 0.6:
                customer_id = random.choice(medium_freq_customers)
            else:
                customer_id = random.choice(list(REFERENCE_DATA['customers'].keys()))

            customer_data = REFERENCE_DATA['customers'].get(customer_id, {})
            district = customer_data.get('district', random.choice(DISTRICTS))

            if district in DISTRICT_FACILITY_MAP and DISTRICT_FACILITY_MAP[district]:
                facility_id = random.choice(DISTRICT_FACILITY_MAP[district])
            else:
                facility_id = random.randint(1, len(REFERENCE_DATA['facilities']))

            issue_category = random.choices(
                list(DATA_VOLUMES['tickets']['category_distribution'].keys()),
                weights=list(DATA_VOLUMES['tickets']['category_distribution'].values()),
                k=1
            )[0]

            issue_channel = random.choices(
                list(DATA_VOLUMES['tickets']['channel_distribution'].keys()),
                weights=list(DATA_VOLUMES['tickets']['channel_distribution'].values()),
                k=1
            )[0]

            if "Supervisor" in role:
                base_time = random.uniform(0.5, 3)
            elif "Team Lead" in role:
                base_time = random.uniform(1, 4)
            elif "Senior" in role:
                base_time = random.uniform(2, 6)
            else:
                base_time = random.uniform(3, 8)

            if issue_category in ["Billing Dispute", "Water Quality"]:
                complexity_factor = 1.5
            elif issue_category in ["Leakage", "Pressure Issues"]:
                complexity_factor = 1.3
            else:
                complexity_factor = 1.0

            resolution_time_hrs = round(base_time * complexity_factor, 2)

            if "Supervisor" in role:
                fcr_chance = 0.85
            elif "Team Lead" in role:
                fcr_chance = 0.75
            elif "Senior" in role:
                fcr_chance = 0.65
            else:
                fcr_chance = 0.55

            resolved_in_first_contact = 1 if random.random() < fcr_chance else 0

            reopened = 0
            if not resolved_in_first_contact:
                reopen_chance = 0.05 if "Supervisor" in role else (
                               0.10 if "Team Lead" in role else (
                               0.15 if "Senior" in role else 0.25))
                reopened = 1 if random.random() < reopen_chance else 0

            sla_threshold = {
                "Supervisor": 0.05,
                "Team Lead": 0.10,
                "Senior Agent": 0.15,
                "Agent": 0.25
            }.get(role, 0.15)

            sla_compliance = 0 if random.random() < sla_threshold else 1

            if "Supervisor" in role:
                base_response = random.randint(1, 5)
            elif "Team Lead" in role:
                base_response = random.randint(2, 8)
            elif "Senior" in role:
                base_response = random.randint(3, 12)
            else:
                base_response = random.randint(5, 15)

            if issue_channel == "Phone":
                response_time_mins = max(1, int(base_response * 0.8))
            elif issue_channel == "Web Portal" or issue_channel == "Mobile App":
                response_time_mins = max(1, int(base_response * 1.2))
            else:
                response_time_mins = max(1, int(base_response * 2))

            satisfaction_base = 3.5
            if resolved_in_first_contact:
                satisfaction_base += 0.5
            if reopened:
                satisfaction_base -= 1.0
            if not sla_compliance:
                satisfaction_base -= 0.5

            satisfaction_score = round(max(1.0, min(5.0, satisfaction_base + random.uniform(-0.5, 0.5))), 1)
            nps_score = int((satisfaction_score - 3) * 4)
            effort_score = round(6 - satisfaction_score, 1) if satisfaction_score > 3 else random.uniform(3.5, 5.0)

            ticket_id = ticket_counter
            ticket_counter += 1

            agent_metrics[agent_id]['total_tickets'] += 1
            agent_metrics[agent_id]['reopened'] += reopened
            agent_metrics[agent_id]['sla_breached'] += (1 - sla_compliance)
            agent_metrics[agent_id]['handle_time_mins'].append(resolution_time_hrs * 60)

            ticket_data = {
                'ticket_id': ticket_id,
                'date_id': simulation_date,
                'customer_id': customer_id,
                'facility_id': facility_id,
                'department_id': department_id,
                'agent_id': agent_id,
                'issue_category': issue_category,
                'issue_channel': issue_channel,
                'resolution_time_hrs': resolution_time_hrs,
                'reopened': reopened,
                'resolved_in_first_contact': resolved_in_first_contact,
                'satisfaction_score': satisfaction_score,
                'nps_score': nps_score,
                'effort_score': effort_score,
                'sla_compliance': sla_compliance,
                'response_time_mins': response_time_mins
            }
            
            if resolved_in_first_contact == 1:
                resolved_tickets.append(ticket_data)
            else:
                agent_metrics[agent_id]['backlog_count'] += 1
                
                is_escalated = (sla_compliance == 0) or (reopened == 1)
                if is_escalated:
                    agent_metrics[agent_id]['escalated_count'] += 1
                
                backlog_tickets.append({
                    'ticket': ticket_data,
                    'customer_data': customer_data,
                    'is_escalated': is_escalated
                })

    if resolved_tickets:
        client.execute("""
            INSERT INTO fact_tickets (
                ticket_id, date_id, customer_id, facility_id, department_id, agent_id,
                issue_category, issue_channel, resolution_time_hrs,
                reopened, resolved_in_first_contact, satisfaction_score,
                nps_score, effort_score, sla_compliance, response_time_mins
            ) VALUES
        """, resolved_tickets)
        print(f"✅ Inserted {len(resolved_tickets)} RESOLVED tickets into fact_tickets for {simulation_date}")
    else:
        print(f"⚠️  No resolved tickets to insert into fact_tickets for {simulation_date}")
    
    if backlog_tickets:
        insert_fact_backlog(client, simulation_date, backlog_tickets)
    else:
        print(f"✅ No backlog cases for {simulation_date}")
    
    return agent_metrics

def insert_fact_agent_performance(client, simulation_date, agent_metrics):
    """Insert agent performance data based on ticket metrics"""
    
    records = []
    
    # Create records in agent_id order (1-20)
    for agent_id in range(1, 21):
        agent_data = REFERENCE_DATA['agents'].get(agent_id, {})
        department_id = agent_data.get('department_id', 4)  # Default to Customer Service
        metrics = agent_metrics[agent_id]
        
        total_tickets = metrics['total_tickets']
        reopened_tickets = metrics['reopened']
        sla_breaches = metrics['sla_breached']
        backlog_count = metrics['backlog_count']  # NEW
        escalated_count = metrics['escalated_count']  # NEW
        
        # Calculate average handle time
        handle_times = metrics['handle_time_mins']
        avg_handle_time = round(sum(handle_times) / max(1, len(handle_times)), 1)
        
        # Calculate escalation rate (based on actual escalations)
        escalation_rate = round(escalated_count / max(1, total_tickets), 3)
        
        # Compute resolution_rate %:
        # Handled = tickets completed by agent = total_tickets - backlog
        handled = max(0, int(total_tickets) - int(backlog_count))
        denom = handled + int(backlog_count)
        resolution_rate = round((handled / denom) * 100.0, 2) if denom > 0 else 0.0

        # More experienced agents have higher utilization
        role = agent_data.get('role_title', '')
        if "Supervisor" in role:
            base_utilization = 0.85
        elif "Team Lead" in role:
            base_utilization = 0.80
        elif "Senior" in role:
            base_utilization = 0.75
        else:
            base_utilization = 0.70
            
        utilization_rate = round(min(0.98, base_utilization + (total_tickets / 100)), 2)
        
        records.append({
            'performance_id': int(simulation_date.strftime('%Y%m%d')) * 100 + agent_id,
            'date_id': simulation_date,
            'agent_id': agent_id,
            'department_id': department_id,
            'total_tickets_handled': total_tickets,
            'avg_handle_time_mins': avg_handle_time,
            'escalation_rate': escalation_rate,  # NOW ALIGNED with actual escalations
            'backlog': backlog_count,  # NOW ALIGNED with actual backlog cases
            'reopened_tickets': reopened_tickets,
            'utilization_rate': utilization_rate,
            'sla_breaches': sla_breaches,
            'resolution_rate': resolution_rate
        })
    
    client.execute("""
        INSERT INTO fact_agent_performance (
            performance_id, date_id, agent_id, department_id,
            total_tickets_handled, avg_handle_time_mins, escalation_rate,
            backlog, reopened_tickets, utilization_rate, sla_breaches, resolution_rate
        ) VALUES
    """, records)
    
    print(f"✅ Inserted performance records for 20 agents on {simulation_date}")

def insert_fact_billing(client, simulation_date, n=30):
    """Insert billing records for a specific date"""
    
    records = []
    for _ in range(n):
        # Use the simulation date
        fact_date = simulation_date
        
        # Select a customer based on tariff distribution
        tariff_weights = list(DATA_VOLUMES['billing']['distribution'].items())
        tariff = random.choices(
            [t[0] for t in tariff_weights],
            weights=[t[1] for t in tariff_weights],
            k=1
        )[0]
        
        tariff_customers = [cid for cid, cdata in REFERENCE_DATA['customers'].items() 
                           if cdata['tariff_category'] == tariff]
        
        if tariff_customers:
            customer_id = random.choice(tariff_customers)
        else:
            customer_id = random.choice(list(REFERENCE_DATA['customers'].keys()))
            
        # Get customer profile and district
        profile = get_customer_profile(customer_id)
        customer_data = REFERENCE_DATA['customers'].get(customer_id, {})
        district = customer_data.get('district', random.choice(DISTRICTS))
        
        # Get facility in the same district
        if district in DISTRICT_FACILITY_MAP and DISTRICT_FACILITY_MAP[district]:
            facility_id = random.choice(DISTRICT_FACILITY_MAP[district])
        else:
            facility_id = random.randint(1, len(REFERENCE_DATA['facilities']))
        
        # Generate consumption based on tariff with seasonal factors
        seasonal_factor = get_seasonal_factor(fact_date)
        customer_factor = profile['consumption_variance']
        
        if tariff == "Domestic":
            base_consumption = random.uniform(5, 50)
            rate_per_m3 = 0.57
        elif tariff == "Commercial":
            base_consumption = random.uniform(50, 300)
            rate_per_m3 = 1.35
        elif tariff == "Industrial":
            base_consumption = random.uniform(100, 800)
            rate_per_m3 = 1.50
        elif tariff == "Government":
            base_consumption = random.uniform(100, 600)
            rate_per_m3 = 1.20
        else:  # Non-Profit
            base_consumption = random.uniform(20, 200)
            rate_per_m3 = 0.85
            
        # Apply seasonal and customer factors
        consumption = round(base_consumption * seasonal_factor * customer_factor, 2)
        
        # Apply discount if applicable
        discount = 0.0
        if tariff == "Non-Profit" and random.random() < 0.7:
            discount = round(random.uniform(0.05, 0.20), 2)  # 5-20% discount
        elif customer_data.get('customer_segment') == "Government" and random.random() < 0.5:
            discount = round(random.uniform(0.05, 0.15), 2)  # 5-15% discount
            
        billed_amount = round(consumption * rate_per_m3 * (1 - discount), 2)
        
        # Determine payment status based on customer reliability
        if profile['payment_reliability'] == 'reliable':
            status_weights = [0.9, 0.1, 0.0]  # 90% Paid, 10% Pending
        elif profile['payment_reliability'] == 'occasional_late':
            status_weights = [0.6, 0.3, 0.1]  # 60% Paid, 30% Pending, 10% Overdue
        else:  # problematic
            status_weights = [0.4, 0.4, 0.2]  # 40% Paid, 40% Pending, 20% Overdue
            
        status = random.choices(BILLING_STATUSES, weights=status_weights, k=1)[0]
        
        # Days late for payment
        days_late = 0
        if status == "Paid":
            if profile['payment_reliability'] == 'reliable':
                days_late = random.randint(0, 7)
            elif profile['payment_reliability'] == 'occasional_late':
                days_late = random.randint(3, 20)
            else:  # problematic
                days_late = random.randint(15, 45)
        
        # Payment method
        payment_methods = ["Online Banking", "Direct Debit", "Cash", "Credit Card", "Check", "Mobile Payment"]
        method_weights = [0.4, 0.2, 0.15, 0.15, 0.05, 0.05]
        payment_method = random.choices(payment_methods, weights=method_weights, k=1)[0]
        
        records.append({
            'billing_id': random.randint(1, 1000000),
            'date_id': fact_date,
            'customer_id': customer_id,
            'facility_id': facility_id,
            'billed_amount': billed_amount,
            'consumption_m3': consumption,
            'payment_status': status,
            'days_late': days_late,
            'discount_applied': discount,
            'payment_method': payment_method
        })
    
    client.execute("""
        INSERT INTO fact_billing (
            billing_id, date_id, customer_id, facility_id, 
            billed_amount, consumption_m3, payment_status,
            days_late, discount_applied, payment_method
        ) VALUES
    """, records)
    
    print(f"✅ Inserted {n} billing records for {simulation_date}")

def insert_fact_backlog(client, simulation_date, backlog_tickets):
    """Insert backlog entries for unresolved tickets"""
    
    # ✅ REVERTED: Use simple counter for backlog_id (no date prefix)
    result = client.execute(f"""
        SELECT COALESCE(MAX(backlog_id), 0) 
        FROM fact_backlog
    """)
    last_backlog_id = result[0][0]
    
    backlog_counter = last_backlog_id + 1
    
    backlog_data = []
    
    for backlog_item in backlog_tickets:
        ticket = backlog_item['ticket']
        customer_data = backlog_item['customer_data']
        is_escalated = backlog_item['is_escalated']
        
        days_in_backlog = random.randint(1, 30)
        
        if days_in_backlog <= 3:
            age_category = "0-3 days"
        elif days_in_backlog <= 7:
            age_category = "4-7 days"
        elif days_in_backlog <= 14:
            age_category = "8-14 days"
        else:
            age_category = "15+ days"
        
        if ticket['sla_compliance'] == 0 and ticket['reopened'] == 1:
            priority = "Critical"
        elif ticket['sla_compliance'] == 0:
            priority = "High"
        elif ticket['reopened'] == 1:
            priority = "Medium"
        else:
            priority = "Low"
        
        if is_escalated:
            backlog_status = random.choice(['Escalated', 'In Progress'])
        elif days_in_backlog > 14:
            backlog_status = random.choice(['Pending', 'On Hold'])
        else:
            backlog_status = random.choice(['Pending', 'In Progress'])
        
        if is_escalated:
            escalation_level = min(3, days_in_backlog // 7)
        else:
            escalation_level = 0
        
        sla_breached = 1 if (days_in_backlog > 7 or ticket['sla_compliance'] == 0) else 0
        
        if backlog_status == "Escalated":
            backlog_reason = random.choice([
                'Escalated to Technical Team',
                'Complex Investigation Required',
                'Senior Specialist Review Needed'
            ])
        elif ticket['issue_category'] in ["Billing Dispute", "Water Quality"]:
            backlog_reason = random.choice([
                'Pending Customer Response',
                'Document Verification Required',
                'Third Party Dependency'
            ])
        elif ticket['issue_category'] in ["Leakage", "No Water"]:
            backlog_reason = random.choice([
                'Parts Unavailable',
                'Field Team Scheduled',
                'Weather Delay'
            ])
        else:
            backlog_reason = random.choice([
                'Pending Customer Response',
                'Complex Investigation Required',
                'Third Party Dependency'
            ])
        
        backlog_entry = {
            'backlog_id': backlog_counter,
            'ticket_id': ticket['ticket_id'],
            'date_id': simulation_date,
            'customer_id': ticket['customer_id'],
            'facility_id': ticket['facility_id'],
            'department_id': ticket['department_id'],
            'agent_id': ticket['agent_id'],
            'issue_category': ticket['issue_category'],
            'issue_channel': ticket['issue_channel'],
            'priority': priority,
            'backlog_status': backlog_status,
            'days_in_backlog': days_in_backlog,
            'backlog_age_category': age_category,
            'assigned_date': simulation_date - timedelta(days=days_in_backlog),
            'expected_resolution_date': simulation_date + timedelta(days=random.randint(1, 7)),
            'last_updated': simulation_date,
            'sla_breached': sla_breached,
            'escalation_level': escalation_level,
            'reopened_count': ticket['reopened'],
            'customer_segment': customer_data.get('customer_segment', ''),
            'tariff_category': customer_data.get('tariff_category', ''),
            'district': customer_data.get('district', ''),
            'backlog_reason': backlog_reason,
            'resolution_notes': ''
        }
        
        backlog_counter += 1
        backlog_data.append(backlog_entry)
    
    if backlog_data:
        client.execute("""
            INSERT INTO fact_backlog (
                backlog_id, ticket_id, date_id, customer_id, facility_id, 
                department_id, agent_id, issue_category, issue_channel, priority,
                backlog_status, days_in_backlog, backlog_age_category,
                assigned_date, expected_resolution_date, last_updated,
                sla_breached, escalation_level, reopened_count,
                customer_segment, tariff_category, district,
                backlog_reason, resolution_notes
            ) VALUES
        """, backlog_data)
        
        print(f"✅ Inserted {len(backlog_data)} backlog cases for {simulation_date}")



# ============================================================================
# MAIN PRODUCER LOOP - DAILY SIMULATION
# ============================================================================

def main():
    """Main producer loop with daily simulation"""
    print("\n" + "="*80)
    print("💧 PBAPP - ClickHouse Data Warehouse Producer (Daily Simulation)")
    print("="*80 + "\n")
    
    client = connect_clickhouse()
    print(f"✅ Connected to ClickHouse: {CLICKHOUSE_DB}\n")
    
    # Build district-facility mapping
    build_district_facility_map()
    
    # One-time dimension table population
    print("="*80)
    print("PHASE 1: Populating Dimension Tables (one-time)")
    print("="*80)
    
    populate_dim_time(client)
    populate_dim_customer(client, n=1000)
    populate_dim_facility(client)
    populate_dim_department(client)
    populate_dim_agent(client)
    
    # Load reference data
    populate_reference_data(client)

    print("\n" + "="*80)
    print("PHASE 2: Daily Simulation (one day per 30 seconds)")
    print("="*80 + "\n")
    
    cycle = 0
    try:
        while True:
            cycle += 1
            simulation_date = get_next_simulation_date()
            ensure_dim_time_up_to(client, simulation_date)

            print(f"\n{'='*80}")
            print(f"📊 Cycle #{cycle} - Simulating day: {simulation_date.strftime('%Y-%m-%d')}")
            print(f"{'='*80}")
            
            # Generate tickets (80-120 per day)
            num_tickets = random.randint(80, 120)
            
            # Insert tickets (resolved → fact_tickets, unresolved → fact_backlog)
            agent_metrics = insert_fact_tickets(client, simulation_date, n=num_tickets)
            
            # Calculate totals
            total_backlog = sum(metrics['backlog_count'] for metrics in agent_metrics.values())
            total_resolved = sum(metrics['total_tickets'] for metrics in agent_metrics.values()) - total_backlog
            
            # Insert agent performance
            insert_fact_agent_performance(client, simulation_date, agent_metrics)
            
            # Insert billing
            num_billing = random.randint(30, 50)
            insert_fact_billing(client, simulation_date, n=num_billing)
            
            print(f"\n✅ Day simulation complete: {simulation_date}")
            print(f"   ↪ Total tickets generated: {num_tickets}")
            print(f"   ↪ Resolved tickets (fact_tickets): {total_resolved}")
            print(f"   ↪ Backlog cases (fact_backlog): {total_backlog}")
            print(f"   ↪ Agent performance records: {len(agent_metrics)}")
            print(f"   ↪ Billing records: {num_billing}")

            #RUN ML PREDICTIONS AFTER DATA GENERATION
            print(f"\n🤖 Running burnout prediction model...")
            try:
                ml_script = Path(__file__).parent.parent.parent / "ml_agent_burnout" / "predict_daily.py"
                exec(open(ml_script, encoding='utf-8').read())
                print(f"✅ Predictions generated successfully")
            except Exception as e:
                print(f"⚠️  Could not run predictions: {e}")

            if cycle == 3:
                print("\n🎉 Reached 200 cycles, stopping producer...")
                break
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping producer...")
    finally:
        client.disconnect()
        print("✅ Disconnected from ClickHouse\n")

if __name__ == "__main__":
    main()