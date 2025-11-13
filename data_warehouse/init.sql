-- Switch to database
CREATE DATABASE IF NOT EXISTS pbapp_poc;
USE pbapp_poc;

--------------------------------------
-- DIMENSION TABLES
--------------------------------------

-- Time Dimension
CREATE TABLE IF NOT EXISTS dim_time (
    date_id Date,
    full_date Date,
    day UInt8,
    month UInt8,
    quarter UInt8,
    year UInt16
)
ENGINE = MergeTree()
ORDER BY (date_id);

-- Customer Dimension
CREATE TABLE IF NOT EXISTS dim_customer (
    customer_id UInt32,
    account_no String,
    customer_name String,
    district String,
    tariff_category String,
    customer_segment String DEFAULT 'Residential'
)
ENGINE = MergeTree()
ORDER BY (customer_id);

-- Facility Dimension
CREATE TABLE IF NOT EXISTS dim_facility (
    facility_id UInt32,
    facility_name String,
    facility_type String,
    district String,
    latitude Float64,
    longitude Float64
)
ENGINE = MergeTree()
ORDER BY (facility_id);

-- Department Dimension
CREATE TABLE IF NOT EXISTS dim_department (
    department_id UInt32,
    department_name String,
    manager_name String,
    team_size UInt16
)
ENGINE = MergeTree()
ORDER BY (department_id);

-- Agent Dimension 
CREATE TABLE IF NOT EXISTS dim_agent (
    agent_id UInt32,
    agent_name String,
    department_id UInt32,
    role_title String,
    hire_date Date,
    status String DEFAULT 'Active'
)
ENGINE = MergeTree()
ORDER BY (agent_id);

--------------------------------------
-- FACT TABLES
--------------------------------------

-- Customer Service Performance
CREATE TABLE IF NOT EXISTS fact_tickets (
    ticket_id UInt32,
    date_id Date,
    customer_id UInt32,
    facility_id UInt32,
    department_id UInt32,
    agent_id UInt32,
    issue_category String,
    issue_channel String,
    resolution_time_hrs Float32,
    reopened UInt8,
    resolved_in_first_contact UInt8,
    satisfaction_score Float32,  -- CSAT
    nps_score Int8,              -- Net Promoter Score
    effort_score Float32,        -- Customer Effort Score
    sla_compliance UInt8,        -- 1 if met, 0 if breached
    response_time_mins UInt16
)
ENGINE = MergeTree()
ORDER BY (date_id, ticket_id);

️-- Operational Efficiency
CREATE TABLE IF NOT EXISTS fact_agent_performance (
    performance_id UInt32,
    date_id Date,
    agent_id UInt32,
    department_id UInt32,
    total_tickets_handled UInt16,
    avg_handle_time_mins Float32,
    escalation_rate Float32,
    backlog UInt16,
    reopened_tickets UInt16,
    utilization_rate Float32,
    sla_breaches UInt16,
    resolution_rate Float32
)
ENGINE = MergeTree()
ORDER BY (date_id, department_id);

--️ Billing & Revenue
CREATE TABLE IF NOT EXISTS fact_billing (
    billing_id UInt32,
    date_id Date,
    customer_id UInt32,
    facility_id UInt32,
    billed_amount Float64,
    consumption_m3 Float64,
    payment_status String,
    days_late UInt8,
    discount_applied Float32,
    payment_method String
)
ENGINE = MergeTree()
ORDER BY (date_id, customer_id);

--------------------------------------
-- BACKLOG MANAGEMENT TABLE
--------------------------------------
CREATE TABLE IF NOT EXISTS fact_backlog (
    backlog_id UInt32,
    ticket_id UInt32,
    date_id Date,
    customer_id UInt32,
    facility_id UInt32,
    department_id UInt32,
    agent_id UInt32,
    
    -- Ticket Information
    issue_category String,
    issue_channel String,
    priority String,  -- Low, Medium, High, Critical
    
    -- Backlog Status
    backlog_status String DEFAULT 'Pending',  -- Pending, In Progress, Escalated, On Hold
    days_in_backlog UInt16,
    backlog_age_category String,  -- 0-3 days, 4-7 days, 8-14 days, 15+ days
    
    -- Assignment & Timeline
    assigned_date Date,
    expected_resolution_date Date,
    last_updated Date DEFAULT today(),
    
    -- Metrics
    sla_breached UInt8 DEFAULT 0,  -- 1 if SLA breached
    escalation_level UInt8 DEFAULT 0,  -- 0 = None, 1 = L1, 2 = L2, 3 = L3
    reopened_count UInt8 DEFAULT 0,
    
    -- Additional Context
    customer_segment String,
    tariff_category String,
    district String,
    
    -- Notes
    backlog_reason String DEFAULT '',
    resolution_notes String DEFAULT ''
)
ENGINE = MergeTree()
ORDER BY (date_id, backlog_status, priority, ticket_id);

--------------------------------------
-- Predictive Table (optional advanced analytics)
--------------------------------------
CREATE TABLE IF NOT EXISTS fact_agent_burnout_prediction (
    prediction_id UInt32,
    date_id Date,
    agent_id UInt32,
    
    -- Risk Scores
    burnout_risk_score Float32,  -- Probability of Critical class (0.0 - 1.0)
    productivity_index Float32,  -- Agent Productivity Index (0.0 - 1.0)
    risk_category String,  -- Low, Medium, High, Critical
    risk_category_encoded UInt8,  -- 0=Low, 1=Medium, 2=High, 3=Critical
    
    -- Current State Snapshot
    current_utilization Float32,
    current_backlog UInt16,
    current_escalation_rate Float32,
    current_sla_breaches UInt8,
    current_stress_index Float32,
    current_handle_time_mins Float32,
    current_total_tickets UInt16,
    current_reopened_tickets UInt16,
    
    -- Engineered Features (7-day rolling)
    avg_utilization_7d Float32,
    avg_handle_time_7d Float32,
    avg_backlog_7d Float32,
    
    -- Engineered Features (30-day rolling)
    avg_utilization_30d Float32,
    avg_handle_time_30d Float32,
    avg_backlog_30d Float32,
    
    -- Trend Indicators
    utilization_trend Float32,  -- Change in utilization (last 7 days)
    escalation_trend Float32,  -- Change in escalation rate (last 7 days)
    
    -- Derived Flags
    performance_declining UInt8,  -- 1 if handle time increasing
    persistent_backlog UInt8,  -- 1 if backlog > 10
    workload_velocity Float32,  -- Change in backlog (last 7 days)
    
    -- Recommendations
    recommended_action String,
    workload_reduction_pct UInt8,
    
    -- Model Metadata
    model_version String DEFAULT 'v1.0_xgboost',
    confidence_score Float32,
)
ENGINE = MergeTree()
ORDER BY (date_id, agent_id, prediction_id);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_risk_category ON fact_agent_burnout_prediction(risk_category) TYPE set(0);
CREATE INDEX IF NOT EXISTS idx_risk_encoded ON fact_agent_burnout_prediction(risk_category_encoded) TYPE set(0);
CREATE INDEX IF NOT EXISTS idx_date ON fact_agent_burnout_prediction(date_id) TYPE minmax;

--------------------------------------
-- Executive Summary Measurement Table
--------------------------------------
CREATE TABLE IF NOT EXISTS meas_executive_summary (
    measure_id UInt32,
    measure_name String,
    measure_value Float64,
    measure_date Date DEFAULT today()
)
ENGINE = MergeTree()
ORDER BY (measure_id);

--------------------------------------
-- Customer Service Measurement Table
--------------------------------------
CREATE TABLE IF NOT EXISTS meas_customer_service (
    measure_id UInt32,
    measure_name String,
    measure_value Float64,
    measure_date Date DEFAULT today(),
    district String DEFAULT '',
    issue_category String DEFAULT ''
)
ENGINE = MergeTree()
ORDER BY (measure_id);

--------------------------------------
-- Agent Performance Measurement Table
--------------------------------------
CREATE TABLE IF NOT EXISTS meas_agent_performance (
    measure_id UInt32,
    agent_id UInt32,
    measure_name String,
    measure_value Float64,
    measure_date Date DEFAULT today()
)
ENGINE = MergeTree()
ORDER BY (measure_id);

--------------------------------------
-- Optional: Billing & Revenue Measurement Table
--------------------------------------
CREATE TABLE IF NOT EXISTS meas_billing (
    measure_id UInt32,
    measure_name String,
    measure_value Float64,
    measure_date Date DEFAULT today(),
    customer_segment String DEFAULT ''
)
ENGINE = MergeTree()
ORDER BY (measure_id);

--------------------------------------
-- KPI Drill-Down Measurement Tables (Placeholders for Power BI)
--------------------------------------

--️ Resolution Rate Drill-Down by Department
CREATE TABLE IF NOT EXISTS meas_resolution_rate_drilldown (
    measure_id UInt32,
    department_id UInt32,
    department_name String,
    measure_value Decimal(18, 4),  -- Changed from Float64 to Decimal for Power BI compatibility
    measure_date Date DEFAULT today(),
    parent_kpi String DEFAULT 'Overall Resolution Rate %'
)
ENGINE = MergeTree()
ORDER BY (measure_date, department_id, measure_id);

-- CSAT Score Drill-Down by Department
CREATE TABLE IF NOT EXISTS meas_csat_drilldown (
    measure_id UInt32,
    department_id UInt32,
    department_name String,
    measure_value Decimal(18, 4),  -- Changed from Float64 to Decimal for Power BI compatibility
    measure_date Date DEFAULT today(),
    parent_kpi String DEFAULT 'Overall CSAT Score'
)
ENGINE = MergeTree()
ORDER BY (measure_date, department_id, measure_id);

-- SLA Compliance Drill-Down by Department
CREATE TABLE IF NOT EXISTS meas_sla_compliance_drilldown (
    measure_id UInt32,
    department_id UInt32,
    department_name String,
    measure_value Decimal(18, 4),  -- Changed from Float64 to Decimal for Power BI compatibility
    measure_date Date DEFAULT today(),
    parent_kpi String DEFAULT 'Overall SLA Compliance %'
)
ENGINE = MergeTree()
ORDER BY (measure_date, department_id, measure_id);

--️ Avg Resolution Time Drill-Down by Department
CREATE TABLE IF NOT EXISTS meas_avg_resolution_time_drilldown (
    measure_id UInt32,
    department_id UInt32,
    department_name String,
    measure_value Decimal(18, 4),  -- Changed from Float64 to Decimal for Power BI compatibility
    measure_date Date DEFAULT today(),
    parent_kpi String DEFAULT 'Overall Avg Resolution Time'
)
ENGINE = MergeTree()
ORDER BY (measure_date, department_id, measure_id);

--------------------------------------
-- Optional: Predictive Insights Measurement Table
--------------------------------------
CREATE TABLE IF NOT EXISTS meas_predictive (
    measure_id UInt32,
    customer_id UInt32,
    measure_name String,
    measure_value Float64,
    prediction_date Date DEFAULT today()
)
ENGINE = MergeTree()
ORDER BY (measure_id);
