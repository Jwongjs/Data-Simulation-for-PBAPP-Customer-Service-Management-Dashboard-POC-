from kafka import KafkaProducer
from faker import Faker
import json
import random
import time
import os
from datetime import datetime
from pathlib import Path

fake = Faker()

# Load configuration
CONFIG_PATH = Path(__file__).parent / "config.json"

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# Kafka connection details
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", config["kafka"]["bootstrap_servers"])
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", config["kafka"]["topic"])
INTERVAL_SECONDS = float(os.getenv("INTERVAL_SECONDS", config["simulation"]["interval_seconds"]))

# Penang Water Authority (PBA) - Real Locations
WATER_FACILITIES = config["facilities"]

# Sensor types with realistic ranges for Penang's tropical climate
SENSOR_CONFIGS = config["sensors"]


def generate_sensor_reading(sensor_type, facility_type):
    """Generate realistic sensor reading with occasional anomalies"""
    sensor_config = SENSOR_CONFIGS[sensor_type]
    
    # Determine if this reading should be anomalous
    is_anomaly = random.random() < sensor_config["anomaly_chance"]
    
    if is_anomaly:
        # Generate anomaly (either too high or too low)
        if random.random() < 0.5:
            # Low anomaly
            value = round(random.uniform(sensor_config["critical_low"], sensor_config["normal_range"][0]), 2)
            status = "warning" if value > sensor_config["critical_low"] else "critical"
        else:
            # High anomaly
            value = round(random.uniform(sensor_config["normal_range"][1], sensor_config["critical_high"]), 2)
            status = "warning" if value < sensor_config["critical_high"] else "critical"
    else:
        # Normal reading with slight variation
        value = round(random.uniform(sensor_config["normal_range"][0], sensor_config["normal_range"][1]), 2)
        status = "normal"
    
    # Adjust readings based on facility type
    if facility_type == "dam" and sensor_type == "water_level":
        # Dams typically have higher water levels
        value = value * 1.5
    elif facility_type == "pumping_station" and sensor_type == "pressure":
        # Pumping stations have higher pressure
        value = value * 1.3
    elif facility_type == "reservoir" and sensor_type == "water_level":
        # Reservoirs have moderate water levels
        value = value * 1.2
    
    return {
        "value": value,
        "unit": sensor_config["unit"],
        "status": status,
        "threshold_min": sensor_config["normal_range"][0],
        "threshold_max": sensor_config["normal_range"][1]
    }


def generate_water_event(facility_name, facility_info):
    """Generate a complete water IoT sensor event using priority_sensors"""
    
    # Use priority sensors from config
    priority_sensors = facility_info.get("priority_sensors", [])
    
    # Filter to only use sensors that exist in SENSOR_CONFIGS
    selected_sensors = [s for s in priority_sensors if s in SENSOR_CONFIGS]
    
    # If no priority sensors or they don't exist, fall back to random selection
    if not selected_sensors:
        num_sensors = random.randint(3, 6)
        selected_sensors = random.sample(list(SENSOR_CONFIGS.keys()), num_sensors)
    
    # Generate sensor readings
    sensor_readings = {}
    for sensor_type in selected_sensors:
        sensor_readings[sensor_type] = generate_sensor_reading(
            sensor_type, 
            facility_info["type"]
        )
    
    # Determine overall facility status
    statuses = [reading["status"] for reading in sensor_readings.values()]
    if "critical" in statuses:
        overall_status = "critical"
        alert_level = 3
    elif "warning" in statuses:
        overall_status = "warning"
        alert_level = 2
    else:
        overall_status = "operational"
        alert_level = 0
    
    # Create the event payload with supply chain info
    event = {
        "event_id": fake.uuid4(),
        "timestamp": datetime.now().isoformat(),
        "facility": {
            "name": facility_name,
            "type": facility_info["type"],
            "district": facility_info["district"],
            "capacity_mld": facility_info["capacity_mld"],
            "location": facility_info["location"],
            "supply_chain": facility_info.get("supply_chain"),
            "upstream_facility": facility_info.get("upstream_facility"),
            "downstream_facility": facility_info.get("downstream_facility")
        },
        "sensors": sensor_readings,
        "operational_metrics": {
            "status": overall_status,
            "alert_level": alert_level,
            "active_sensors": len(sensor_readings),
            "uptime_percentage": round(random.uniform(95.0, 99.9), 2)
        },
        "metadata": {
            "device_id": f"PBA-{facility_info['type'].upper()}-{hash(facility_name) % 10000:04d}",
            "firmware_version": f"{random.randint(1,3)}.{random.randint(0,9)}.{random.randint(0,20)}",
            "battery_level": round(random.uniform(60, 100), 1) if facility_info["type"] in ["pumping_station", "distribution_center"] else None,
            "network_signal": round(random.uniform(70, 100), 1)
        }
    }
    
    return event


def create_kafka_producer():
    """Create and configure Kafka producer with retry logic"""
    max_retries = 10
    producer_config = config["kafka"]["producer_config"]
    
    for attempt in range(max_retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks=producer_config["acks"],
                retries=producer_config["retries"],
                max_in_flight_requests_per_connection=producer_config["max_in_flight_requests_per_connection"]
            )
            print(f"✅ Connected to Kafka at {KAFKA_BOOTSTRAP_SERVERS}")
            return producer
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⏳ Waiting for Kafka... (attempt {attempt + 1}/{max_retries})")
                time.sleep(5)
            else:
                raise e


def main():
    """Main producer loop - generates data for ALL facilities simultaneously"""
    producer = create_kafka_producer()
    
    print("\n" + "="*80)
    print("💧 Penang Water Authority - IoT Data Stream Producer")
    print("="*80)
    print(f"📍 Monitoring {len(WATER_FACILITIES)} water facilities across Penang")
    print(f"📊 Active sensor types: {len(SENSOR_CONFIGS)}")
    print(f"📡 Kafka topic: {KAFKA_TOPIC}")
    print(f"⏱️  Interval: {INTERVAL_SECONDS} seconds (ALL facilities simultaneously)")
    print("="*80 + "\n")
    
    # Group facilities by supply chain
    supply_chains = {}
    for facility, info in WATER_FACILITIES.items():
        chain = info.get("supply_chain", "Unknown")
        if chain not in supply_chains:
            supply_chains[chain] = []
        supply_chains[chain].append(facility)
    
    print("🌊 Water Supply Chains:")
    for chain, facilities in supply_chains.items():
        print(f"\n   {chain}:")
        for facility in facilities:
            info = WATER_FACILITIES[facility]
            print(f"      → {facility} ({info['type']}) - {info['district']}")
    
    print("\n" + "="*80)
    print("[Water IoT Producer] Running... Press Ctrl+C to stop.")
    print("="*80 + "\n")
    
    event_count = 0
    anomaly_count = 0
    cycle_count = 0
    
    try:
        while True:
            cycle_count += 1
            cycle_start = time.time()
            
            print(f"\n{'='*80}")
            print(f"📊 Cycle #{cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}")
            
            # Generate and send events for ALL facilities in this cycle
            cycle_anomalies = 0
            
            for facility_name, facility_info in WATER_FACILITIES.items():
                # Generate sensor event
                event = generate_water_event(facility_name, facility_info)
                
                # Send to Kafka
                producer.send(KAFKA_TOPIC, value=event)
                
                # Track statistics
                event_count += 1
                status = event["operational_metrics"]["status"]
                
                if status in ["warning", "critical"]:
                    anomaly_count += 1
                    cycle_anomalies += 1
                    
                    # Show detailed info for anomalies
                    anomaly_sensors = [
                        f"{sensor}: {data['value']}{data['unit']}" 
                        for sensor, data in event['sensors'].items() 
                        if data['status'] != 'normal'
                    ]
                    print(f"   ⚠️  {status.upper()}: {facility_name}")
                    print(f"      Chain: {facility_info.get('supply_chain')}")
                    print(f"      Anomalies: {', '.join(anomaly_sensors)}")
                else:
                    # Compact display for normal readings
                    print(f"   ✓ {facility_name} ({facility_info['type']}) - {len(event['sensors'])} sensors OK")
            
            # Cycle summary
            cycle_duration = time.time() - cycle_start
            print(f"\n{'─'*80}")
            print(f"Cycle Summary: {len(WATER_FACILITIES)} facilities processed in {cycle_duration:.2f}s")
            if cycle_anomalies > 0:
                print(f"⚠️  {cycle_anomalies} facilities with anomalies detected this cycle")
            print(f"{'─'*80}")
            
            # Sleep for the remaining interval
            sleep_time = max(0, INTERVAL_SECONDS - cycle_duration)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n" + "="*80)
        print("🛑 Stopping Water IoT Producer...")
        print("="*80)
        print(f"📈 Final Statistics:")
        print(f"   Total cycles: {cycle_count}")
        print(f"   Total events generated: {event_count}")
        print(f"   Events per cycle: {len(WATER_FACILITIES)}")
        print(f"   Anomalies detected: {anomaly_count} ({anomaly_count/event_count*100:.2f}%)")
        print(f"   Average events/second: {event_count/(cycle_count*INTERVAL_SECONDS):.2f}")
        print("="*80)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        producer.close()
        print("✅ [Water IoT Producer] Connection closed\n")


if __name__ == "__main__":
    main()