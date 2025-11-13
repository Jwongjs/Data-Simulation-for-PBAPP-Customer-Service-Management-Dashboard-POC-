"""
Simple script to run ML training pipeline in sequence
Usage: python run_training.py
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name):
    """Run a Python script and show output"""
    print("\n" + "="*80)
    print(f"▶️  Running {script_name}...")
    print("="*80 + "\n")
    
    script_path = Path(__file__).parent / "model prep" / script_name
    
    # Run the script and show output in real-time
    result = subprocess.run([sys.executable, str(script_path)])
    
    if result.returncode != 0:
        print(f"\n❌ Error running {script_name}")
        return False
    
    return True

def main():
    print("\n" + "="*80)
    print("🤖 ML AGENT BURNOUT - TRAINING PIPELINE")
    print("="*80)
    
    # Step 1: Export data from ClickHouse
    if not run_script("export_data.py"):
        print("\n❌ Failed at Step 1: Data Export")
        sys.exit(1)
    
    # Step 2: Prepare features
    if not run_script("prepare_features.py"):
        print("\n❌ Failed at Step 2: Feature Preparation")
        sys.exit(1)
    
    # Step 3: Train model
    if not run_script("train_model.py"):
        print("\n❌ Failed at Step 3: Model Training")
        sys.exit(1)
    
    # Success!
    print("\n" + "="*80)
    print("✅ ML TRAINING PIPELINE COMPLETED!")
    print("="*80)
    print("\n📁 Model saved in: ml_agent_burnout/models/")
    print("💡 Next: Run predict_daily.py to generate predictions")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()