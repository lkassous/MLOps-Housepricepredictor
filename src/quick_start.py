"""
Quick Start Script
Run this to quickly test the entire pipeline
"""

import subprocess
import sys

def run_command(command, description):
    """Run a shell command and print the result"""
    print(f"\n{'='*70}")
    print(f"🚀 {description}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=False
        )
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error: {e}")
        return False

def main():
    """Quick start the entire pipeline"""
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║         HOUSE PRICES MLOPS - QUICK START                       ║
    ║         Automated MLflow Pipeline Execution                    ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    steps = [
        ("python src/data_preparation.py", "Step 1: Data Preparation"),
        ("python src/train_models.py", "Step 2: Model Training & MLflow Tracking"),
        ("python src/register_model.py", "Step 3: Register Best Model"),
    ]
    
    for command, description in steps:
        success = run_command(command, description)
        if not success:
            print("\n❌ Pipeline failed. Please check the errors above.")
            sys.exit(1)
    
    print(f"\n{'='*70}")
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    print("\n📊 Next Steps:")
    print("  1. Open MLflow UI: http://localhost:5000")
    print("  2. Compare model performances")
    print("  3. Check the registered model in Model Registry")
    print("  4. Deploy the production model\n")

if __name__ == "__main__":
    main()
