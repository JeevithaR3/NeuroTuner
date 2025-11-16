# main.py
"""
NeuroTuner Main Pipeline
Runs:
1. Hyperparameter tuning (Optuna)
2. Model training for best config
3. Hardware monitoring
4. Adaptive policy update
Outputs JSON results for judges demo
"""

import json
from tuner import run_tuning
from trainer import train_model
from monitor import get_hardware_state
from policy import adapt_config

def main():
    print("\n=== 🧠 NEUROTUNER — AI SELF-OPTIMIZER ===")

    # 1. Run Optuna hyperparameter tuning
    print("\n🔎 Running Hyperparameter Search...")
    best_config = run_tuning(n_trials=5)
    print("\n🏆 Best Hyperparameters Found:")
    print(best_config)

    # 2. Train model once with best config
    print("\n📚 Training Final Model With Best Config...")
    metrics = train_model(best_config)
    print("\n📊 Final Training Metrics:")
    print(metrics)

    # 3. Check hardware status (CPU/GPU load)
    print("\n⚙️ Checking Hardware Usage...")
    hw = get_hardware_state()
    print(hw)

    # 4. Adapt config using policy logic
    print("\n🔁 Applying Adaptive Policy...")
    new_config = adapt_config(best_config, metrics, hw)
    print("\n✨ Updated Config After Policy Adjustment:")
    print(new_config)

    # 5. Prepare final output for judges (JSON)
    output = {
        "initial_best_config": best_config,
        "final_metrics": metrics,
        "hardware_state": hw,
        "adapted_config": new_config
    }

    print("\n📦 Final Output JSON:")
    print(json.dumps(output, indent=4))

    # Save results
    with open("results.json", "w") as f:
        json.dump(output, f, indent=4)

    print("\n💾 Results saved to results.json")
    print("\n🚀 NeuroTuner pipeline complete!")


if __name__ == "__main__":
    main()
