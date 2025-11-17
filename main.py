# main.py
"""
NeuroTuner Main Pipeline
"""

import json
from tuner import run_tuning
from trainer import run_training
from monitor import get_hardware_state
from policy import adapt_config


def main():
    print("\n=== 🧠 NEUROTUNER — AI SELF-OPTIMIZER ===")

    # 1. Hyperparameter tuning
    print("\n🔎 Running Hyperparameter Search...")
    best_config = run_tuning(n_trials=3)  # small for fast demo
    print("\n🏆 Best Hyperparameters Found:")
    print(best_config)

    # Add default training extras
    best_config["epochs"] = 1
    best_config["device"] = "cpu"

    # 2. Train with best config
    print("\n📚 Training final model...")
    metrics = run_training(best_config)
    print("\n📊 Metrics:", metrics)

    # 3. Hardware monitoring
    print("\n⚙️ Checking hardware usage...")
    hw = get_hardware_state()
    print(hw)

    # 4. Apply adaptive policy
    print("\n🔁 Applying adaptive policy...")
    new_config = adapt_config(best_config, metrics, hw)
    print("\n✨ Updated Config:", new_config)

    # 5. Save output
    output = {
        "initial_best_config": best_config,
        "final_metrics": metrics,
        "hardware_state": hw,
        "adapted_config": new_config,
    }

    with open("results.json", "w") as f:
        json.dump(output, f, indent=4)

    print("\n📦 Final Output JSON written to results.json")
    print("\n🚀 DONE!")


if __name__ == "__main__":
    main()
